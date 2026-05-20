import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# --- 1. Configuration des paramètres (Unités SI) ---
# Ces paramètres définissent les propriétés biophysiques du neurone [1, 2].
params = {
        "tau_m": 0.02,      # Constante de temps membranaire (s) : définit la vitesse de la "fuite" [1, 3].
        "el": -0.06,        # Potentiel de repos (V) : la tension de base du neurone sans stimulus [1, 4].
        "vr": -0.07,        # Potentiel de réinitialisation (V) : valeur après un spike [1, 5].
        "vth": -0.05,       # Seuil de décharge (V) : niveau de tension déclenchant un potentiel d'action [1, 6].
        "r": 100e6,         # Résistance membranaire (Ohm) : opposition au passage du courant ionique [1, 7].
        "dt": 0.0001,       # Pas de temps (s) : discrétisation pour la simulation numérique [2, 8].
        "t_ref": 0.005      # Période réfractaire (s) : temps mort durant lequel le neurone ne peut plus tirer [9, 10].
}

# --- 2. Fonction de mise à jour (Un pas de temps) ---
# Cette fonction définit la dynamique du "modèle comment" (How model) [11, 12].
def lif_step(state, i_input):
    """
    Simule une étape de temps discrète pour un neurone LIF.
    L'utilisation de jnp.where permet la compilation XLA efficace sur JAX.
    """
    v, ref_time = state # État actuel : potentiel v et temps réfractaire restant

    # Extraction des constantes pour la formule d'Euler [13, 14].
    dt, tau, el, vr, vth, r = params["dt"], params["tau_m"], params["el"], params["vr"], params["vth"], params["r"]

    # 2.1 Gestion de la période réfractaire
    # On vérifie si le neurone est encore dans son temps mort biologique [15, 16].
    is_refractory = ref_time > 0

    # 2.2 Calcul du potentiel de membrane
    # Si réfractaire : le potentiel reste au niveau de reset (vr).
    # Sinon : on applique l'intégration d'Euler : V(n+1) = V(n) + (dt/tau)*(EL - V(n) + R*I) [14, 17].
    v_new = jnp.where(
            is_refractory,
            vr,
            v + (dt / tau) * (el - v + r * i_input)
    )

    # 2.3 Condition de décharge (Fire)
    # Le neurone émet un spike si le seuil (vth) est atteint [7, 18].
    spiked = v_new >= vth

    # 2.4 Réinitialisation et mise à jour de l'horloge réfractaire
    # Si spike : retour immédiat au potentiel vr [5, 19].
    v_final = jnp.where(spiked, vr, v_new)

    # Mise à jour du temps réfractaire : on le réinitialise si spike, sinon on le décrémente [9].
    new_ref_time = jnp.where(spiked, params["t_ref"], jnp.maximum(0, ref_time - dt))

    # Retourne le nouvel état et les données à enregistrer (potentiel et événement de spike)
    return (v_final, new_ref_time), (v_final, spiked)

# --- 3. Simulation avec jax.lax.scan ---
def run_simulation(i_array):
    """
    Utilise jax.lax.scan pour itérer la fonction lif_step sur tout le vecteur de courant.
    C'est beaucoup plus rapide qu'une boucle 'for' Python classique [20, 21].
    """
    init_state = (params["el"], 0.0) # Initialisation au repos [22].
    _, (v_history, spike_history) = jax.lax.scan(lif_step, init_state, i_array)
    return v_history, spike_history

# --- 4. Préparation des données et exécution ---
t_max = 0.15 # Durée totale de 150 ms [2].
time = jnp.arange(0, t_max, params["dt"])

# 4.1 Génération d'un courant d'entrée sinusoïdal
# Ce type d'entrée simule un stimulus périodique ou rythmique [23, 24].
i_mean = 2.5e-10 # Courant moyen standard de 250 pA [2].
i_input = i_mean * (1 + jnp.sin((time * 2 * jnp.pi) / 0.01))

# Exécution de la simulation compilée
v_trace, spike_trace = run_simulation(i_input)

# --- 5. Visualisation ---
# On affiche la dynamique pour relier la théorie au comportement réel [21, 25].
plt.figure(figsize=(10, 6))

# Graphique du potentiel de membrane
plt.subplot(2, 1, 1)
plt.plot(time, v_trace, label="Potentiel de membrane (V)")
plt.axhline(params["vth"], color='r', linestyle='--', label="Seuil (Vth)")
plt.title("Simulation LIF avec JAX : Intégration, Fuite et Réinitialisation")
plt.ylabel("Voltage (V)")
plt.legend()

# Graphique du courant d'entrée
plt.subplot(2, 1, 2)
plt.plot(time, i_input, color='g', label="Courant d'entrée (A)")
plt.ylabel("Courant (A)")
plt.xlabel("Temps (s)")
plt.legend()

plt.tight_layout()
plt.show()