import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Définition des constantes
g = 9.81  # Accélération due à la gravité (m/s^2)
L1 = 1.0  # Longueur du premier pendule (m)
L2 = 1.0  # Longueur du deuxième pendule (m)
m1 = 1.0  # Masse du premier pendule (kg)
m2 = 1.0  # Masse du deuxième pendule (kg)

# Fonction pour calculer les dérivées des angles et des vitesses angulaires
def derivatives(t, state):
    theta1, omega1, theta2, omega2 = state
    
    # Équations de mouvement du double pendule
    delta_theta = theta2 - theta1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta_theta) ** 2
    den2 = (L2 / L1) * den1
    
    dtheta1_dt = omega1
    domega1_dt = (m2 * L1 * omega1 ** 2 * np.sin(delta_theta) * np.cos(delta_theta)
                  + m2 * g * np.sin(theta2) * np.cos(delta_theta)
                  + m2 * L2 * omega2 ** 2 * np.sin(delta_theta)
                  - (m1 + m2) * g * np.sin(theta1)) / den1
    
    dtheta2_dt = omega2
    domega2_dt = (-L2 / L1 * omega1 ** 2 * np.sin(delta_theta) * np.cos(delta_theta)
                  + (m1 + m2) * g * np.sin(theta1) * np.cos(delta_theta)
                  - (m1 + m2) * L1 * omega1 ** 2 * np.sin(delta_theta)
                  - (m1 + m2) * g * np.sin(theta2)) / den2
    
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

# Conditions initiales
initial_state = [np.pi / 4, 0, np.pi / 2, 0]  # [theta1, omega1, theta2, omega2]

# Intervalle de temps
t_span = (0, 20)  # Début et fin de la simulation
t_points = np.linspace(*t_span, 1000)

# Résolution des équations différentielles
sol = solve_ivp(derivatives, t_span, initial_state, t_eval=t_points)

# Extraction des résultats
theta1 = sol.y[0]
theta2 = sol.y[2]

# Conversion en coordonnées cartésiennes pour la visualisation
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# Affichage de la trajectoire du double pendule
plt.figure(figsize=(8, 8))
plt.plot(x1, y1, label='Pendule 1')
plt.plot(x2, y2, label='Pendule 2')
plt.xlabel('Position en x')
plt.ylabel('Position en y')
plt.title('Mouvement d\'un double pendule')
plt.legend()
plt.grid()
plt.show()
