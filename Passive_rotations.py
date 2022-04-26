
import biorbd
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import bioviz

"""
Ce code a été écrit pout faire les simulations pour Antoine, 
mais il pourra être modifié pour apporter d'autres infos utiles avec d'autres anthropo comme base.
Attention, ici on considere des mouvements des joints à vitesse constante (Qddot_J = 0).
C'est surement overkill d'utiliser un KR4 à 100 noeuds pour des mouvements à vitesse constante, mais bon
"""

def dynamics_root(m, X, Qddot_J):
    Q = X[:m.nbQ()]
    Qdot = X[m.nbQ():]
    Qddot = np.hstack((np.zeros((6,)), Qddot_J))
    NLEffects = m.InverseDynamics(Q, Qdot, Qddot).to_array()
    mass_matrix = m.massMatrix(Q).to_array()
    Qddot_R = np.linalg.inv(mass_matrix[:6, :6]) @ NLEffects[:6]
    Xdot = np.hstack((Qdot, Qddot_R, Qddot_J))
    return Xdot

def runge_kutta_4(m, x0, t, N, n_step):
    h = t / (N-1) / n_step
    x = np.zeros((x0.shape[0], n_step + 1))
    x[:, 0] = x0
    Qddot_J = np.zeros((m.nbQ()-m.nbRoot(),)) #### mouvement continu, sans acceleration
    for i in range(1, n_step + 1):
        k1 = dynamics_root(m, x[:, i - 1], Qddot_J)
        k2 = dynamics_root(m, x[:, i - 1] + h / 2 * k1, Qddot_J)
        k3 = dynamics_root(m, x[:, i - 1] + h / 2 * k2, Qddot_J)
        k4 = dynamics_root(m, x[:, i - 1] + h * k3, Qddot_J)
        x[:, i] = np.hstack((x[:, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)))
    return x

def integrate_plots(m, X0, t, N, n_step):
    fig, axs = plt.subplots(2, 3)
    axs = np.ravel(axs)
    X_tous = X0
    for i in range(int(N/2)-1): #N - 1):
        X = runge_kutta_4(m, X0, t[-1], N, n_step)
        plot_index = 0
        for k in [3, 4, 5, 15, 24]:
            axs[plot_index].plot(np.arange(i * n_step, (i + 1) * n_step + 1), np.reshape(X[k, :], n_step + 1), ':',
                        label=f'{m.nameDof()[k].to_string()}')
            plot_index += 1
        X_tous = np.vstack((X_tous, X[:, -1]))
        X0 = X[:, -1]

    X0[m_JeCh.nbQ() + 15] = 0  # brasD
    X0[m_JeCh.nbQ() + 24] = 0  # brasG
    for i in range(int(N / 2)-1, N-1):  # N - 1):
        X = runge_kutta_4(m, X0, t[-1], N, n_step)
        plot_index = 0
        for k in [3, 4, 5, 15, 24]:
            axs[plot_index].plot(np.arange(i * n_step, (i + 1) * n_step + 1), np.reshape(X[k, :], n_step + 1), ':',
                                 label=f'{m.nameDof()[k].to_string()}')
            plot_index += 1
        X_tous = np.vstack((X_tous, X[:, -1]))
        X0 = X[:, -1]

    fig.suptitle('Salto bras en haut (integrated with single shooting)')
    plt.show()
    return X_tous

###################################################################################
N = 100
n_step = 5
model_path_JeCh = "/home/user/Documents/Programmation/Eve/AnthropoImpactOnTech/Models/JeCh_201.bioMod"
model_path_SaMi = "/home/user/Documents/Programmation/Eve/AnthropoImpactOnTech/Models/SaMi.bioMod"
m_JeCh = biorbd.Model(model_path_JeCh)
m_SaMi = biorbd.Model(model_path_SaMi)
m_JeCh.setGravity(np.array((0, 0, 0)))
m_SaMi.setGravity(np.array((0, 0, 0)))

# b = bioviz.Viz(model_path_SaMi)
# b = bioviz.Viz(model_path_JeCh)
# b.exec()

t = np.linspace(0, 1, N)

# Salto bras en haut JeCh
X0 = np.zeros((m_JeCh.nbQ()*2, ))
X0[15] = -np.pi
X0[24] = np.pi
X0[m_JeCh.nbQ()+3] = 2 * np.pi #Salto
X_tous = integrate_plots(m_JeCh, X0, t, N, n_step)
print("Salto bras en haut JeCh")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}\n\n")

b = bioviz.Viz(model_path_JeCh)
b.load_movement(X_tous[:, :m_JeCh.nbQ()].T)
b.exec()

# Salto bras qui descendent JeCh
X0 = np.zeros((m_JeCh.nbQ()*2, ))
X0[15] = -np.pi
X0[24] = np.pi
X0[m_JeCh.nbQ()+3] = 2*np.pi #Salto
X0[m_JeCh.nbQ()+15] = 2*np.pi #brasD
X0[m_JeCh.nbQ()+24] = -2*np.pi #brasG
X_tous = integrate_plots(m_JeCh, X0, t, N, n_step)
print("Salto bras qui descendent JeCh")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}")

b = bioviz.Viz(model_path_JeCh)
b.load_movement(X_tous[:, :m_JeCh.nbQ()].T)
b.exec()

# Salto un bras qui descend JeCh
X0 = np.zeros((m_JeCh.nbQ()*2, ))
X0[15] = -np.pi
X0[24] = np.pi
X0[m_JeCh.nbQ()+3] = 2*np.pi #Salto
X0[m_JeCh.nbQ()+24] = -2*np.pi #brasG
X_tous = integrate_plots(m_JeCh, X0, t, N, n_step)
print("Salto un bras qui descend JeCh")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}")

b = bioviz.Viz(model_path_JeCh)
b.load_movement(X_tous[:, :m_JeCh.nbQ()].T)
b.exec()

# Salto bras en haut SaMi
X0 = np.zeros((m_SaMi.nbQ()*2, ))
X0[15] = -np.pi
X0[24] = np.pi
X0[m_SaMi.nbQ()+3] = 2*np.pi #Salto
X_tous = integrate_plots(m_SaMi, X0, t, N, n_step)
print("Salto bras en haut SaMi")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}\n\n")

b = bioviz.Viz(model_path_SaMi)
b.load_movement(X_tous[:, :m_SaMi.nbQ()].T)
b.exec()

# Salto bras qui descendent SaMi
X0 = np.zeros((m_SaMi.nbQ()*2, ))
X0[15] = -np.pi
X0[24] = np.pi
X0[m_SaMi.nbQ()+3] = 2 * np.pi #Salto
X0[m_SaMi.nbQ()+15] = 2*np.pi #brasD
X0[m_SaMi.nbQ()+24] = -2*np.pi #brasG
X_tous = integrate_plots(m_SaMi, X0, t, N, n_step)
print("Salto bras qui descendent SaMi")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}")

b = bioviz.Viz(model_path_SaMi)
b.load_movement(X_tous[:, :m_SaMi.nbQ()].T)
b.exec()

# Salto un bras qui descend SaMi
X0 = np.zeros((m_SaMi.nbQ()*2, ))
X0[15] = -np.pi
X0[24] = np.pi
X0[m_SaMi.nbQ()+3] = 2*np.pi #Salto
X0[m_SaMi.nbQ()+24] = -2*np.pi #brasG
X_tous = integrate_plots(m_SaMi, X0, t, N, n_step)
print("Salto un bras qui descend SaMi")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}")

b = bioviz.Viz(model_path_SaMi)
b.load_movement(X_tous[:, :m_SaMi.nbQ()].T)
b.exec()
















