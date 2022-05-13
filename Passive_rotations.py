
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

def Quintic(t, Ti, Tj, Qi, Qj):
    if t < Ti:
        t = Ti
    elif t > Tj:
        t = Tj

    tp0 = Tj - Ti
    tp1 = (t - Ti) / tp0

    tp2 = tp1 ** 3 * (6 * tp1 ** 2 - 15 * tp1 + 10)
    tp3 = Qj - Qi  # x1 x2
    tp4 = t - Ti
    tp5 = Tj - t

    p = Qi + tp3 * tp2
    v = 30 * tp3 * tp4 ** 2 * tp5 ** 2 / tp0 ** 5
    a = 60 * tp3 * tp4 * tp5 * (Tj + Ti - 2 * t) / tp0 ** 5
    #return p, v, a  # qddot_j
    return a

def dynamics_root(m, X, Qddot_J):
    Q = X[:m.nbQ()]
    Qdot = X[m.nbQ():]
    Qddot = np.hstack((np.zeros((6,)), Qddot_J)) #qddot2
    NLEffects = m.InverseDynamics(Q, Qdot, Qddot).to_array()
    mass_matrix = m.massMatrix(Q).to_array()
    Qddot_R = np.linalg.inv(mass_matrix[:6, :6]) @ NLEffects[:6]
    Xdot = np.hstack((Qdot, Qddot_R, Qddot_J))
    return Xdot

def runge_kutta_4_neutre(m, x0, t, tf, N):
    h = tf / (N-1)

    Qddot_J = np.zeros((m.nbQ()-m.nbRoot(),)) #### mouvement continu, sans acceleration

    k1 = dynamics_root(m, x0, Qddot_J)
    k2 = dynamics_root(m, x0 + h / 2 * k1, Qddot_J)
    k3 = dynamics_root(m, x0 + h / 2 * k2, Qddot_J)
    k4 = dynamics_root(m, x0 + h * k3, Qddot_J)

    x = np.hstack((x0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)))

    return x

def runge_kutta_4_brasG(m, x0, t, tf, N):
    h = tf / (N-1)

    Qddot_Jt1 = np.zeros((m.nbQ()-m.nbRoot(),)) #### mouvement continu, sans acceleration
    Qddot_Jt23 = np.zeros((m.nbQ() - m.nbRoot(),))
    Qddot_Jt4 = np.zeros((m.nbQ() - m.nbRoot(),))

    Qddot_Jt1[24-6] = Quintic(t, 0, 1, 2.9, .18)  # accelere bras gauche  TODO modifier les bornes
    Qddot_Jt23[24-6] = Quintic(t + h/2, 0, 1, 2.9, .18)
    Qddot_Jt4[24-6] = Quintic(t + h, 0, 1, 2.9, .18)

    k1 = dynamics_root(m, x0, Qddot_Jt1)
    k2 = dynamics_root(m, x0 + h / 2 * k1, Qddot_Jt23)
    k3 = dynamics_root(m, x0 + h / 2 * k2, Qddot_Jt23)
    k4 = dynamics_root(m, x0 + h * k3, Qddot_Jt4)

    x = x0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x

def runge_kutta_4_2bras(m, x0, t, tf, N):
    h = tf / (N-1)

    Qddot_Jt1 = np.zeros((m.nbQ()-m.nbRoot(),)) #### mouvement continu, sans acceleration
    Qddot_Jt23 = np.zeros((m.nbQ() - m.nbRoot(),))
    Qddot_Jt4 = np.zeros((m.nbQ() - m.nbRoot(),))

    Qddot_Jt1[15-6] = -Quintic(t, 0, 1, 2.9, .18)   # accelere bras droit  TODO modifier les bornes
    Qddot_Jt1[24 - 6] = Quintic(t, 0, 1, 2.9, .18)  # accelere bras gauche
    Qddot_Jt23[15 - 6] = -Quintic(t + h/2, 0, 1, 2.9, .18)  # accelere bras droit  TODO modifier les bornes
    Qddot_Jt23[24 - 6] = Quintic(t + h/2, 0, 1, 2.9, .18)  # accelere bras gauche
    Qddot_Jt4[15 - 6] = -Quintic(t + h, 0, 1, 2.9, .18)  # accelere bras droit  TODO modifier les bornes
    Qddot_Jt4[24 - 6] = Quintic(t + h, 0, 1, 2.9, .18)  # accelere bras gauche

    k1 = dynamics_root(m, x0, Qddot_Jt1)
    k2 = dynamics_root(m, x0 + h / 2 * k1, Qddot_Jt23)
    k3 = dynamics_root(m, x0 + h / 2 * k2, Qddot_Jt23)
    k4 = dynamics_root(m, x0 + h * k3, Qddot_Jt4)

    x = x0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x

def integrate_plots(m, X0, t, N, runge_kutta_4):
    # fig, axs = plt.subplots(2, 3)
    # axs = np.ravel(axs)
    X_tous = X0
    for i in range(N - 1): #N//2 - 1):
        X = runge_kutta_4(m, X0, t[i], t[-1], N)
        # plot_index = 0
        # for k in [3, 4, 5, 15, 24]:
        #     axs[plot_index].plot(np.arange(i * n_step, (i + 1) * n_step + 1), np.reshape(X[k, :], n_step + 1), ':',
        #                 label=f'{m.nameDof()[k].to_string()}')
        #     plot_index += 1
        X_tous = np.vstack((X_tous, X))
        X0 = X
    """
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
    """
    return X_tous

###################################################################################
N = 100
model_path_JeCh = "Models/JeCh_201.bioMod"
model_path_SaMi = "Models/SaMi.bioMod"
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
X_tous = integrate_plots(m_JeCh, X0, t, N, runge_kutta_4_neutre)
print("Salto bras en haut JeCh")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}\n\n")

b = bioviz.Viz(model_path_JeCh, show_floor=False)
b.load_movement(X_tous[:, :m_JeCh.nbQ()].T)
b.exec()

# Salto bras qui descendent JeCh
X0 = np.zeros((m_JeCh.nbQ()*2, ))
X0[15] = -np.pi
X0[24] = np.pi
X0[m_JeCh.nbQ()+3] = 2*np.pi #Salto
#X0[m_JeCh.nbQ()+15] = 2*np.pi #brasD
#X0[m_JeCh.nbQ()+24] = -2*np.pi #brasG
X_tous = integrate_plots(m_JeCh, X0, t, N, runge_kutta_4_2bras)
print("Salto bras qui descendent JeCh")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}")

b = bioviz.Viz(model_path_JeCh, show_floor=False)
b.load_movement(X_tous[:, :m_JeCh.nbQ()].T)
b.exec()

# Salto un bras qui descend JeCh
X0 = np.zeros((m_JeCh.nbQ()*2, ))
X0[15] = -np.pi
X0[24] = np.pi
X0[m_JeCh.nbQ()+3] = 2*np.pi #Salto
#X0[m_JeCh.nbQ()+24] = -2*np.pi #brasG
X_tous = integrate_plots(m_JeCh, X0, t, N, runge_kutta_4_brasG)
print("Salto un bras qui descend JeCh")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}")

b = bioviz.Viz(model_path_JeCh, show_floor=False)
b.load_movement(X_tous[:, :m_JeCh.nbQ()].T)
b.exec()

print("exit")
exit()  # ce qui suit est inaccessible

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
















