
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

def integrate(m, X0, t, N, runge_kutta_4):
    X_tous = X0
    for i in range(N - 1):
        X = runge_kutta_4(m, X0, t[i], t[-1], N)
        X_tous = np.vstack((X_tous, X))
        X0 = X

    return X_tous

def plot_Q_Qdot_bras(m, t, X_tous):
    nb_q = m.nbQ()
    QbrasD = X_tous[:, 15]
    QbrasG = X_tous[:, 24]
    QdotbrasD = X_tous[:, nb_q + 15]
    QdotbrasG = X_tous[:, nb_q + 24]

    fig, ((axQG, axQD), (axQdG, axQdD)) = plt.subplots(2, 2)
    axQD.plot(t, QbrasD)
    axQD.set_title("Q droit")
    axQG.plot(t, QbrasG)
    axQG.set_title("Q gauche")
    axQdD.plot(t, QdotbrasD)
    axQdD.set_title("Qdot droit")
    axQdG.plot(t, QdotbrasG)
    axQdG.set_title("Qdot gauche")

    plt.tight_layout()
    plt.show(block=False)


def plot_Q_Qdot_bassin(m, t, X_tous):
    nb_q = m.nbQ()
    QdotrotX = X_tous[:, nb_q + 3]
    QdotrotY = X_tous[:, nb_q + 4]
    QdotrotZ = X_tous[:, nb_q + 5]

    fig, (axX, axY, axZ) = plt.subplots(3, 1)
    axX.plot(t, QdotrotX)
    axX.set_title("Rot X")
    axY.plot(t, QdotrotY)
    axY.set_title("Rot Y")
    axZ.plot(t, QdotrotZ)
    axZ.set_title("Rot Z")

    plt.tight_layout()
    plt.show(block=False)


###################################################################################
N = 100
model_path_JeCh = "Models/JeCh_201.bioMod"
model_path_SaMi = "Models/SaMi.bioMod"
m_JeCh = biorbd.Model(model_path_JeCh)
m_SaMi = biorbd.Model(model_path_SaMi)
m_JeCh.setGravity(np.array((0, 0, 0)))
m_SaMi.setGravity(np.array((0, 0, 0)))

# b = bioviz.Viz(model_path_JeCh)
# b.exec()

t = np.linspace(0, 1, N)

# Salto bras en haut JeCh
X0 = np.zeros((m_JeCh.nbQ()*2, ))
X0[15] = -np.pi
X0[24] = np.pi
X0[m_JeCh.nbQ()+3] = 2 * np.pi  # Salto rot
X_tous = integrate(m_JeCh, X0, t, N, runge_kutta_4_neutre)
print("Salto bras en haut JeCh")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}\n")
plot_Q_Qdot_bras(m_JeCh, t, X_tous)
plot_Q_Qdot_bassin(m_JeCh, t, X_tous)

b = bioviz.Viz(model_path_JeCh, show_floor=False)
b.load_movement(X_tous[:, :m_JeCh.nbQ()].T)
b.exec()

# Salto bras qui descendent JeCh
X0 = np.zeros((m_JeCh.nbQ()*2, ))
X0[15] = -np.pi
X0[24] = np.pi
X0[m_JeCh.nbQ()+3] = 2*np.pi  # Salto rot
X_tous = integrate(m_JeCh, X0, t, N, runge_kutta_4_2bras)
print("Salto bras qui descendent JeCh")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}\n")
plot_Q_Qdot_bras(m_JeCh, t, X_tous)
plot_Q_Qdot_bassin(m_JeCh, t, X_tous)

b = bioviz.Viz(model_path_JeCh, show_floor=False)
b.load_movement(X_tous[:, :m_JeCh.nbQ()].T)
b.exec()

# Salto un bras qui descend JeCh
X0 = np.zeros((m_JeCh.nbQ()*2, ))
X0[15] = -np.pi
X0[24] = np.pi
X0[m_JeCh.nbQ()+3] = 2*np.pi  # Salto rot
X_tous = integrate(m_JeCh, X0, t, N, runge_kutta_4_brasG)
print("Salto un bras qui descend JeCh")
print(f"Salto : {X_tous[-1, 3] / 2/np.pi}\nTilt : {X_tous[-1, 4] / 2/np.pi}\nTwist : {X_tous[-1, 5] / 2/np.pi}\n")
plot_Q_Qdot_bras(m_JeCh, t, X_tous)
plot_Q_Qdot_bassin(m_JeCh, t, X_tous)

b = bioviz.Viz(model_path_JeCh, show_floor=False)
b.load_movement(X_tous[:, :m_JeCh.nbQ()].T)
b.exec()
