import biorbd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import bioviz
import pickle
import xlsxwriter

"""
Ce code a été écrit pout faire les simulations pour Antoine, 
mais il pourra être modifié pour apporter d'autres infos utiles avec d'autres anthropo comme base.
Attention, ici on considere des mouvements des joints à vitesse constante (Qddot_J = 0).
C'est surement overkill d'utiliser un KR4 à 100 noeuds pour des mouvements à vitesse constante, mais bon
"""
#
# Physique
#


def Quintic(t, Ti, Tj, Qi, Qj):  # Quintic est bonne
    if t < Ti:
        t = Ti
    elif t > Tj:
        t = Tj

    tp0 = Tj - Ti
    tp1 = (t - Ti) / tp0

    tp2 = tp1**3 * (6 * tp1**2 - 15 * tp1 + 10)
    tp3 = Qj - Qi  # x1 x2
    tp4 = t - Ti
    tp5 = Tj - t

    p = Qi + tp3 * tp2
    v = 30 * tp3 * tp4**2 * tp5**2 / tp0**5
    a = 60 * tp3 * tp4 * tp5 * (Tj + Ti - 2 * t) / tp0**5

    return p, v, a


def dynamics_root(m, X, Qddot_J):
    Q = X[: m.nbQ()]
    Qdot = X[m.nbQ() :]
    Qddot = np.hstack((np.zeros((6,)), Qddot_J))  # qddot2
    NLEffects = m.InverseDynamics(Q, Qdot, Qddot).to_array()
    mass_matrix = m.massMatrix(Q).to_array()
    Qddot_R = np.linalg.solve(mass_matrix[:6, :6], -NLEffects[:6])
    Xdot = np.hstack((Qdot, Qddot_R, Qddot_J))
    return Xdot


def dsym2(f: np.array, h: float, out: np.array = None) -> np.array:
    if len(f) < 5:
        raise ValueError("len(f) must be >= 5 for this function")

    if out is None:
        out = np.zeros(len(f))

    out[0] = (f[1] - f[0]) / h  # naively first
    out[1] = (f[2] - f[0]) / 2 / h  # symetric 1

    out[2:-2] = (f[:-4] - 8 * f[1:-3] + 8 * f[3:-1] - f[4:]) / 12 / h

    out[-2] = (f[-1] - f[-3]) / 2 / h
    out[-1] = (f[-1] - f[-2]) / h

    return out


def bras_en_haut(m, x0, t, T0, Tf, Q0, Qf):
    Qddot_J = np.zeros(m.nbQ() - m.nbRoot())
    x = dynamics_root(m, x0, Qddot_J)
    return x


def bras_descendent(m, x0, t, T0, Tf, Q0, Qf):
    global GAUCHE
    global DROITE
    Kp = 10.0
    Kv = 3.0
    p, v, a = Quintic(t, T0, Tf, Q0, Qf)
    Qddot_J = np.zeros(m.nbQ() - m.nbRoot())
    Qddot_J[GAUCHE - m.nbRoot()] = -a + Kp * (-p - x0[GAUCHE]) + Kv * (-v - x0[m.nbQ() + GAUCHE])
    Qddot_J[DROITE - m.nbRoot()] = a + Kp * (p - x0[DROITE]) + Kv * (v - x0[m.nbQ() + DROITE])

    x = dynamics_root(m, x0, Qddot_J)
    return x


def bras_gauche_descend(m, x0, t, T0, Tf, Q0, Qf):
    global GAUCHE
    Kp = 10.0
    Kv = 3.0
    p, v, a = Quintic(t, T0, Tf, Q0, Qf)
    Qddot_J = np.zeros(m.nbQ() - m.nbRoot())
    Qddot_J[GAUCHE - m.nbRoot()] = a + Kp * (p - x0[GAUCHE]) + Kv * (v - x0[m.nbQ() + GAUCHE])

    x = dynamics_root(m, x0, Qddot_J)
    return x


def bras_droit_descend(m, x0, t, T0, Tf, Q0, Qf):
    global DROITE
    Kp = 10.0
    Kv = 3.0
    p, v, a = Quintic(t, T0, Tf, Q0, Qf)
    Qddot_J = np.zeros(m.nbQ() - m.nbRoot())
    Qddot_J[DROITE - m.nbRoot()] = -a + Kp * (-p - x0[DROITE]) + Kv * (-v - x0[m.nbQ() + DROITE])

    x = dynamics_root(m, x0, Qddot_J)
    return x


#
# Visualisation
#
def plot_Q_Qdot_bras(m, t, X_tous, Qddot, titre=""):
    global GAUCHE
    global DROITE

    nb_q = m.nbQ()
    QbrasD = X_tous[:, DROITE]
    QbrasG = X_tous[:, GAUCHE]
    QdotbrasD = X_tous[:, nb_q + DROITE]
    QdotbrasG = X_tous[:, nb_q + GAUCHE]
    QddotbrasD = Qddot[:, DROITE]
    QddotbrasG = Qddot[:, GAUCHE]

    titles = ["QbrasD", "QbrasG", "QdotbrasD", "QdotbrasG", "QddotbrasG", "QddotbrasG"]
    values = [QbrasD, QbrasG, QdotbrasD, QdotbrasG, QddotbrasG, QddotbrasG]
    athlete = titre.partition("debut")[0]
    n = len(values)
    for i in range(n):
        file = open(f"Q_passive_rotations/{titre}-{titles[i]}.pkl", "wb")
        pickle.dump(values[i], file)
        file.close()

    fig, ((axQG, axQD), (axQdG, axQdD), (axQddG, axQddD)) = plt.subplots(3, 2, sharex=True)
    axQD.plot(t, QbrasD)
    axQG.plot(t, QbrasG)
    axQG.set_ylabel("position (rad)")

    axQdD.plot(t, QdotbrasD)
    axQdG.plot(t, QdotbrasG)
    axQdG.set_ylabel("vitesse (rad/s)")

    axQddD.plot(t, QddotbrasD)
    axQddG.plot(t, QddotbrasG)
    axQddG.set_ylabel("acceleration (rad/s$^2$)")

    axQddG.set_xlabel("temps (s)")
    axQddD.set_xlabel("temps (s)")

    axQD.set_title("Droit")
    axQG.set_title("Gauche")
    suptitre = f"Mouvement des bras - {titre}" if titre != "" else ""
    fig.suptitle(suptitre)

    fig.tight_layout()
    # fig.savefig(f'Videos/{suptitre}.pdf')
    # fig.show()


def plot_Q_Qdot_bassin(m, t, X_tous, Qddot, titre=""):
    nb_q = m.nbQ()

    # position
    QX = X_tous[:, 0]
    QY = X_tous[:, 1]
    QZ = X_tous[:, 2]
    QrotX = X_tous[:, 3]
    QrotY = X_tous[:, 4]
    QrotZ = X_tous[:, 5]

    # vitesses
    QdotX = X_tous[:, nb_q + 0]
    QdotY = X_tous[:, nb_q + 1]
    QdotZ = X_tous[:, nb_q + 2]
    QdotrotX = X_tous[:, nb_q + 3]
    QdotrotY = X_tous[:, nb_q + 4]
    QdotrotZ = X_tous[:, nb_q + 5]

    # acceleration
    QddotX = Qddot[:, 0]
    QddotY = Qddot[:, 1]
    QddotZ = Qddot[:, 2]
    QddotrotX = Qddot[:, 3]
    QddotrotY = Qddot[:, 4]
    QddotrotZ = Qddot[:, 5]

    titles = [
        "QX",
        "QY",
        "QZ",
        "QrotX",
        "QrotY",
        "QrotZ",
        "QdotX",
        "QdotY",
        "QdotZ",
        "QdotrotX",
        "QdotrotY",
        "QdotrotZ",
        "QddotX",
        "QddotY",
        "QddotZ",
        "QddotrotX",
        "QddotrotY",
        "QddotrotZ",
    ]
    values = [
        QX,
        QY,
        QZ,
        QrotX,
        QrotY,
        QrotZ,
        QdotX,
        QdotY,
        QdotZ,
        QdotrotX,
        QdotrotY,
        QdotrotZ,
        QddotX,
        QddotY,
        QddotZ,
        QddotrotX,
        QddotrotY,
        QddotrotZ,
    ]
    n = len(values)
    athlete = titre.partition("debut")[0]
    for i in range(n):
        file = open(f"Q_passive_rotations/{titre}-{titles[i]}.pkl", "wb")
        pickle.dump(values[i], file)
        file.close()

    fig, (axp, axv, axa) = plt.subplots(3, 1, sharex=True)
    axp.plot(t, QX, label="X")
    axp.plot(t, QY, label="Y")
    axp.plot(t, QZ, label="Z")
    axp.set_ylabel("position (m)")
    axp.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    axv.plot(t, QdotX, label="Xdot")
    axv.plot(t, QdotY, label="Ydot")
    axv.plot(t, QdotZ, label="Zdot")
    axv.set_ylabel("vitesse (m/s)")
    axv.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    axa.plot(t, QddotX, label="Xddot")
    axa.plot(t, QddotY, label="Yddot")
    axa.plot(t, QddotZ, label="Zddot")
    axa.set_ylabel("acceleration (m/s$^2$)")
    axa.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    axa.set_xlabel("temps (s)")
    suptitre = "Translation du bassin" + f" - {titre}" if titre != "" else ""
    fig.suptitle(suptitre)
    fig.tight_layout()
    # fig.savefig(f'Videos/{suptitre}.pdf')
    # fig.show()

    figrot, (axprot, axvrot, axarot) = plt.subplots(3, 1, sharex=True)
    axprot.plot(t, QrotX, label="Rot X")
    axprot.plot(t, QrotY, label="Rot Y")
    axprot.plot(t, QrotZ, label="Rot Z")
    axprot.set_ylabel("position (rad)")
    axprot.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    axvrot.plot(t, QdotrotX, label="Rot Xdot")
    axvrot.plot(t, QdotrotY, label="Rot Ydot")
    axvrot.plot(t, QdotrotZ, label="Rot Zdot")
    axvrot.set_ylabel("vitesse (rad/s)")
    axvrot.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    axarot.plot(t, QddotrotX, label="Rot Xddot")
    axarot.plot(t, QddotrotY, label="Rot Yddot")
    axarot.plot(t, QddotrotZ, label="Rot Zddot")
    axarot.set_ylabel("acceleration (rad/s$^2$)")
    axarot.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    axarot.set_xlabel("temps (s)")
    suptitre = "Rotation du bassin" + f" - {titre}" if titre != "" else ""
    figrot.suptitle(suptitre)
    figrot.tight_layout()
    # figrot.savefig(f'Videos/{suptitre}.pdf')
    # figrot.show()


workbook = xlsxwriter.Workbook(
    "/home/lim/Documents/Stage_Lisa/AnthropoImpactOnTech/Passive_rotations/degrees_of_liberty.xlsx"
)

# The workbook object is then used to add new
# worksheet via the add_worksheet() method.
worksheet = workbook.add_worksheet()

# Use the worksheet object to write
# data via the write() method.
worksheet.write("A1", "Athlete")
worksheet.write("B1", "Position")
worksheet.write("C1", "Salto")
worksheet.write("D1", "Tilt")
worksheet.write("E1", "Twist")


# Simulation
#
def simuler(nom, m, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras, row, column, situation, viz=True):
    m.setGravity(np.array((0, 0, 0)))
    t, dt = np.linspace(t0, tf, num=N + 1, retstep=True)
    situation += "," + nom.partition("bioMod")[2]
    nom = nom.partition(".bioMod")[0].removeprefix("Models/")
    func = lambda t, y: action_bras(m, y, t, T0, Tf, Q0, Qf)

    r = scipy.integrate.ode(func).set_integrator("dop853").set_initial_value(X0, t0)
    X_tous = X0
    while (
        r.successful() and r.t < tf
    ):  # inspire de la doc de scipy [https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.integrate.ode.html]
        r.integrate(r.t + dt)
        X_tous = np.vstack((X_tous, r.y))

    worksheet.write(row, column, f"{nom}")
    column += 1
    worksheet.write(row, column, f"{situation}")
    column += 1
    worksheet.write(row, column, X_tous[-1, 3] / 2 / np.pi)
    column += 1
    worksheet.write(row, column, X_tous[-1, 4] / 2 / np.pi)
    column += 1
    worksheet.write(row, column, X_tous[-1, 5] / 2 / np.pi)

    # print(f"Salto : {X_tous[-1, 3] / 2 / np.pi}\nTilt : {X_tous[-1, 4] / 2 / np.pi}\nTwist : {X_tous[-1, 5] / 2 / np.pi}\n")

    if viz:
        Qddot = np.zeros(X_tous.shape)
        dsym2(X_tous, dt, Qddot)
        Qddot[np.logical_and(Qddot < 1e-14, Qddot > -1e-14)] = 0
        Qddot = Qddot[:, m.nbQ() :]

        nom = nom.partition(".bioMod")[0].removeprefix("Models/")
        suptitre = nom.removeprefix("Models/") + " " + situation
        plot_Q_Qdot_bras(model, t, X_tous, Qddot, titre=suptitre)
        plot_Q_Qdot_bassin(model, t, X_tous, Qddot, titre=suptitre)

        b = bioviz.Viz(models[i], show_floor=False)
        b.load_movement(X_tous[:, : model.nbQ()].T)
        b.exec()


N = 100
JeCh = "Models/JeCh.bioMod"
WeEm = "Models/WeEm.bioMod"
SoMe = "Models/SoMe.bioMod"
Sarah = "Models/Sarah.bioMod"
OlGa = "Models/OlGa.bioMod"
MaJa = "Models/MaJa.bioMod"
MaCu = "Models/MaCu.bioMod"
LaDe = "Models/LaDe.bioMod"
FeBl = "Models/FeBl.bioMod"
EvZl = "Models/EvZl.bioMod"
Benjamin = "Models/Benjamin.bioMod"
AuJo = "Models/AuJo.bioMod"
AlAd = "Models/AlAd.bioMod"
AdCh = "Models/AdCh.bioMod"


# JeCh_2 = "Models/JeCh_2.bioMod"
# SaMi = "Models/SaMi_pr.bioMod"
# ElMe = "Models/ElMe.bioMod"
# ZoTs = "Models/ZoTs.bioMod"
# KaMi = "Models/KaMi.bioMod"
# KaFu = "Models/KaFu.bioMod"
# B


# m_JeCh = biorbd.Model(model_path_JeCh)
# m_SaMi = biorbd.Model(model_path_SaMi)
# m_ElMe = biorbd.Model(model_path_ElMe)
# m_ZoTs = biorbd.Model(model_path_ZoTs)


GAUCHE = 11  # 42 -> 24; 10 -> 9
DROITE = 7  # 42 -> 15; 10 -> 7

t0 = 0.0
tf = 1.0
T0 = 0.0
Tf = 0.2
Q0 = 2.9
Qf = 0.0

models = [JeCh]
for i in range(len(models)):
    model = biorbd.Model(models[i])
    name = models[i]
    column = 0
    row = i * 6 + 1

    # JeCh
    # debut bras en haut
    X0 = np.zeros(model.nbQ() * 2)
    X0[DROITE] = Q0
    X0[GAUCHE] = -Q0

    CoM_func = model.CoM(X0[: model.nbQ()]).to_array()
    bassin = model.globalJCS(0).to_array()
    QCoM = CoM_func.reshape(1, 3)
    Qbassin = bassin[-1, :3]
    r = QCoM - Qbassin

    X0[model.nbQ() + 3] = -2 * np.pi  # Salto rot
    X0[model.nbQ() : model.nbQ() + 3] = X0[model.nbQ() : model.nbQ() + 3] + np.cross(
        r, X0[model.nbQ() + 3 : model.nbQ() + 6]
    )  # correction pour la translation

    # row = i +1
    situation = "starts with arms up"
    simuler(
        f"{name} arms stay up",
        model,
        N,
        t0,
        tf,
        T0,
        Tf,
        Q0,
        Qf,
        X0,
        action_bras=bras_en_haut,
        viz=False,
        row=row,
        column=column,
        situation=situation,
    )
    row += 1
    simuler(
        f"{name} arms go down",
        model,
        N,
        t0,
        tf,
        T0,
        Tf,
        Q0,
        Qf,
        X0,
        action_bras=bras_descendent,
        viz=True,
        row=row,
        column=column,
        situation=situation,
    )

    row += 1
    simuler(
        f"{name} left arm goes down",
        model,
        N,
        t0,
        tf,
        T0,
        Tf,
        Q0,
        Qf,
        X0,
        action_bras=bras_gauche_descend,
        viz=False,
        row=row,
        column=column,
        situation=situation,
    )
    row += 1
    simuler(
        f"{name} right arm goes down",
        model,
        N,
        t0,
        tf,
        T0,
        Tf,
        Q0,
        Qf,
        X0,
        action_bras=bras_droit_descend,
        viz=False,
        row=row,
        column=column,
        situation=situation,
    )

    # debut bras droit en haut, gauche bas
    situation = "starts with right arm up and left arm down"
    X0 = np.zeros(model.nbQ() * 2)
    X0[DROITE] = -Q0
    X0[GAUCHE] = Qf

    CoM_func = model.CoM(X0[: model.nbQ()]).to_array()
    bassin = model.globalJCS(0).to_array()
    QCoM = CoM_func.reshape(1, 3)
    Qbassin = bassin[-1, :3]
    r = QCoM - Qbassin

    X0[model.nbQ() + 3] = 2 * np.pi  # Salto rot
    X0[model.nbQ() : model.nbQ() + 3] = X0[model.nbQ() : model.nbQ() + 3] + np.cross(
        r, X0[model.nbQ() + 3 : model.nbQ() + 6]
    )  # correction pour la translation
    row += 1
    simuler(
        f"{name} right goes down",
        model,
        N,
        t0,
        tf,
        T0,
        Tf,
        Q0,
        Qf,
        X0,
        action_bras=bras_droit_descend,
        viz=False,
        row=row,
        column=column,
        situation=situation,
    )

    situation = "starts with left arm up, right arm down"
    X0 = np.zeros(model.nbQ() * 2)
    X0[DROITE] = -Qf
    X0[GAUCHE] = Q0

    CoM_func = model.CoM(X0[: model.nbQ()]).to_array()
    bassin = model.globalJCS(0).to_array()
    QCoM = CoM_func.reshape(1, 3)
    Qbassin = bassin[-1, :3]
    r = QCoM - Qbassin

    X0[model.nbQ() + 3] = 2 * np.pi  # Salto rot
    X0[model.nbQ() : model.nbQ() + 3] = X0[model.nbQ() : model.nbQ() + 3] + np.cross(
        r, X0[model.nbQ() + 3 : model.nbQ() + 6]
    )  # correction pour la translation
    row += 1
    simuler(
        f"{name} left arm goes down",
        model,
        N,
        t0,
        tf,
        T0,
        Tf,
        Q0,
        Qf,
        X0,
        action_bras=bras_gauche_descend,
        viz=False,
        row=row,
        column=column,
        situation=situation,
    )

print("fin")
workbook.close()


#
# # SaMi
# # debut bras en haut
# X0 = np.zeros(m_SaMi.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_SaMi.CoM(X0[:m_SaMi.nbQ()]).to_array()
# bassin = m_SaMi.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_SaMi.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_SaMi.nbQ():m_SaMi.nbQ()+3] = X0[m_SaMi.nbQ():m_SaMi.nbQ() + 3] + np.cross(r, X0[m_SaMi.nbQ()+3:m_SaMi.nbQ()+6])  # correction pour la translation
#
# simuler("SaMi bras en haut", m_SaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("SaMi bras descendent", m_SaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("SaMi bras gauche descend", m_SaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("SaMi bras droit descend", m_SaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_SaMi.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_SaMi.CoM(X0[:m_SaMi.nbQ()]).to_array()
# bassin = m_SaMi.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_SaMi.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_SaMi.nbQ():m_SaMi.nbQ()+3] = X0[m_SaMi.nbQ():m_SaMi.nbQ() + 3] + np.cross(r, X0[m_SaMi.nbQ()+3:m_SaMi.nbQ()+6])  # correction pour la translation
#
# simuler("SaMi bras gauche bas, droit descend", m_SaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_SaMi.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_SaMi.CoM(X0[:m_SaMi.nbQ()]).to_array()
# bassin = m_SaMi.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_SaMi.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_SaMi.nbQ():m_SaMi.nbQ()+3] = X0[m_SaMi.nbQ():m_SaMi.nbQ() + 3] + np.cross(r, X0[m_SaMi.nbQ()+3:m_SaMi.nbQ()+6])  # correction pour la translation
#
# simuler("SaMi bras droit bas, gauche descend", m_SaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
# # El Me
# # debut bras en haut
# X0 = np.zeros(m_ElMe.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_ElMe.CoM(X0[:m_ElMe.nbQ()]).to_array()
# bassin = m_ElMe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_ElMe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_ElMe.nbQ():m_ElMe.nbQ()+3] = X0[m_ElMe.nbQ():m_ElMe.nbQ() + 3] + np.cross(r, X0[m_ElMe.nbQ()+3:m_ElMe.nbQ()+6])  # correction pour la translation
#
# simuler("ELMe bras en haut", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("ElMe bras descendent", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("ElMe bras gauche descend", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("ElMe bras droit descend", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_ElMe.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_ElMe.CoM(X0[:m_ElMe.nbQ()]).to_array()
# bassin = m_ElMe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_ElMe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_ElMe.nbQ():m_ElMe.nbQ()+3] = X0[m_ElMe.nbQ():m_ElMe.nbQ() + 3] + np.cross(r, X0[m_ElMe.nbQ()+3:m_ElMe.nbQ()+6])  # correction pour la translation
#
# simuler("ElMe bras gauche bas, droit descend", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_ElMe.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_ElMe.CoM(X0[:m_ElMe.nbQ()]).to_array()
# bassin = m_ElMe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_ElMe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_ElMe.nbQ():m_ElMe.nbQ()+3] = X0[m_ElMe.nbQ():m_ElMe.nbQ() + 3] + np.cross(r, X0[m_ElMe.nbQ()+3:m_ElMe.nbQ()+6])  # correction pour la translation
#
# simuler("ElMe bras droit bas, gauche descend", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
# # ZoTs
# # debut bras en haut
# X0 = np.zeros(m_ZoTs.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_ZoTs.CoM(X0[:m_ZoTs.nbQ()]).to_array()
# bassin = m_ZoTs.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_ZoTs.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_ZoTS.nbQ():m_ZoTs.nbQ()+3] = X0[m_ZoTs.nbQ():m_ZoTs.nbQ() + 3] + np.cross(r, X0[m_ZoTs.nbQ()+3:m_ZoTs.nbQ()+6])  # correction pour la translation
#
# simuler("ZoTs bras en haut", m_ZoTs, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("ZoTs bras descendent", m_ZoTs, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("ZoTs bras gauche descend", m_ZoTs, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("ZoTs bras droit descend", m_ZoTs, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_ZoTs.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_ZoTs.CoM(X0[:m_ZoTs.nbQ()]).to_array()
# bassin = m_ZoTs.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_ZoTs.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_ZoTs.nbQ():m_ZoTs.nbQ()+3] = X0[m_ZoTs.nbQ():m_ZoTs.nbQ() + 3] + np.cross(r, X0[m_ZoTs.nbQ()+3:m_ZoTs.nbQ()+6])  # correction pour la translation
#
# simuler("ZoTs bras gauche bas, droit descend", m_ZoTs, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_ZoTs.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_ZoTs.CoM(X0[:m_ZoTs.nbQ()]).to_array()
# bassin = m_ZoTs.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_ZoTs.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_ZoTs.nbQ():m_ZoTs.nbQ()+3] = X0[m_ZoTs.nbQ():m_ZoTs.nbQ() + 3] + np.cross(r, X0[m_ZoTs.nbQ()+3:m_ZoTs.nbQ()+6])  # correction pour la translation
#
# simuler("ZoTs bras droit bas, gauche descend", m_ZoTs, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
#
# # WeEm
# # debut bras en haut
# X0 = np.zeros(m_WeEm.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_WeEm.CoM(X0[:m_WeEm.nbQ()]).to_array()
# bassin = m_WeEm.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_WeEm.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_WeEm.nbQ():m_WeEm.nbQ()+3] = X0[m_WeEm.nbQ():m_WeEm.nbQ() + 3] + np.cross(r, X0[m_WeEm.nbQ()+3:m_WeEm.nbQ()+6])  # correction pour la translation
#
# simuler("WeEm bras en haut", m_WeEm, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("WeEm bras descendent", m_WeEm, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("WeEm bras gauche descend", m_WeEm, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("WeEm bras droit descend", m_WeEm, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_WeEm.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_WeEm.CoM(X0[:m_WeEm.nbQ()]).to_array()
# bassin = m_WeEm.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_WeEm.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_WeEm.nbQ():m_WeEm.nbQ()+3] = X0[m_WeEm.nbQ():m_WeEm.nbQ() + 3] + np.cross(r, X0[m_WeEm.nbQ()+3:m_WeEm.nbQ()+6])  # correction pour la translation
#
# simuler("WeEm bras gauche bas, droit descend", m_WeEm, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_WeEm.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_WeEm.CoM(X0[:m_WeEm.nbQ()]).to_array()
# bassin = m_WeEm.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_WeEm.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_WeEm.nbQ():m_WeEm.nbQ()+3] = X0[m_WeEm.nbQ():m_WeEm.nbQ() + 3] + np.cross(r, X0[m_WeEm.nbQ()+3:m_WeEm.nbQ()+6])  # correction pour la translation
#
# simuler("WeEm bras droit bas, gauche descend", m_WeEm, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
#
# # SoMe
# # debut bras en haut
# X0 = np.zeros(m_SoMe.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_SoMe.CoM(X0[:m_SoMe.nbQ()]).to_array()
# bassin = m_SoMe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_SoMe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_SoMe.nbQ():m_SoMe.nbQ()+3] = X0[m_SoMe.nbQ():m_SoMe.nbQ() + 3] + np.cross(r, X0[m_SoMe.nbQ()+3:m_SoMe.nbQ()+6])  # correction pour la translation
#
# simuler("SoMe bras en haut", m_SoMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("SoMe bras descendent", m_SoMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("SoMe bras gauche descend", m_SoMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("SoMe bras droit descend", m_SoMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_SoMe.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_SoMe.CoM(X0[:m_SoMe.nbQ()]).to_array()
# bassin = m_SoMe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_SoMe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_SoMe.nbQ():m_SoMe.nbQ()+3] = X0[m_SoMe.nbQ():m_SoMe.nbQ() + 3] + np.cross(r, X0[m_SoMe.nbQ()+3:m_SoMe.nbQ()+6])  # correction pour la translation
#
# simuler("SoMe bras gauche bas, droit descend", m_SoMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_SoMe.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_SoMe.CoM(X0[:m_SoMe.nbQ()]).to_array()
# bassin = m_SoMe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_SoMe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_SoMe.nbQ():m_SoMe.nbQ()+3] = X0[m_SoMe.nbQ():m_SoMe.nbQ() + 3] + np.cross(r, X0[m_SoMe.nbQ()+3:m_SoMe.nbQ()+6])  # correction pour la translation
#
# simuler("SoMe bras droit bas, gauche descend", m_SoMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
# # Sarah
# # debut bras en haut
# X0 = np.zeros(m_Sarah.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_Sarah.CoM(X0[:m_Sarah.nbQ()]).to_array()
# bassin = m_Sarah.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_Sarah.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_Sarah.nbQ():m_Sarah.nbQ()+3] = X0[m_Sarah.nbQ():m_Sarah.nbQ() + 3] + np.cross(r, X0[m_Sarah.nbQ()+3:m_Sarah.nbQ()+6])  # correction pour la translation
#
# simuler("Sarah bras en haut", m_Sarah, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("Sarah bras descendent", m_Sarah, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("Sarah bras gauche descend", m_Sarah, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("Sarah bras droit descend", m_Sarah, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_Sarah.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_Sarah.CoM(X0[:m_Sarah.nbQ()]).to_array()
# bassin = m_Sarah.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_Sarah.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_Sarah.nbQ():m_Sarah.nbQ()+3] = X0[m_Sarah.nbQ():m_Sarah.nbQ() + 3] + np.cross(r, X0[m_Sarah.nbQ()+3:m_Sarah.nbQ()+6])  # correction pour la translation
#
# simuler("Sarah bras gauche bas, droit descend", m_Sarah, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_Sarah.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_Sarah.CoM(X0[:m_Sarah.nbQ()]).to_array()
# bassin = m_Sarah.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_Sarah.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_Sarah.nbQ():m_Sarah.nbQ()+3] = X0[m_Sarah.nbQ():m_Sarah.nbQ() + 3] + np.cross(r, X0[m_Sarah.nbQ()+3:m_Sarah.nbQ()+6])  # correction pour la translation
#
# simuler("Sarah bras droit bas, gauche descend", m_Sarah, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
# # OlGa
# # debut bras en haut
# X0 = np.zeros(m_OlGa.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_OlGa.CoM(X0[:m_OlGa.nbQ()]).to_array()
# bassin = m_OlGa.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_OlGa.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_OlGa.nbQ():m_OlGa.nbQ()+3] = X0[m_OlGa.nbQ():m_OlGa.nbQ() + 3] + np.cross(r, X0[m_OlGa.nbQ()+3:m_OlGa.nbQ()+6])  # correction pour la translation
#
# simuler("OlGa bras en haut", m_OlGa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("OlGa bras descendent", m_OlGa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("OlGa bras gauche descend", m_OlGa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("OlGa bras droit descend", m_OlGa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_OlGa.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_OlGa.CoM(X0[:m_OlGa.nbQ()]).to_array()
# bassin = m_OlGa.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_OlGa.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_OlGa.nbQ():m_OlGa.nbQ()+3] = X0[m_OlGa.nbQ():m_OlGa.nbQ() + 3] + np.cross(r, X0[m_OlGa.nbQ()+3:m_OlGa.nbQ()+6])  # correction pour la translation
#
# simuler("OlGa bras gauche bas, droit descend", m_OlGa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_OlGa.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_OlGa.CoM(X0[:m_OlGa.nbQ()]).to_array()
# bassin = m_OlGa.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_OlGa.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_OlGa.nbQ():m_OlGa.nbQ()+3] = X0[m_OlGa.nbQ():m_OlGa.nbQ() + 3] + np.cross(r, X0[m_OlGa.nbQ()+3:m_OlGa.nbQ()+6])  # correction pour la translation
#
# simuler("OlGa bras droit bas, gauche descend", m_OlGa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
# # MaJa
# # debut bras en haut
# X0 = np.zeros(m_MaJa.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_MaJa.CoM(X0[:m_MaJa.nbQ()]).to_array()
# bassin = m_MaJa.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_MaJa.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_MaJa.nbQ():m_MaJa.nbQ()+3] = X0[m_MaJa.nbQ():m_MaJa.nbQ() + 3] + np.cross(r, X0[m_MaJa.nbQ()+3:m_MaJa.nbQ()+6])  # correction pour la translation
#
# simuler("MaJa bras en haut", m_MaJa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("MaJa bras descendent", m_MaJa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("MaJa bras gauche descend", m_MaJa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("MaJa bras droit descend", m_MaJa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_MaJa.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_MaJa.CoM(X0[:m_MaJa.nbQ()]).to_array()
# bassin = m_MaJa.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_MaJa.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_MaJa.nbQ():m_MaJa.nbQ()+3] = X0[m_MaJa.nbQ():m_MaJa.nbQ() + 3] + np.cross(r, X0[m_MaJa.nbQ()+3:m_MaJa.nbQ()+6])  # correction pour la translation
#
# simuler("MaJa bras gauche bas, droit descend", m_MaJa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_MaJa.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_MaJa.CoM(X0[:m_MaJa.nbQ()]).to_array()
# bassin = m_MaJa.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_MaJa.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_MaJa.nbQ():m_MaJa.nbQ()+3] = X0[m_MaJa.nbQ():m_MaJa.nbQ() + 3] + np.cross(r, X0[m_MaJa.nbQ()+3:m_MaJa.nbQ()+6])  # correction pour la translation
#
# simuler("MaJa bras droit bas, gauche descend", m_MaJa, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
#
# # MaCu
# # debut bras en haut
# X0 = np.zeros(m_MaCu.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_MaCu.CoM(X0[:m_MaCu.nbQ()]).to_array()
# bassin = m_MaCu.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_MaCu.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_MaCu.nbQ():m_MaCu.nbQ()+3] = X0[m_MaCu.nbQ():m_MaCu.nbQ() + 3] + np.cross(r, X0[m_MaCu.nbQ()+3:m_MaCu.nbQ()+6])  # correction pour la translation
#
# simuler("MaCu bras en haut", m_MaCu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("MaCu bras descendent", m_MaCu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("MaCu bras gauche descend", m_MaCu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("MaCu bras droit descend", m_MaCu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_MaCu.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_MaCu.CoM(X0[:m_MaCu.nbQ()]).to_array()
# bassin = m_MaCu.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_MaCu.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_MaCu.nbQ():m_MaCu.nbQ()+3] = X0[m_MaCu.nbQ():m_MaCu.nbQ() + 3] + np.cross(r, X0[m_MaCu.nbQ()+3:m_MaCu.nbQ()+6])  # correction pour la translation
#
# simuler("MaCu bras gauche bas, droit descend", m_MaCu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_MaCu.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_MaCu.CoM(X0[:m_MaCu.nbQ()]).to_array()
# bassin = m_MaCu.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_MaCu.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_MaCu.nbQ():m_MaCu.nbQ()+3] = X0[m_MaCu.nbQ():m_MaCu.nbQ() + 3] + np.cross(r, X0[m_MaCu.nbQ()+3:m_MaCu.nbQ()+6])  # correction pour la translation
#
# simuler("MaCu bras droit bas, gauche descend", m_MaCu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
#
#
# # LaDe
# # debut bras en haut
# X0 = np.zeros(m_LaDe.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_LaDe.CoM(X0[:m_LaDe.nbQ()]).to_array()
# bassin = m_LaDe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_LaDe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_LaDe.nbQ():m_LaDe.nbQ()+3] = X0[m_LaDe.nbQ():m_LaDe.nbQ() + 3] + np.cross(r, X0[m_LaDe.nbQ()+3:m_LaDe.nbQ()+6])  # correction pour la translation
#
# simuler("LaDe bras en haut", m_LaDe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("LaDe bras descendent", m_LaDe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("LaDe bras gauche descend", m_LaDe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("LaDe bras droit descend", m_LaDe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_LaDe.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_LaDe.CoM(X0[:m_LaDe.nbQ()]).to_array()
# bassin = m_LaDe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_LaDe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_LaDe.nbQ():m_LaDe.nbQ()+3] = X0[m_LaDe.nbQ():m_LaDe.nbQ() + 3] + np.cross(r, X0[m_LaDe.nbQ()+3:m_LaDe.nbQ()+6])  # correction pour la translation
#
# simuler("LaDe bras gauche bas, droit descend", m_LaDe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_LaDe.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_LaDe.CoM(X0[:m_LaDe.nbQ()]).to_array()
# bassin = m_LaDe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_LaDe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_LaDe.nbQ():m_LaDe.nbQ()+3] = X0[m_LaDe.nbQ():m_LaDe.nbQ() + 3] + np.cross(r, X0[m_LaDe.nbQ()+3:m_LaDe.nbQ()+6])  # correction pour la translation
#
# simuler("LaDe bras droit bas, gauche descend", m_LaDe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
#
#
# # KaMi
# # debut bras en haut
# X0 = np.zeros(m_KaMi.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_KaMi.CoM(X0[:m_KaMi.nbQ()]).to_array()
# bassin = m_KaMi.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_KaMi.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_KaMi.nbQ():m_KaMi.nbQ()+3] = X0[m_KaMi.nbQ():m_KaMi.nbQ() + 3] + np.cross(r, X0[m_KaMi.nbQ()+3:m_KaMi.nbQ()+6])  # correction pour la translation
#
# simuler("KaMi bras en haut", m_KaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("KaMi bras descendent", m_KaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("KaMi bras gauche descend", m_KaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("KaMi bras droit descend", m_KaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_KaMi.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_KaMi.CoM(X0[:m_KaMi.nbQ()]).to_array()
# bassin = m_KaMi.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_KaMi.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_KaMi.nbQ():m_KaMi.nbQ()+3] = X0[m_KaMi.nbQ():m_KaMi.nbQ() + 3] + np.cross(r, X0[m_KaMi.nbQ()+3:m_KaMi.nbQ()+6])  # correction pour la translation
#
# simuler("KaMi bras gauche bas, droit descend", m_KaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_KaMi.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_KaMi.CoM(X0[:m_KaMi.nbQ()]).to_array()
# bassin = m_KaMi.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_KaMi.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_KaMi.nbQ():m_KaMi.nbQ()+3] = X0[m_KaMi.nbQ():m_KaMi.nbQ() + 3] + np.cross(r, X0[m_KaMi.nbQ()+3:m_KaMi.nbQ()+6])  # correction pour la translation
#
# simuler("KaMi bras droit bas, gauche descend", m_KaMi, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
#
# # KaFu
# # debut bras en haut
# X0 = np.zeros(m_KaFu.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_KaFu.CoM(X0[:m_KaFu.nbQ()]).to_array()
# bassin = m_KaFu.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_KaFu.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_KaFu.nbQ():m_KaFu.nbQ()+3] = X0[m_KaFu.nbQ():m_KaFu.nbQ() + 3] + np.cross(r, X0[m_KaFu.nbQ()+3:m_KaFu.nbQ()+6])  # correction pour la translation
#
# simuler("KaFu bras en haut", m_KaFu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("KaFu bras descendent", m_KaFu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("KaFu bras gauche descend", m_KaFu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("KaFu bras droit descend", m_KaFu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_KaFu.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_KaFu.CoM(X0[:m_KaFu.nbQ()]).to_array()
# bassin = m_KaFu.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_KaFu.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_KaFu.nbQ():m_KaFu.nbQ()+3] = X0[m_KaFu.nbQ():m_KaFu.nbQ() + 3] + np.cross(r, X0[m_KaFu.nbQ()+3:m_KaFu.nbQ()+6])  # correction pour la translation
#
# simuler("KaFu bras gauche bas, droit descend", m_KaFu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_KaFu.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_KaFu.CoM(X0[:m_KaFu.nbQ()]).to_array()
# bassin = m_KaFu.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_KaFu.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_KaFu.nbQ():m_KaFu.nbQ()+3] = X0[m_KaFu.nbQ():m_KaFu.nbQ() + 3] + np.cross(r, X0[m_KaFu.nbQ()+3:m_KaFu.nbQ()+6])  # correction pour la translation
#
# simuler("KaFu bras droit bas, gauche descend", m_KaFu, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
#
# # FeBl
# # debut bras en haut
# X0 = np.zeros(m_FeBl.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_FeBl.CoM(X0[:m_FeBl.nbQ()]).to_array()
# bassin = m_FeBl.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_FeBl.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_FeBl.nbQ():m_FeBl.nbQ()+3] = X0[m_FeBl.nbQ():m_FeBl.nbQ() + 3] + np.cross(r, X0[m_FeBl.nbQ()+3:m_FeBl.nbQ()+6])  # correction pour la translation
#
# simuler("FeBl bras en haut", m_FeBl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("FeBl bras descendent", m_FeBl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("FeBl bras gauche descend", m_FeBl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("FeBl bras droit descend", m_FeBl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_FeBl.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_FeBl.CoM(X0[:m_FeBl.nbQ()]).to_array()
# bassin = m_FeBl.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_FeBl.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_FeBl.nbQ():m_FeBl.nbQ()+3] = X0[m_FeBl.nbQ():m_FeBl.nbQ() + 3] + np.cross(r, X0[m_FeBl.nbQ()+3:m_FeBl.nbQ()+6])  # correction pour la translation
#
# simuler("FeBl bras gauche bas, droit descend", m_FeBl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_FeBl.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_FeBl.CoM(X0[:m_FeBl.nbQ()]).to_array()
# bassin = m_FeBl.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_FeBl.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_FeBl.nbQ():m_FeBl.nbQ()+3] = X0[m_FeBl.nbQ():m_KaFu.nbQ() + 3] + np.cross(r, X0[m_FeBl.nbQ()+3:m_FeBl.nbQ()+6])  # correction pour la translation
#
# simuler("FeBl bras droit bas, gauche descend", m_FeBl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
# # EvZl
# # debut bras en haut
# X0 = np.zeros(m_EvZl.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_EvZl.CoM(X0[:m_EvZl.nbQ()]).to_array()
# bassin = m_EvZl.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_EvZl.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_EvZl.nbQ():m_EvZl.nbQ()+3] = X0[m_EvZl.nbQ():m_EvZl.nbQ() + 3] + np.cross(r, X0[m_EvZl.nbQ()+3:m_EvZl.nbQ()+6])  # correction pour la translation
#
# simuler("EvZl bras en haut", m_EvZl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("EvZl bras descendent", m_EvZl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("EvZl bras gauche descend", m_EvZl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("EvZl bras droit descend", m_EvZl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_EvZl.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_EvZl.CoM(X0[:m_EvZl.nbQ()]).to_array()
# bassin = m_EvZl.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_EvZl.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_EvZl.nbQ():m_EvZl.nbQ()+3] = X0[m_EvZl.nbQ():m_EvZl.nbQ() + 3] + np.cross(r, X0[m_EvZl.nbQ()+3:m_EvZl.nbQ()+6])  # correction pour la translation
#
# simuler("EvZl bras gauche bas, droit descend", m_EvZl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_EvZl.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_EvZl.CoM(X0[:m_EvZl.nbQ()]).to_array()
# bassin = m_EvZl.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_EvZl.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_EvZl.nbQ():m_EvZl.nbQ()+3] = X0[m_EvZl.nbQ():m_EvZl.nbQ() + 3] + np.cross(r, X0[m_EvZl.nbQ()+3:m_EvZl.nbQ()+6])  # correction pour la translation
#
# simuler("EvZl bras droit bas, gauche descend", m_EvZl, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
#
# # ElMe
# # debut bras en haut
# X0 = np.zeros(m_ElMe.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_ElMe.CoM(X0[:m_ElMe.nbQ()]).to_array()
# bassin = m_ElMe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_ElMe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_ElMe.nbQ():m_ElMe.nbQ()+3] = X0[m_ElMe.nbQ():m_ElMe.nbQ() + 3] + np.cross(r, X0[m_ElMe.nbQ()+3:m_ElMe.nbQ()+6])  # correction pour la translation
#
# simuler("ElMe bras en haut", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("ElMe bras descendent", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("ElMe bras gauche descend", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("ElMe bras droit descend", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_ElMe.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_ElMe.CoM(X0[:m_ElMe.nbQ()]).to_array()
# bassin = m_ElMe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_ElMe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_ElMe.nbQ():m_ElMe.nbQ()+3] = X0[m_ElMe.nbQ():m_ElMe.nbQ() + 3] + np.cross(r, X0[m_ElMe.nbQ()+3:m_ElMe.nbQ()+6])  # correction pour la translation
#
# simuler("ElMe bras gauche bas, droit descend", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_ElMe.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_ElMe.CoM(X0[:m_ElMe.nbQ()]).to_array()
# bassin = m_ElMe.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_ElMe.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_ElMe.nbQ():m_ElMe.nbQ()+3] = X0[m_ElMe.nbQ():m_ElMe.nbQ() + 3] + np.cross(r, X0[m_ElMe.nbQ()+3:m_ElMe.nbQ()+6])  # correction pour la translation
#
# simuler("ElMe bras droit bas, gauche descend", m_ElMe, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
# # Benjamin
# # debut bras en haut
# X0 = np.zeros(m_Benjamin.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_Benjamin.CoM(X0[:m_Benjamin.nbQ()]).to_array()
# bassin = m_Benjamin.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_Benjamin.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_Benjamin.nbQ():m_Benjamin.nbQ()+3] = X0[m_Benjamin.nbQ():m_Benjamin.nbQ() + 3] + np.cross(r, X0[m_Benjamin.nbQ()+3:m_Benjamin.nbQ()+6])  # correction pour la translation
#
# simuler("Benjamin bras en haut", m_Benjamin, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("Benjamin bras descendent", m_Benjamin, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("Benjamin bras gauche descend", m_Benjamin, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("Benjamin bras droit descend", m_Benjamin, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_Benjamin.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_Benjamin.CoM(X0[:m_Benjamin.nbQ()]).to_array()
# bassin = m_Benjamin.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_Benjamin.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_Benjamin.nbQ():m_Benjamin.nbQ()+3] = X0[m_Benjamin.nbQ():m_Benjamin.nbQ() + 3] + np.cross(r, X0[m_Benjamin.nbQ()+3:m_Benjamin.nbQ()+6])  # correction pour la translation
#
# simuler("Benjamin bras gauche bas, droit descend", m_Benjamin, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_Benjamin.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_Benjamin.CoM(X0[:m_Benjamin.nbQ()]).to_array()
# bassin = m_Benjamin.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_Benjamin.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_Benjamin.nbQ():m_Benjamin.nbQ()+3] = X0[m_Benjamin.nbQ():m_Benjamin.nbQ() + 3] + np.cross(r, X0[m_Benjamin.nbQ()+3:m_Benjamin.nbQ()+6])  # correction pour la translation
#
# simuler("Benjamin bras droit bas, gauche descend", m_Benjamin, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
# # AlAd
# # debut bras en haut
# X0 = np.zeros(m_AlAd.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_AlAd.CoM(X0[:m_AlAd.nbQ()]).to_array()
# bassin = m_AlAd.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_AlAd.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_AlAd.nbQ():m_AlAd.nbQ()+3] = X0[m_AlAd.nbQ():m_AlAd.nbQ() + 3] + np.cross(r, X0[m_AlAd.nbQ()+3:m_AlAd.nbQ()+6])  # correction pour la translation
#
# simuler("AlAd bras en haut", m_AlAd, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("AlAd bras descendent", m_AlAd, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("AlAd bras gauche descend", m_AlAd, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("AlAd bras droit descend", m_AlAd, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_AlAd.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_AlAd.CoM(X0[:m_AlAd.nbQ()]).to_array()
# bassin = m_AlAd.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_AlAd.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_AlAd.nbQ():m_AlAd.nbQ()+3] = X0[m_AlAd.nbQ():m_AlAd.nbQ() + 3] + np.cross(r, X0[m_AlAd.nbQ()+3:m_AlAd.nbQ()+6])  # correction pour la translation
#
# simuler("AlAd bras gauche bas, droit descend", m_AlAd, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_AlAd.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_AlAd.CoM(X0[:m_AlAd.nbQ()]).to_array()
# bassin = m_AlAd.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_AlAd.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_AlAd.nbQ():m_AlAd.nbQ()+3] = X0[m_AlAd.nbQ():m_AlAd.nbQ() + 3] + np.cross(r, X0[m_AlAd.nbQ()+3:m_AlAd.nbQ()+6])  # correction pour la translation
#
# simuler("AlAd bras droit bas, gauche descend", m_AlAd, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
#
# # AdCh
# # debut bras en haut
# X0 = np.zeros(m_AdCh.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Q0
#
# CoM_func = m_AdCh.CoM(X0[:m_AdCh.nbQ()]).to_array()
# bassin = m_AdCh.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_AdCh.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_AdCh.nbQ():m_AdCh.nbQ()+3] = X0[m_AdCh.nbQ():m_AdCh.nbQ() + 3] + np.cross(r, X0[m_AdCh.nbQ()+3:m_AdCh.nbQ()+6])  # correction pour la translation
#
# simuler("AdCh bras en haut", m_AdCh, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_en_haut, viz=True)
# simuler("AdCh bras descendent", m_AdCh, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_descendent, viz=True)
# simuler("AdCh bras gauche descend", m_AdCh, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
# simuler("AdCh bras droit descend", m_AdCh, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras droit en haut, gauche bas
# X0 = np.zeros(m_AdCh.nbQ() * 2)
# X0[DROITE] = -Q0
# X0[GAUCHE] = Qf
#
# CoM_func = m_AdCh.CoM(X0[:m_AdCh.nbQ()]).to_array()
# bassin = m_AdCh.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_AdCh.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_AdCh.nbQ():m_AdCh.nbQ()+3] = X0[m_AdCh.nbQ():m_AdCh.nbQ() + 3] + np.cross(r, X0[m_AdCh.nbQ()+3:m_AdCh.nbQ()+6])  # correction pour la translation
#
# simuler("AdCh bras gauche bas, droit descend", m_AdCh, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_droit_descend, viz=True)
#
# # debut bras gauche en haut, droit bas
# X0 = np.zeros(m_AdCh.nbQ() * 2)
# X0[DROITE] = -Qf
# X0[GAUCHE] = Q0
#
# CoM_func = m_AdCh.CoM(X0[:m_AdCh.nbQ()]).to_array()
# bassin = m_AdCh.globalJCS(0).to_array()
# QCoM = CoM_func.reshape(1, 3)
# Qbassin = bassin[-1, :3]
# r = QCoM - Qbassin
#
# X0[m_AdCh.nbQ() + 3] = 2 * np.pi  # Salto rot
# X0[m_AdCh.nbQ():m_AdCh.nbQ()+3] = X0[m_AdCh.nbQ():m_AdCh.nbQ() + 3] + np.cross(r, X0[m_AdCh.nbQ()+3:m_AdCh.nbQ()+6])  # correction pour la translation
#
# simuler("AdCh bras droit bas, gauche descend", m_AdCh, N, t0, tf, T0, Tf, Q0, Qf, X0, action_bras=bras_gauche_descend, viz=True)
#
