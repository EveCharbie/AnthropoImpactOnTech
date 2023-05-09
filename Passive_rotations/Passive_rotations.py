import os

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
    Qddot_J[GAUCHE - m.nbRoot()] = a + Kp * (p - x0[GAUCHE]) + Kv * (v - x0[m.nbQ() + GAUCHE])
    Qddot_J[DROITE - m.nbRoot()] = -a + Kp * (-p - x0[DROITE]) + Kv * (-v - x0[m.nbQ() + DROITE])

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

def jambes_tilt(m, x0, t, T0, Tf, Q0, Qf):
    global YrotC

    Kp = 10.0
    Kv = 3.0
    p, v, a = Quintic(t, T0, Tf, Q0, Qf)
    Qddot_J = np.zeros(m.nbQ() - m.nbRoot())
    Qddot_J[YrotC - m.nbRoot()] = -a + Kp * (-p - x0[YrotC]) + Kv * (-v - x0[m.nbQ() + YrotC])
    x = dynamics_root(m, x0, Qddot_J)
    return x


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
    n = len(values)
    for i in range(n):
        file = open(f"/passive rotations results/Q_passive_rotations/{titre}{titles[i]}.pkl", 'wb')
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
    # fig.savefig(f'/passive rotations results/Graphs/{suptitre}.pdf')
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
        file = open(f"/passive rotations results/Q_passive_rotations/{titre}{titles[i]}.pkl", 'wb')
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
    # figrot.savefig(f"passive rotations results/Graphs/{suptitre}.pdf")
    # figrot.show()


workbook = xlsxwriter.Workbook("/passive rotations results/degrees_of_liberty.xlsx")

# The workbook object is then used to add new
# worksheet via the add_worksheet() method.
worksheet = workbook.add_worksheet()

# Use the worksheet object to write
# data via the write() method.
worksheet.write("A1", "Athlete")
worksheet.write("B1", "Starting")
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
    ):
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


    if viz:
        Qddot = np.zeros(X_tous.shape)
        dsym2(X_tous, dt, Qddot)
        Qddot[np.logical_and(Qddot < 1e-14, Qddot > -1e-14)] = 0
        Qddot = Qddot[:, m.nbQ() :]

        nom = nom.partition(".bioMod")[0].removeprefix("Models/")
        suptitre = nom.removeprefix("Models/") + " " + situation
        plot_Q_Qdot_bras(model, t, X_tous, Qddot, titre=suptitre)
        plot_Q_Qdot_bassin(model, t, X_tous, Qddot, titre=suptitre)

        b = bioviz.Viz(f'{models_path}{model_name}', show_floor=False)
        b.load_movement(X_tous[:, : model.nbQ()].T)
        b.exec()


N = 100


GAUCHE =  11 #32 -> 7 # 42 -> 24; 10 -> 9
DROITE = 7 # 32->11  # 42 -> 15; 10 -> 7
YrotC = 15

t0 = 0.0
tf = 1.0
T0 = 0.0
Tf = 0.2
Q0 = -2.9
Qf = 0
Q0_tilt = 0
Qf_tilt = 3.14/16

models_path = '/Models/Models_Lisa/'
for i, model_name in enumerate(os.listdir(models_path)):
    if model_name.endswith('bioMod'):
        model = biorbd.Model(f'{models_path}/{model_name}')
        name = model_name.removesuffix('.bioMod')
        column = 0
        row = i * 7 + 1

        # debut bras en haut
        X0 = np.zeros(model.nbQ() * 2)
        X0[DROITE] = -Q0
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

        # row = i +1
        situation = "debut bras en haut"
        simuler(
            f"{name} bras en haut",
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
            f"{name} bras descendent",
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
            viz=False,
            row=row,
            column=column,
            situation=situation,
        )

        row += 1
        simuler(
            f"{name} bras gauche descend",
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
            f"{name} bras droit descend",
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
        situation = "debut bras droit en haut, gauche bas"
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
            f"{name} bras gauche bas, droit descend",
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

        situation = "debut bras gauche en haut, droit bas"
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
            f"{name} bras droit bas, gauche descend",
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

        situation = "bras en bas"
        X0 = np.zeros(model.nbQ() * 2)
        X0[DROITE] = Qf
        X0[GAUCHE] = Qf
        X0[YrotC] = Q0_tilt

        CoM_func = model.CoM(X0[: model.nbQ()]).to_array()
        bassin = model.globalJCS(0).to_array()
        QCoM = CoM_func.reshape(1, 3)
        Qbassin = bassin[-1, :3]
        r = QCoM - Qbassin

        X0[model.nbQ() + 3] = 2 * np.pi  # Salto rot
        X0[model.nbQ(): model.nbQ() + 3] = X0[model.nbQ(): model.nbQ() + 3] + np.cross(
            r, X0[model.nbQ() + 3: model.nbQ() + 6]
        )  # correction pour la translation
        row += 1
        simuler(
            f"{name} bras en  bas, jambes tilt",
            model,
            N,
            t0,
            tf,
            T0,
            Tf,
            Q0_tilt,
            Qf_tilt,
            X0,
            action_bras=jambes_tilt,
            viz=False,
            row=row,
            column=column,
            situation=situation,
        )
workbook.close()
print('fin')
