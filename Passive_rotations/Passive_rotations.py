import os

import biorbd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.integrate
import bioviz
import pickle
import xlsxwriter

"""
This code allows to simulate the effect one segment's movement on the rotations of the floating base.
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
    nb_root = m.nbRoot()
    Qddot = np.hstack((np.zeros((nb_root,)), Qddot_J))  # qddot2
    NLEffects = m.InverseDynamics(Q, Qdot, Qddot).to_array()
    mass_matrix = m.massMatrix(Q).to_array()
    Qddot_R = np.linalg.solve(mass_matrix[:nb_root, :nb_root], -NLEffects[:nb_root])
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

def bras_gauche_descend_root_YZ_fixe(m, x0, t, T0, Tf, Q0, Qf):
    global GAUCHE
    Kp = 10.0
    Kv = 3.0
    p, v, a = Quintic(t, T0, Tf, Q0, Qf)
    Qddot_J = np.zeros(m.nbQ() - m.nbRoot())
    Qddot_J[GAUCHE-2 - m.nbRoot()] = a + Kp * (p - x0[GAUCHE-2]) + Kv * (v - x0[m.nbQ() + GAUCHE-2])

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
        file = open(f"passive rotations results/Q_passive_rotations/{titre}{titles[i]}.pkl", 'wb')
        pickle.dump(values[i], file)
        file.close()

    fig, ((axQG, axQD), (axQdG, axQdD), (axQddG, axQddD)) = plt.subplots(3, 2, figsize=(6, 10))
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
    fig.savefig(f'passive rotations results/{suptitre[:-1]}.png', dpi=300)
    fig.show()


def plot_Q_Qdot_bassin(m, t, X_tous, Qddot, titre=""):
    nb_q = m.nbQ()
    cmap = cm.get_cmap('viridis')

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
        file = open(f"passive rotations results/Q_passive_rotations/{titre}{titles[i]}.pkl", 'wb')
        pickle.dump(values[i], file)
        file.close()

    fig, (axp, axv, axa) = plt.subplots(3, 1, figsize=(6, 10))
    axp.plot(t, QX, label="X", color=cmap(3/3))
    axp.plot(t, QY, label="Y", color=cmap(2/3))
    axp.plot(t, QZ, label="Z", color=cmap(1/3))
    axp.set_ylabel("position (m)")
    axp.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    axv.plot(t, QdotX, label="Xdot", color=cmap(3/3))
    axv.plot(t, QdotY, label="Ydot", color=cmap(2/3))
    axv.plot(t, QdotZ, label="Zdot", color=cmap(1/3))
    axv.set_ylabel("vitesse (m/s)")
    axv.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    axa.plot(t, QddotX, label="Xddot", color=cmap(3/3))
    axa.plot(t, QddotY, label="Yddot", color=cmap(2/3))
    axa.plot(t, QddotZ, label="Zddot", color=cmap(1/3))
    axa.set_ylabel("acceleration (m/s$^2$)")
    axa.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    axa.set_xlabel("temps (s)")
    suptitre = "Translation du bassin" + f" - {titre}" if titre != "" else ""
    fig.suptitle(suptitre)
    fig.tight_layout()
    # fig.show()

    figrot, (axprot, axvrot, axarot) = plt.subplots(3, 1, figsize=(6, 10))
    axprot.plot(t, QrotX * 180 / np.pi, label="Somersault", color=cmap(3/3))
    axprot.plot(t, QrotY * 180 / np.pi, label="Tilt", color=cmap(2/3))
    axprot.plot(t, QrotZ * 180 / np.pi, label="Twist", color=cmap(1/3))
    axprot.set_ylabel("Joint angles [$^\circ$]")
    axprot.legend(bbox_to_anchor=(0.4, 0.99))

    axvrot.plot(t, QdotrotX, label="Rot Xdot", color=cmap(3/3))
    axvrot.plot(t, QdotrotY, label="Rot Ydot", color=cmap(2/3))
    axvrot.plot(t, QdotrotZ, label="Rot Zdot", color=cmap(1/3))
    axvrot.set_ylabel("vitesse (rad/s)")
    axvrot.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    axarot.plot(t, QddotrotX, label="Rot Xddot", color=cmap(3/3))
    axarot.plot(t, QddotrotY, label="Rot Yddot", color=cmap(2/3))
    axarot.plot(t, QddotrotZ, label="Rot Zddot", color=cmap(1/3))
    axarot.set_ylabel("acceleration (rad/s$^2$)")
    axarot.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    axarot.set_xlabel("Time [s]")
    suptitre = "Rotation du bassin" + f" - {titre}" if titre != "" else ""
    figrot.suptitle(suptitre)
    figrot.tight_layout()
    figrot.savefig(f"passive rotations results/{suptitre[:-1]}.png", dpi=300)
    figrot.show()


workbook = xlsxwriter.Workbook("passive rotations results/degrees_of_liberty.xlsx")
worksheet = workbook.add_worksheet()
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
    if not nom.endswith("YZ fixe"):
        worksheet.write(row, column, X_tous[-1, 4] / 2 / np.pi)
        column += 1
        worksheet.write(row, column, X_tous[-1, 5] / 2 / np.pi)


    if viz:
        Qddot = np.zeros(X_tous.shape)
        dsym2(X_tous, dt, Qddot)
        Qddot[np.logical_and(Qddot < 1e-14, Qddot > -1e-14)] = 0
        Qddot = Qddot[:, m.nbQ() :]

        if nom.endswith("YZ fixe"):
            model_path = f'{models_path}{model_name.removesuffix(".bioMod")}_rootYZfixed.bioMod'
        else:
            model_path = f'{models_path}{model_name}'
        model = biorbd.Model(model_path)
        nom = nom.partition(".bioMod")[0].removeprefix("Models/")
        suptitre = nom.removeprefix("Models/") + " " + situation
        plot_Q_Qdot_bras(model, t, X_tous, Qddot, titre=suptitre)
        plot_Q_Qdot_bassin(model, t, X_tous, Qddot, titre=suptitre)

        # b = bioviz.Viz(model_path, show_floor=False)
        # b.load_movement(X_tous[:, : model.nbQ()].T)
        # b.exec()

        return X_tous

def plot_spline_limb_movements(time, X_simulated_arms, X_simulated_hips):
    cmap = cm.get_cmap('viridis')
    global DROITE, YrotC
    fig, ax = plt.subplots(1, 1, figsize=(4*1.11, 3.3*1.11))
    ax.plot(time, X_simulated_arms[:, DROITE] * 180 / np.pi, label="Right arm elevation", color=cmap(3/3))
    ax.plot(time, np.zeros(time.shape), label="Left arm elevation", color=cmap(2/3))
    ax.plot(time, X_simulated_hips[:, YrotC] * 180 / np.pi, label="Hips lateral flexion", color=cmap(1/3))
    ax.set_ylabel("Joint angles [$^\circ$]")
    ax.set_xlabel("Time [s]")

    fig.suptitle("Splines")
    fig.legend(bbox_to_anchor=(0.95, 0.85))
    fig.tight_layout()
    fig.savefig(f'passive rotations results/Splines.png', dpi=300)
    fig.show()


N = 100

GAUCHE = 11  #32 -> 6 # 42 -> 24; 10 -> 9
DROITE = 7  # 32->10  # 42 -> 15; 10 -> 7
YrotC = 15

t0 = 0.0
tf = 1.0
T0 = 0.0
Tf = 0.2
Q0 = -2.9
Qf = 0
Q0_tilt = 0
Qf_tilt = 3.14/16

models_path = '../Models/Models_Lisa/'
row = 0
for model_name in os.listdir(models_path):
    if not model_name.endswith('bioMod'):
        continue
    if model_name.endswith('_rootYZfixed.bioMod'):
        continue
    model = biorbd.Model(f'{models_path}/{model_name}')
    model_fixe = biorbd.Model(f"{models_path}/{model_name.removesuffix('.bioMod')}_rootYZfixed.bioMod")
    name = model_name.removesuffix('.bioMod')

    column = 0

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

    row += 1
    situation = "debut bras en haut"
    X_simulated = simuler(
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
    X_simulated = simuler(
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
    X_simulated = simuler(
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
    X_simulated_arm = simuler(
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
    X0[model.nbQ(): model.nbQ() + 3] = X0[model.nbQ(): model.nbQ() + 3] + np.cross(
        r, X0[model.nbQ() + 3: model.nbQ() + 6]
    )  # correction pour la translation
    row += 1
    X_simulated = simuler(
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
    X_simulated_hips = simuler(
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

    # debut bras en haut
    nb_root = model_fixe.nbRoot()
    X0 = np.zeros(model_fixe.nbQ() * 2)
    X0[DROITE-2] = -Q0
    X0[GAUCHE-2] = Q0

    CoM_func = model_fixe.CoM(X0[: model_fixe.nbQ()]).to_array()
    bassin = model_fixe.globalJCS(0).to_array()
    QCoM = CoM_func.reshape(1, 3)
    Qbassin = bassin[-1, :3]
    r = QCoM - Qbassin

    X0[model_fixe.nbQ() + 3] = 2 * np.pi  # Salto rot
    pelvis_angular_velocity = np.array([X0[model.nbQ() + 3], 0, 0])
    Correction = X0[model_fixe.nbQ(): model_fixe.nbQ() + 3] + np.cross(
        r, pelvis_angular_velocity
    )  # correction pour la translation
    X0[model_fixe.nbQ() : model_fixe.nbQ() + 3] = Correction[0]


    row += 1
    simuler(
        f"{name} bras gauche descend, YZ fixe",
        model_fixe,
        N,
        t0,
        tf,
        T0,
        Tf,
        Q0,
        Qf,
        X0,
        action_bras=bras_gauche_descend_root_YZ_fixe,
        viz=False,
        row=row,
        column=column,
        situation=situation,
    )

    time, _ = np.linspace(t0, tf, num=N + 1, retstep=True)
    plot_spline_limb_movements(time, X_simulated_arm, X_simulated_hips)

workbook.close()
print('fin')
