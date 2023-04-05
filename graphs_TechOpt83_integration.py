import biorbd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from decimal import *
# import Colormap as normalize
import bioviz

# from ..misc.enums import (
#             Shooting,
#             )
# import Shooting

from IPython import embed
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os

import numpy as np
import biorbd_casadi as biorbd
from casadi import MX, Function
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    BiorbdModel,
    InitialGuessList,
    InterpolationType,
    OdeSolver,
    Node,
    Solver,
    Solution,
    BiMappingList,
    CostType,
    ConstraintList,
    ConstraintFcn,
    PenaltyNodeList,
    MultiStart,
    Solution,
    MagnitudeType,
    NoisedInitialGuess,
    BiorbdModel,
    Shooting,
)
import time


def minimize_dofs(all_pn: PenaltyNodeList, dofs: list, targets: list) -> MX:
    diff = 0
    for i, dof in enumerate(dofs):
        diff += (all_pn.nlp.states["q"].mx[dof] - targets[i]) ** 2
    return all_pn.nlp.mx_to_cx("minimize_dofs", diff, all_pn.nlp.states["q"])


def prepare_ocp(
    biorbd_model_path: str,
    nb_twist: int,
    seed: int,
    n_threads: int = 1,
    ode_solver: OdeSolver = OdeSolver.RK4(),
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    final_time = 1.87
    n_shooting = (40, 100, 100, 100, 40)

    bio_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
    )

    nb_q = bio_model[0].nb_q
    nb_qdot = bio_model[0].nb_qdot
    nb_qddot_joints = nb_q - bio_model[0].nb_root

    # Pour la lisibilite
    X = 0
    Y = 1
    Z = 2
    Xrot = 3
    Yrot = 4
    Zrot = 5
    ZrotBD = 6
    YrotBD = 7
    ZrotABD = 8
    XrotABD = 9
    ZrotBG = 10
    YrotBG = 11
    ZrotABG = 12
    XrotABG = 13
    XrotC = 14
    YrotC = 15
    vX = 0 + nb_q
    vY = 1 + nb_q
    vZ = 2 + nb_q
    vXrot = 3 + nb_q
    vYrot = 4 + nb_q
    vZrot = 5 + nb_q
    vZrotBD = 6 + nb_q
    vYrotBD = 7 + nb_q
    vZrotABD = 8 + nb_q
    vYrotABD = 9 + nb_q
    vZrotBG = 10 + nb_q
    vYrotBG = 11 + nb_q
    vZrotABG = 12 + nb_q
    vYrotABG = 13 + nb_q
    vXrotC = 14 + nb_q
    vYrotC = 15 + nb_q

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, marker_index=1, weight=-1)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=1
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=2
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=3
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=4
    )

    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.0, max_bound=1.0, weight=100000, phase=0)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=final_time, weight=.01, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.0, max_bound=1.0, weight=100000, phase=2)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=final_time, weight=.01, phase=3)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=final_time, weight=.01, phase=4)

    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="MidMainG",
        second_marker="CibleMainG",
        weight=1000,
        phase=0,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="MidMainD",
        second_marker="CibleMainD",
        weight=1000,
        phase=0,
    )

    # arrete de gigoter les bras
    les_bras = [ZrotBD, YrotBD, ZrotABD, XrotABD, ZrotBG, YrotBG, ZrotABG, XrotABG]
    les_coudes = [ZrotABD, XrotABD, ZrotABG, XrotABG]
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        dofs=les_coudes,
        targets=np.zeros(len(les_coudes)),
        weight=1000,
        phase=0,
    )
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        dofs=les_bras,
        targets=np.zeros(len(les_bras)),
        weight=10,
        phase=0,
    )
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        dofs=les_bras,
        targets=np.zeros(len(les_bras)),
        weight=10,
        phase=1,
    )
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        dofs=les_bras,
        targets=np.zeros(len(les_bras)),
        weight=10,
        phase=2,
    )
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        dofs=les_bras,
        targets=np.zeros(len(les_bras)),
        weight=10,
        phase=3,
    )
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        dofs=les_bras,
        targets=np.zeros(len(les_bras)),
        weight=10,
        phase=4,
    )
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        dofs=les_coudes,
        targets=np.zeros(len(les_coudes)),
        weight=1000,
        phase=4,
    )
    # ouvre les hanches rapidement apres la vrille
    objective_functions.add(
        minimize_dofs, custom_type=ObjectiveFcn.Mayer, node=Node.END, dofs=[XrotC], targets=[0], weight=10000, phase=3
    )

    # Dynamics
    dynamics = DynamicsList()

    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)

    qddot_joints_min, qddot_joints_max, qddot_joints_init = -500, 500, 0
    u_bounds = BoundsList()
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)

    u_init = InitialGuessList()
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)

    u_init.add_noise(
        bounds=u_bounds,
        magnitude=0.2,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=n_shooting,
        seed=seed,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))

    # Pour la lisibilite
    DEBUT, MILIEU, FIN = 0, 1, 2

    #
    # Contraintes de position: PHASE 0 la montee en carpe
    #

    zmax = 8
    # 12 / 8 * final_time**2 + 1  # une petite marge

    # deplacement
    x_bounds[0].min[X, :] = -0.1
    x_bounds[0].max[X, :] = 0.1
    x_bounds[0].min[Y, :] = -1.0
    x_bounds[0].max[Y, :] = 1.0
    x_bounds[0].min[: Z + 1, DEBUT] = 0
    x_bounds[0].max[: Z + 1, DEBUT] = 0
    x_bounds[0].min[Z, MILIEU:] = 0
    x_bounds[0].max[Z, MILIEU:] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[0].min[Xrot, :] = 0
    # 2 * 3.14 + 3 / 2 * 3.14 - .2
    x_bounds[0].max[Xrot, :] = -0.50 + 3.14
    x_bounds[0].min[Xrot, DEBUT] = 0.50  # penche vers l'avant un peu carpe
    x_bounds[0].max[Xrot, DEBUT] = 0.50
    x_bounds[0].min[Xrot, MILIEU:] = 0
    x_bounds[0].max[Xrot, MILIEU:] = 4 * 3.14 + 0.1  # salto
    # limitation du tilt autour de y
    x_bounds[0].min[Yrot, DEBUT] = 0
    x_bounds[0].max[Yrot, DEBUT] = 0
    x_bounds[0].min[Yrot, MILIEU:] = -3.14 / 16  # vraiment pas suppose tilte
    x_bounds[0].max[Yrot, MILIEU:] = 3.14 / 16
    # la vrille autour de z
    x_bounds[0].min[Zrot, DEBUT] = 0
    x_bounds[0].max[Zrot, DEBUT] = 0
    x_bounds[0].min[Zrot, MILIEU:] = -0.1  # pas de vrille dans cette phase
    x_bounds[0].max[Zrot, MILIEU:] = 0.1

    # bras droit
    x_bounds[0].min[YrotBD, DEBUT] = 2.9  # debut bras aux oreilles
    x_bounds[0].max[YrotBD, DEBUT] = 2.9
    x_bounds[0].min[ZrotBD, DEBUT] = 0
    x_bounds[0].max[ZrotBD, DEBUT] = 0
    # bras gauche
    x_bounds[0].min[YrotBG, DEBUT] = -2.9  # debut bras aux oreilles
    x_bounds[0].max[YrotBG, DEBUT] = -2.9
    x_bounds[0].min[ZrotBG, DEBUT] = 0
    x_bounds[0].max[ZrotBG, DEBUT] = 0

    # coude droit
    x_bounds[0].min[ZrotABD : XrotABD + 1, DEBUT] = 0
    x_bounds[0].max[ZrotABD : XrotABD + 1, DEBUT] = 0
    # coude gauche
    x_bounds[0].min[ZrotABG : XrotABG + 1, DEBUT] = 0
    x_bounds[0].max[ZrotABG : XrotABG + 1, DEBUT] = 0

    # le carpe
    x_bounds[0].min[XrotC, DEBUT] = -0.50  # depart un peu ferme aux hanches
    x_bounds[0].max[XrotC, DEBUT] = -0.50
    x_bounds[0].max[XrotC, FIN] = -2.5
    # x_bounds[0].min[XrotC, FIN] = 2.7  # min du modele
    # le dehanchement
    x_bounds[0].min[YrotC, DEBUT] = 0
    x_bounds[0].max[YrotC, DEBUT] = 0
    x_bounds[0].min[YrotC, MILIEU:] = -0.1
    x_bounds[0].max[YrotC, MILIEU:] = 0.1

    # Contraintes de vitesse: PHASE 0 la montee en carpe

    vzinit = 9.81 / (2 * final_time)  # vitesse initiale en z du CoM pour revenir a terre au temps final

    # decalage entre le bassin et le CoM
    CoM_Q_sym = MX.sym("CoM", nb_q)
    CoM_Q_init = x_bounds[0].min[
        :nb_q, DEBUT
    ]  # min ou max ne change rien a priori, au DEBUT ils sont egaux normalement
    CoM_Q_func = Function("CoM_Q_func", [CoM_Q_sym], [bio_model[0].center_of_mass(CoM_Q_sym)])
    bassin_Q_func = Function(
        "bassin_Q_func", [CoM_Q_sym], [bio_model[0].homogeneous_matrices_in_global(CoM_Q_sym, 0).to_mx()]
    )  # retourne la RT du bassin

    r = (
        np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1, :3]
    )  # selectionne seulement la translation de la RT

    # en xy bassin
    x_bounds[0].min[vX : vY + 1, :] = -10
    x_bounds[0].max[vX : vY + 1, :] = 10
    x_bounds[0].min[vX : vY + 1, DEBUT] = -0.5
    x_bounds[0].max[vX : vY + 1, DEBUT] = 0.5
    # z bassin
    x_bounds[0].min[vZ, :] = -50
    x_bounds[0].max[vZ, :] = 50
    x_bounds[0].min[vZ, DEBUT] = vzinit - 0.5
    x_bounds[0].max[vZ, DEBUT] = vzinit + 0.5

    # autour de x
    x_bounds[0].min[vXrot, :] = 0.5  # d'apres une observation video
    x_bounds[0].max[vXrot, :] = 20  # aussi vite que nécessaire, mais ne devrait pas atteindre cette vitesse
    # autour de y
    x_bounds[0].min[vYrot, :] = -50
    x_bounds[0].max[vYrot, :] = 50
    x_bounds[0].min[vYrot, DEBUT] = 0
    x_bounds[0].max[vYrot, DEBUT] = 0
    # autour de z
    x_bounds[0].min[vZrot, :] = -50
    x_bounds[0].max[vZrot, :] = 50
    x_bounds[0].min[vZrot, DEBUT] = 0
    x_bounds[0].max[vZrot, DEBUT] = 0

    # tenir compte du decalage entre bassin et CoM avec la rotation
    # Qtransdot = Qtransdot + v cross Qrotdot
    borne_inf = (x_bounds[0].min[vX : vZ + 1, DEBUT] + np.cross(r, x_bounds[0].min[vXrot : vZrot + 1, DEBUT]))[0]
    borne_sup = (x_bounds[0].max[vX : vZ + 1, DEBUT] + np.cross(r, x_bounds[0].max[vXrot : vZrot + 1, DEBUT]))[0]
    x_bounds[0].min[vX : vZ + 1, DEBUT] = (
        min(borne_sup[0], borne_inf[0]),
        min(borne_sup[1], borne_inf[1]),
        min(borne_sup[2], borne_inf[2]),
    )
    x_bounds[0].max[vX : vZ + 1, DEBUT] = (
        max(borne_sup[0], borne_inf[0]),
        max(borne_sup[1], borne_inf[1]),
        max(borne_sup[2], borne_inf[2]),
    )

    # bras droit
    x_bounds[0].min[vZrotBD : vYrotBD + 1, :] = -50
    x_bounds[0].max[vZrotBD : vYrotBD + 1, :] = 50
    x_bounds[0].min[vZrotBD : vYrotBD + 1, DEBUT] = 0
    x_bounds[0].max[vZrotBD : vYrotBD + 1, DEBUT] = 0
    # bras droit
    x_bounds[0].min[vZrotBG : vYrotBG + 1, :] = -50
    x_bounds[0].max[vZrotBG : vYrotBG + 1, :] = 50
    x_bounds[0].min[vZrotBG : vYrotBG + 1, DEBUT] = 0
    x_bounds[0].max[vZrotBG : vYrotBG + 1, DEBUT] = 0

    # coude droit
    x_bounds[0].min[vZrotABD : vYrotABD + 1, :] = -50
    x_bounds[0].max[vZrotABD : vYrotABD + 1, :] = 50
    x_bounds[0].min[vZrotABD : vYrotABD + 1, DEBUT] = 0
    x_bounds[0].max[vZrotABD : vYrotABD + 1, DEBUT] = 0
    # coude gauche
    x_bounds[0].min[vZrotABD : vYrotABG + 1, :] = -50
    x_bounds[0].max[vZrotABD : vYrotABG + 1, :] = 50
    x_bounds[0].min[vZrotABG : vYrotABG + 1, DEBUT] = 0
    x_bounds[0].max[vZrotABG : vYrotABG + 1, DEBUT] = 0

    # du carpe
    x_bounds[0].min[vXrotC, :] = -50
    x_bounds[0].max[vXrotC, :] = 50
    x_bounds[0].min[vXrotC, DEBUT] = 0
    x_bounds[0].max[vXrotC, DEBUT] = 0
    # du dehanchement
    x_bounds[0].min[vYrotC, :] = -50
    x_bounds[0].max[vYrotC, :] = 50
    x_bounds[0].min[vYrotC, DEBUT] = 0
    x_bounds[0].max[vYrotC, DEBUT] = 0

    #
    # Contraintes de position: PHASE 1 le salto carpe
    #

    # deplacement
    x_bounds[1].min[X, :] = -0.1
    x_bounds[1].max[X, :] = 0.1
    x_bounds[1].min[Y, :] = -1.0
    x_bounds[1].max[Y, :] = 1.0
    x_bounds[1].min[Z, :] = 0
    x_bounds[1].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[1].min[Xrot, :] = 0
    # 2 * 3.14 + 3 / 2 * 3.14 - .2
    x_bounds[1].max[Xrot, :] = -0.50 + 4 * 3.14
    x_bounds[1].min[Xrot, FIN] = 2 * 3.14 - 0.1
    # limitation du tilt autour de y
    x_bounds[1].min[Yrot, :] = -3.14 / 16
    x_bounds[1].max[Yrot, :] = 3.14 / 16
    # la vrille autour de z
    x_bounds[1].min[Zrot, :] = -0.1
    x_bounds[1].max[Zrot, :] = 0.1

    # bras droit  f4a a l'ouverture
    # x_bounds[1].min[YrotD, DEBUT] = -2.9  # debut bras aux oreilles
    # x_bounds[1].max[YrotD, DEBUT] = -2.9
    # x_bounds[1].min[ZrotD, DEBUT] = 0
    # x_bounds[1].max[ZrotD, DEBUT] = 0
    # bras gauche
    # x_bounds[1].min[YrotG, DEBUT] = 2.9  # debut bras aux oreilles
    # x_bounds[1].max[YrotG, DEBUT] = 2.9
    # x_bounds[1].min[ZrotG, DEBUT] = 0
    # x_bounds[1].max[ZrotG, DEBUT] = 0

    # le carpe
    x_bounds[1].max[XrotC, :] = -2.5
    # x_bounds[1].min[XrotC, :] = 2.7  # contraint par le model
    # le dehanchement
    x_bounds[1].min[YrotC, DEBUT] = -0.1
    x_bounds[1].max[YrotC, DEBUT] = 0.1

    # Contraintes de vitesse: PHASE 1 le salto carpe

    # en xy bassin
    x_bounds[1].min[vX : vY + 1, :] = -10
    x_bounds[1].max[vX : vY + 1, :] = 10

    # z bassin
    x_bounds[1].min[vZ, :] = -50
    x_bounds[1].max[vZ, :] = 50

    # autour de x
    x_bounds[1].min[vXrot, :] = -50
    x_bounds[1].max[vXrot, :] = 50
    # autour de y
    x_bounds[1].min[vYrot, :] = -50
    x_bounds[1].max[vYrot, :] = 50

    # autour de z
    x_bounds[1].min[vZrot, :] = -50
    x_bounds[1].max[vZrot, :] = 50

    # bras droit
    x_bounds[1].min[vZrotBD : vYrotBD + 1, :] = -50
    x_bounds[1].max[vZrotBD : vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[1].min[vZrotBG : vYrotBG + 1, :] = -50
    x_bounds[1].max[vZrotBG : vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[1].min[vZrotABD : vYrotABD + 1, :] = -50
    x_bounds[1].max[vZrotABD : vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[1].min[vZrotABD : vYrotABG + 1, :] = -50
    x_bounds[1].max[vZrotABD : vYrotABG + 1, :] = 50

    # du carpe
    x_bounds[1].min[vXrotC, :] = -50
    x_bounds[1].max[vXrotC, :] = 50
    # du dehanchement
    x_bounds[1].min[vYrotC, :] = -50
    x_bounds[1].max[vYrotC, :] = 50

    #
    # Contraintes de position: PHASE 2 l'ouverture
    #

    # deplacement
    x_bounds[2].min[X, :] = -0.2
    x_bounds[2].max[X, :] = 0.2
    x_bounds[2].min[Y, :] = -1.0
    x_bounds[2].max[Y, :] = 1.0
    x_bounds[2].min[Z, :] = 0
    x_bounds[2].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[2].min[Xrot, :] = 2 * 3.14 - 0.1
    # 2 * 3.14 + 3 / 2 * 3.14 - .2 #2 * 3.14 - .1       # 2 * 3.14 + 3 / 2 * 3.14 - .2  # 1 salto 3/4
    x_bounds[2].max[Xrot, :] = -0.50 + 4 * 3.14
    # limitation du tilt autour de y
    x_bounds[2].min[Yrot, :] = -3.14 / 4
    x_bounds[2].max[Yrot, :] = 3.14 / 4
    # la vrille autour de z
    x_bounds[2].min[Zrot, :] = 0
    x_bounds[2].max[Zrot, :] = 3.14  # 5 * 3.14

    # bras droit  f4a a l'ouverture
    # x_bounds[2].min[YrotD, DEBUT] = -2.9  # debut bras aux oreilles
    # x_bounds[2].max[YrotD, DEBUT] = -2.9
    # x_bounds[2].min[ZrotD, DEBUT] = 0
    # x_bounds[2].max[ZrotD, DEBUT] = 0
    # bras gauche
    # x_bounds[2].min[YrotG, DEBUT] = 2.9  # debut bras aux oreilles
    # x_bounds[2].max[YrotG, DEBUT] = 2.9
    # x_bounds[2].min[ZrotG, DEBUT] = 0
    # x_bounds[2].max[ZrotG, DEBUT] = 0

    # le carpe
    # x_bounds[2].min[XrotC, DEBUT] = 0
    # x_bounds[2].max[XrotC, DEBUT] = 0
    # x_bounds[2].min[XrotC, FIN] = 2.8  # min du modele
    x_bounds[2].min[XrotC, FIN] = -0.4
    # le dehanchement
    # x_bounds[2].min[YrotC, DEBUT] = -.05
    # x_bounds[2].max[YrotC, DEBUT] = .05
    # x_bounds[2].min[YrotC, MILIEU:] = -.05  # f4a a l'ouverture
    # x_bounds[2].max[YrotC, MILIEU:] = .05

    # Contraintes de vitesse: PHASE 2 l'ouverture

    # en xy bassin
    x_bounds[2].min[vX : vY + 1, :] = -10
    x_bounds[2].max[vX : vY + 1, :] = 10

    # z bassin
    x_bounds[2].min[vZ, :] = -50
    x_bounds[2].max[vZ, :] = 50

    # autour de x
    x_bounds[2].min[vXrot, :] = -50
    x_bounds[2].max[vXrot, :] = 50
    # autour de y
    x_bounds[2].min[vYrot, :] = -50
    x_bounds[2].max[vYrot, :] = 50

    # autour de z
    x_bounds[2].min[vZrot, :] = -50
    x_bounds[2].max[vZrot, :] = 50

    # bras droit
    x_bounds[2].min[vZrotBD : vYrotBD + 1, :] = -50
    x_bounds[2].max[vZrotBD : vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[2].min[vZrotBG : vYrotBG + 1, :] = -50
    x_bounds[2].max[vZrotBG : vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[2].min[vZrotABD : vYrotABD + 1, :] = -50
    x_bounds[2].max[vZrotABD : vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[2].min[vZrotABD : vYrotABG + 1, :] = -50
    x_bounds[2].max[vZrotABD : vYrotABG + 1, :] = 50

    # du carpe
    x_bounds[2].min[vXrotC, :] = -50
    x_bounds[2].max[vXrotC, :] = 50
    # du dehanchement
    x_bounds[2].min[vYrotC, :] = -50
    x_bounds[2].max[vYrotC, :] = 50

    #
    # Contraintes de position: PHASE 3 la vrille et demie
    #

    # deplacement
    x_bounds[3].min[X, :] = -0.2
    x_bounds[3].max[X, :] = 0.2
    x_bounds[3].min[Y, :] = -1.0
    x_bounds[3].max[Y, :] = 1.0
    x_bounds[3].min[Z, :] = 0
    x_bounds[3].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[3].min[Xrot, :] = 0
    x_bounds[3].min[Xrot, :] = 2 * 3.14 - 0.1

    x_bounds[3].max[Xrot, :] = 2 * 3.14 + 3 / 2 * 3.14 + 0.1  # 1 salto 3/4
    x_bounds[3].min[Xrot, FIN] = 2 * 3.14 + 3 / 2 * 3.14 - 0.1
    x_bounds[3].max[Xrot, FIN] = 2 * 3.14 + 3 / 2 * 3.14 + 0.1  # 1 salto 3/4
    # x_bounds[3].max[Xrot, :] = -.50 + 4 * 3.14  # 1 salto 3/4
    # x_bounds[3].min[Xrot, FIN] = 0
    # 2 * 3.14 + 2 * 3.14 - .1
    # x_bounds[3].max[Xrot, FIN] = 2 * 3.14 + 2 * 3.14 + .1  # 1 salto 3/4
    # limitation du tilt autour de y
    x_bounds[3].min[Yrot, :] = -3.14 / 4
    x_bounds[3].max[Yrot, :] = 3.14 / 4
    x_bounds[3].min[Yrot, FIN] = -3.14 / 8
    x_bounds[3].max[Yrot, FIN] = 3.14 / 8
    # la vrille autour de z
    x_bounds[3].min[Zrot, :] = 0
    x_bounds[3].max[Zrot, :] = 5 * 3.14
    x_bounds[3].min[Zrot, FIN] = nb_twist * 3.14 - 0.1  # complete la vrille
    x_bounds[3].max[Zrot, FIN] = nb_twist * 3.14 + 0.1

    # bras droit  f4a la vrille
    # x_bounds[3].min[YrotD, DEBUT] = -2.9  # debut bras aux oreilles
    # x_bounds[3].max[YrotD, DEBUT] = -2.9
    # x_bounds[3].min[ZrotD, DEBUT] = 0
    # x_bounds[3].max[ZrotD, DEBUT] = 0
    # bras gauche
    # x_bounds[3].min[YrotG, DEBUT] = 2.9  # debut bras aux oreilles
    # x_bounds[3].max[YrotG, DEBUT] = 2.9
    # x_bounds[3].min[ZrotG, DEBUT] = 0
    # x_bounds[3].max[ZrotG, DEBUT] = 0

    # le carpe  f4a les jambes
    # x_bounds[3].max[XrotC, :] = 2.8  # max du modele
    x_bounds[3].min[XrotC, :] = -0.4
    # le dehanchement
    # x_bounds[3].min[YrotC, DEBUT] = -.05
    # x_bounds[3].max[YrotC, DEBUT] = .05
    # x_bounds[3].min[YrotC, MILIEU:] = -.05  # f4a a l'ouverture
    # x_bounds[3].max[YrotC, MILIEU:] = .05

    # Contraintes de vitesse: PHASE 3 la vrille et demie

    # en xy bassin
    x_bounds[3].min[vX : vY + 1, :] = -10
    x_bounds[3].max[vX : vY + 1, :] = 10

    # z bassin
    x_bounds[3].min[vZ, :] = -50
    x_bounds[3].max[vZ, :] = 50

    # autour de x
    x_bounds[3].min[vXrot, :] = -50
    x_bounds[3].max[vXrot, :] = 50
    # autour de y
    x_bounds[3].min[vYrot, :] = -50
    x_bounds[3].max[vYrot, :] = 50

    # autour de z
    x_bounds[3].min[vZrot, :] = -50
    x_bounds[3].max[vZrot, :] = 50

    # bras droit
    x_bounds[3].min[vZrotBD : vYrotBD + 1, :] = -50
    x_bounds[3].max[vZrotBD : vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[3].min[vZrotBG : vYrotBG + 1, :] = -50
    x_bounds[3].max[vZrotBG : vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[3].min[vZrotABD : vYrotABD + 1, :] = -50
    x_bounds[3].max[vZrotABD : vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[3].min[vZrotABD : vYrotABG + 1, :] = -50
    x_bounds[3].max[vZrotABD : vYrotABG + 1, :] = 50

    # du carpe
    x_bounds[3].min[vXrotC, :] = -50
    x_bounds[3].max[vXrotC, :] = 50
    # du dehanchement
    x_bounds[3].min[vYrotC, :] = -50
    x_bounds[3].max[vYrotC, :] = 50

    #
    # Contraintes de position: PHASE 4 la reception
    #

    # deplacement
    x_bounds[4].min[X, :] = -0.1
    x_bounds[4].max[X, :] = 0.1
    x_bounds[4].min[Y, FIN] = -0.1
    x_bounds[4].max[Y, FIN] = 0.1
    x_bounds[4].min[Z, :] = 0
    x_bounds[4].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    x_bounds[4].min[Z, FIN] = 0
    x_bounds[4].max[Z, FIN] = 0.1

    # le salto autour de x
    x_bounds[4].min[Xrot, :] = 2 * 3.14 + 3 / 2 * 3.14 - 0.2  # penche vers avant -> moins de salto
    x_bounds[4].max[Xrot, :] = -0.50 + 4 * 3.14  # un peu carpe a la fin
    x_bounds[4].min[Xrot, FIN] = -0.50 + 4 * 3.14 - 0.1  # 2 salto fin un peu carpe
    x_bounds[4].max[Xrot, FIN] = -0.50 + 4 * 3.14 + 0.1  # 2 salto fin un peu carpe
    # limitation du tilt autour de y
    x_bounds[4].min[Yrot, :] = -3.14 / 16
    x_bounds[4].max[Yrot, :] = 3.14 / 16
    # la vrille autour de z
    x_bounds[4].min[Zrot, :] = nb_twist * 3.14 - 0.1  # complete la vrille
    x_bounds[4].max[Zrot, :] = nb_twist * 3.14 + 0.1

    # bras droit
    x_bounds[4].min[YrotBD, FIN] = 2.9 - 0.1  # debut bras aux oreilles
    x_bounds[4].max[YrotBD, FIN] = 2.9 + 0.1
    x_bounds[4].min[ZrotBD, FIN] = -0.1
    x_bounds[4].max[ZrotBD, FIN] = 0.1
    # bras gauche
    x_bounds[4].min[YrotBG, FIN] = -2.9 - 0.1  # debut bras aux oreilles
    x_bounds[4].max[YrotBG, FIN] = -2.9 + 0.1
    x_bounds[4].min[ZrotBG, FIN] = -0.1
    x_bounds[4].max[ZrotBG, FIN] = 0.1

    # coude droit
    x_bounds[4].min[ZrotABD : XrotABD + 1, FIN] = -0.1
    x_bounds[4].max[ZrotABD : XrotABD + 1, FIN] = 0.1
    # coude gauche
    x_bounds[4].min[ZrotABG : XrotABG + 1, FIN] = -0.1
    x_bounds[4].max[ZrotABG : XrotABG + 1, FIN] = 0.1

    # le carpe
    x_bounds[4].min[XrotC, :] = -0.4
    x_bounds[4].min[XrotC, FIN] = -0.60
    x_bounds[4].max[XrotC, FIN] = -0.40  # fin un peu carpe
    # le dehanchement
    x_bounds[4].min[YrotC, FIN] = -0.1
    x_bounds[4].max[YrotC, FIN] = 0.1

    # Contraintes de vitesse: PHASE 4 la reception

    # en xy bassin
    x_bounds[4].min[vX : vY + 1, :] = -10
    x_bounds[4].max[vX : vY + 1, :] = 10

    # z bassin
    x_bounds[4].min[vZ, :] = -50
    x_bounds[4].max[vZ, :] = 50

    # autour de x
    x_bounds[4].min[vXrot, :] = -50
    x_bounds[4].max[vXrot, :] = 50
    # autour de y
    x_bounds[4].min[vYrot, :] = -50
    x_bounds[4].max[vYrot, :] = 50

    # autour de z
    x_bounds[4].min[vZrot, :] = -50
    x_bounds[4].max[vZrot, :] = 50

    # bras droit
    x_bounds[4].min[vZrotBD : vYrotBD + 1, :] = -50
    x_bounds[4].max[vZrotBD : vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[4].min[vZrotBG : vYrotBG + 1, :] = -50
    x_bounds[4].max[vZrotBG : vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[4].min[vZrotABD : vYrotABD + 1, :] = -50
    x_bounds[4].max[vZrotABD : vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[4].min[vZrotABD : vYrotABG + 1, :] = -50
    x_bounds[4].max[vZrotABD : vYrotABG + 1, :] = 50

    # du carpe
    x_bounds[4].min[vXrotC, :] = -50
    x_bounds[4].max[vXrotC, :] = 50
    # du dehanchement
    x_bounds[4].min[vYrotC, :] = -50
    x_bounds[4].max[vYrotC, :] = 50

    #
    # Initial guesses
    #
    x0 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x1 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x2 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x3 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x4 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))

    # bras droit  f4a la vrille
    # décollage prise del aposition carpée
    x0[Xrot, 0] = 0.50
    x0[ZrotBG] = -0.75
    x0[ZrotBD] = 0.75
    x0[YrotBG, 0] = -2.9
    x0[YrotBD, 0] = 2.9
    x0[YrotBG, 1] = -1.35
    x0[YrotBD, 1] = 1.35
    x0[XrotC, 0] = -0.5
    x0[XrotC, 1] = -2.6

    # rotater en salto (x) en carpé
    x1[ZrotBG] = -0.75
    x1[ZrotBD] = 0.75
    x1[Xrot, 1] = 2 * 3.14
    x1[YrotBG] = -1.35
    x1[YrotBD] = 1.35
    x1[XrotC] = -2.6

    # ouverture des hanches
    x2[Xrot] = 2 * 3.14
    x2[Zrot, 1] = 0.2
    x2[ZrotBG, 0] = -0.75
    x2[ZrotBD, 0] = 0.75
    x2[YrotBG, 0] = -1.35
    x2[YrotBD, 0] = 1.35
    x2[XrotC, 0] = -2.6

    # Vrille en position tendue
    x3[Xrot, 0] = 2 * 3.14
    x3[Xrot, 1] = 2 * 3.14 + 3 / 2 * 3.14
    x3[Zrot, 0] = 0  # METTRE 0 ?
    x3[Zrot, 1] = nb_twist * 3.14

    # Aterrissage (réduire le tilt)
    x4[Xrot, 0] = 2 * 3.14 + 3 / 2 * 3.14
    x4[Xrot, 1] = 4 * 3.14
    x4[Zrot] = nb_twist * 3.14
    x4[XrotC, 1] = -0.5

    x_init = InitialGuessList()

    x_init.add(x0, interpolation=InterpolationType.LINEAR)
    x_init.add(x1, interpolation=InterpolationType.LINEAR)
    x_init.add(x2, interpolation=InterpolationType.LINEAR)
    x_init.add(x3, interpolation=InterpolationType.LINEAR)
    x_init.add(x4, interpolation=InterpolationType.LINEAR)

    x_init.add_noise(
        bounds=x_bounds,
        n_shooting=np.array(n_shooting) + 1,
        magnitude=0.2,
        magnitude_type=MagnitudeType.RELATIVE,
        seed=seed,
    )
    #
    constraints = ConstraintList()
    # on verra si remet cette contrainte plus stricte (-0.05, 0.05)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.ALL_SHOOTING,
        min_bound=-0.1,
        max_bound=0.1,
        first_marker="MidMainG",
        second_marker="CibleMainG",
        phase=1,
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.ALL_SHOOTING,
        min_bound=-0.1,
        max_bound=0.1,
        first_marker="MidMainD",
        second_marker="CibleMainD",
        phase=1,
    )
    #    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0, max_bound=final_time, phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=1.5, phase=1)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=1.5, phase=2)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=0.7, phase=3)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=0.5, phase=4)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        [final_time / len(bio_model)] * len(bio_model),
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        n_threads=3,
    )


model_path = "Models/JeCh_TechOpt83.bioMod"

folder_per_athlete = {
    "AuJo": "AuJo/",
    "ElMe": "ElMe/",
    "EvZl": "EvZl/",
    "FeBl": "FeBl/",
    "JeCh_2": "JeCh_2/",
    "KaFu": "KaFu/",
    "KaMi": "KaMi/",
    "LaDe": "LaDe/",
    "MaCu": "MaCu/",
    "MaJa": "MaJa/",
    "OlGa": "OlGa/",
    "Sarah": "Sarah/",
    "SoMe": "SoMe/",
}

folder_per_twist_nb = {"3": "Solutions_vrille_et_demi/"} # , "5": "Solutions_double_vrille_et_demi/"}
# , "2": "Solutions_double_vrille_et_demi/", "3": "Solutions_triple_vrille_et_demi/"}

results_path = "solutions_multi_start/"
folder_graphs = "kinematics_graphs"


FLAG_SAME_FIG = True

model_path = "Models"
num_half_twist = "3"
folder = folder_per_twist_nb[num_half_twist].removeprefix('Solutions_').removesuffix('/')
# done_athlete = os.listdir(f'{folder_graphs}/{folder}')
athlete_done = []
# for i in range(len(done_athlete)):
#     filename = done_athlete[i]
#     athlete_done.append(filename.split(' ')[2].split('_')[0])
for athlete in folder_per_athlete:
    results_path_this_time = results_path + folder_per_twist_nb[num_half_twist] + folder_per_athlete[athlete]

    if athlete in athlete_done:
        print(f'{athlete} for {folder} has already a graph')
        continue
    else:
        print(f'Building graph for {athlete} doing {folder}')

    Bruit = []
    C = []
    Q = []
    Q_integrated = []
    Error = []
    nb = 0
    if FLAG_SAME_FIG:
        fig = None
        axs = None
        fig, axs = plt.subplots(4, 4, figsize=(18, 9))
        axs = axs.ravel()
    if not FLAG_SAME_FIG:
        fig, axs = plt.subplots(4, 4, figsize=(18, 9))
        axs = axs.ravel()
    nb_twists = int(num_half_twist)
    for filename in os.listdir(results_path_this_time):
        nb += 1

        athlete = filename.split("_")[0]
        if filename.removesuffix(".pkl")[-3] == "C":
            file_name = f'/kinematics_graph for {athlete}_{folder_per_twist_nb[num_half_twist].removesuffix("/")}.png'
            Bruit += filename.split("_")[-2]
            print(filename)
            f = os.path.join(results_path_this_time, filename)
            filename = results_path_this_time + filename
            model = biorbd.Model(f"{model_path}/{athlete}.bioMod")
            ocp = prepare_ocp(
                biorbd_model_path=model, nb_twist=int(num_half_twist), seed=int(filename.split("_")[-2]), n_threads=3
            )
            if os.path.isfile(f):
                if filename.endswith(".pkl"):
                    with open(filename, "rb") as f:
                        data = pickle.load(f)
                        q = data["q"]

                        C.append(data["sol"].cost.toarray()[0][0])

                        # integrated
                        sol = data["sol"]
                        sol.ocp = ocp

                        sol_integrated = sol.integrate(
                            shooting_type=Shooting.SINGLE, keep_intermediate_points=False, merge_phases=True
                        )
                        q_integrated = sol_integrated.states["q"]
                        Q_integrated.append(q_integrated)

                        erreur = 0
                        for degree in range(len(q[0])):

                            if degree not in [0, 1, 2]:
                                erreur += abs(q[-1][degree][-1] - q_integrated[degree][-1])
                        Q.append(q)
                        Error.append(erreur)
    if Error != []:
        min_error = np.array(Error).min()
        max_error = np.array(Error).max()
    #if C != []:
        COST = C
        C = np.array(C)

        max = C.max()
        min = C.min()

        cmap = cm.get_cmap("viridis")

        C = C[:, np.newaxis]
        fig.subplots_adjust()
        cbar_ax = fig.add_axes([0.85, 0.11, 0.07, 0.8])

        im = fig.figimage(C)
        fig.colorbar(im, cax=cbar_ax)

        for i in range(len(Bruit)):
            bruit = Bruit[i]
            cost = COST[i]
            q = Q[i]
            q_integrated = Q_integrated[i]  # pour chaque opti
            error = Error[i]
            alpha =abs((error - max_error)/(min_error - max_error))
            alpha_decimal = Decimal(alpha)
            alpha_roundresult = alpha_decimal.quantize(Decimal('.0001'), rounding=ROUND_HALF_UP)
            linewidth_max = 3
            linewidth_min =0.4
            linewidth = lambda alpha : (linewidth_max-linewidth_min)*alpha +linewidth_min


            if FLAG_SAME_FIG:
                if min != max:
                    ratio = (cost - min) / (max - min)

                else:
                    ratio = 1

                color = cmap(ratio)

                print(f"alpha is {alpha}")
                for degree in range(len(q[0])):
                    q_plot = []
                    for phase in range(len(q)):
                        q_plot += q[phase][degree].tolist()[:]

                    if degree == 0:
                        axs[degree].plot(q_plot, color=color, label=f'{bruit}, {alpha_roundresult}', linewidth = linewidth(alpha))

                    else:
                        axs[degree].plot(q_plot, color=color,  linewidth =linewidth(alpha))

                    axs[degree].set_title(f"{model.nameDof()[degree].to_string()}")

    # fig.subplots_adjust()
    # alphabar_ax = fig.add_axes([0.4, 0.1, 0.1, 0.4])
    #alphabar_ax = fig.add_axes([0.05, 0.11, 0.07, 0.8])

    #im = fig.figimage(alpha_list)
    #fig.colorbar(im, cax=alphabar_ax)

    if FLAG_SAME_FIG and Q!= []:
        axs[0].legend(bbox_to_anchor=(0.5, 1), loc="upper left", borderaxespad=-5, ncols=nb, fontsize=12)
        plt.subplots_adjust(left=0.05, right=0.8, hspace=0.4)
        plt.savefig(f"{folder_graphs}/{folder}/{file_name}", dpi=300)
        plt.show()
