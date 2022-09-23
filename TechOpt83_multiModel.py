"""
The goal of this program is to optimize the movement to achieve a rudi out pike (803<).
"""
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
    QAndQDotBounds,
    InitialGuessList,
    InterpolationType,
    OdeSolver,
    Node,
    Solver,
    BiMappingList,
    CostType,
    ConstraintList,
    ConstraintFcn,
    PenaltyNodeList,
    BiorbdInterface,
)
import time

try:
    import IPython
    IPYTHON = True
except ImportError:
    print("No IPython.")
    IPYTHON = False


def minimize_dofs(all_pn: PenaltyNodeList, dofs: list, targets: list) -> MX:
    diff = 0
    for i, dof in enumerate(dofs):
        diff += (all_pn.nlp.states['q'].mx[dof] - targets[i])**2
    return BiorbdInterface.mx_to_cx('minimize_dofs', diff, all_pn.nlp.states['q'])


def prepare_ocp(
    biorbd_model_path: str, n_shooting: int, final_time: float, n_threads: int, ode_solver: OdeSolver = OdeSolver.RK4()
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    n_shooting: int
        The number of shooting points
    final_time: float
        The time at the final node
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = ( biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path) )

    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_qddot_joints = nb_q - biorbd_model[0].nbRoot()

    # Pour la lisibilite
    X_AuJo = 0
    Y_AuJo = 1
    Z_AuJo = 2
    Xrot_AuJo = 3
    Yrot_AuJo = 4
    Zrot_AuJo = 5
    ZrotBD_AuJo = 6
    YrotBD_AuJo = 7
    ZrotABD_AuJo = 8
    XrotABD_AuJo = 9
    ZrotBG_AuJo = 10
    YrotBG_AuJo = 11
    ZrotABG_AuJo = 12
    XrotABG_AuJo = 13
    XrotC_AuJo = 14
    YrotC_AuJo = 15
    X_JeCh = 16
    Y_JeCh = 17
    Z_JeCh = 18
    Xrot_JeCh = 19
    Yrot_JeCh = 20
    Zrot_JeCh = 21
    ZrotBD_JeCh = 22
    YrotBD_JeCh = 23
    ZrotABD_JeCh = 24
    XrotABD_JeCh = 25
    ZrotBG_JeCh = 26
    YrotBG_JeCh = 27
    ZrotABG_JeCh = 28
    XrotABG_JeCh = 29
    XrotC_JeCh = 30
    YrotC_JeCh = 31
    vX_AuJo = 0 + nb_q
    vY_AuJo = 1 + nb_q
    vZ_AuJo = 2 + nb_q
    vXrot_AuJo = 3 + nb_q
    vYrot_AuJo = 4 + nb_q
    vZrot_AuJo = 5 + nb_q
    vZrotBD_AuJo = 6 + nb_q
    vYrotBD_AuJo = 7 + nb_q
    vZrotABD_AuJo = 8 + nb_q
    vYrotABD_AuJo = 9 + nb_q
    vZrotBG_AuJo = 10 + nb_q
    vYrotBG_AuJo = 11 + nb_q
    vZrotABG_AuJo = 12 + nb_q
    vYrotABG_AuJo = 13 + nb_q
    vXrotC_AuJo = 14 + nb_q
    vYrotC_AuJo = 15 + nb_q
    vX_JeCh = 16 + nb_q
    vY_JeCh = 17 + nb_q
    vZ_JeCh = 18 + nb_q
    vXrot_JeCh = 19 + nb_q
    vYrot_JeCh = 20 + nb_q
    vZrot_JeCh = 21 + nb_q
    vZrotBD_JeCh = 22 + nb_q
    vYrotBD_JeCh = 23 + nb_q
    vZrotABD_JeCh = 24 + nb_q
    vYrotABD_JeCh = 25 + nb_q
    vZrotBG_JeCh = 26 + nb_q
    vYrotBG_JeCh = 27 + nb_q
    vZrotABG_JeCh = 28 + nb_q
    vYrotABG_JeCh = 29 + nb_q
    vXrotC_JeCh = 30 + nb_q
    vYrotC_JeCh = 31 + nb_q

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, marker_index=1, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=4)

    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=final_time, weight=100000, phase=0)

    # TODO: peut-etre changer le nom des cibles pour plus generique
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, node=Node.END, first_marker='MidMainGAuJo', second_marker='CibleMainGAuJo', weight=1000, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, node=Node.END, first_marker='MidMainDAuJo', second_marker='CibleMainDAuJo', weight=1000, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, node=Node.END, first_marker='MidMainGJeCh', second_marker='CibleMainGJeCh', weight=1000, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, node=Node.END, first_marker='MidMainDJeCh', second_marker='CibleMainDJeCh', weight=1000, phase=0)

    # arrete de gigoter les bras
    les_bras = [ZrotBD_AuJo, YrotBD_AuJo, ZrotABD_AuJo, XrotABD_AuJo, ZrotBG_AuJo, YrotBG_AuJo, ZrotABG_AuJo, XrotABG_AuJo,
                ZrotBD_JeCh, YrotBD_JeCh, ZrotABD_JeCh, XrotABD_JeCh, ZrotBG_JeCh, YrotBG_JeCh, ZrotABG_JeCh, XrotABG_JeCh]
    les_coudes = [ZrotABD_AuJo, XrotABD_AuJo, ZrotABG_AuJo, XrotABG_AuJo, ZrotABD_JeCh, XrotABD_JeCh, ZrotABG_JeCh, XrotABG_JeCh]
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_coudes, targets=np.zeros(len(les_coudes)), weight=10000, phase=0)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_bras, targets=np.zeros(len(les_bras)), weight=10000, phase=2)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_bras, targets=np.zeros(len(les_bras)), weight=10000, phase=3)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING, dofs=les_coudes, targets=np.zeros(len(les_coudes)), weight=10000, phase=4)
    # ouvre les hanches rapidement apres la vrille
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Mayer, node=Node.END, dofs=[XrotC_AuJo, XrotC_JeCh], targets=[0, 0], weight=10000, phase=3)

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

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    # Pour la lisibilite
    DEBUT, MILIEU, FIN = 0, 1, 2

    #
    # Contraintes de position: PHASE 0 la montee en carpe
    #

    zmax = 9.81 / 8 * final_time**2 + 1  # une petite marge

    # deplacement
    x_bounds[0].min[X_AuJo, :] = -.1
    x_bounds[0].max[X_AuJo, :] = .1
    x_bounds[0].min[Y_AuJo, :] = -1.
    x_bounds[0].max[Y_AuJo, :] = 1.
    x_bounds[0].min[:Z_AuJo+1, DEBUT] = 0
    x_bounds[0].max[:Z_AuJo+1, DEBUT] = 0
    x_bounds[0].min[Z_AuJo, MILIEU:] = 0
    x_bounds[0].max[Z_AuJo, MILIEU:] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    x_bounds[0].min[X_JeCh, :] = -.1
    x_bounds[0].max[X_JeCh, :] = .1
    x_bounds[0].min[Y_JeCh, :] = -1.
    x_bounds[0].max[Y_JeCh, :] = 1.
    x_bounds[0].min[:Z_JeCh + 1, DEBUT] = 0
    x_bounds[0].max[:Z_JeCh + 1, DEBUT] = 0
    x_bounds[0].min[Z_JeCh, MILIEU:] = 0
    x_bounds[0].max[Z_JeCh, MILIEU:] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[0].min[Xrot_AuJo, DEBUT] = .50  # penche vers l'avant un peu carpe
    x_bounds[0].max[Xrot_AuJo, DEBUT] = .50
    x_bounds[0].min[Xrot_AuJo, MILIEU:] = 0
    x_bounds[0].max[Xrot_AuJo, MILIEU:] = 4 * 3.14 + .1  # salto

    x_bounds[0].min[Xrot_JeCh, DEBUT] = .50  # penche vers l'avant un peu carpe
    x_bounds[0].max[Xrot_JeCh, DEBUT] = .50
    x_bounds[0].min[Xrot_JeCh, MILIEU:] = 0
    x_bounds[0].max[Xrot_JeCh, MILIEU:] = 4 * 3.14 + .1  # salto

    # limitation du tilt autour de y
    x_bounds[0].min[Yrot_AuJo, DEBUT] = 0
    x_bounds[0].max[Yrot_AuJo, DEBUT] = 0
    x_bounds[0].min[Yrot_AuJo, MILIEU:] = - 3.14 / 16  # vraiment pas suppose tilte
    x_bounds[0].max[Yrot_AuJo, MILIEU:] = 3.14 / 16

    x_bounds[0].min[Yrot_JeCh, DEBUT] = 0
    x_bounds[0].max[Yrot_JeCh, DEBUT] = 0
    x_bounds[0].min[Yrot_JeCh, MILIEU:] = - 3.14 / 16  # vraiment pas suppose tilte
    x_bounds[0].max[Yrot_JeCh, MILIEU:] = 3.14 / 16

    # la vrille autour de z
    x_bounds[0].min[Zrot_AuJo, DEBUT] = 0
    x_bounds[0].max[Zrot_AuJo, DEBUT] = 0
    x_bounds[0].min[Zrot_AuJo, MILIEU:] = -.1  # pas de vrille dans cette phase
    x_bounds[0].max[Zrot_AuJo, MILIEU:] = .1

    x_bounds[0].min[Zrot_JeCh, DEBUT] = 0
    x_bounds[0].max[Zrot_JeCh, DEBUT] = 0
    x_bounds[0].min[Zrot_JeCh, MILIEU:] = -.1  # pas de vrille dans cette phase
    x_bounds[0].max[Zrot_JeCh, MILIEU:] = .1

    # bras droit
    x_bounds[0].min[YrotBD_AuJo, DEBUT] = 2.9  # debut bras aux oreilles
    x_bounds[0].max[YrotBD_AuJo, DEBUT] = 2.9
    x_bounds[0].min[ZrotBD_AuJo, DEBUT] = 0
    x_bounds[0].max[ZrotBD_AuJo, DEBUT] = 0

    x_bounds[0].min[YrotBD_JeCh, DEBUT] = 2.9  # debut bras aux oreilles
    x_bounds[0].max[YrotBD_JeCh, DEBUT] = 2.9
    x_bounds[0].min[ZrotBD_JeCh, DEBUT] = 0
    x_bounds[0].max[ZrotBD_JeCh, DEBUT] = 0

    # bras gauche
    x_bounds[0].min[YrotBG_AuJo, DEBUT] = -2.9  # debut bras aux oreilles
    x_bounds[0].max[YrotBG_AuJo, DEBUT] = -2.9
    x_bounds[0].min[ZrotBG_AuJo, DEBUT] = 0
    x_bounds[0].max[ZrotBG_AuJo, DEBUT] = 0

    x_bounds[0].min[YrotBG_JeCh, DEBUT] = -2.9  # debut bras aux oreilles
    x_bounds[0].max[YrotBG_JeCh, DEBUT] = -2.9
    x_bounds[0].min[ZrotBG_JeCh, DEBUT] = 0
    x_bounds[0].max[ZrotBG_JeCh, DEBUT] = 0

    # coude droit
    x_bounds[0].min[ZrotABD_AuJo:XrotABD_AuJo+1, DEBUT] = 0
    x_bounds[0].max[ZrotABD_AuJo:XrotABD_AuJo+1, DEBUT] = 0

    x_bounds[0].min[ZrotABD_JeCh:XrotABD_JeCh + 1, DEBUT] = 0
    x_bounds[0].max[ZrotABD_JeCh:XrotABD_JeCh + 1, DEBUT] = 0

    # coude gauche
    x_bounds[0].min[ZrotABG_AuJo:XrotABG_AuJo+1, DEBUT] = 0
    x_bounds[0].max[ZrotABG_AuJo:XrotABG_AuJo+1, DEBUT] = 0

    x_bounds[0].min[ZrotABG_JeCh:XrotABG_JeCh + 1, DEBUT] = 0
    x_bounds[0].max[ZrotABG_JeCh:XrotABG_JeCh + 1, DEBUT] = 0

    # le carpe
    x_bounds[0].min[XrotC_AuJo, DEBUT] = -.50  # depart un peu ferme aux hanches
    x_bounds[0].max[XrotC_AuJo, DEBUT] = -.50
    x_bounds[0].max[XrotC_AuJo, FIN] = -2.5

    x_bounds[0].min[XrotC_JeCh, DEBUT] = -.50  # depart un peu ferme aux hanches
    x_bounds[0].max[XrotC_JeCh, DEBUT] = -.50
    x_bounds[0].max[XrotC_JeCh, FIN] = -2.5

    # le dehanchement
    x_bounds[0].min[YrotC_AuJo, DEBUT] = 0
    x_bounds[0].max[YrotC_AuJo, DEBUT] = 0
    x_bounds[0].min[YrotC_AuJo, MILIEU:] = -.1
    x_bounds[0].max[YrotC_AuJo, MILIEU:] = .1

    x_bounds[0].min[YrotC_JeCh, DEBUT] = 0
    x_bounds[0].max[YrotC_JeCh, DEBUT] = 0
    x_bounds[0].min[YrotC_JeCh, MILIEU:] = -.1
    x_bounds[0].max[YrotC_JeCh, MILIEU:] = .1

    # Contraintes de vitesse: PHASE 0 la montee en carpe

    vzinit = 9.81 / 2 * final_time  # vitesse initiale en z du CoM pour revenir a terre au temps final

    # decalage entre le bassin et le CoM
    CoM_Q_sym = MX.sym('CoM', nb_q)
    CoM_Q_init = x_bounds[0].min[:nb_q, DEBUT]  # min ou max ne change rien a priori, au DEBUT ils sont egaux normalement
    CoM_Q_func = Function('CoM_Q_func', [CoM_Q_sym], [biorbd_model[0].CoM(CoM_Q_sym).to_mx()])
    bassin_Q_func = Function('bassin_Q_func', [CoM_Q_sym],
                             [biorbd_model[0].globalJCS(0).to_mx()])  # retourne la RT du bassin

    r = np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1, :3]  # selectionne seulement la translation de la RT

    # en xy bassin
    x_bounds[0].min[vX_AuJo:vY_AuJo+1, :] = -10
    x_bounds[0].max[vX_AuJo:vY_AuJo+1, :] = 10
    x_bounds[0].min[vX_AuJo:vY_AuJo+1, DEBUT] = -.5
    x_bounds[0].max[vX_AuJo:vY_AuJo+1, DEBUT] = .5

    x_bounds[0].min[vX_JeCh:vY_JeCh + 1, :] = -10
    x_bounds[0].max[vX_JeCh:vY_JeCh + 1, :] = 10
    x_bounds[0].min[vX_JeCh:vY_JeCh + 1, DEBUT] = -.5
    x_bounds[0].max[vX_JeCh:vY_JeCh + 1, DEBUT] = .5

    # z bassin
    x_bounds[0].min[vZ_AuJo, :] = -100
    x_bounds[0].max[vZ_AuJo, :] = 100
    x_bounds[0].min[vZ_AuJo, DEBUT] = vzinit - .5
    x_bounds[0].max[vZ_AuJo, DEBUT] = vzinit + .5

    x_bounds[0].min[vZ_JeCh, :] = -100
    x_bounds[0].max[vZ_JeCh, :] = 100
    x_bounds[0].min[vZ_JeCh, DEBUT] = vzinit - .5
    x_bounds[0].max[vZ_JeCh, DEBUT] = vzinit + .5

    # autour de x
    x_bounds[0].min[vXrot_AuJo, :] = .5  # d'apres une observation video
    x_bounds[0].max[vXrot_AuJo, :] = 20  # aussi vite que nécessaire, mais ne devrait pas atteindre cette vitesse

    x_bounds[0].min[vXrot_JeCh, :] = .5  # d'apres une observation video
    x_bounds[0].max[vXrot_JeCh, :] = 20  # aussi vite que nécessaire, mais ne devrait pas atteindre cette vitesse

    # autour de y
    x_bounds[0].min[vYrot_AuJo, :] = -100
    x_bounds[0].max[vYrot_AuJo, :] = 100
    x_bounds[0].min[vYrot_AuJo, DEBUT] = 0
    x_bounds[0].max[vYrot_AuJo, DEBUT] = 0

    x_bounds[0].min[vYrot_JeCh, :] = -100
    x_bounds[0].max[vYrot_JeCh, :] = 100
    x_bounds[0].min[vYrot_JeCh, DEBUT] = 0
    x_bounds[0].max[vYrot_JeCh, DEBUT] = 0

    # autour de z
    x_bounds[0].min[vZrot_AuJo, :] = -100
    x_bounds[0].max[vZrot_AuJo, :] = 100
    x_bounds[0].min[vZrot_AuJo, DEBUT] = 0
    x_bounds[0].max[vZrot_AuJo, DEBUT] = 0

    x_bounds[0].min[vZrot_JeCh, :] = -100
    x_bounds[0].max[vZrot_JeCh, :] = 100
    x_bounds[0].min[vZrot_JeCh, DEBUT] = 0
    x_bounds[0].max[vZrot_JeCh, DEBUT] = 0

    # tenir compte du decalage entre bassin et CoM avec la rotation
    # Qtransdot = Qtransdot + v cross Qrotdot
    borne_inf = ( x_bounds[0].min[vX_AuJo:vZ_AuJo+1, DEBUT] + np.cross(r, x_bounds[0].min[vXrot_AuJo:vZrot_AuJo+1, DEBUT]) )[0]
    borne_sup = ( x_bounds[0].max[vX_AuJo:vZ_AuJo+1, DEBUT] + np.cross(r, x_bounds[0].max[vXrot_AuJo:vZrot_AuJo+1, DEBUT]) )[0]
    x_bounds[0].min[vX_AuJo:vZ_AuJo+1, DEBUT] = min(borne_sup[0], borne_inf[0]), min(borne_sup[1], borne_inf[1]), min(borne_sup[2], borne_inf[2])
    x_bounds[0].max[vX_AuJo:vZ_AuJo+1, DEBUT] = max(borne_sup[0], borne_inf[0]), max(borne_sup[1], borne_inf[1]), max(borne_sup[2], borne_inf[2])

    borne_inf = (x_bounds[0].min[vX_JeCh:vZ_JeCh + 1, DEBUT] + np.cross(r, x_bounds[0].min[vXrot_JeCh:vZrot_JeCh + 1, DEBUT]))[0]
    borne_sup = (x_bounds[0].max[vX_JeCh:vZ_JeCh + 1, DEBUT] + np.cross(r, x_bounds[0].max[vXrot_JeCh:vZrot_JeCh + 1, DEBUT]))[0]
    x_bounds[0].min[vX_JeCh:vZ_JeCh + 1, DEBUT] = min(borne_sup[0], borne_inf[0]), min(borne_sup[1], borne_inf[1]), min(borne_sup[2], borne_inf[2])
    x_bounds[0].max[vX_JeCh:vZ_JeCh + 1, DEBUT] = max(borne_sup[0], borne_inf[0]), max(borne_sup[1], borne_inf[1]), max(borne_sup[2], borne_inf[2])

    # bras droit
    x_bounds[0].min[vZrotBD_AuJo:vYrotBD_AuJo+1, :] = -100
    x_bounds[0].max[vZrotBD_AuJo:vYrotBD_AuJo+1, :] = 100
    x_bounds[0].min[vZrotBD_AuJo:vYrotBD_AuJo+1, DEBUT] = 0
    x_bounds[0].max[vZrotBD_AuJo:vYrotBD_AuJo+1, DEBUT] = 0

    x_bounds[0].min[vZrotBD_JeCh:vYrotBD_JeCh + 1, :] = -100
    x_bounds[0].max[vZrotBD_JeCh:vYrotBD_JeCh + 1, :] = 100
    x_bounds[0].min[vZrotBD_JeCh:vYrotBD_JeCh + 1, DEBUT] = 0
    x_bounds[0].max[vZrotBD_JeCh:vYrotBD_JeCh + 1, DEBUT] = 0

    # bras droit
    x_bounds[0].min[vZrotBG_AuJo:vYrotBG_AuJo+1, :] = -100
    x_bounds[0].max[vZrotBG_AuJo:vYrotBG_AuJo+1, :] = 100
    x_bounds[0].min[vZrotBG_AuJo:vYrotBG_AuJo+1, DEBUT] = 0
    x_bounds[0].max[vZrotBG_AuJo:vYrotBG_AuJo+1, DEBUT] = 0

    x_bounds[0].min[vZrotBG_JeCh:vYrotBG_JeCh + 1, :] = -100
    x_bounds[0].max[vZrotBG_JeCh:vYrotBG_JeCh + 1, :] = 100
    x_bounds[0].min[vZrotBG_JeCh:vYrotBG_JeCh + 1, DEBUT] = 0
    x_bounds[0].max[vZrotBG_JeCh:vYrotBG_JeCh + 1, DEBUT] = 0

    # coude droit
    x_bounds[0].min[vZrotABD_AuJo:vYrotABD_AuJo+1, :] = -100
    x_bounds[0].max[vZrotABD_AuJo:vYrotABD_AuJo+1, :] = 100
    x_bounds[0].min[vZrotABD_AuJo:vYrotABD_AuJo+1, DEBUT] = 0
    x_bounds[0].max[vZrotABD_AuJo:vYrotABD_AuJo+1, DEBUT] = 0

    x_bounds[0].min[vZrotABD_JeCh:vYrotABD_JeCh + 1, :] = -100
    x_bounds[0].max[vZrotABD_JeCh:vYrotABD_JeCh + 1, :] = 100
    x_bounds[0].min[vZrotABD_JeCh:vYrotABD_JeCh + 1, DEBUT] = 0
    x_bounds[0].max[vZrotABD_JeCh:vYrotABD_JeCh + 1, DEBUT] = 0
    # coude gauche
    x_bounds[0].min[vZrotABD_AuJo:vYrotABG_AuJo+1, :] = -100
    x_bounds[0].max[vZrotABD_AuJo:vYrotABG_AuJo+1, :] = 100
    x_bounds[0].min[vZrotABG_AuJo:vYrotABG_AuJo+1, DEBUT] = 0
    x_bounds[0].max[vZrotABG_AuJo:vYrotABG_AuJo+1, DEBUT] = 0

    x_bounds[0].min[vZrotABD_JeCh:vYrotABG_JeCh + 1, :] = -100
    x_bounds[0].max[vZrotABD_JeCh:vYrotABG_JeCh + 1, :] = 100
    x_bounds[0].min[vZrotABG_JeCh:vYrotABG_JeCh + 1, DEBUT] = 0
    x_bounds[0].max[vZrotABG_JeCh:vYrotABG_JeCh + 1, DEBUT] = 0

    # du carpe
    x_bounds[0].min[vXrotC_AuJo, :] = -100
    x_bounds[0].max[vXrotC_AuJo, :] = 100
    x_bounds[0].min[vXrotC_AuJo, DEBUT] = 0
    x_bounds[0].max[vXrotC_AuJo, DEBUT] = 0

    x_bounds[0].min[vXrotC_JeCh, :] = -100
    x_bounds[0].max[vXrotC_JeCh, :] = 100
    x_bounds[0].min[vXrotC_JeCh, DEBUT] = 0
    x_bounds[0].max[vXrotC_JeCh, DEBUT] = 0

    # du dehanchement
    x_bounds[0].min[vYrotC_AuJo, :] = -100
    x_bounds[0].max[vYrotC_AuJo, :] = 100
    x_bounds[0].min[vYrotC_AuJo, DEBUT] = 0
    x_bounds[0].max[vYrotC_AuJo, DEBUT] = 0

    x_bounds[0].min[vYrotC_JeCh, :] = -100
    x_bounds[0].max[vYrotC_JeCh, :] = 100
    x_bounds[0].min[vYrotC_JeCh, DEBUT] = 0
    x_bounds[0].max[vYrotC_JeCh, DEBUT] = 0

    #
    # Contraintes de position: PHASE 1 le salto carpe
    #

    # deplacement
    x_bounds[1].min[X_AuJo, :] = -.1
    x_bounds[1].max[X_AuJo, :] = .1
    x_bounds[1].min[Y_AuJo, :] = -1.
    x_bounds[1].max[Y_AuJo, :] = 1.
    x_bounds[1].min[Z_AuJo, :] = 0
    x_bounds[1].max[Z_AuJo, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    x_bounds[1].min[X_JeCh, :] = -.1
    x_bounds[1].max[X_JeCh, :] = .1
    x_bounds[1].min[Y_JeCh, :] = -1.
    x_bounds[1].max[Y_JeCh, :] = 1.
    x_bounds[1].min[Z_JeCh, :] = 0
    x_bounds[1].max[Z_JeCh, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[1].min[Xrot_AuJo, :] = 0
    x_bounds[1].max[Xrot_AuJo, :] = 4 * 3.14
    x_bounds[1].min[Xrot_AuJo, FIN] = 2 * 3.14 - .1

    x_bounds[1].min[Xrot_JeCh, :] = 0
    x_bounds[1].max[Xrot_JeCh, :] = 4 * 3.14
    x_bounds[1].min[Xrot_JeCh, FIN] = 2 * 3.14 - .1

    # limitation du tilt autour de y
    x_bounds[1].min[Yrot_AuJo, :] = - 3.14 / 16
    x_bounds[1].max[Yrot_AuJo, :] = 3.14 / 16

    x_bounds[1].min[Yrot_JeCh, :] = - 3.14 / 16
    x_bounds[1].max[Yrot_JeCh, :] = 3.14 / 16
    # la vrille autour de z
    x_bounds[1].min[Zrot_AuJo, :] = -.1
    x_bounds[1].max[Zrot_AuJo, :] = .1

    x_bounds[1].min[Zrot_JeCh, :] = -.1
    x_bounds[1].max[Zrot_JeCh, :] = .1

    # bras f4a a l'ouverture

    # le carpe
    x_bounds[1].max[XrotC_AuJo, :] = -2.5
    x_bounds[1].max[XrotC_JeCh, :] = -2.5

    # le dehanchement
    x_bounds[1].min[YrotC_AuJo, DEBUT] = -.1
    x_bounds[1].max[YrotC_AuJo, DEBUT] = .1

    x_bounds[1].min[YrotC_JeCh, DEBUT] = -.1
    x_bounds[1].max[YrotC_JeCh, DEBUT] = .1


    # Contraintes de vitesse: PHASE 1 le salto carpe

    # en xy bassin
    x_bounds[1].min[vX_AuJo:vY_AuJo + 1, :] = -10
    x_bounds[1].max[vX_AuJo:vY_AuJo + 1, :] = 10

    x_bounds[1].min[vX_JeCh:vY_JeCh + 1, :] = -10
    x_bounds[1].max[vX_JeCh:vY_JeCh + 1, :] = 10

    # z bassin
    x_bounds[1].min[vZ_AuJo, :] = -100
    x_bounds[1].max[vZ_AuJo, :] = 100

    x_bounds[1].min[vZ_JeCh, :] = -100
    x_bounds[1].max[vZ_JeCh, :] = 100

    # autour de x
    x_bounds[1].min[vXrot_AuJo, :] = -100
    x_bounds[1].max[vXrot_AuJo, :] = 100

    x_bounds[1].min[vXrot_JeCh, :] = -100
    x_bounds[1].max[vXrot_JeCh, :] = 100

    # autour de y
    x_bounds[1].min[vYrot_AuJo, :] = -100
    x_bounds[1].max[vYrot_AuJo, :] = 100

    x_bounds[1].min[vYrot_JeCh, :] = -100
    x_bounds[1].max[vYrot_JeCh, :] = 100

    # autour de z
    x_bounds[1].min[vZrot_AuJo, :] = -100
    x_bounds[1].max[vZrot_AuJo, :] = 100

    x_bounds[1].min[vZrot_JeCh, :] = -100
    x_bounds[1].max[vZrot_JeCh, :] = 100

    # bras droit
    x_bounds[1].min[vZrotBD_AuJo:vYrotBD_AuJo + 1, :] = -100
    x_bounds[1].max[vZrotBD_AuJo:vYrotBD_AuJo + 1, :] = 100

    x_bounds[1].min[vZrotBD_JeCh:vYrotBD_JeCh + 1, :] = -100
    x_bounds[1].max[vZrotBD_JeCh:vYrotBD_JeCh + 1, :] = 100

    # bras droit
    x_bounds[1].min[vZrotBG_AuJo:vYrotBG_AuJo + 1, :] = -100
    x_bounds[1].max[vZrotBG_AuJo:vYrotBG_AuJo + 1, :] = 100

    x_bounds[1].min[vZrotBG_JeCh:vYrotBG_JeCh + 1, :] = -100
    x_bounds[1].max[vZrotBG_JeCh:vYrotBG_JeCh + 1, :] = 100

    # coude droit
    x_bounds[1].min[vZrotABD_AuJo:vYrotABD_AuJo + 1, :] = -100
    x_bounds[1].max[vZrotABD_AuJo:vYrotABD_AuJo + 1, :] = 100

    x_bounds[1].min[vZrotABD_JeCh:vYrotABD_JeCh + 1, :] = -100
    x_bounds[1].max[vZrotABD_JeCh:vYrotABD_JeCh + 1, :] = 100

    # coude gauche
    x_bounds[1].min[vZrotABD_AuJo:vYrotABG_AuJo + 1, :] = -100
    x_bounds[1].max[vZrotABD_AuJo:vYrotABG_AuJo + 1, :] = 100

    x_bounds[1].min[vZrotABD_JeCh:vYrotABG_JeCh + 1, :] = -100
    x_bounds[1].max[vZrotABD_JeCh:vYrotABG_JeCh + 1, :] = 100

    # du carpe
    x_bounds[1].min[vXrotC_AuJo, :] = -100
    x_bounds[1].max[vXrotC_AuJo, :] = 100

    x_bounds[1].min[vXrotC_JeCh, :] = -100
    x_bounds[1].max[vXrotC_JeCh, :] = 100

    # du dehanchement
    x_bounds[1].min[vYrotC_AuJo, :] = -100
    x_bounds[1].max[vYrotC_AuJo, :] = 100

    x_bounds[1].min[vYrotC_JeCh, :] = -100
    x_bounds[1].max[vYrotC_JeCh, :] = 100

    #
    # Contraintes de position: PHASE 2 l'ouverture
    #

    # deplacement
    x_bounds[2].min[X_AuJo, :] = -.2
    x_bounds[2].max[X_AuJo, :] = .2
    x_bounds[2].min[Y_AuJo, :] = -1.
    x_bounds[2].max[Y_AuJo, :] = 1.
    x_bounds[2].min[Z_AuJo, :] = 0
    x_bounds[2].max[Z_AuJo, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    x_bounds[2].min[X_JeCh, :] = -.2
    x_bounds[2].max[X_JeCh, :] = .2
    x_bounds[2].min[Y_JeCh, :] = -1.
    x_bounds[2].max[Y_JeCh, :] = 1.
    x_bounds[2].min[Z_JeCh, :] = 0
    x_bounds[2].max[Z_JeCh, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[2].min[Xrot_AuJo, :] = 2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[2].max[Xrot_AuJo, :] = 4 * 3.14

    x_bounds[2].min[Xrot_JeCh, :] = 2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[2].max[Xrot_JeCh, :] = 4 * 3.14

    # limitation du tilt autour de y
    x_bounds[2].min[Yrot_AuJo, :] = - 3.14 / 4
    x_bounds[2].max[Yrot_AuJo, :] = 3.14 / 4

    x_bounds[2].min[Yrot_JeCh, :] = - 3.14 / 4
    x_bounds[2].max[Yrot_JeCh, :] = 3.14 / 4

    # la vrille autour de z
    x_bounds[2].min[Zrot_AuJo, :] = 0
    x_bounds[2].max[Zrot_AuJo, :] = 3 * 3.14

    x_bounds[2].min[Zrot_JeCh, :] = 0
    x_bounds[2].max[Zrot_JeCh, :] = 3 * 3.14

    # bras f4a a l'ouverture

    # le carpe
    x_bounds[2].min[XrotC_AuJo, FIN] = -.4
    x_bounds[2].min[XrotC_JeCh, FIN] = -.4

    # le dehanchement f4a a l'ouverture

    # Contraintes de vitesse: PHASE 2 l'ouverture

    # en xy bassin
    x_bounds[2].min[vX_AuJo:vY_AuJo + 1, :] = -10
    x_bounds[2].max[vX_AuJo:vY_AuJo + 1, :] = 10

    x_bounds[2].min[vX_JeCh:vY_JeCh + 1, :] = -10
    x_bounds[2].max[vX_JeCh:vY_JeCh + 1, :] = 10

    # z bassin
    x_bounds[2].min[vZ_AuJo, :] = -100
    x_bounds[2].max[vZ_AuJo, :] = 100

    x_bounds[2].min[vZ_JeCh, :] = -100
    x_bounds[2].max[vZ_JeCh, :] = 100

    # autour de x
    x_bounds[2].min[vXrot_AuJo, :] = -100
    x_bounds[2].max[vXrot_AuJo, :] = 100

    x_bounds[2].min[vXrot_JeCh, :] = -100
    x_bounds[2].max[vXrot_JeCh, :] = 100

    # autour de y
    x_bounds[2].min[vYrot_AuJo, :] = -100
    x_bounds[2].max[vYrot_AuJo, :] = 100

    x_bounds[2].min[vYrot_JeCh, :] = -100
    x_bounds[2].max[vYrot_JeCh, :] = 100

    # autour de z
    x_bounds[2].min[vZrot_AuJo, :] = -100
    x_bounds[2].max[vZrot_AuJo, :] = 100

    x_bounds[2].min[vZrot_JeCh, :] = -100
    x_bounds[2].max[vZrot_JeCh, :] = 100

    # bras droit
    x_bounds[2].min[vZrotBD_AuJo:vYrotBD_AuJo + 1, :] = -100
    x_bounds[2].max[vZrotBD_AuJo:vYrotBD_AuJo + 1, :] = 100

    x_bounds[2].min[vZrotBD_JeCh:vYrotBD_JeCh + 1, :] = -100
    x_bounds[2].max[vZrotBD_JeCh:vYrotBD_JeCh + 1, :] = 100

    # bras droit
    x_bounds[2].min[vZrotBG_AuJo:vYrotBG_AuJo + 1, :] = -100
    x_bounds[2].max[vZrotBG_AuJo:vYrotBG_AuJo + 1, :] = 100

    x_bounds[2].min[vZrotBG_JeCh:vYrotBG_JeCh + 1, :] = -100
    x_bounds[2].max[vZrotBG_JeCh:vYrotBG_JeCh + 1, :] = 100

    # coude droit
    x_bounds[2].min[vZrotABD_AuJo:vYrotABD_AuJo + 1, :] = -100
    x_bounds[2].max[vZrotABD_AuJo:vYrotABD_AuJo + 1, :] = 100

    x_bounds[2].min[vZrotABD_JeCh:vYrotABD_JeCh + 1, :] = -100
    x_bounds[2].max[vZrotABD_JeCh:vYrotABD_JeCh + 1, :] = 100

    # coude gauche
    x_bounds[2].min[vZrotABD_AuJo:vYrotABG_AuJo + 1, :] = -100
    x_bounds[2].max[vZrotABD_AuJo:vYrotABG_AuJo + 1, :] = 100

    x_bounds[2].min[vZrotABD_JeCh:vYrotABG_JeCh + 1, :] = -100
    x_bounds[2].max[vZrotABD_JeCh:vYrotABG_JeCh + 1, :] = 100

    # du carpe
    x_bounds[2].min[vXrotC_AuJo, :] = -100
    x_bounds[2].max[vXrotC_AuJo, :] = 100

    x_bounds[2].min[vXrotC_JeCh, :] = -100
    x_bounds[2].max[vXrotC_JeCh, :] = 100

    # du dehanchement
    x_bounds[2].min[vYrotC_AuJo, :] = -100
    x_bounds[2].max[vYrotC_AuJo, :] = 100

    x_bounds[2].min[vYrotC_JeCh, :] = -100
    x_bounds[2].max[vYrotC_JeCh, :] = 100

    #
    # Contraintes de position: PHASE 3 la vrille et demie
    #

    # deplacement
    x_bounds[3].min[X_AuJo, :] = -.2
    x_bounds[3].max[X_AuJo, :] = .2
    x_bounds[3].min[Y_AuJo, :] = -1.
    x_bounds[3].max[Y_AuJo, :] = 1.
    x_bounds[3].min[Z_AuJo, :] = 0
    x_bounds[3].max[Z_AuJo, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    x_bounds[3].min[X_JeCh, :] = -.2
    x_bounds[3].max[X_JeCh, :] = .2
    x_bounds[3].min[Y_JeCh, :] = -1.
    x_bounds[3].max[Y_JeCh, :] = 1.
    x_bounds[3].min[Z_JeCh, :] = 0
    x_bounds[3].max[Z_JeCh, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[3].min[Xrot_AuJo, :] = 2 * 3.14 - .1
    x_bounds[3].max[Xrot_AuJo, :] = 2 * 3.14 + 3/2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[3].min[Xrot_AuJo, FIN] = 2 * 3.14 + 3/2 * 3.14 - .1
    x_bounds[3].max[Xrot_AuJo, FIN] = 2 * 3.14 + 3/2 * 3.14 + .1  # 1 salto 3/4

    x_bounds[3].min[Xrot_JeCh, :] = 2 * 3.14 - .1
    x_bounds[3].max[Xrot_JeCh, :] = 2 * 3.14 + 3 / 2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[3].min[Xrot_JeCh, FIN] = 2 * 3.14 + 3 / 2 * 3.14 - .1
    x_bounds[3].max[Xrot_JeCh, FIN] = 2 * 3.14 + 3 / 2 * 3.14 + .1  # 1 salto 3/4

    # limitation du tilt autour de y
    x_bounds[3].min[Yrot_AuJo, :] = - 3.14 / 4
    x_bounds[3].max[Yrot_AuJo, :] = 3.14 / 4
    x_bounds[3].min[Yrot_AuJo, FIN] = - 3.14 / 8
    x_bounds[3].max[Yrot_AuJo, FIN] = 3.14 / 8

    x_bounds[3].min[Yrot_JeCh, :] = - 3.14 / 4
    x_bounds[3].max[Yrot_JeCh, :] = 3.14 / 4
    x_bounds[3].min[Yrot_JeCh, FIN] = - 3.14 / 8
    x_bounds[3].max[Yrot_JeCh, FIN] = 3.14 / 8

    # la vrille autour de z
    x_bounds[3].min[Zrot_AuJo, :] = 0
    x_bounds[3].max[Zrot_AuJo, :] = 3 * 3.14
    x_bounds[3].min[Zrot_AuJo, FIN] = 3 * 3.14 - .1  # complete la vrille
    x_bounds[3].max[Zrot_AuJo, FIN] = 3 * 3.14 + .1

    x_bounds[3].min[Zrot_JeCh, :] = 0
    x_bounds[3].max[Zrot_JeCh, :] = 3 * 3.14
    x_bounds[3].min[Zrot_JeCh, FIN] = 3 * 3.14 - .1  # complete la vrille
    x_bounds[3].max[Zrot_JeCh, FIN] = 3 * 3.14 + .1

    # bras f4a la vrille

    # le carpe
    x_bounds[3].min[XrotC_AuJo, :] = -.4

    # le dehanchement f4a la vrille

    # Contraintes de vitesse: PHASE 3 la vrille et demie

    # en xy bassin
    x_bounds[3].min[vX_AuJo:vY_AuJo + 1, :] = -10
    x_bounds[3].max[vX_AuJo:vY_AuJo + 1, :] = 10

    x_bounds[3].min[vX_JeCh:vY_JeCh + 1, :] = -10
    x_bounds[3].max[vX_JeCh:vY_JeCh + 1, :] = 10

    # z bassin
    x_bounds[3].min[vZ_AuJo, :] = -100
    x_bounds[3].max[vZ_AuJo, :] = 100

    x_bounds[3].min[vZ_JeCh, :] = -100
    x_bounds[3].max[vZ_JeCh, :] = 100

    # autour de x
    x_bounds[3].min[vXrot_AuJo, :] = -100
    x_bounds[3].max[vXrot_AuJo, :] = 100

    x_bounds[3].min[vXrot_JeCh, :] = -100
    x_bounds[3].max[vXrot_JeCh, :] = 100

    # autour de y
    x_bounds[3].min[vYrot_AuJo, :] = -100
    x_bounds[3].max[vYrot_AuJo, :] = 100

    x_bounds[3].min[vYrot_JeCh, :] = -100
    x_bounds[3].max[vYrot_JeCh, :] = 100

    # autour de z
    x_bounds[3].min[vZrot_AuJo, :] = -100
    x_bounds[3].max[vZrot_AuJo, :] = 100

    x_bounds[3].min[vZrot_JeCh, :] = -100
    x_bounds[3].max[vZrot_JeCh, :] = 100

    # bras droit
    x_bounds[3].min[vZrotBD_AuJo:vYrotBD_AuJo + 1, :] = -100
    x_bounds[3].max[vZrotBD_AuJo:vYrotBD_AuJo + 1, :] = 100

    x_bounds[3].min[vZrotBD_JeCh:vYrotBD_JeCh + 1, :] = -100
    x_bounds[3].max[vZrotBD_JeCh:vYrotBD_JeCh + 1, :] = 100

    # bras droit
    x_bounds[3].min[vZrotBG_AuJo:vYrotBG_AuJo + 1, :] = -100
    x_bounds[3].max[vZrotBG_AuJo:vYrotBG_AuJo + 1, :] = 100

    x_bounds[3].min[vZrotBG_JeCh:vYrotBG_JeCh + 1, :] = -100
    x_bounds[3].max[vZrotBG_JeCh:vYrotBG_JeCh + 1, :] = 100

    # coude droit
    x_bounds[3].min[vZrotABD_AuJo:vYrotABD_AuJo + 1, :] = -100
    x_bounds[3].max[vZrotABD_AuJo:vYrotABD_AuJo + 1, :] = 100

    x_bounds[3].min[vZrotABD_JeCh:vYrotABD_JeCh + 1, :] = -100
    x_bounds[3].max[vZrotABD_JeCh:vYrotABD_JeCh + 1, :] = 100

    # coude gauche
    x_bounds[3].min[vZrotABD_AuJo:vYrotABG_AuJo + 1, :] = -100
    x_bounds[3].max[vZrotABD_AuJo:vYrotABG_AuJo + 1, :] = 100

    x_bounds[3].min[vZrotABD_JeCh:vYrotABG_JeCh + 1, :] = -100
    x_bounds[3].max[vZrotABD_JeCh:vYrotABG_JeCh + 1, :] = 100

    # du carpe
    x_bounds[3].min[vXrotC_AuJo, :] = -100
    x_bounds[3].max[vXrotC_AuJo, :] = 100

    x_bounds[3].min[vXrotC_JeCh, :] = -100
    x_bounds[3].max[vXrotC_JeCh, :] = 100

    # du dehanchement
    x_bounds[3].min[vYrotC_AuJo, :] = -100
    x_bounds[3].max[vYrotC_AuJo, :] = 100

    x_bounds[3].min[vYrotC_JeCh, :] = -100
    x_bounds[3].max[vYrotC_JeCh, :] = 100

    #
    # Contraintes de position: PHASE 4 la reception
    #

    # deplacement
    x_bounds[4].min[X_AuJo, :] = -.1
    x_bounds[4].max[X_AuJo, :] = .1
    x_bounds[4].min[Y_AuJo, FIN] = -.1
    x_bounds[4].max[Y_AuJo, FIN] = .1
    x_bounds[4].min[Z_AuJo, :] = 0
    x_bounds[4].max[Z_AuJo, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    x_bounds[4].min[Z_AuJo, FIN] = 0
    x_bounds[4].max[Z_AuJo, FIN] = .1

    x_bounds[4].min[X_JeCh, :] = -.1
    x_bounds[4].max[X_JeCh, :] = .1
    x_bounds[4].min[Y_JeCh, FIN] = -.1
    x_bounds[4].max[Y_JeCh, FIN] = .1
    x_bounds[4].min[Z_JeCh, :] = 0
    x_bounds[4].max[Z_JeCh, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    x_bounds[4].min[Z_JeCh, FIN] = 0
    x_bounds[4].max[Z_JeCh, FIN] = .1

    # le salto autour de x
    x_bounds[4].min[Xrot_AuJo, :] = 2 * 3.14 + 3 / 2 * 3.14 - .2  # penche vers avant -> moins de salto
    x_bounds[4].max[Xrot_AuJo, :] = -.50 + 4 * 3.14  # un peu carpe a la fin
    x_bounds[4].min[Xrot_AuJo, FIN] = -.50 + 4 * 3.14 - .1
    x_bounds[4].max[Xrot_AuJo, FIN] = -.50 + 4 * 3.14 + .1  # 2 salto fin un peu carpe

    x_bounds[4].min[Xrot_JeCh, :] = 2 * 3.14 + 3 / 2 * 3.14 - .2  # penche vers avant -> moins de salto
    x_bounds[4].max[Xrot_JeCh, :] = -.50 + 4 * 3.14  # un peu carpe a la fin
    x_bounds[4].min[Xrot_JeCh, FIN] = -.50 + 4 * 3.14 - .1
    x_bounds[4].max[Xrot_JeCh, FIN] = -.50 + 4 * 3.14 + .1  # 2 salto fin un peu carpe

    # limitation du tilt autour de y
    x_bounds[4].min[Yrot_AuJo, :] = - 3.14 / 16
    x_bounds[4].max[Yrot_AuJo, :] = 3.14 / 16

    x_bounds[4].min[Yrot_JeCh, :] = - 3.14 / 16
    x_bounds[4].max[Yrot_JeCh, :] = 3.14 / 16

    # la vrille autour de z
    x_bounds[4].min[Zrot_AuJo, :] = 3 * 3.14 - .1  # complete la vrille
    x_bounds[4].max[Zrot_AuJo, :] = 3 * 3.14 + .1

    x_bounds[4].min[Zrot_JeCh, :] = 3 * 3.14 - .1  # complete la vrille
    x_bounds[4].max[Zrot_JeCh, :] = 3 * 3.14 + .1

    # bras droit
    x_bounds[4].min[YrotBD_AuJo, FIN] = 2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[YrotBD_AuJo, FIN] = 2.9 + .1
    x_bounds[4].min[ZrotBD_AuJo, FIN] = -.1
    x_bounds[4].max[ZrotBD_AuJo, FIN] = .1

    x_bounds[4].min[YrotBD_JeCh, FIN] = 2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[YrotBD_JeCh, FIN] = 2.9 + .1
    x_bounds[4].min[ZrotBD_JeCh, FIN] = -.1
    x_bounds[4].max[ZrotBD_JeCh, FIN] = .1

    # bras gauche
    x_bounds[4].min[YrotBG_AuJo, FIN] = -2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[YrotBG_AuJo, FIN] = -2.9 + .1
    x_bounds[4].min[ZrotBG_AuJo, FIN] = -.1
    x_bounds[4].max[ZrotBG_AuJo, FIN] = .1

    x_bounds[4].min[YrotBG_JeCh, FIN] = -2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[YrotBG_JeCh, FIN] = -2.9 + .1
    x_bounds[4].min[ZrotBG_JeCh, FIN] = -.1
    x_bounds[4].max[ZrotBG_JeCh, FIN] = .1

    # coude droit
    x_bounds[4].min[ZrotABD_AuJo:XrotABD_AuJo + 1, FIN] = -.1
    x_bounds[4].max[ZrotABD_AuJo:XrotABD_AuJo + 1, FIN] = .1

    x_bounds[4].min[ZrotABD_JeCh:XrotABD_JeCh + 1, FIN] = -.1
    x_bounds[4].max[ZrotABD_JeCh:XrotABD_JeCh + 1, FIN] = .1
    # coude gauche
    x_bounds[4].min[ZrotABG_AuJo:XrotABG_AuJo + 1, FIN] = -.1
    x_bounds[4].max[ZrotABG_AuJo:XrotABG_AuJo + 1, FIN] = .1

    x_bounds[4].min[ZrotABG_JeCh:XrotABG_JeCh + 1, FIN] = -.1
    x_bounds[4].max[ZrotABG_JeCh:XrotABG_JeCh + 1, FIN] = .1

    # le carpe
    x_bounds[4].min[XrotC_AuJo, :] = -.4
    x_bounds[4].min[XrotC_AuJo, FIN] = -.60
    x_bounds[4].max[XrotC_AuJo, FIN] = -.40  # fin un peu carpe

    x_bounds[4].min[XrotC_JeCh, :] = -.4
    x_bounds[4].min[XrotC_JeCh, FIN] = -.60
    x_bounds[4].max[XrotC_JeCh, FIN] = -.40  # fin un peu carpe

    # le dehanchement
    x_bounds[4].min[YrotC_AuJo, FIN] = -.1
    x_bounds[4].max[YrotC_AuJo, FIN] = .1

    x_bounds[4].min[YrotC_JeCh, FIN] = -.1
    x_bounds[4].max[YrotC_JeCh, FIN] = .1

    # Contraintes de vitesse: PHASE 4 la reception

    # en xy bassin
    x_bounds[4].min[vX_AuJo:vY_AuJo + 1, :] = -10
    x_bounds[4].max[vX_AuJo:vY_AuJo + 1, :] = 10

    x_bounds[4].min[vX_JeCh:vY_JeCh + 1, :] = -10
    x_bounds[4].max[vX_JeCh:vY_JeCh + 1, :] = 10

    # z bassin
    x_bounds[4].min[vZ_AuJo, :] = -100
    x_bounds[4].max[vZ_AuJo, :] = 100

    x_bounds[4].min[vZ_JeCh, :] = -100
    x_bounds[4].max[vZ_JeCh, :] = 100

    # autour de x
    x_bounds[4].min[vXrot_AuJo, :] = -100
    x_bounds[4].max[vXrot_AuJo, :] = 100

    x_bounds[4].min[vXrot_JeCh, :] = -100
    x_bounds[4].max[vXrot_JeCh, :] = 100

    # autour de y
    x_bounds[4].min[vYrot_AuJo, :] = -100
    x_bounds[4].max[vYrot_AuJo, :] = 100

    x_bounds[4].min[vYrot_JeCh, :] = -100
    x_bounds[4].max[vYrot_JeCh, :] = 100

    # autour de z
    x_bounds[4].min[vZrot_AuJo, :] = -100
    x_bounds[4].max[vZrot_AuJo, :] = 100

    x_bounds[4].min[vZrot_JeCh, :] = -100
    x_bounds[4].max[vZrot_JeCh, :] = 100

    # bras droit
    x_bounds[4].min[vZrotBD_AuJo:vYrotBD_AuJo + 1, :] = -100
    x_bounds[4].max[vZrotBD_AuJo:vYrotBD_AuJo + 1, :] = 100

    x_bounds[4].min[vZrotBD_JeCh:vYrotBD_JeCh + 1, :] = -100
    x_bounds[4].max[vZrotBD_JeCh:vYrotBD_JeCh + 1, :] = 100

    # bras droit
    x_bounds[4].min[vZrotBG_AuJo:vYrotBG_AuJo + 1, :] = -100
    x_bounds[4].max[vZrotBG_AuJo:vYrotBG_AuJo + 1, :] = 100

    x_bounds[4].min[vZrotBG_JeCh:vYrotBG_JeCh + 1, :] = -100
    x_bounds[4].max[vZrotBG_JeCh:vYrotBG_JeCh + 1, :] = 100

    # coude droit
    x_bounds[4].min[vZrotABD_AuJo:vYrotABD_AuJo + 1, :] = -100
    x_bounds[4].max[vZrotABD_AuJo:vYrotABD_AuJo + 1, :] = 100

    x_bounds[4].min[vZrotABD_JeCh:vYrotABD_JeCh + 1, :] = -100
    x_bounds[4].max[vZrotABD_JeCh:vYrotABD_JeCh + 1, :] = 100

    # coude gauche
    x_bounds[4].min[vZrotABD_AuJo:vYrotABG_AuJo + 1, :] = -100
    x_bounds[4].max[vZrotABD_AuJo:vYrotABG_AuJo + 1, :] = 100

    x_bounds[4].min[vZrotABD_JeCh:vYrotABG_JeCh + 1, :] = -100
    x_bounds[4].max[vZrotABD_JeCh:vYrotABG_JeCh + 1, :] = 100

    # du carpe
    x_bounds[4].min[vXrotC_AuJo, :] = -100
    x_bounds[4].max[vXrotC_AuJo, :] = 100

    x_bounds[4].min[vXrotC_JeCh, :] = -100
    x_bounds[4].max[vXrotC_JeCh, :] = 100

    # du dehanchement
    x_bounds[4].min[vYrotC_AuJo, :] = -100
    x_bounds[4].max[vYrotC_AuJo, :] = 100

    x_bounds[4].min[vYrotC_JeCh, :] = -100
    x_bounds[4].max[vYrotC_JeCh, :] = 100

    #
    # Initial guesses
    #
    x0 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x1 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x2 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x3 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x4 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))

    x0[Xrot_AuJo, 0] = .50
    x0[ZrotBG_AuJo] = -.75
    x0[ZrotBD_AuJo] = .75
    x0[YrotBG_AuJo, 0] = -2.9
    x0[YrotBD_AuJo, 0] = 2.9
    x0[YrotBG_AuJo, 1] = -1.35
    x0[YrotBD_AuJo, 1] = 1.35
    x0[XrotC_AuJo, 0] = -.5
    x0[XrotC_AuJo, 1] = -2.6

    x0[Xrot_JeCh, 0] = .50
    x0[ZrotBG_JeCh] = -.75
    x0[ZrotBD_JeCh] = .75
    x0[YrotBG_JeCh, 0] = -2.9
    x0[YrotBD_JeCh, 0] = 2.9
    x0[YrotBG_JeCh, 1] = -1.35
    x0[YrotBD_JeCh, 1] = 1.35
    x0[XrotC_JeCh, 0] = -.5
    x0[XrotC_JeCh, 1] = -2.6

    x1[ZrotBG_AuJo] = -.75
    x1[ZrotBD_AuJo] = .75
    x1[Xrot_AuJo, 1] = 2 * 3.14
    x1[YrotBG_AuJo] = -1.35
    x1[YrotBD_AuJo] = 1.35
    x1[XrotC_AuJo] = -2.6

    x1[ZrotBG_JeCh] = -.75
    x1[ZrotBD_JeCh] = .75
    x1[Xrot_JeCh, 1] = 2 * 3.14
    x1[YrotBG_JeCh] = -1.35
    x1[YrotBD_JeCh] = 1.35
    x1[XrotC_JeCh] = -2.6

    x2[Xrot_AuJo] = 2 * 3.14
    x2[Zrot_AuJo, 1] = 3.14
    x2[ZrotBG_AuJo, 0] = -.75
    x2[ZrotBD_AuJo, 0] = .75
    x2[YrotBG_AuJo, 0] = -1.35
    x2[YrotBD_AuJo, 0] = 1.35
    x2[XrotC_AuJo, 0] = -2.6

    x2[Xrot_JeCh] = 2 * 3.14
    x2[Zrot_JeCh, 1] = 3.14
    x2[ZrotBG_JeCh, 0] = -.75
    x2[ZrotBD_JeCh, 0] = .75
    x2[YrotBG_JeCh, 0] = -1.35
    x2[YrotBD_JeCh, 0] = 1.35
    x2[XrotC_JeCh, 0] = -2.6

    x3[Xrot_AuJo, 0] = 2 * 3.14
    x3[Xrot_AuJo, 1] = 2 * 3.14 + 3/2 * 3.14
    x3[Zrot_AuJo, 0] = 3.14
    x3[Zrot_AuJo, 1] = 3 * 3.14

    x3[Xrot_JeCh, 0] = 2 * 3.14
    x3[Xrot_JeCh, 1] = 2 * 3.14 + 3 / 2 * 3.14
    x3[Zrot_JeCh, 0] = 3.14
    x3[Zrot_JeCh, 1] = 3 * 3.14

    x4[Xrot_AuJo, 0] = 2 * 3.14 + 3/2 * 3.14
    x4[Xrot_AuJo, 1] = 4 * 3.14
    x4[Zrot_AuJo] = 3 * 3.14
    x4[XrotC_AuJo, 1] = -.5

    x4[Xrot_JeCh, 0] = 2 * 3.14 + 3 / 2 * 3.14
    x4[Xrot_JeCh, 1] = 4 * 3.14
    x4[Zrot_JeCh] = 3 * 3.14
    x4[XrotC_JeCh, 1] = -.5

    x_init = InitialGuessList()
    x_init.add(x0, interpolation=InterpolationType.LINEAR)
    x_init.add(x1, interpolation=InterpolationType.LINEAR)
    x_init.add(x2, interpolation=InterpolationType.LINEAR)
    x_init.add(x3, interpolation=InterpolationType.LINEAR)
    x_init.add(x4, interpolation=InterpolationType.LINEAR)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.05, max_bound=.05, first_marker='MidMainGAuJo', second_marker='CibleMainGAuJo', phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.05, max_bound=.05, first_marker='MidMainDAuJo', second_marker='CibleMainDAuJo', phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.05, max_bound=.05, first_marker='MidMainGJeCh', second_marker='CibleMainGJeCh', phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.05, max_bound=.05, first_marker='MidMainDJeCh', second_marker='CibleMainDJeCh', phase=1)
#    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0, max_bound=final_time, phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=1)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=2)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=3)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=4)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        [final_time/len(biorbd_model)] * len(biorbd_model),
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
        n_threads=n_threads
    )


def main():
    """
    Prepares and solves an ocp for a 803<. Animates the results
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="the bioMod file")
    parser.add_argument("--no-hsl", dest='with_hsl', action='store_false', help="do not use libhsl")
    parser.add_argument("-j", default=1, dest='n_threads', type=int, help="number of threads in the solver")
    parser.add_argument("--no-sol", action='store_false', dest='savesol', help="do not save the solution")
    parser.add_argument("--no-show-online", action='store_false', dest='show_online', help="do not show graphs during optimization")
    parser.add_argument("--print-ocp", action='store_true', dest='print_ocp', help="print the ocp")
    args = parser.parse_args()

    n_shooting = (40, 100, 100, 100, 40)
    ocp = prepare_ocp(args.model, n_shooting=n_shooting, n_threads=args.n_threads, final_time=1.87)
    ocp.add_plot_penalty(CostType.ALL)
    if args.print_ocp:
        ocp.print(to_graph=True)
    solver = Solver.IPOPT(show_online_optim=args.show_online, show_options=dict(show_bounds=True))
    if args.with_hsl:
        solver.set_linear_solver('ma57')
    else:
        print("Not using ma57")
    solver.set_maximum_iterations(10000)
    solver.set_convergence_tolerance(1e-4)
    sol = ocp.solve(solver)

    temps = time.strftime("%Y-%m-%d-%H%M")
    nom = args.model.split('/')[-1].removesuffix('.bioMod')
    qs = sol.states[0]['q']
    qdots = sol.states[0]['qdot']
    for i in range(1, len(sol.states)):
        qs = np.hstack((qs, sol.states[i]['q']))
        qdots = np.hstack((qdots, sol.states[i]['qdot']))
    if args.savesol:  # switch manuelle
        np.save(f"Solutions/{nom}-{str(n_shooting).replace(', ', '_')}-{temps}-q.npy", qs)
        np.save(f"Solutions/{nom}-{str(n_shooting).replace(', ', '_')}-{temps}-qdot.npy", qdots)
        np.save(f"Solutions/{nom}-{str(n_shooting).replace(', ', '_')}-{temps}-t.npy", sol.phase_time)

    if IPYTHON:
        IPython.embed()  # afin de pouvoir explorer plus en details la solution

    # Print the last solution
    #sol.animate(n_frames=-1, show_floor=False)
    # sol.graphs(show_bounds=True)

if __name__ == "__main__":
    main()
