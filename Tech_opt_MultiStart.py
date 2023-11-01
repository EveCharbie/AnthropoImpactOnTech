"""
The goal of this program is to optimize the movement to achieve a rudi out pike (803<) for left twisters.
"""
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
            InitialGuessList,
            InterpolationType,
            OdeSolver,
            Node,
            Solver,
            BiMappingList,
            CostType,
            ConstraintList,
            ConstraintFcn,
            PenaltyController,
            MultiStart,
            Solution,
            MagnitudeType,
            BiorbdModel,
        )
import time
import pickle


class Model:
    """
    Attributes
    ----------
    model: str
        A reference to the name of the model
    with_hsl  :
        no hsl, don't use libhsl
    n_threads : int
        refers to the numbers of threads in the solver
    savesol :
        returns true if empty, else returns False
    show_online : bool
        returns true if empty, else returns False
    print_ocp : bool
        returns False if empty, else returns True """

    def __init__(self, model, n_threads=5, with_hsl=False, savesol=False, show_online=False, print_ocp=False):

        self.model = model
        self.with_hsl = with_hsl
        self.n_threads = n_threads
        self.savesol = savesol
        self.show_online = show_online
        self.print_ocp = print_ocp

        #
        # # if savesol :
        # #    return False
        #
        # if show_online:
        #     return False
        #
        # if print_ocp:
        #     return True


        #   parser = argparse.ArgumentParser()
        # parser.add_argument("model", type=str, help="the bioMod file")
        # parser.add_argument("--no-hsl", dest='with_hsl', action='store_false', help="do not use libhsl")
        # parser.add_argument("-j", default=1, dest='n_threads', type=int, help="number of threads in the solver")
        # parser.add_argument("--no-sol", action='store_false', dest='savesol', help="do not save the solution")
        # parser.add_argument("--no-show-online", action='store_false', dest='show_online', help="do not show graphs during optimization")
        # parser.add_argument("--print-ocp", action='store_true', dest='print_ocp', help="print the ocp")
        # args = parser.parse_args()
        #

try:
    import IPython

    IPYTHON = True
except ImportError:
    print("No IPython.")
    IPYTHON = False


def minimize_dofs(controller: PenaltyController, dofs: list, targets: list):
    diff = 0
    for i, dof in enumerate(dofs):
        diff += (controller.states['q'].cx_start[dof] - targets[i]) ** 2
    return diff


def prepare_ocp(
        biorbd_model_path: str,
        nb_twist: int,
        seed : int,
        athlete_to_copy = None,
        save_folder = None,
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

    biomodel = (BiorbdModel(biorbd_model_path))
    biorbd_model = (biomodel, biomodel, biomodel, biomodel, biomodel)

    nb_q = biorbd_model[0].nb_q
    nb_qdot = biorbd_model[0].nb_qdot
    nb_qddot_joints = nb_q - biorbd_model[0].nb_root

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
    vX = 0
    vY = 1
    vZ = 2
    vXrot = 3
    vYrot = 4
    vZrot = 5
    vZrotBD = 6
    vYrotBD = 7
    vZrotABD = 8
    vYrotABD = 9
    vZrotBG = 10
    vYrotBG = 11
    vZrotABG = 12
    vYrotABG = 13
    vXrotC = 14
    vYrotC = 15

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, marker_index=1, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING,
                            weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING,
                            weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING,
                            weight=1, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING,
                            weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING,
                            weight=1, phase=4)

    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=1.0, weight=100000,
                            phase=0)

    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=1.0, weight=100000,
                            phase=2)

    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, node=Node.END, first_marker='MidMainG',
                            second_marker='CibleMainG', weight=1000, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, node=Node.END, first_marker='MidMainD',
                            second_marker='CibleMainD', weight=1000, phase=0)

    # arrete de gigoter les bras
    les_bras = [ZrotBD, YrotBD, ZrotABD, XrotABD, ZrotBG, YrotBG, ZrotABG, XrotABG]
    les_coudes = [ZrotABD, XrotABD, ZrotABG, XrotABG]
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING,
                            dofs=les_coudes, targets=np.zeros(len(les_coudes)), weight=1000, phase=0)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING,
                            dofs=les_bras, targets=np.zeros(len(les_bras)), weight=10, phase=0)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING,
                            dofs=les_bras, targets=np.zeros(len(les_bras)), weight=10, phase=1)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING,
                            dofs=les_bras, targets=np.zeros(len(les_bras)), weight=10, phase=2)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING,
                            dofs=les_bras, targets=np.zeros(len(les_bras)), weight=10, phase=3)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING,
                            dofs=les_bras, targets=np.zeros(len(les_bras)), weight=10, phase=4)
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL_SHOOTING,
                            dofs=les_coudes, targets=np.zeros(len(les_coudes)), weight=1000, phase=4)

    # ouvre les hanches rapidement apres la vrille
    objective_functions.add(minimize_dofs, custom_type=ObjectiveFcn.Mayer, node=Node.END, dofs=[XrotC],
                            targets=[0], weight=10000, phase=3)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)

    qddot_joints_min, qddot_joints_max, qddot_joints_init = -500, 500, 0
    u_bounds = BoundsList()
    for i in range(5):
        u_bounds.add("qddot_joints", min_bound=[qddot_joints_min] * nb_qddot_joints, max_bound=[qddot_joints_max] * nb_qddot_joints, phase=i)
        
    u_init = InitialGuessList()
    u0 = np.ones((nb_qddot_joints, n_shooting[0])) * qddot_joints_init
    u1 = np.ones((nb_qddot_joints, n_shooting[1])) * qddot_joints_init
    u2 = np.ones((nb_qddot_joints, n_shooting[2])) * qddot_joints_init
    u3 = np.ones((nb_qddot_joints, n_shooting[3])) * qddot_joints_init
    u4 = np.ones((nb_qddot_joints, n_shooting[4])) * qddot_joints_init

    # Path constraint
    x_bounds = BoundsList()
    for i in range(5):
        x_bounds.add("q", min_bound=biorbd_model[0].bounds_from_ranges("q").min, max_bound=biorbd_model[0].bounds_from_ranges("q").max, phase=i)
        x_bounds.add("qdot", min_bound=biorbd_model[0].bounds_from_ranges("qdot").min, max_bound=biorbd_model[0].bounds_from_ranges("qdot").max, phase=i)

    # Pour la lisibilite
    DEBUT, MILIEU, FIN = 0, 1, 2

    #
    # Contraintes de position: PHASE 0 la montee en carpe
    #

    zmax = 8
    # 12 / 8 * final_time**2 + 1  # une petite marge

    # deplacement
    x_bounds[0]["q"].min[X, :] = -.1
    x_bounds[0]["q"].max[X, :] = .1
    x_bounds[0]["q"].min[Y, :] = -1.
    x_bounds[0]["q"].max[Y, :] = 1.
    x_bounds[0]["q"].min[:Z + 1, DEBUT] = 0
    x_bounds[0]["q"].max[:Z + 1, DEBUT] = 0
    x_bounds[0]["q"].min[Z, MILIEU:] = 0
    x_bounds[0]["q"].max[Z, MILIEU:] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[0]["q"].min[Xrot, :] = 0
    # 2 * 3.14 + 3 / 2 * 3.14 - .2
    x_bounds[0]["q"].max[Xrot, :] = -.50 + 3.14
    x_bounds[0]["q"].min[Xrot, DEBUT] = .50  # penche vers l'avant un peu carpe
    x_bounds[0]["q"].max[Xrot, DEBUT] = .50
    x_bounds[0]["q"].min[Xrot, MILIEU:] = 0
    x_bounds[0]["q"].max[Xrot, MILIEU:] = 4 * 3.14 + .1  # salto
    # limitation du tilt autour de y
    x_bounds[0]["q"].min[Yrot, DEBUT] = 0
    x_bounds[0]["q"].max[Yrot, DEBUT] = 0
    x_bounds[0]["q"].min[Yrot, MILIEU:] = - 3.14 / 16  # vraiment pas suppose tilte
    x_bounds[0]["q"].max[Yrot, MILIEU:] = 3.14 / 16
    # la vrille autour de z
    x_bounds[0]["q"].min[Zrot, DEBUT] = 0
    x_bounds[0]["q"].max[Zrot, DEBUT] = 0
    x_bounds[0]["q"].min[Zrot, MILIEU:] = -.1  # pas de vrille dans cette phase
    x_bounds[0]["q"].max[Zrot, MILIEU:] = .1

    # bras droit
    x_bounds[0]["q"].min[YrotBD, DEBUT] = 2.9  # debut bras aux oreilles
    x_bounds[0]["q"].max[YrotBD, DEBUT] = 2.9
    x_bounds[0]["q"].min[ZrotBD, DEBUT] = 0
    x_bounds[0]["q"].max[ZrotBD, DEBUT] = 0
    # bras gauche
    x_bounds[0]["q"].min[YrotBG, DEBUT] = -2.9  # debut bras aux oreilles
    x_bounds[0]["q"].max[YrotBG, DEBUT] = -2.9
    x_bounds[0]["q"].min[ZrotBG, DEBUT] = 0
    x_bounds[0]["q"].max[ZrotBG, DEBUT] = 0

    # coude droit
    x_bounds[0]["q"].min[ZrotABD:XrotABD + 1, DEBUT] = 0
    x_bounds[0]["q"].max[ZrotABD:XrotABD + 1, DEBUT] = 0
    # coude gauche
    x_bounds[0]["q"].min[ZrotABG:XrotABG + 1, DEBUT] = 0
    x_bounds[0]["q"].max[ZrotABG:XrotABG + 1, DEBUT] = 0

    # le carpe
    x_bounds[0]["q"].min[XrotC, DEBUT] = -.50  # depart un peu ferme aux hanches
    x_bounds[0]["q"].max[XrotC, DEBUT] = -.50
    x_bounds[0]["q"].max[XrotC, FIN] = -2.5
    # x_bounds[0].min[XrotC, FIN] = 2.7  # min du modele
    # le dehanchement
    x_bounds[0]["q"].min[YrotC, DEBUT] = 0
    x_bounds[0]["q"].max[YrotC, DEBUT] = 0
    x_bounds[0]["q"].min[YrotC, MILIEU:] = -.1
    x_bounds[0]["q"].max[YrotC, MILIEU:] = .1

    # Contraintes de vitesse: PHASE 0 la montee en carpe

    vzinit = 9.81 / (2 * final_time ) # vitesse initiale en z du CoM pour revenir a terre au temps final

    # decalage entre le bassin et le CoM
    CoM_Q_sym = MX.sym('CoM', nb_q)
    CoM_Q_init = x_bounds[0]["q"].min[:nb_q,
                 DEBUT]  # min ou max ne change rien a priori, au DEBUT ils sont egaux normalement
    CoM_Q_func = Function('CoM_Q_func', [CoM_Q_sym], [biorbd_model[0].center_of_mass(CoM_Q_sym)])
    bassin_Q_func = Function('bassin_Q_func', [CoM_Q_sym],
                             [biorbd_model[0].homogeneous_matrices_in_global(CoM_Q_sym, 0).to_mx()])  # retourne la RT du bassin

    r = np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1,
                                                         :3]  # selectionne seulement la translation de la RT

    # en xy bassin
    x_bounds[0]["qdot"].min[vX:vY + 1, :] = -10
    x_bounds[0]["qdot"].max[vX:vY + 1, :] = 10
    x_bounds[0]["qdot"].min[vX:vY + 1, DEBUT] = -.5
    x_bounds[0]["qdot"].max[vX:vY + 1, DEBUT] = .5
    # z bassin
    x_bounds[0]["qdot"].min[vZ, :] = -50
    x_bounds[0]["qdot"].max[vZ, :] = 50
    x_bounds[0]["qdot"].min[vZ, DEBUT] = vzinit - .5
    x_bounds[0]["qdot"].max[vZ, DEBUT] = vzinit + .5

    # autour de x
    x_bounds[0]["qdot"].min[vXrot, :] = .5  # d'apres une observation video
    x_bounds[0]["qdot"].max[vXrot, :] = 20  # aussi vite que nécessaire, mais ne devrait pas atteindre cette vitesse
    # autour de y
    x_bounds[0]["qdot"].min[vYrot, :] = -50
    x_bounds[0]["qdot"].max[vYrot, :] = 50
    x_bounds[0]["qdot"].min[vYrot, DEBUT] = 0
    x_bounds[0]["qdot"].max[vYrot, DEBUT] = 0
    # autour de z
    x_bounds[0]["qdot"].min[vZrot, :] = -50
    x_bounds[0]["qdot"].max[vZrot, :] = 50
    x_bounds[0]["qdot"].min[vZrot, DEBUT] = 0
    x_bounds[0]["qdot"].max[vZrot, DEBUT] = 0

    # tenir compte du decalage entre bassin et CoM avec la rotation
    # Qtransdot = Qtransdot + v cross Qrotdot
    borne_inf = (x_bounds[0]["qdot"].min[vX:vZ + 1, DEBUT] + np.cross(r, x_bounds[0]["qdot"].min[vXrot:vZrot + 1, DEBUT]))[0]
    borne_sup = (x_bounds[0]["qdot"].max[vX:vZ + 1, DEBUT] + np.cross(r, x_bounds[0]["qdot"].max[vXrot:vZrot + 1, DEBUT]))[0]
    x_bounds[0]["qdot"].min[vX:vZ + 1, DEBUT] = min(borne_sup[0], borne_inf[0]), min(borne_sup[1], borne_inf[1]), min(
        borne_sup[2], borne_inf[2])
    x_bounds[0]["qdot"].max[vX:vZ + 1, DEBUT] = max(borne_sup[0], borne_inf[0]), max(borne_sup[1], borne_inf[1]), max(
        borne_sup[2], borne_inf[2])

    # bras droit
    x_bounds[0]["qdot"].min[vZrotBD:vYrotBD + 1, :] = -50
    x_bounds[0]["qdot"].max[vZrotBD:vYrotBD + 1, :] = 50
    x_bounds[0]["qdot"].min[vZrotBD:vYrotBD + 1, DEBUT] = 0
    x_bounds[0]["qdot"].max[vZrotBD:vYrotBD + 1, DEBUT] = 0
    # bras droit
    x_bounds[0]["qdot"].min[vZrotBG:vYrotBG + 1, :] = -50
    x_bounds[0]["qdot"].max[vZrotBG:vYrotBG + 1, :] = 50
    x_bounds[0]["qdot"].min[vZrotBG:vYrotBG + 1, DEBUT] = 0
    x_bounds[0]["qdot"].max[vZrotBG:vYrotBG + 1, DEBUT] = 0

    # coude droit
    x_bounds[0]["qdot"].min[vZrotABD:vYrotABD + 1, :] = -50
    x_bounds[0]["qdot"].max[vZrotABD:vYrotABD + 1, :] = 50
    x_bounds[0]["qdot"].min[vZrotABD:vYrotABD + 1, DEBUT] = 0
    x_bounds[0]["qdot"].max[vZrotABD:vYrotABD + 1, DEBUT] = 0
    # coude gauche
    x_bounds[0]["qdot"].min[vZrotABD:vYrotABG + 1, :] = -50
    x_bounds[0]["qdot"].max[vZrotABD:vYrotABG + 1, :] = 50
    x_bounds[0]["qdot"].min[vZrotABG:vYrotABG + 1, DEBUT] = 0
    x_bounds[0]["qdot"].max[vZrotABG:vYrotABG + 1, DEBUT] = 0

    # du carpe
    x_bounds[0]["qdot"].min[vXrotC, :] = -50
    x_bounds[0]["qdot"].max[vXrotC, :] = 50
    x_bounds[0]["qdot"].min[vXrotC, DEBUT] = 0
    x_bounds[0]["qdot"].max[vXrotC, DEBUT] = 0
    # du dehanchement
    x_bounds[0]["qdot"].min[vYrotC, :] = -50
    x_bounds[0]["qdot"].max[vYrotC, :] = 50
    x_bounds[0]["qdot"].min[vYrotC, DEBUT] = 0
    x_bounds[0]["qdot"].max[vYrotC, DEBUT] = 0

    #
    # Contraintes de position: PHASE 1 le salto carpe
    #

    # deplacement
    x_bounds[1]["q"].min[X, :] = -.1
    x_bounds[1]["q"].max[X, :] = .1
    x_bounds[1]["q"].min[Y, :] = -1.
    x_bounds[1]["q"].max[Y, :] = 1.
    x_bounds[1]["q"].min[Z, :] = 0
    x_bounds[1]["q"].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[1]["q"].min[Xrot, :] = 0
    x_bounds[1]["q"].max[Xrot, :] = -.50 + 4 * 3.14
    x_bounds[1]["q"].min[Xrot, FIN] = 2 * 3.14 - .1
    # limitation du tilt autour de y
    x_bounds[1]["q"].min[Yrot, :] = - 3.14 / 16
    x_bounds[1]["q"].max[Yrot, :] = 3.14 / 16
    # la vrille autour de z
    x_bounds[1]["q"].min[Zrot, :] = -.1
    x_bounds[1]["q"].max[Zrot, :] = .1

    # le carpe
    x_bounds[1]["q"].max[XrotC, :] = -2.5
    # le dehanchement
    x_bounds[1]["q"].min[YrotC, DEBUT] = -.1
    x_bounds[1]["q"].max[YrotC, DEBUT] = .1

    # Contraintes de vitesse: PHASE 1 le salto carpe

    # en xy bassin
    x_bounds[1]["qdot"].min[vX:vY + 1, :] = -10
    x_bounds[1]["qdot"].max[vX:vY + 1, :] = 10

    # z bassin
    x_bounds[1]["qdot"].min[vZ, :] = -50
    x_bounds[1]["qdot"].max[vZ, :] = 50

    # autour de x
    x_bounds[1]["qdot"].min[vXrot, :] = -50
    x_bounds[1]["qdot"].max[vXrot, :] = 50
    # autour de y
    x_bounds[1]["qdot"].min[vYrot, :] = -50
    x_bounds[1]["qdot"].max[vYrot, :] = 50

    # autour de z
    x_bounds[1]["qdot"].min[vZrot, :] = -50
    x_bounds[1]["qdot"].max[vZrot, :] = 50

    # bras droit
    x_bounds[1]["qdot"].min[vZrotBD:vYrotBD + 1, :] = -50
    x_bounds[1]["qdot"].max[vZrotBD:vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[1]["qdot"].min[vZrotBG:vYrotBG + 1, :] = -50
    x_bounds[1]["qdot"].max[vZrotBG:vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[1]["qdot"].min[vZrotABD:vYrotABD + 1, :] = -50
    x_bounds[1]["qdot"].max[vZrotABD:vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[1]["qdot"].min[vZrotABD:vYrotABG + 1, :] = -50
    x_bounds[1]["qdot"].max[vZrotABD:vYrotABG + 1, :] = 50

    # du carpe
    x_bounds[1]["qdot"].min[vXrotC, :] = -50
    x_bounds[1]["qdot"].max[vXrotC, :] = 50
    # du dehanchement
    x_bounds[1]["qdot"].min[vYrotC, :] = -50
    x_bounds[1]["qdot"].max[vYrotC, :] = 50

    #
    # Contraintes de position: PHASE 2 l'ouverture
    #

    # deplacement
    x_bounds[2]["q"].min[X, :] = -.2
    x_bounds[2]["q"].max[X, :] = .2
    x_bounds[2]["q"].min[Y, :] = -1.
    x_bounds[2]["q"].max[Y, :] = 1.
    x_bounds[2]["q"].min[Z, :] = 0
    x_bounds[2]["q"].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[2]["q"].min[Xrot, :] = 2 * 3.14 - .1
    x_bounds[2]["q"].max[Xrot, :] = -.50 + 4 * 3.14
    # limitation du tilt autour de y
    x_bounds[2]["q"].min[Yrot, :] = - 3.14 / 4
    x_bounds[2]["q"].max[Yrot, :] = 3.14 / 4
    # la vrille autour de z
    x_bounds[2]["q"].min[Zrot, :] = 0
    x_bounds[2]["q"].max[Zrot, :] = 3.14  # 5 * 3.14

    x_bounds[2]["q"].min[XrotC, FIN] = -.4

    # Contraintes de vitesse: PHASE 2 l'ouverture

    # en xy bassin
    x_bounds[2]["qdot"].min[vX:vY + 1, :] = -10
    x_bounds[2]["qdot"].max[vX:vY + 1, :] = 10

    # z bassin
    x_bounds[2]["qdot"].min[vZ, :] = -50
    x_bounds[2]["qdot"].max[vZ, :] = 50

    # autour de x
    x_bounds[2]["qdot"].min[vXrot, :] = -50
    x_bounds[2]["qdot"].max[vXrot, :] = 50
    # autour de y
    x_bounds[2]["qdot"].min[vYrot, :] = -50
    x_bounds[2]["qdot"].max[vYrot, :] = 50

    # autour de z
    x_bounds[2]["qdot"].min[vZrot, :] = -50
    x_bounds[2]["qdot"].max[vZrot, :] = 50

    # bras droit
    x_bounds[2]["qdot"].min[vZrotBD:vYrotBD + 1, :] = -50
    x_bounds[2]["qdot"].max[vZrotBD:vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[2]["qdot"].min[vZrotBG:vYrotBG + 1, :] = -50
    x_bounds[2]["qdot"].max[vZrotBG:vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[2]["qdot"].min[vZrotABD:vYrotABD + 1, :] = -50
    x_bounds[2]["qdot"].max[vZrotABD:vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[2]["qdot"].min[vZrotABD:vYrotABG + 1, :] = -50
    x_bounds[2]["qdot"].max[vZrotABD:vYrotABG + 1, :] = 50

    # du carpe
    x_bounds[2]["qdot"].min[vXrotC, :] = -50
    x_bounds[2]["qdot"].max[vXrotC, :] = 50
    # du dehanchement
    x_bounds[2]["qdot"].min[vYrotC, :] = -50
    x_bounds[2]["qdot"].max[vYrotC, :] = 50

    #
    # Contraintes de position: PHASE 3 la vrille et demie
    #

    # deplacement
    x_bounds[3]["q"].min[X, :] = -.2
    x_bounds[3]["q"].max[X, :] = .2
    x_bounds[3]["q"].min[Y, :] = -1.
    x_bounds[3]["q"].max[Y, :] = 1.
    x_bounds[3]["q"].min[Z, :] = 0
    x_bounds[3]["q"].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[3]["q"].min[Xrot, :] = 0
    x_bounds[3]["q"].min[Xrot, :] = 2 * 3.14 - .1

    x_bounds[3]["q"].max[Xrot, :] = 2 * 3.14 + 3 / 2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[3]["q"].min[Xrot, FIN] = 2 * 3.14 + 3 / 2 * 3.14 - .1
    x_bounds[3]["q"].max[Xrot, FIN] = 2 * 3.14 + 3 / 2 * 3.14 + .1  # 1 salto 3/4

    # limitation du tilt autour de y
    x_bounds[3]["q"].min[Yrot, :] = - 3.14 / 4
    x_bounds[3]["q"].max[Yrot, :] = 3.14 / 4
    x_bounds[3]["q"].min[Yrot, FIN] = - 3.14 / 8
    x_bounds[3]["q"].max[Yrot, FIN] = 3.14 / 8
    # la vrille autour de z
    x_bounds[3]["q"].min[Zrot, :] = 0
    x_bounds[3]["q"].max[Zrot, :] = 5 * 3.14
    x_bounds[3]["q"].min[Zrot, FIN] = nb_twist * 3.14 - .1  # complete la vrille
    x_bounds[3]["q"].max[Zrot, FIN] = nb_twist * 3.14 + .1

    # le carpe  f4a les jambes
    x_bounds[3]["q"].min[XrotC, :] = -.4
    # le dehanchement

    # Contraintes de vitesse: PHASE 3 la vrille et demie

    # en xy bassin
    x_bounds[3]["qdot"].min[vX:vY + 1, :] = -10
    x_bounds[3]["qdot"].max[vX:vY + 1, :] = 10

    # z bassin
    x_bounds[3]["qdot"].min[vZ, :] = -50
    x_bounds[3]["qdot"].max[vZ, :] = 50

    # autour de x
    x_bounds[3]["qdot"].min[vXrot, :] = -50
    x_bounds[3]["qdot"].max[vXrot, :] = 50
    # autour de y
    x_bounds[3]["qdot"].min[vYrot, :] = -50
    x_bounds[3]["qdot"].max[vYrot, :] = 50

    # autour de z
    x_bounds[3]["qdot"].min[vZrot, :] = -50
    x_bounds[3]["qdot"].max[vZrot, :] = 50

    # bras droit
    x_bounds[3]["qdot"].min[vZrotBD:vYrotBD + 1, :] = -50
    x_bounds[3]["qdot"].max[vZrotBD:vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[3]["qdot"].min[vZrotBG:vYrotBG + 1, :] = -50
    x_bounds[3]["qdot"].max[vZrotBG:vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[3]["qdot"].min[vZrotABD:vYrotABD + 1, :] = -50
    x_bounds[3]["qdot"].max[vZrotABD:vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[3]["qdot"].min[vZrotABD:vYrotABG + 1, :] = -50
    x_bounds[3]["qdot"].max[vZrotABD:vYrotABG + 1, :] = 50

    # du carpe
    x_bounds[3]["qdot"].min[vXrotC, :] = -50
    x_bounds[3]["qdot"].max[vXrotC, :] = 50
    # du dehanchement
    x_bounds[3]["qdot"].min[vYrotC, :] = -50
    x_bounds[3]["qdot"].max[vYrotC, :] = 50

    #
    # Contraintes de position: PHASE 4 la reception
    #

    # deplacement
    x_bounds[4]["q"].min[X, :] = -.1
    x_bounds[4]["q"].max[X, :] = .1
    x_bounds[4]["q"].min[Y, FIN] = -.1
    x_bounds[4]["q"].max[Y, FIN] = .1
    x_bounds[4]["q"].min[Z, :] = 0
    x_bounds[4]["q"].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    x_bounds[4]["q"].min[Z, FIN] = 0
    x_bounds[4]["q"].max[Z, FIN] = .1

    # le salto autour de x
    x_bounds[4]["q"].min[Xrot, :] = 2 * 3.14 + 3 / 2 * 3.14 - .2  # penche vers avant -> moins de salto
    x_bounds[4]["q"].max[Xrot, :] = -.50 + 4 * 3.14  # un peu carpe a la fin
    x_bounds[4]["q"].min[Xrot, FIN] = -.50 + 4 * 3.14 - .1  #  salto fin un peu carpe
    x_bounds[4]["q"].max[Xrot, FIN] = -.50 + 4 * 3.14 + .1  #  salto fin un peu carpe
    # limitation du tilt autour de y
    x_bounds[4]["q"].min[Yrot, :] = - 3.14 / 16
    x_bounds[4]["q"].max[Yrot, :] = 3.14 / 16
    # la vrille autour de z
    x_bounds[4]["q"].min[Zrot, :] = nb_twist * 3.14 - .1  # complete la vrille
    x_bounds[4]["q"].max[Zrot, :] = nb_twist * 3.14 + .1

    # bras droit
    x_bounds[4]["q"].min[YrotBD, FIN] = 2.9 - .1  # debut bras aux oreilles
    x_bounds[4]["q"].max[YrotBD, FIN] = 2.9 + .1
    x_bounds[4]["q"].min[ZrotBD, FIN] = -.1
    x_bounds[4]["q"].max[ZrotBD, FIN] = .1
    # bras gauche
    x_bounds[4]["q"].min[YrotBG, FIN] = -2.9 - .1  # debut bras aux oreilles
    x_bounds[4]["q"].max[YrotBG, FIN] = -2.9 + .1
    x_bounds[4]["q"].min[ZrotBG, FIN] = -.1
    x_bounds[4]["q"].max[ZrotBG, FIN] = .1

    # coude droit
    x_bounds[4]["q"].min[ZrotABD:XrotABD + 1, FIN] = -.1
    x_bounds[4]["q"].max[ZrotABD:XrotABD + 1, FIN] = .1
    # coude gauche
    x_bounds[4]["q"].min[ZrotABG:XrotABG + 1, FIN] = -.1
    x_bounds[4]["q"].max[ZrotABG:XrotABG + 1, FIN] = .1

    # le carpe
    x_bounds[4]["q"].min[XrotC, :] = -.4
    x_bounds[4]["q"].min[XrotC, FIN] = -.60
    x_bounds[4]["q"].max[XrotC, FIN] = -.40  # fin un peu carpe
    # le dehanchement
    x_bounds[4]["q"].min[YrotC, FIN] = -.1
    x_bounds[4]["q"].max[YrotC, FIN] = .1

    # Contraintes de vitesse: PHASE 4 la reception

    # en xy bassin
    x_bounds[4]["qdot"].min[vX:vY + 1, :] = -10
    x_bounds[4]["qdot"].max[vX:vY + 1, :] = 10

    # z bassin
    x_bounds[4]["qdot"].min[vZ, :] = -50
    x_bounds[4]["qdot"].max[vZ, :] = 50

    # autour de x
    x_bounds[4]["qdot"].min[vXrot, :] = -50
    x_bounds[4]["qdot"].max[vXrot, :] = 50
    # autour de y
    x_bounds[4]["qdot"].min[vYrot, :] = -50
    x_bounds[4]["qdot"].max[vYrot, :] = 50

    # autour de z
    x_bounds[4]["qdot"].min[vZrot, :] = -50
    x_bounds[4]["qdot"].max[vZrot, :] = 50

    # bras droit
    x_bounds[4]["qdot"].min[vZrotBD:vYrotBD + 1, :] = -50
    x_bounds[4]["qdot"].max[vZrotBD:vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[4]["qdot"].min[vZrotBG:vYrotBG + 1, :] = -50
    x_bounds[4]["qdot"].max[vZrotBG:vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[4]["qdot"].min[vZrotABD:vYrotABD + 1, :] = -50
    x_bounds[4]["qdot"].max[vZrotABD:vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[4]["qdot"].min[vZrotABD:vYrotABG + 1, :] = -50
    x_bounds[4]["qdot"].max[vZrotABD:vYrotABG + 1, :] = 50

    # du carpe
    x_bounds[4]["qdot"].min[vXrotC, :] = -50
    x_bounds[4]["qdot"].max[vXrotC, :] = 50
    # du dehanchement
    x_bounds[4]["qdot"].min[vYrotC, :] = -50
    x_bounds[4]["qdot"].max[vYrotC, :] = 50

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
    x0[Xrot, 0] = .50
    x0[ZrotBG] = -.75
    x0[ZrotBD] = .75
    x0[YrotBG, 0] = -2.9
    x0[YrotBD, 0] = 2.9
    x0[YrotBG, 1] = -1.35
    x0[YrotBD, 1] = 1.35
    x0[XrotC, 0] = -.5
    x0[XrotC, 1] = -2.6

    # rotater en salto (x) en carpé
    x1[ZrotBG] = -.75
    x1[ZrotBD] = .75
    x1[Xrot, 1] = 2 * 3.14
    x1[YrotBG] = -1.35
    x1[YrotBD] = 1.35
    x1[XrotC] = -2.6

    # ouverture des hanches
    x2[Xrot] = 2 * 3.14
    x2[Zrot, 1] = 0.2
    x2[ZrotBG, 0] = -.75
    x2[ZrotBD, 0] = .75
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
    x4[XrotC, 1] = -.5

    x_init = InitialGuessList()
    interpolation = InterpolationType.LINEAR
    t_init = [final_time / len(biorbd_model)] * len(biorbd_model)

    name = biorbd_model_path[-11:-7]
    if name in athlete_to_copy.keys():
        if save_folder == "Multistart_double_vrille":
            save_prename = "_double_vrille_et_demi_"
        elif save_folder == "Multistart_vrille_et_demi":
            save_prename = "_vrille_et_demi_"
        else:
            raise RuntimeError("Wrong type of OCP, see l.756.")
        save_name_to_copy = (save_folder + '/' + athlete_to_copy[name] + save_prename + str(seed) + '_' + "CVG.pkl")
        if os.path.isfile(save_name_to_copy):
            with open(save_name_to_copy, 'rb') as f:
                data = pickle.load(f)
                interpolation = InterpolationType.EACH_FRAME
                x0 = np.vstack((data["q"][0], data["qdot"][0]))
                x1 = np.vstack((data["q"][1], data["qdot"][1]))
                x2 = np.vstack((data["q"][2], data["qdot"][2]))
                x3 = np.vstack((data["q"][3], data["qdot"][3]))
                x4 = np.vstack((data["q"][4], data["qdot"][4]))

                u0 = data["tau"][0][:, :-1]
                u1 = data["tau"][1][:, :-1]
                u2 = data["tau"][2][:, :-1]
                u3 = data["tau"][3][:, :-1]
                u4 = data["tau"][4][:, :-1]

                t_init = [float(data["sol"].parameters["time"][i][0]) for i in range(5)]


    x_init.add("q", initial_guess=x0[:nb_q, :], interpolation=interpolation, phase=0)
    x_init.add("qdot", initial_guess=x0[nb_q:, :], interpolation=interpolation, phase=0)
    x_init.add("q", initial_guess=x1[:nb_q, :], interpolation=interpolation, phase=1)
    x_init.add("qdot", initial_guess=x1[nb_q:, :], interpolation=interpolation, phase=1)
    x_init.add("q", initial_guess=x2[:nb_q, :], interpolation=interpolation, phase=2)
    x_init.add("qdot", initial_guess=x2[nb_q:, :], interpolation=interpolation, phase=2)
    x_init.add("q", initial_guess=x3[:nb_q, :], interpolation=interpolation, phase=3)
    x_init.add("qdot", initial_guess=x3[nb_q:, :], interpolation=interpolation, phase=3)
    x_init.add("q", initial_guess=x4[:nb_q, :], interpolation=interpolation, phase=4)
    x_init.add("qdot", initial_guess=x4[nb_q:, :], interpolation=interpolation, phase=4)

    u_init.add("qddot_joints", initial_guess=u0, interpolation=InterpolationType.EACH_FRAME, phase=0)
    u_init.add("qddot_joints", initial_guess=u1, interpolation=InterpolationType.EACH_FRAME, phase=1)
    u_init.add("qddot_joints", initial_guess=u2, interpolation=InterpolationType.EACH_FRAME, phase=2)
    u_init.add("qddot_joints", initial_guess=u3, interpolation=InterpolationType.EACH_FRAME, phase=3)
    u_init.add("qddot_joints", initial_guess=u4, interpolation=InterpolationType.EACH_FRAME, phase=4)

    if interpolation == InterpolationType.LINEAR:
        for i in range(5):
            x_init[i]["q"].add_noise(
                bounds=x_bounds[i]["q"],
                n_shooting=np.array(n_shooting[i])+1,
                magnitude=0.2,
                magnitude_type=MagnitudeType.RELATIVE,
                seed=seed,
                )
            x_init[i]["qdot"].add_noise(
                bounds=x_bounds[i]["qdot"],
                n_shooting=np.array(n_shooting[i])+1,
                magnitude=0.2,
                magnitude_type=MagnitudeType.RELATIVE,
                seed=seed,
                )

            u_init[i]["qddot_joints"].add_noise(
                bounds=u_bounds[i]["qddot_joints"],
                magnitude=0.2,
                magnitude_type=MagnitudeType.RELATIVE,
                n_shooting=n_shooting[i],
                seed=seed,
            )

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.1, max_bound=.1,
                    first_marker='MidMainG', second_marker='CibleMainG', phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.1, max_bound=.1,
                    first_marker='MidMainD', second_marker='CibleMainD', phase=1)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=1.5, phase=1)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=0.7, phase=3)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=0.5, phase=4)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        t_init,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        n_threads=5,
    )


def construct_filepath(biorbd_model_path, nb_twist, seed):
    stunts = dict({3: "vrille_et_demi", 5: "double_vrille_et_demi", 7: "triple_vrille_et_demi"})
    stunt = stunts[nb_twist]
    athlete = biorbd_model_path.split('/')[-1].removesuffix('.bioMod')
    title_before_solve = f"{athlete}_{stunt}_{seed}"
    return title_before_solve


def save_results(sol: Solution, 
                 *combinatorial_parameters,
                 **extra_parameter):
    """
    Solving the ocp
    Parameters
    ----------
    sol: Solution
        The solution to the ocp at the current pool
    """

    biorbd_model_path, nb_twist, seed, _, _ = combinatorial_parameters
    save_folder = extra_parameter["save_folder"]

    title_before_solve = construct_filepath(biorbd_model_path, nb_twist, seed)

    convergence = sol.status
    dict_state = {}
    q = []
    qdot = []
    tau = []

    for i in range(len(sol.states)) :
        q.append(sol.states[i]['q'])
        qdot.append(sol.states[i]['qdot'])
        tau.append(sol.controls[i]['qddot_joints'])

    dict_state['q'] = q
    dict_state['qdot'] = qdot
    dict_state['tau'] = tau
    del sol.ocp
    dict_state['sol'] = sol

    if convergence == 0 :
        convergence = 'CVG'
        print(f'{biorbd_model_path}  doing' + f' {nb_twist}' + ' converge')
    else:
        convergence = 'DVG'
        print(f'{biorbd_model_path} doing ' + f'{nb_twist}' + ' doesn t converge')
        
    if save_folder:
        with open(f'{save_folder}/{title_before_solve}_{convergence}.pkl', "wb") as file:
            pickle.dump(dict_state, file)
    else:
        raise RuntimeError(f"This folder {save_folder} does not exist")


def should_solve(*combinatorial_parameters, **extra_parameters):
    """
    Check if the filename already appears in the folder where files are saved, if not ocp must be solved
    """
    biorbd_model_path, nb_twist, seed, _, _ = combinatorial_parameters
    save_folder = extra_parameters["save_folder"]
    file_path = construct_filepath(biorbd_model_path, nb_twist, seed)
    already_done_filenames = os.listdir(f"{save_folder}")
    
    if file_path not in already_done_filenames:
        return True
    else:
        return False


def prepare_multi_start(
    combinatorial_parameters: dict[tuple,...],
    save_folder: str = None,
    athlete_to_copy = None,
    n_pools: int = 6
) -> MultiStart:

    """
    The initialization of the multi-start
    """
    return MultiStart(
        combinatorial_parameters=combinatorial_parameters,
        prepare_ocp_callback=prepare_ocp,
        post_optimization_callback=(save_results, {'save_folder': save_folder}),
        should_solve_callback=(should_solve, {'save_folder': save_folder}),
        solver=Solver.IPOPT(show_online_optim=False),  # You cannot use show_online_optim with multi-start
        n_pools=n_pools,
    )

def main():
    """
    Prepares and solves an ocp for a 803<. Animates the results
    """

    seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    nb_twist = [3, 5]
    athletes = [
        "Athlete_03",
        "Athlete_05",
        "Athlete_18",
        "Athlete_07",
        "Athlete_14",
        "Athlete_17",
        "Athlete_02",
        "Athlete_06",
        "Athlete_11",
        "Athlete_13",
        "Athlete_16",
        "Athlete_12",
        "Athlete_04",
        "Athlete_10",
        "Athlete_08",
        "Athlete_09",
        "Athlete_01",
        "Athlete_15"
        ]

    athlete_to_copy = {"Athlete_18": "Athlete_14",
                       # "Athlete_12": "Athlete_08",
                       "Athlete_12": "Athlete_01"}

    all_paths = []
    for athlete in athletes :
        path = f'{athlete}'+'.bioMod'
        biorbd_model_path = "Models/Models_Lisa/" + f'{path}'
        all_paths.append(biorbd_model_path)

    save_folder = "Multistart_double_vrille"
    # save_folder = "Multistart_vrille_et_demi"

    combinatorial_parameters = {'bio_model_path': all_paths,
                                'nb_twist': nb_twist,
                                'seed': seed,
                                'athlete_to_copy': [athlete_to_copy],
                                'save_folder': [save_folder]}

    multi_start = prepare_multi_start(combinatorial_parameters=combinatorial_parameters, save_folder=save_folder, n_pools=6)

    multi_start.solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=False))
    multi_start.solver.set_linear_solver('ma57')
    multi_start.solver.set_maximum_iterations(3000)
    multi_start.solver.set_convergence_tolerance(1e-4)
    #multi_start.solver.set_print_level(0)

    multi_start.solve()


if __name__ == "__main__":
    main()

