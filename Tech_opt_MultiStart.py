"initiates TechOpt83 for all the athletes for different stunts"
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
            #QAndQDotBounds,
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
            MultiStart,
            Solution,
            MagnitudeType,
            NoisedInitialGuess,
            BiorbdModel,
        )
import time
import pickle

"""
The goal of this program is to optimize the movement to achieve a rudi out pike (803<).
"""


def check_already_done(self, args):
    """
    Check if the filename already appears in the folder where files are saved, if not ocp must be solved
    """
    already_done_filenames = os.listdir(f"/home/mickaelbegon/Documents/Stage_Lisa/AnthropoImpactOnTech/new_sol")
    for i, title in enumerate(already_done_filenames):
        title = title[0:-8]
        already_done_filenames[i] = title
    return self.post_optimization_callback([None], *args, only_save_filename=True) not in already_done_filenames

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


def minimize_dofs(all_pn: PenaltyNodeList, dofs: list, targets: list) -> MX:
    diff = 0
    for i, dof in enumerate(dofs):
        diff += (all_pn.nlp.states['q'].mx[dof] - targets[i]) ** 2
    return all_pn.nlp.mx_to_cx('minimize_dofs', diff, all_pn.nlp.states['q'])


def prepare_ocp(
        biorbd_model_path: str, nb_twist: int, seed : int,  n_threads: int = 5 ,
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
    #n_shooting = (1, 1, 1, 1, 1)

   # nom = biorbd_model_path[0].split('/')[-1].removesuffix('.bioMod')
    #print(nom)
    biorbd_model = (
    BiorbdModel(biorbd_model_path),BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path),
    BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

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
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=final_time, weight=.01, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=1.0, weight=100000,
                            phase=2)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=final_time, weight=.01, phase=3)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=final_time, weight=.01, phase=4)

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
    x_bounds.add(bounds=biorbd_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_model[0].bounds_from_ranges(["q", "qdot"]))

    # Pour la lisibilite
    DEBUT, MILIEU, FIN = 0, 1, 2

    #
    # Contraintes de position: PHASE 0 la montee en carpe
    #

    zmax = 8
    # 12 / 8 * final_time**2 + 1  # une petite marge

    # deplacement
    x_bounds[0].min[X, :] = -.1
    x_bounds[0].max[X, :] = .1
    x_bounds[0].min[Y, :] = -1.
    x_bounds[0].max[Y, :] = 1.
    x_bounds[0].min[:Z + 1, DEBUT] = 0
    x_bounds[0].max[:Z + 1, DEBUT] = 0
    x_bounds[0].min[Z, MILIEU:] = 0
    x_bounds[0].max[Z, MILIEU:] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[0].min[Xrot, :] = 0
    # 2 * 3.14 + 3 / 2 * 3.14 - .2
    x_bounds[0].max[Xrot, :] = -.50 + 3.14
    x_bounds[0].min[Xrot, DEBUT] = .50  # penche vers l'avant un peu carpe
    x_bounds[0].max[Xrot, DEBUT] = .50
    x_bounds[0].min[Xrot, MILIEU:] = 0
    x_bounds[0].max[Xrot, MILIEU:] = 4 * 3.14 + .1  # salto
    # limitation du tilt autour de y
    x_bounds[0].min[Yrot, DEBUT] = 0
    x_bounds[0].max[Yrot, DEBUT] = 0
    x_bounds[0].min[Yrot, MILIEU:] = - 3.14 / 16  # vraiment pas suppose tilte
    x_bounds[0].max[Yrot, MILIEU:] = 3.14 / 16
    # la vrille autour de z
    x_bounds[0].min[Zrot, DEBUT] = 0
    x_bounds[0].max[Zrot, DEBUT] = 0
    x_bounds[0].min[Zrot, MILIEU:] = -.1  # pas de vrille dans cette phase
    x_bounds[0].max[Zrot, MILIEU:] = .1

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
    x_bounds[0].min[ZrotABD:XrotABD + 1, DEBUT] = 0
    x_bounds[0].max[ZrotABD:XrotABD + 1, DEBUT] = 0
    # coude gauche
    x_bounds[0].min[ZrotABG:XrotABG + 1, DEBUT] = 0
    x_bounds[0].max[ZrotABG:XrotABG + 1, DEBUT] = 0

    # le carpe
    x_bounds[0].min[XrotC, DEBUT] = -.50  # depart un peu ferme aux hanches
    x_bounds[0].max[XrotC, DEBUT] = -.50
    x_bounds[0].max[XrotC, FIN] = -2.5
    # x_bounds[0].min[XrotC, FIN] = 2.7  # min du modele
    # le dehanchement
    x_bounds[0].min[YrotC, DEBUT] = 0
    x_bounds[0].max[YrotC, DEBUT] = 0
    x_bounds[0].min[YrotC, MILIEU:] = -.1
    x_bounds[0].max[YrotC, MILIEU:] = .1

    # Contraintes de vitesse: PHASE 0 la montee en carpe

    vzinit = 9.81 / (2 * final_time ) # vitesse initiale en z du CoM pour revenir a terre au temps final

    # decalage entre le bassin et le CoM
    CoM_Q_sym = MX.sym('CoM', nb_q)
    CoM_Q_init = x_bounds[0].min[:nb_q,
                 DEBUT]  # min ou max ne change rien a priori, au DEBUT ils sont egaux normalement
    CoM_Q_func = Function('CoM_Q_func', [CoM_Q_sym], [biorbd_model[0].center_of_mass(CoM_Q_sym)])
    bassin_Q_func = Function('bassin_Q_func', [CoM_Q_sym],
                             [biorbd_model[0].homogeneous_matrices_in_global(CoM_Q_sym,0).to_mx()])  # retourne la RT du bassin

    r = np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1,
                                                         :3]  # selectionne seulement la translation de la RT

    # en xy bassin
    x_bounds[0].min[vX:vY + 1, :] = -10
    x_bounds[0].max[vX:vY + 1, :] = 10
    x_bounds[0].min[vX:vY + 1, DEBUT] = -.5
    x_bounds[0].max[vX:vY + 1, DEBUT] = .5
    # z bassin
    x_bounds[0].min[vZ, :] = -50
    x_bounds[0].max[vZ, :] = 50
    x_bounds[0].min[vZ, DEBUT] = vzinit - .5
    x_bounds[0].max[vZ, DEBUT] = vzinit + .5

    # autour de x
    x_bounds[0].min[vXrot, :] = .5  # d'apres une observation video
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
    borne_inf = (x_bounds[0].min[vX:vZ + 1, DEBUT] + np.cross(r, x_bounds[0].min[vXrot:vZrot + 1, DEBUT]))[0]
    borne_sup = (x_bounds[0].max[vX:vZ + 1, DEBUT] + np.cross(r, x_bounds[0].max[vXrot:vZrot + 1, DEBUT]))[0]
    x_bounds[0].min[vX:vZ + 1, DEBUT] = min(borne_sup[0], borne_inf[0]), min(borne_sup[1], borne_inf[1]), min(
        borne_sup[2], borne_inf[2])
    x_bounds[0].max[vX:vZ + 1, DEBUT] = max(borne_sup[0], borne_inf[0]), max(borne_sup[1], borne_inf[1]), max(
        borne_sup[2], borne_inf[2])

    # bras droit
    x_bounds[0].min[vZrotBD:vYrotBD + 1, :] = -50
    x_bounds[0].max[vZrotBD:vYrotBD + 1, :] = 50
    x_bounds[0].min[vZrotBD:vYrotBD + 1, DEBUT] = 0
    x_bounds[0].max[vZrotBD:vYrotBD + 1, DEBUT] = 0
    # bras droit
    x_bounds[0].min[vZrotBG:vYrotBG + 1, :] = -50
    x_bounds[0].max[vZrotBG:vYrotBG + 1, :] = 50
    x_bounds[0].min[vZrotBG:vYrotBG + 1, DEBUT] = 0
    x_bounds[0].max[vZrotBG:vYrotBG + 1, DEBUT] = 0

    # coude droit
    x_bounds[0].min[vZrotABD:vYrotABD + 1, :] = -50
    x_bounds[0].max[vZrotABD:vYrotABD + 1, :] = 50
    x_bounds[0].min[vZrotABD:vYrotABD + 1, DEBUT] = 0
    x_bounds[0].max[vZrotABD:vYrotABD + 1, DEBUT] = 0
    # coude gauche
    x_bounds[0].min[vZrotABD:vYrotABG + 1, :] = -50
    x_bounds[0].max[vZrotABD:vYrotABG + 1, :] = 50
    x_bounds[0].min[vZrotABG:vYrotABG + 1, DEBUT] = 0
    x_bounds[0].max[vZrotABG:vYrotABG + 1, DEBUT] = 0

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
    x_bounds[1].min[X, :] = -.1
    x_bounds[1].max[X, :] = .1
    x_bounds[1].min[Y, :] = -1.
    x_bounds[1].max[Y, :] = 1.
    x_bounds[1].min[Z, :] = 0
    x_bounds[1].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[1].min[Xrot, :] = 0
    # 2 * 3.14 + 3 / 2 * 3.14 - .2
    x_bounds[1].max[Xrot, :] = -.50 + 4 * 3.14
    x_bounds[1].min[Xrot, FIN] = 2 * 3.14 - .1
    # limitation du tilt autour de y
    x_bounds[1].min[Yrot, :] = - 3.14 / 16
    x_bounds[1].max[Yrot, :] = 3.14 / 16
    # la vrille autour de z
    x_bounds[1].min[Zrot, :] = -.1
    x_bounds[1].max[Zrot, :] = .1

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
    x_bounds[1].min[YrotC, DEBUT] = -.1
    x_bounds[1].max[YrotC, DEBUT] = .1

    # Contraintes de vitesse: PHASE 1 le salto carpe

    # en xy bassin
    x_bounds[1].min[vX:vY + 1, :] = -10
    x_bounds[1].max[vX:vY + 1, :] = 10

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
    x_bounds[1].min[vZrotBD:vYrotBD + 1, :] = -50
    x_bounds[1].max[vZrotBD:vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[1].min[vZrotBG:vYrotBG + 1, :] = -50
    x_bounds[1].max[vZrotBG:vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[1].min[vZrotABD:vYrotABD + 1, :] = -50
    x_bounds[1].max[vZrotABD:vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[1].min[vZrotABD:vYrotABG + 1, :] = -50
    x_bounds[1].max[vZrotABD:vYrotABG + 1, :] = 50

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
    x_bounds[2].min[X, :] = -.2
    x_bounds[2].max[X, :] = .2
    x_bounds[2].min[Y, :] = -1.
    x_bounds[2].max[Y, :] = 1.
    x_bounds[2].min[Z, :] = 0
    x_bounds[2].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[2].min[Xrot, :] = 2 * 3.14 - .1
    # 2 * 3.14 + 3 / 2 * 3.14 - .2 #2 * 3.14 - .1       # 2 * 3.14 + 3 / 2 * 3.14 - .2  # 1 salto 3/4
    x_bounds[2].max[Xrot, :] = -.50 + 4 * 3.14
    # limitation du tilt autour de y
    x_bounds[2].min[Yrot, :] = - 3.14 / 4
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
    x_bounds[2].min[XrotC, FIN] = -.4
    # le dehanchement
    # x_bounds[2].min[YrotC, DEBUT] = -.05
    # x_bounds[2].max[YrotC, DEBUT] = .05
    # x_bounds[2].min[YrotC, MILIEU:] = -.05  # f4a a l'ouverture
    # x_bounds[2].max[YrotC, MILIEU:] = .05

    # Contraintes de vitesse: PHASE 2 l'ouverture

    # en xy bassin
    x_bounds[2].min[vX:vY + 1, :] = -10
    x_bounds[2].max[vX:vY + 1, :] = 10

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
    x_bounds[2].min[vZrotBD:vYrotBD + 1, :] = -50
    x_bounds[2].max[vZrotBD:vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[2].min[vZrotBG:vYrotBG + 1, :] = -50
    x_bounds[2].max[vZrotBG:vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[2].min[vZrotABD:vYrotABD + 1, :] = -50
    x_bounds[2].max[vZrotABD:vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[2].min[vZrotABD:vYrotABG + 1, :] = -50
    x_bounds[2].max[vZrotABD:vYrotABG + 1, :] = 50

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
    x_bounds[3].min[X, :] = -.2
    x_bounds[3].max[X, :] = .2
    x_bounds[3].min[Y, :] = -1.
    x_bounds[3].max[Y, :] = 1.
    x_bounds[3].min[Z, :] = 0
    x_bounds[3].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[3].min[Xrot, :] = 0
    x_bounds[3].min[Xrot, :] = 2 * 3.14 - .1

    x_bounds[3].max[Xrot, :] = 2 * 3.14 + 3 / 2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[3].min[Xrot, FIN] = 2 * 3.14 + 3 / 2 * 3.14 - .1
    x_bounds[3].max[Xrot, FIN] = 2 * 3.14 + 3 / 2 * 3.14 + .1  # 1 salto 3/4
    # x_bounds[3].max[Xrot, :] = -.50 + 4 * 3.14  # 1 salto 3/4
    # x_bounds[3].min[Xrot, FIN] = 0
    # 2 * 3.14 + 2 * 3.14 - .1
    # x_bounds[3].max[Xrot, FIN] = 2 * 3.14 + 2 * 3.14 + .1  # 1 salto 3/4
    # limitation du tilt autour de y
    x_bounds[3].min[Yrot, :] = - 3.14 / 4
    x_bounds[3].max[Yrot, :] = 3.14 / 4
    x_bounds[3].min[Yrot, FIN] = - 3.14 / 8
    x_bounds[3].max[Yrot, FIN] = 3.14 / 8
    # la vrille autour de z
    x_bounds[3].min[Zrot, :] = 0
    x_bounds[3].max[Zrot, :] = 5 * 3.14
    x_bounds[3].min[Zrot, FIN] = nb_twist * 3.14 - .1  # complete la vrille
    x_bounds[3].max[Zrot, FIN] = nb_twist * 3.14 + .1

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
    x_bounds[3].min[XrotC, :] = -.4
    # le dehanchement
    # x_bounds[3].min[YrotC, DEBUT] = -.05
    # x_bounds[3].max[YrotC, DEBUT] = .05
    # x_bounds[3].min[YrotC, MILIEU:] = -.05  # f4a a l'ouverture
    # x_bounds[3].max[YrotC, MILIEU:] = .05

    # Contraintes de vitesse: PHASE 3 la vrille et demie

    # en xy bassin
    x_bounds[3].min[vX:vY + 1, :] = -10
    x_bounds[3].max[vX:vY + 1, :] = 10

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
    x_bounds[3].min[vZrotBD:vYrotBD + 1, :] = -50
    x_bounds[3].max[vZrotBD:vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[3].min[vZrotBG:vYrotBG + 1, :] = -50
    x_bounds[3].max[vZrotBG:vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[3].min[vZrotABD:vYrotABD + 1, :] = -50
    x_bounds[3].max[vZrotABD:vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[3].min[vZrotABD:vYrotABG + 1, :] = -50
    x_bounds[3].max[vZrotABD:vYrotABG + 1, :] = 50

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
    x_bounds[4].min[X, :] = -.1
    x_bounds[4].max[X, :] = .1
    x_bounds[4].min[Y, FIN] = -.1
    x_bounds[4].max[Y, FIN] = .1
    x_bounds[4].min[Z, :] = 0
    x_bounds[4].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    x_bounds[4].min[Z, FIN] = 0
    x_bounds[4].max[Z, FIN] = .1

    # le salto autour de x
    x_bounds[4].min[Xrot, :] = 2 * 3.14 + 3 / 2 * 3.14 - .2  # penche vers avant -> moins de salto
    x_bounds[4].max[Xrot, :] = -.50 + 4 * 3.14  # un peu carpe a la fin
    x_bounds[4].min[Xrot, FIN] = -.50 + 4 * 3.14 - .1  # 2 salto fin un peu carpe
    x_bounds[4].max[Xrot, FIN] = -.50 + 4 * 3.14 + .1  # 2 salto fin un peu carpe
    # limitation du tilt autour de y
    x_bounds[4].min[Yrot, :] = - 3.14 / 16
    x_bounds[4].max[Yrot, :] = 3.14 / 16
    # la vrille autour de z
    x_bounds[4].min[Zrot, :] = nb_twist * 3.14 - .1  # complete la vrille
    x_bounds[4].max[Zrot, :] = nb_twist * 3.14 + .1

    # bras droit
    x_bounds[4].min[YrotBD, FIN] = 2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[YrotBD, FIN] = 2.9 + .1
    x_bounds[4].min[ZrotBD, FIN] = -.1
    x_bounds[4].max[ZrotBD, FIN] = .1
    # bras gauche
    x_bounds[4].min[YrotBG, FIN] = -2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[YrotBG, FIN] = -2.9 + .1
    x_bounds[4].min[ZrotBG, FIN] = -.1
    x_bounds[4].max[ZrotBG, FIN] = .1

    # coude droit
    x_bounds[4].min[ZrotABD:XrotABD + 1, FIN] = -.1
    x_bounds[4].max[ZrotABD:XrotABD + 1, FIN] = .1
    # coude gauche
    x_bounds[4].min[ZrotABG:XrotABG + 1, FIN] = -.1
    x_bounds[4].max[ZrotABG:XrotABG + 1, FIN] = .1

    # le carpe
    x_bounds[4].min[XrotC, :] = -.4
    x_bounds[4].min[XrotC, FIN] = -.60
    x_bounds[4].max[XrotC, FIN] = -.40  # fin un peu carpe
    # le dehanchement
    x_bounds[4].min[YrotC, FIN] = -.1
    x_bounds[4].max[YrotC, FIN] = .1

    # Contraintes de vitesse: PHASE 4 la reception

    # en xy bassin
    x_bounds[4].min[vX:vY + 1, :] = -10
    x_bounds[4].max[vX:vY + 1, :] = 10

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
    x_bounds[4].min[vZrotBD:vYrotBD + 1, :] = -50
    x_bounds[4].max[vZrotBD:vYrotBD + 1, :] = 50
    # bras droit
    x_bounds[4].min[vZrotBG:vYrotBG + 1, :] = -50
    x_bounds[4].max[vZrotBG:vYrotBG + 1, :] = 50

    # coude droit
    x_bounds[4].min[vZrotABD:vYrotABD + 1, :] = -50
    x_bounds[4].max[vZrotABD:vYrotABD + 1, :] = 50
    # coude gauche
    x_bounds[4].min[vZrotABD:vYrotABG + 1, :] = -50
    x_bounds[4].max[vZrotABD:vYrotABG + 1, :] = 50

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
  #  for i in range(nb_phases):
   #    x_init.add(NoisedInitialGuess(
 #   self[i],
  #  interpolation = self[i].type,
  #  bounds = bounds[i],
 #   n_shooting = n_shooting[i],
  #  bound_push = bound_push[i],
  #  seed = seed[i],
   # magnitude = magnitude[i],
   # magnitude_type = magnitude_type,
    #))
    x_init.add(x0, interpolation=InterpolationType.LINEAR)
    x_init.add(x1, interpolation=InterpolationType.LINEAR)
    x_init.add(x2, interpolation=InterpolationType.LINEAR)
    x_init.add(x3, interpolation=InterpolationType.LINEAR)
    x_init.add(x4, interpolation=InterpolationType.LINEAR)
        #x_init = InitialGuess([i] * (nb_q + nb_qdot), interpolation=InterpolationType.LINEAR)

    x_init.add_noise(
        bounds=x_bounds,
        n_shooting=np.array(n_shooting)+1,
        magnitude=0.2,
        magnitude_type=MagnitudeType.RELATIVE,
        seed=seed,
        )
#
    constraints = ConstraintList()
    # on verra si remet cette contrainte plus stricte (-0.05, 0.05)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.1, max_bound=.1,
                    first_marker='MidMainG', second_marker='CibleMainG', phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.1, max_bound=.1,
                    first_marker='MidMainD', second_marker='CibleMainD', phase=1)
    #    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0, max_bound=final_time, phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=1.5, phase=1)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=1.5, phase=2)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=0.7, phase=3)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=0.5, phase=4)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        [final_time / len(biorbd_model)] * len(biorbd_model),
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        n_threads=n_threads
    )




def save_results(sol: Solution, biorbd_model_path: str,  nb_twist : int , seed: int,save_folder:str=None, only_save_filename : bool = False):
    """
    Solving the ocp
    Parameters
    ----------
    sol: Solution
        The solution to the ocp at the current pool
    biorbd_model_path: str
        The path to the biorbd model
    seed: int
        The seed to use for the random initial guess
    only_save_filename : bool
        True if you only want to return the name of the file that would be saved
    """

    stunts = dict({3: "vrille_et_demi", 5: "double_vrille_et_demi", 7: "triple_vrille_et_demi"})
   # print('do somethin with biorbd_model_path to ge the name')
    # OptimalControlProgram.save(sol, f"solutions/pendulum_multi_start_random{seed}.bo", stand_alone=True)
    #states = sol.states["all"]
    stunt = stunts[nb_twist]
    athlete=biorbd_model_path.split('/')[-1].removesuffix('.bioMod')
    path_folder = '/home/mickaelbegon/Documents/Stage_Lisa/Anthropo Lisa/new_sol_with_updated_models'
    title_before_solve = f"{athlete}_{stunt}_{seed}"

    if only_save_filename == True :
        return title_before_solve

    convergence = sol.status
    dict_state = {}
    q = []
    qdot = []
    tau = []
  #  sol = []

    for i in range(len(sol.states)) :
        q.append( sol.states[i]['q'])
        qdot.append(sol.states[i]['qdot'])
        tau.append(sol.controls[i]['qddot_joints'])

    dict_state['q'] = q
    dict_state['qdot'] = qdot
    dict_state['tau'] = tau
    del sol.ocp
    dict_state['sol'] = sol

    if convergence == 0 :
        convergence = 'CVG'
        print(f'{athlete}  doing' + f' {stunt}' + ' converge')
    else:
        convergence = 'DVG'
        print( f'{athlete} doing ' + f'{stunt}' + ' doesn t converge')
    if save_folder :
        with open(f'{path_folder}/{title_before_solve}_{convergence}.pkl', "wb") as file:
            pickle.dump(dict_state, file)

def check_already_done(args,save_folder, save_results= save_results):
    """
    Check if the filename already appears in the folder where files are saved, if not ocp must be solved
    """
    already_done_filenames = os.listdir(f"/home/laseche/Documents/Stage_Lisa/AnthropoImpactOnTech/Solutions_multistart/")
    for i, title in enumerate(already_done_filenames):
        title = title[0:-8]
        already_done_filenames[i] = title
    return save_results([None], *args,save_folder, only_save_filename=True) not in already_done_filenames


def prepare_multi_start(
    combinatorial_parameters: dict[tuple,...],
    save_folder: str = None,
    n_pools: int = 1
) -> MultiStart:

    """
    The initialization of the multi-start
    """
    return MultiStart(
        combinatorial_parameters=combinatorial_parameters,
        prepare_ocp_callback=prepare_ocp,
        post_optimization_callback=(save_results,{'save_folder': save_folder}),
        should_solve_callback=(check_already_done, {'save_folder':save_folder}),
        solver=Solver.IPOPT(show_online_optim=False),  # You cannot use show_online_optim with multi-start
        n_pools=n_pools,
        # save_folder= save_folder,
    )

def main():
    """
    Prepares and solves an ocp for a 803<. Animates the results
    """


    #biorbd_model_path = "/home/mickaelbegon/Documents/Stage_Lisa/AnthropoImpactOnTech/models/Models/Sarah.bioMod"
    #Mod = Model(savesol= True, with_hsl=True)

    n_threads = 25

    seed = [0,1,2,3,4,5,6,7,8,9]
    nb_twist = [3]
    athletes = ["AdCh", "AlAd", "AuJo", "Benjamin", "ElMe", "EvZl", "FeBl", "JeCh", "KaFu", "KaMi", "LaDe", "MaCu", "MaJa",
                "MeVa", "OlGa", "Sarah", "SoMe", "WeEm", "ZoTs"]

    all_paths = []
    for athlete in athletes :
        path = f'{athlete}'+'.bioMod'
        biorbd_model_path = "/home/mickaelbegon/Documents/Stage_Lisa/AnthropoImpactOnTech/Models/Models_Lisa/" + f'{path}'
        all_paths.append(biorbd_model_path)


    #path = "/home/mickaelbegon/Documents/Stage_Lisa/AnthropoImpactOnTech/Models/"
    combinatorial_parameters = {'bio_model_path': biorbd_model_path,'nb_twist':nb_twist,
                                'seed': seed}
    save_folder = "./temporary_results"

    multi_start = prepare_multi_start(combinatorial_parameters=combinatorial_parameters, save_folder=save_folder)

    # multi_start = prepare_multi_start(biorbd_model_path=all_paths, nb_twist=nb_twist, seed=seed, should_solve=check_already_done, use_multi_process=True)


    multi_start.solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=False))
        #if Mod.with_hsl:
    multi_start.solver.set_linear_solver('ma57')
    #else:
    #    print("Not using ma57")
    multi_start.solver.set_maximum_iterations(10000)
    multi_start.solver.set_convergence_tolerance(1e-4)
    #multi_start.solver.set_print_level(0)

    multi_start.solve()

    #solve
    #ocp.solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    # if Mod.with_hsl:
    #ocp.solver.set_linear_solver('ma57')
    # else:
    #    print("Not using ma57")

    #ocp.solver.set_maximum_iterations(10000)
    #ocp.solver.set_convergence_tolerance(1e-4)
    #ocp.solve()
    #ocp.solver.set_print_level()

    #multi_start.solve()

    temps = time.strftime("%Y-%m-%d-%H%M")
    #    if Mod.savesol:  # switch manuelle
    #        np.save(f"{folder}/{athlete}/{athlete}-{str(n_shooting).replace(', ', '_')}-{temps}-q.npy", qs)
    #        np.save(f"{folder}/{athlete}/{athlete}-{str(n_shooting).replace(', ', '_')}-{temps}-qdot.npy", qdots)
     #       np.save(f"{folder}/{athlete}/{athlete}-{str(n_shooting).replace(', ', '_')}-{temps}-t.npy", sol.phase_time)
     #   sol.graphs(show_bounds=True, show_now=False, save_path=None)
     #   print(f'{athlete}')
        #sol.graphs(show_bounds=True, show_now=False, save_path=f'{folder}/{athlete}')



#file = open('/home/mickaelbegon/Documents/Stage_Lisa/AnthropoImpactOnTech/Sol/Sarah_vrille_et_demi_0_1.pkl', 'rb')
#data = pickle.load(file)
#
# with open('/home/mickaelbegon/Documents/Stage_Lisa/AnthropoImpactOnTech/Sol/Sarah_vrille_et_demi_0_1.pkl', 'rb') as f:
#     data = pickle.load(f)


if __name__ == "__main__":
    main()

