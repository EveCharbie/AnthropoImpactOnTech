"""
The goal of this program is to optimize the movement to achieve a rudi out pike (803<).
Simultaneously for two anthorpometric models.
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


def set_fancy_names_index():
    """
    For readability
    """
    fancy_names_index = {}
    fancy_names_index["X_AuJo"] = 0
    fancy_names_index["Y_AuJo"] = 1
    fancy_names_index["Z_AuJo"] = 2
    fancy_names_index["Xrot_AuJo"] = 3
    fancy_names_index["Yrot_AuJo"] = 4
    fancy_names_index["Zrot_AuJo"] = 5
    fancy_names_index["ZrotBD_AuJo"] = 6
    fancy_names_index["YrotBD_AuJo"] = 7
    fancy_names_index["ZrotABD_AuJo"] = 8
    fancy_names_index["XrotABD_AuJo"] = 9
    fancy_names_index["ZrotBG_AuJo"] = 10
    fancy_names_index["YrotBG_AuJo"] = 11
    fancy_names_index["ZrotABG_AuJo"] = 12
    fancy_names_index["XrotABG_AuJo"] = 13
    fancy_names_index["XrotC_AuJo"] = 14
    fancy_names_index["YrotC_AuJo"] = 15
    fancy_names_index["X_JeCh"] = 16
    fancy_names_index["Y_JeCh"] = 17
    fancy_names_index["Z_JeCh"] = 18
    fancy_names_index["Xrot_JeCh"] = 19
    fancy_names_index["Yrot_JeCh"] = 20
    fancy_names_index["Zrot_JeCh"] = 21
    fancy_names_index["ZrotBD_JeCh"] = 22
    fancy_names_index["YrotBD_JeCh"] = 23
    fancy_names_index["ZrotABD_JeCh"] = 24
    fancy_names_index["XrotABD_JeCh"] = 25
    fancy_names_index["ZrotBG_JeCh"] = 26
    fancy_names_index["YrotBG_JeCh"] = 27
    fancy_names_index["ZrotABG_JeCh"] = 28
    fancy_names_index["XrotABG_JeCh"] = 29
    fancy_names_index["XrotC_JeCh"] = 30
    fancy_names_index["YrotC_JeCh"] = 31
    fancy_names_index["vX_AuJo"] = 0 + nb_q
    fancy_names_index["vY_AuJo"] = 1 + nb_q
    fancy_names_index["vZ_AuJo"] = 2 + nb_q
    fancy_names_index["vXrot_AuJo"] = 3 + nb_q
    fancy_names_index["vYrot_AuJo"] = 4 + nb_q
    fancy_names_index["vZrot_AuJo"] = 5 + nb_q
    fancy_names_index["vZrotBD_AuJo"] = 6 + nb_q
    fancy_names_index["vYrotBD_AuJo"] = 7 + nb_q
    fancy_names_index["vZrotABD_AuJo"] = 8 + nb_q
    fancy_names_index["vYrotABD_AuJo"] = 9 + nb_q
    fancy_names_index["vZrotBG_AuJo"] = 10 + nb_q
    fancy_names_index["vYrotBG_AuJo"] = 11 + nb_q
    fancy_names_index["vZrotABG_AuJo"] = 12 + nb_q
    fancy_names_index["vYrotABG_AuJo"] = 13 + nb_q
    fancy_names_index["vXrotC_AuJo"] = 14 + nb_q
    fancy_names_index["vYrotC_AuJo"] = 15 + nb_q
    fancy_names_index["vX_JeCh"] = 16 + nb_q
    fancy_names_index["vY_JeCh"] = 17 + nb_q
    fancy_names_index["vZ_JeCh"] = 18 + nb_q
    fancy_names_index["vXrot_JeCh"] = 19 + nb_q
    fancy_names_index["vYrot_JeCh"] = 20 + nb_q
    fancy_names_index["vZrot_JeCh"] = 21 + nb_q
    fancy_names_index["vZrotBD_JeCh"] = 22 + nb_q
    fancy_names_index["vYrotBD_JeCh"] = 23 + nb_q
    fancy_names_index["vZrotABD_JeCh"] = 24 + nb_q
    fancy_names_index["vYrotABD_JeCh"] = 25 + nb_q
    fancy_names_index["vZrotBG_JeCh"] = 26 + nb_q
    fancy_names_index["vYrotBG_JeCh"] = 27 + nb_q
    fancy_names_index["vZrotABG_JeCh"] = 28 + nb_q
    fancy_names_index["vYrotABG_JeCh"] = 29 + nb_q
    fancy_names_index["vXrotC_JeCh"] = 30 + nb_q
    fancy_names_index["vYrotC_JeCh"] = 31 + nb_q

    return fancy_names_index

def set_x_bounds(biorbd_model, fancy_names_index):

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
    x_bounds[0].min[fancy_names_index["X_AuJo"], :] = -.1
    x_bounds[0].max[fancy_names_index["X_AuJo"], :] = .1
    x_bounds[0].min[fancy_names_index["Y_AuJo"], :] = -1.
    x_bounds[0].max[fancy_names_index["Y_AuJo"], :] = 1.
    x_bounds[0].min[:fancy_names_index["Z_AuJo"]+1, DEBUT] = 0
    x_bounds[0].max[:fancy_names_index["Z_AuJo"]+1, DEBUT] = 0
    x_bounds[0].min[fancy_names_index["Z_AuJo"], MILIEU:] = 0
    x_bounds[0].max[fancy_names_index["Z_AuJo"], MILIEU:] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    x_bounds[0].min[fancy_names_index["X_JeCh"], :] = -.1
    x_bounds[0].max[fancy_names_index["X_JeCh"], :] = .1
    x_bounds[0].min[fancy_names_index["Y_JeCh"], :] = -1.
    x_bounds[0].max[fancy_names_index["Y_JeCh"], :] = 1.
    x_bounds[0].min[:fancy_names_index["Z_JeCh"] + 1, DEBUT] = 0
    x_bounds[0].max[:fancy_names_index["Z_JeCh"] + 1, DEBUT] = 0
    x_bounds[0].min[fancy_names_index["Z_JeCh"], MILIEU:] = 0
    x_bounds[0].max[fancy_names_index["Z_JeCh"], MILIEU:] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[0].min[fancy_names_index["Xrot_AuJo"], DEBUT] = .50  # penche vers l'avant un peu carpe
    x_bounds[0].max[fancy_names_index["Xrot_AuJo"], DEBUT] = .50
    x_bounds[0].min[fancy_names_index["Xrot_AuJo"], MILIEU:] = 0
    x_bounds[0].max[fancy_names_index["Xrot_AuJo"], MILIEU:] = 4 * 3.14 + .1  # salto

    x_bounds[0].min[fancy_names_index["Xrot_JeCh"], DEBUT] = .50  # penche vers l'avant un peu carpe
    x_bounds[0].max[fancy_names_index["Xrot_JeCh"], DEBUT] = .50
    x_bounds[0].min[fancy_names_index["Xrot_JeCh"], MILIEU:] = 0
    x_bounds[0].max[fancy_names_index["Xrot_JeCh"], MILIEU:] = 4 * 3.14 + .1  # salto

    # limitation du tilt autour de y
    x_bounds[0].min[fancy_names_index["Yrot_AuJo"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["Yrot_AuJo"], DEBUT] = 0
    x_bounds[0].min[fancy_names_index["Yrot_AuJo"], MILIEU:] = - 3.14 / 16  # vraiment pas suppose tilte
    x_bounds[0].max[fancy_names_index["Yrot_AuJo"], MILIEU:] = 3.14 / 16

    x_bounds[0].min[fancy_names_index["Yrot_JeCh"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["Yrot_JeCh"], DEBUT] = 0
    x_bounds[0].min[fancy_names_index["Yrot_JeCh"], MILIEU:] = - 3.14 / 16  # vraiment pas suppose tilte
    x_bounds[0].max[fancy_names_index["Yrot_JeCh"], MILIEU:] = 3.14 / 16

    # la vrille autour de z
    x_bounds[0].min[fancy_names_index["Zrot_AuJo"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["Zrot_AuJo"], DEBUT] = 0
    x_bounds[0].min[fancy_names_index["Zrot_AuJo"], MILIEU:] = -.1  # pas de vrille dans cette phase
    x_bounds[0].max[fancy_names_index["Zrot_AuJo"], MILIEU:] = .1

    x_bounds[0].min[fancy_names_index["Zrot_JeCh"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["Zrot_JeCh"], DEBUT] = 0
    x_bounds[0].min[fancy_names_index["Zrot_JeCh"], MILIEU:] = -.1  # pas de vrille dans cette phase
    x_bounds[0].max[fancy_names_index["Zrot_JeCh"], MILIEU:] = .1

    # bras droit
    x_bounds[0].min[fancy_names_index["YrotBD_AuJo"], DEBUT] = 2.9  # debut bras aux oreilles
    x_bounds[0].max[fancy_names_index["YrotBD_AuJo"], DEBUT] = 2.9
    x_bounds[0].min[fancy_names_index["ZrotBD_AuJo"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotBD_AuJo"], DEBUT] = 0

    x_bounds[0].min[fancy_names_index["YrotBD_JeCh"], DEBUT] = 2.9  # debut bras aux oreilles
    x_bounds[0].max[fancy_names_index["YrotBD_JeCh"], DEBUT] = 2.9
    x_bounds[0].min[fancy_names_index["ZrotBD_JeCh"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotBD_JeCh"], DEBUT] = 0

    # bras gauche
    x_bounds[0].min[fancy_names_index["YrotBG_AuJo"], DEBUT] = -2.9  # debut bras aux oreilles
    x_bounds[0].max[fancy_names_index["YrotBG_AuJo"], DEBUT] = -2.9
    x_bounds[0].min[fancy_names_index["ZrotBG_AuJo"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotBG_AuJo"], DEBUT] = 0

    x_bounds[0].min[fancy_names_index["YrotBG_JeCh"], DEBUT] = -2.9  # debut bras aux oreilles
    x_bounds[0].max[fancy_names_index["YrotBG_JeCh"], DEBUT] = -2.9
    x_bounds[0].min[fancy_names_index["ZrotBG_JeCh"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotBG_JeCh"], DEBUT] = 0

    # coude droit
    x_bounds[0].min[fancy_names_index["ZrotABD_AuJo"]:fancy_names_index["XrotABD_AuJo"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotABD_AuJo"]:fancy_names_index["XrotABD_AuJo"]+1, DEBUT] = 0

    x_bounds[0].min[fancy_names_index["ZrotABD_JeCh"]:fancy_names_index["XrotABD_JeCh"] + 1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotABD_JeCh"]:fancy_names_index["XrotABD_JeCh"] + 1, DEBUT] = 0

    # coude gauche
    x_bounds[0].min[fancy_names_index["ZrotABG_AuJo"]:fancy_names_index["XrotABG_AuJo"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotABG_AuJo"]:fancy_names_index["XrotABG_AuJo"]+1, DEBUT] = 0

    x_bounds[0].min[fancy_names_index["ZrotABG_JeCh"]:fancy_names_index["XrotABG_JeCh"] + 1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["ZrotABG_JeCh"]:fancy_names_index["XrotABG_JeCh"] + 1, DEBUT] = 0

    # le carpe
    x_bounds[0].min[fancy_names_index["XrotC_AuJo"], DEBUT] = -.50  # depart un peu ferme aux hanches
    x_bounds[0].max[fancy_names_index["XrotC_AuJo"], DEBUT] = -.50
    x_bounds[0].min[fancy_names_index["XrotC_AuJo"], FIN] = -2.35
    x_bounds[0].max[fancy_names_index["XrotC_AuJo"], FIN] = -2.35

    x_bounds[0].min[fancy_names_index["XrotC_JeCh"], DEBUT] = -.50  # depart un peu ferme aux hanches
    x_bounds[0].max[fancy_names_index["XrotC_JeCh"], DEBUT] = -.50
    x_bounds[0].min[fancy_names_index["XrotC_JeCh"], FIN] = -2.35
    x_bounds[0].max[fancy_names_index["XrotC_JeCh"], FIN] = -2.35

    # le dehanchement
    x_bounds[0].min[fancy_names_index["YrotC_AuJo"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["YrotC_AuJo"], DEBUT] = 0
    x_bounds[0].min[fancy_names_index["YrotC_AuJo"], MILIEU:] = -.1
    x_bounds[0].max[fancy_names_index["YrotC_AuJo"], MILIEU:] = .1

    x_bounds[0].min[fancy_names_index["YrotC_JeCh"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["YrotC_JeCh"], DEBUT] = 0
    x_bounds[0].min[fancy_names_index["YrotC_JeCh"], MILIEU:] = -.1
    x_bounds[0].max[fancy_names_index["YrotC_JeCh"], MILIEU:] = .1

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
    x_bounds[0].min[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"]+1, :] = -10
    x_bounds[0].max[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"]+1, :] = 10
    x_bounds[0].min[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"]+1, DEBUT] = -.5
    x_bounds[0].max[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"]+1, DEBUT] = .5

    x_bounds[0].min[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, :] = -10
    x_bounds[0].max[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, :] = 10
    x_bounds[0].min[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, DEBUT] = -.5
    x_bounds[0].max[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, DEBUT] = .5

    # z bassin
    x_bounds[0].min[fancy_names_index["vZ_AuJo"], :] = -100
    x_bounds[0].max[fancy_names_index["vZ_AuJo"], :] = 100
    x_bounds[0].min[fancy_names_index["vZ_AuJo"], DEBUT] = vzinit - .5
    x_bounds[0].max[fancy_names_index["vZ_AuJo"], DEBUT] = vzinit + .5

    x_bounds[0].min[fancy_names_index["vZ_JeCh"], :] = -100
    x_bounds[0].max[fancy_names_index["vZ_JeCh"], :] = 100
    x_bounds[0].min[fancy_names_index["vZ_JeCh"], DEBUT] = vzinit - .5
    x_bounds[0].max[fancy_names_index["vZ_JeCh"], DEBUT] = vzinit + .5

    # autour de x
    x_bounds[0].min[fancy_names_index["vXrot_AuJo"], :] = .5  # d'apres une observation video
    x_bounds[0].max[fancy_names_index["vXrot_AuJo"], :] = 20  # aussi vite que nécessaire, mais ne devrait pas atteindre cette vitesse

    x_bounds[0].min[fancy_names_index["vXrot_JeCh"], :] = .5  # d'apres une observation video
    x_bounds[0].max[fancy_names_index["vXrot_JeCh"], :] = 20  # aussi vite que nécessaire, mais ne devrait pas atteindre cette vitesse

    # autour de y
    x_bounds[0].min[fancy_names_index["vYrot_AuJo"], :] = -100
    x_bounds[0].max[fancy_names_index["vYrot_AuJo"], :] = 100
    x_bounds[0].min[fancy_names_index["vYrot_AuJo"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vYrot_AuJo"], DEBUT] = 0

    x_bounds[0].min[fancy_names_index["vYrot_JeCh"], :] = -100
    x_bounds[0].max[fancy_names_index["vYrot_JeCh"], :] = 100
    x_bounds[0].min[fancy_names_index["vYrot_JeCh"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vYrot_JeCh"], DEBUT] = 0

    # autour de z
    x_bounds[0].min[fancy_names_index["vZrot_AuJo"], :] = -100
    x_bounds[0].max[fancy_names_index["vZrot_AuJo"], :] = 100
    x_bounds[0].min[fancy_names_index["vZrot_AuJo"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrot_AuJo"], DEBUT] = 0

    x_bounds[0].min[fancy_names_index["vZrot_JeCh"], :] = -100
    x_bounds[0].max[fancy_names_index["vZrot_JeCh"], :] = 100
    x_bounds[0].min[fancy_names_index["vZrot_JeCh"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrot_JeCh"], DEBUT] = 0

    # tenir compte du decalage entre bassin et CoM avec la rotation
    # Qtransdot = Qtransdot + v cross Qrotdot
    borne_inf = ( x_bounds[0].min[fancy_names_index["vX_AuJo"]:fancy_names_index["vZ_AuJo"]+1, DEBUT] + np.cross(r, x_bounds[0].min[fancy_names_index["vXrot_AuJo"]:fancy_names_index["vZrot_AuJo"]+1, DEBUT]) )[0]
    borne_sup = ( x_bounds[0].max[fancy_names_index["vX_AuJo"]:fancy_names_index["vZ_AuJo"]+1, DEBUT] + np.cross(r, x_bounds[0].max[fancy_names_index["vXrot_AuJo"]:fancy_names_index["vZrot_AuJo"]+1, DEBUT]) )[0]
    x_bounds[0].min[fancy_names_index["vX_AuJo"]:fancy_names_index["vZ_AuJo"]+1, DEBUT] = min(borne_sup[0], borne_inf[0]), min(borne_sup[1], borne_inf[1]), min(borne_sup[2], borne_inf[2])
    x_bounds[0].max[fancy_names_index["vX_AuJo"]:fancy_names_index["vZ_AuJo"]+1, DEBUT] = max(borne_sup[0], borne_inf[0]), max(borne_sup[1], borne_inf[1]), max(borne_sup[2], borne_inf[2])

    borne_inf = (x_bounds[0].min[fancy_names_index["vX_JeCh"]:fancy_names_index["vZ_JeCh"] + 1, DEBUT] + np.cross(r, x_bounds[0].min[fancy_names_index["vXrot_JeCh"]:fancy_names_index["vZrot_JeCh"] + 1, DEBUT]))[0]
    borne_sup = (x_bounds[0].max[fancy_names_index["vX_JeCh"]:fancy_names_index["vZ_JeCh"] + 1, DEBUT] + np.cross(r, x_bounds[0].max[fancy_names_index["vXrot_JeCh"]:fancy_names_index["vZrot_JeCh"] + 1, DEBUT]))[0]
    x_bounds[0].min[fancy_names_index["vX_JeCh"]:fancy_names_index["vZ_JeCh"] + 1, DEBUT] = min(borne_sup[0], borne_inf[0]), min(borne_sup[1], borne_inf[1]), min(borne_sup[2], borne_inf[2])
    x_bounds[0].max[fancy_names_index["vX_JeCh"]:fancy_names_index["vZ_JeCh"] + 1, DEBUT] = max(borne_sup[0], borne_inf[0]), max(borne_sup[1], borne_inf[1]), max(borne_sup[2], borne_inf[2])

    # bras droit
    x_bounds[0].min[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"]+1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"]+1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"]+1, DEBUT] = 0

    x_bounds[0].min[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, DEBUT] = 0

    # bras droit
    x_bounds[0].min[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"]+1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"]+1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"]+1, DEBUT] = 0

    x_bounds[0].min[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, DEBUT] = 0

    # coude droit
    x_bounds[0].min[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"]+1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"]+1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"]+1, DEBUT] = 0

    x_bounds[0].min[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, DEBUT] = 0
    # coude gauche
    x_bounds[0].min[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABG_AuJo"]+1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABG_AuJo"]+1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotABG_AuJo"]:fancy_names_index["vYrotABG_AuJo"]+1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABG_AuJo"]:fancy_names_index["vYrotABG_AuJo"]+1, DEBUT] = 0

    x_bounds[0].min[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, :] = -100
    x_bounds[0].max[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, :] = 100
    x_bounds[0].min[fancy_names_index["vZrotABG_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABG_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, DEBUT] = 0

    # du carpe
    x_bounds[0].min[fancy_names_index["vXrotC_AuJo"], :] = -100
    x_bounds[0].max[fancy_names_index["vXrotC_AuJo"], :] = 100
    x_bounds[0].min[fancy_names_index["vXrotC_AuJo"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vXrotC_AuJo"], DEBUT] = 0

    x_bounds[0].min[fancy_names_index["vXrotC_JeCh"], :] = -100
    x_bounds[0].max[fancy_names_index["vXrotC_JeCh"], :] = 100
    x_bounds[0].min[fancy_names_index["vXrotC_JeCh"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vXrotC_JeCh"], DEBUT] = 0

    # du dehanchement
    x_bounds[0].min[fancy_names_index["vYrotC_AuJo"], :] = -100
    x_bounds[0].max[fancy_names_index["vYrotC_AuJo"], :] = 100
    x_bounds[0].min[fancy_names_index["vYrotC_AuJo"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vYrotC_AuJo"], DEBUT] = 0

    x_bounds[0].min[fancy_names_index["vYrotC_JeCh"], :] = -100
    x_bounds[0].max[fancy_names_index["vYrotC_JeCh"], :] = 100
    x_bounds[0].min[fancy_names_index["vYrotC_JeCh"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vYrotC_JeCh"], DEBUT] = 0

    #
    # Contraintes de position: PHASE 1 le salto carpe
    #

    # deplacement
    x_bounds[1].min[fancy_names_index["X_AuJo"], :] = -.1
    x_bounds[1].max[fancy_names_index["X_AuJo"], :] = .1
    x_bounds[1].min[fancy_names_index["Y_AuJo"], :] = -1.
    x_bounds[1].max[fancy_names_index["Y_AuJo"], :] = 1.
    x_bounds[1].min[fancy_names_index["Z_AuJo"], :] = 0
    x_bounds[1].max[fancy_names_index["Z_AuJo"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    x_bounds[1].min[fancy_names_index["X_JeCh"], :] = -.1
    x_bounds[1].max[fancy_names_index["X_JeCh"], :] = .1
    x_bounds[1].min[fancy_names_index["Y_JeCh"], :] = -1.
    x_bounds[1].max[fancy_names_index["Y_JeCh"], :] = 1.
    x_bounds[1].min[fancy_names_index["Z_JeCh"], :] = 0
    x_bounds[1].max[fancy_names_index["Z_JeCh"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[1].min[fancy_names_index["Xrot_AuJo"], :] = 0
    x_bounds[1].max[fancy_names_index["Xrot_AuJo"], :] = 4 * 3.14
    x_bounds[1].min[fancy_names_index["Xrot_AuJo"], FIN] = 2 * 3.14 - .1

    x_bounds[1].min[fancy_names_index["Xrot_JeCh"], :] = 0
    x_bounds[1].max[fancy_names_index["Xrot_JeCh"], :] = 4 * 3.14
    x_bounds[1].min[fancy_names_index["Xrot_JeCh"], FIN] = 2 * 3.14 - .1

    # limitation du tilt autour de y
    x_bounds[1].min[fancy_names_index["Yrot_AuJo"], :] = - 3.14 / 16
    x_bounds[1].max[fancy_names_index["Yrot_AuJo"], :] = 3.14 / 16

    x_bounds[1].min[fancy_names_index["Yrot_JeCh"], :] = - 3.14 / 16
    x_bounds[1].max[fancy_names_index["Yrot_JeCh"], :] = 3.14 / 16
    # la vrille autour de z
    x_bounds[1].min[fancy_names_index["Zrot_AuJo"], :] = -.1
    x_bounds[1].max[fancy_names_index["Zrot_AuJo"], :] = .1

    x_bounds[1].min[fancy_names_index["Zrot_JeCh"], :] = -.1
    x_bounds[1].max[fancy_names_index["Zrot_JeCh"], :] = .1

    # bras f4a a l'ouverture

    # le carpe
    x_bounds[1].min[fancy_names_index["XrotC_AuJo"], :] = -2.35 - 0.1
    x_bounds[1].max[fancy_names_index["XrotC_AuJo"], :] = -2.35 + 0.1
    x_bounds[1].min[fancy_names_index["XrotC_JeCh"], :] = -2.35 - 0.1
    x_bounds[1].max[fancy_names_index["XrotC_JeCh"], :] = -2.35 + 0.1

    # le dehanchement
    x_bounds[1].min[fancy_names_index["YrotC_AuJo"], DEBUT] = -.1
    x_bounds[1].max[fancy_names_index["YrotC_AuJo"], DEBUT] = .1

    x_bounds[1].min[fancy_names_index["YrotC_JeCh"], DEBUT] = -.1
    x_bounds[1].max[fancy_names_index["YrotC_JeCh"], DEBUT] = .1


    # Contraintes de vitesse: PHASE 1 le salto carpe

    # en xy bassin
    x_bounds[1].min[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"] + 1, :] = -10
    x_bounds[1].max[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"] + 1, :] = 10

    x_bounds[1].min[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, :] = -10
    x_bounds[1].max[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, :] = 10

    # z bassin
    x_bounds[1].min[fancy_names_index["vZ_AuJo"], :] = -100
    x_bounds[1].max[fancy_names_index["vZ_AuJo"], :] = 100

    x_bounds[1].min[fancy_names_index["vZ_JeCh"], :] = -100
    x_bounds[1].max[fancy_names_index["vZ_JeCh"], :] = 100

    # autour de x
    x_bounds[1].min[fancy_names_index["vXrot_AuJo"], :] = -100
    x_bounds[1].max[fancy_names_index["vXrot_AuJo"], :] = 100

    x_bounds[1].min[fancy_names_index["vXrot_JeCh"], :] = -100
    x_bounds[1].max[fancy_names_index["vXrot_JeCh"], :] = 100

    # autour de y
    x_bounds[1].min[fancy_names_index["vYrot_AuJo"], :] = -100
    x_bounds[1].max[fancy_names_index["vYrot_AuJo"], :] = 100

    x_bounds[1].min[fancy_names_index["vYrot_JeCh"], :] = -100
    x_bounds[1].max[fancy_names_index["vYrot_JeCh"], :] = 100

    # autour de z
    x_bounds[1].min[fancy_names_index["vZrot_AuJo"], :] = -100
    x_bounds[1].max[fancy_names_index["vZrot_AuJo"], :] = 100

    x_bounds[1].min[fancy_names_index["vZrot_JeCh"], :] = -100
    x_bounds[1].max[fancy_names_index["vZrot_JeCh"], :] = 100

    # bras droit
    x_bounds[1].min[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"] + 1, :] = 100

    x_bounds[1].min[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, :] = 100

    # bras droit
    x_bounds[1].min[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"] + 1, :] = 100

    x_bounds[1].min[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, :] = 100

    # coude droit
    x_bounds[1].min[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"] + 1, :] = 100

    x_bounds[1].min[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, :] = 100

    # coude gauche
    x_bounds[1].min[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABG_AuJo"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABG_AuJo"] + 1, :] = 100

    x_bounds[1].min[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, :] = -100
    x_bounds[1].max[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, :] = 100

    # du carpe
    x_bounds[1].min[fancy_names_index["vXrotC_AuJo"], :] = -100
    x_bounds[1].max[fancy_names_index["vXrotC_AuJo"], :] = 100

    x_bounds[1].min[fancy_names_index["vXrotC_JeCh"], :] = -100
    x_bounds[1].max[fancy_names_index["vXrotC_JeCh"], :] = 100

    # du dehanchement
    x_bounds[1].min[fancy_names_index["vYrotC_AuJo"], :] = -100
    x_bounds[1].max[fancy_names_index["vYrotC_AuJo"], :] = 100

    x_bounds[1].min[fancy_names_index["vYrotC_JeCh"], :] = -100
    x_bounds[1].max[fancy_names_index["vYrotC_JeCh"], :] = 100

    #
    # Contraintes de position: PHASE 2 l'ouverture
    #

    # deplacement
    x_bounds[2].min[fancy_names_index["X_AuJo"], :] = -.2
    x_bounds[2].max[fancy_names_index["X_AuJo"], :] = .2
    x_bounds[2].min[fancy_names_index["Y_AuJo"], :] = -1.
    x_bounds[2].max[fancy_names_index["Y_AuJo"], :] = 1.
    x_bounds[2].min[fancy_names_index["Z_AuJo"], :] = 0
    x_bounds[2].max[fancy_names_index["Z_AuJo"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    x_bounds[2].min[fancy_names_index["X_JeCh"], :] = -.2
    x_bounds[2].max[fancy_names_index["X_JeCh"], :] = .2
    x_bounds[2].min[fancy_names_index["Y_JeCh"], :] = -1.
    x_bounds[2].max[fancy_names_index["Y_JeCh"], :] = 1.
    x_bounds[2].min[fancy_names_index["Z_JeCh"], :] = 0
    x_bounds[2].max[fancy_names_index["Z_JeCh"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[2].min[fancy_names_index["Xrot_AuJo"], :] = 2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[2].max[fancy_names_index["Xrot_AuJo"], :] = 4 * 3.14

    x_bounds[2].min[fancy_names_index["Xrot_JeCh"], :] = 2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[2].max[fancy_names_index["Xrot_JeCh"], :] = 4 * 3.14

    # limitation du tilt autour de y
    x_bounds[2].min[fancy_names_index["Yrot_AuJo"], :] = - 3.14 / 4
    x_bounds[2].max[fancy_names_index["Yrot_AuJo"], :] = 3.14 / 4

    x_bounds[2].min[fancy_names_index["Yrot_JeCh"], :] = - 3.14 / 4
    x_bounds[2].max[fancy_names_index["Yrot_JeCh"], :] = 3.14 / 4

    # la vrille autour de z
    x_bounds[2].min[fancy_names_index["Zrot_AuJo"], :] = 0
    x_bounds[2].max[fancy_names_index["Zrot_AuJo"], :] = 3 * 3.14

    x_bounds[2].min[fancy_names_index["Zrot_JeCh"], :] = 0
    x_bounds[2].max[fancy_names_index["Zrot_JeCh"], :] = 3 * 3.14

    # bras f4a a l'ouverture

    # le carpe
    x_bounds[2].min[fancy_names_index["XrotC_AuJo"], FIN] = -.4
    x_bounds[2].min[fancy_names_index["XrotC_JeCh"], FIN] = -.4

    # le dehanchement f4a a l'ouverture

    # Contraintes de vitesse: PHASE 2 l'ouverture

    # en xy bassin
    x_bounds[2].min[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"] + 1, :] = -10
    x_bounds[2].max[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"] + 1, :] = 10

    x_bounds[2].min[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, :] = -10
    x_bounds[2].max[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, :] = 10

    # z bassin
    x_bounds[2].min[fancy_names_index["vZ_AuJo"], :] = -100
    x_bounds[2].max[fancy_names_index["vZ_AuJo"], :] = 100

    x_bounds[2].min[fancy_names_index["vZ_JeCh"], :] = -100
    x_bounds[2].max[fancy_names_index["vZ_JeCh"], :] = 100

    # autour de x
    x_bounds[2].min[fancy_names_index["vXrot_AuJo"], :] = -100
    x_bounds[2].max[fancy_names_index["vXrot_AuJo"], :] = 100

    x_bounds[2].min[fancy_names_index["vXrot_JeCh"], :] = -100
    x_bounds[2].max[fancy_names_index["vXrot_JeCh"], :] = 100

    # autour de y
    x_bounds[2].min[fancy_names_index["vYrot_AuJo"], :] = -100
    x_bounds[2].max[fancy_names_index["vYrot_AuJo"], :] = 100

    x_bounds[2].min[fancy_names_index["vYrot_JeCh"], :] = -100
    x_bounds[2].max[fancy_names_index["vYrot_JeCh"], :] = 100

    # autour de z
    x_bounds[2].min[fancy_names_index["vZrot_AuJo"], :] = -100
    x_bounds[2].max[fancy_names_index["vZrot_AuJo"], :] = 100

    x_bounds[2].min[fancy_names_index["vZrot_JeCh"], :] = -100
    x_bounds[2].max[fancy_names_index["vZrot_JeCh"], :] = 100

    # bras droit
    x_bounds[2].min[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"] + 1, :] = 100

    x_bounds[2].min[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, :] = 100

    # bras droit
    x_bounds[2].min[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"] + 1, :] = 100

    x_bounds[2].min[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, :] = 100

    # coude droit
    x_bounds[2].min[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"] + 1, :] = 100

    x_bounds[2].min[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, :] = 100

    # coude gauche
    x_bounds[2].min[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABG_AuJo"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABG_AuJo"] + 1, :] = 100

    x_bounds[2].min[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, :] = -100
    x_bounds[2].max[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, :] = 100

    # du carpe
    x_bounds[2].min[fancy_names_index["vXrotC_AuJo"], :] = -100
    x_bounds[2].max[fancy_names_index["vXrotC_AuJo"], :] = 100

    x_bounds[2].min[fancy_names_index["vXrotC_JeCh"], :] = -100
    x_bounds[2].max[fancy_names_index["vXrotC_JeCh"], :] = 100

    # du dehanchement
    x_bounds[2].min[fancy_names_index["vYrotC_AuJo"], :] = -100
    x_bounds[2].max[fancy_names_index["vYrotC_AuJo"], :] = 100

    x_bounds[2].min[fancy_names_index["vYrotC_JeCh"], :] = -100
    x_bounds[2].max[fancy_names_index["vYrotC_JeCh"], :] = 100

    #
    # Contraintes de position: PHASE 3 la vrille et demie
    #

    # deplacement
    x_bounds[3].min[fancy_names_index["X_AuJo"], :] = -.2
    x_bounds[3].max[fancy_names_index["X_AuJo"], :] = .2
    x_bounds[3].min[fancy_names_index["Y_AuJo"], :] = -1.
    x_bounds[3].max[fancy_names_index["Y_AuJo"], :] = 1.
    x_bounds[3].min[fancy_names_index["Z_AuJo"], :] = 0
    x_bounds[3].max[fancy_names_index["Z_AuJo"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    x_bounds[3].min[fancy_names_index["X_JeCh"], :] = -.2
    x_bounds[3].max[fancy_names_index["X_JeCh"], :] = .2
    x_bounds[3].min[fancy_names_index["Y_JeCh"], :] = -1.
    x_bounds[3].max[fancy_names_index["Y_JeCh"], :] = 1.
    x_bounds[3].min[fancy_names_index["Z_JeCh"], :] = 0
    x_bounds[3].max[fancy_names_index["Z_JeCh"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[3].min[fancy_names_index["Xrot_AuJo"], :] = 2 * 3.14 - .1
    x_bounds[3].max[fancy_names_index["Xrot_AuJo"], :] = 2 * 3.14 + 3/2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[3].min[fancy_names_index["Xrot_AuJo"], FIN] = 2 * 3.14 + 3/2 * 3.14 - .1
    x_bounds[3].max[fancy_names_index["Xrot_AuJo"], FIN] = 2 * 3.14 + 3/2 * 3.14 + .1  # 1 salto 3/4

    x_bounds[3].min[fancy_names_index["Xrot_JeCh"], :] = 2 * 3.14 - .1
    x_bounds[3].max[fancy_names_index["Xrot_JeCh"], :] = 2 * 3.14 + 3 / 2 * 3.14 + .1  # 1 salto 3/4
    x_bounds[3].min[fancy_names_index["Xrot_JeCh"], FIN] = 2 * 3.14 + 3 / 2 * 3.14 - .1
    x_bounds[3].max[fancy_names_index["Xrot_JeCh"], FIN] = 2 * 3.14 + 3 / 2 * 3.14 + .1  # 1 salto 3/4

    # limitation du tilt autour de y
    x_bounds[3].min[fancy_names_index["Yrot_AuJo"], :] = - 3.14 / 4
    x_bounds[3].max[fancy_names_index["Yrot_AuJo"], :] = 3.14 / 4
    x_bounds[3].min[fancy_names_index["Yrot_AuJo"], FIN] = - 3.14 / 8
    x_bounds[3].max[fancy_names_index["Yrot_AuJo"], FIN] = 3.14 / 8

    x_bounds[3].min[fancy_names_index["Yrot_JeCh"], :] = - 3.14 / 4
    x_bounds[3].max[fancy_names_index["Yrot_JeCh"], :] = 3.14 / 4
    x_bounds[3].min[fancy_names_index["Yrot_JeCh"], FIN] = - 3.14 / 8
    x_bounds[3].max[fancy_names_index["Yrot_JeCh"], FIN] = 3.14 / 8

    # la vrille autour de z
    x_bounds[3].min[fancy_names_index["Zrot_AuJo"], :] = 0
    x_bounds[3].max[fancy_names_index["Zrot_AuJo"], :] = 3 * 3.14
    x_bounds[3].min[fancy_names_index["Zrot_AuJo"], FIN] = 3 * 3.14 - .1  # complete la vrille
    x_bounds[3].max[fancy_names_index["Zrot_AuJo"], FIN] = 3 * 3.14 + .1

    x_bounds[3].min[fancy_names_index["Zrot_JeCh"], :] = 0
    x_bounds[3].max[fancy_names_index["Zrot_JeCh"], :] = 3 * 3.14
    x_bounds[3].min[fancy_names_index["Zrot_JeCh"], FIN] = 3 * 3.14 - .1  # complete la vrille
    x_bounds[3].max[fancy_names_index["Zrot_JeCh"], FIN] = 3 * 3.14 + .1

    # bras f4a la vrille

    # le carpe
    x_bounds[3].min[fancy_names_index["XrotC_AuJo"], :] = -.4

    # le dehanchement f4a la vrille

    # Contraintes de vitesse: PHASE 3 la vrille et demie

    # en xy bassin
    x_bounds[3].min[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"] + 1, :] = -10
    x_bounds[3].max[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"] + 1, :] = 10

    x_bounds[3].min[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, :] = -10
    x_bounds[3].max[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, :] = 10

    # z bassin
    x_bounds[3].min[fancy_names_index["vZ_AuJo"], :] = -100
    x_bounds[3].max[fancy_names_index["vZ_AuJo"], :] = 100

    x_bounds[3].min[fancy_names_index["vZ_JeCh"], :] = -100
    x_bounds[3].max[fancy_names_index["vZ_JeCh"], :] = 100

    # autour de x
    x_bounds[3].min[fancy_names_index["vXrot_AuJo"], :] = -100
    x_bounds[3].max[fancy_names_index["vXrot_AuJo"], :] = 100

    x_bounds[3].min[fancy_names_index["vXrot_JeCh"], :] = -100
    x_bounds[3].max[fancy_names_index["vXrot_JeCh"], :] = 100

    # autour de y
    x_bounds[3].min[fancy_names_index["vYrot_AuJo"], :] = -100
    x_bounds[3].max[fancy_names_index["vYrot_AuJo"], :] = 100

    x_bounds[3].min[fancy_names_index["vYrot_JeCh"], :] = -100
    x_bounds[3].max[fancy_names_index["vYrot_JeCh"], :] = 100

    # autour de z
    x_bounds[3].min[fancy_names_index["vZrot_AuJo"], :] = -100
    x_bounds[3].max[fancy_names_index["vZrot_AuJo"], :] = 100

    x_bounds[3].min[fancy_names_index["vZrot_JeCh"], :] = -100
    x_bounds[3].max[fancy_names_index["vZrot_JeCh"], :] = 100

    # bras droit
    x_bounds[3].min[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"] + 1, :] = 100

    x_bounds[3].min[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, :] = 100

    # bras droit
    x_bounds[3].min[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"] + 1, :] = 100

    x_bounds[3].min[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, :] = 100

    # coude droit
    x_bounds[3].min[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"] + 1, :] = 100

    x_bounds[3].min[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, :] = 100

    # coude gauche
    x_bounds[3].min[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABG_AuJo"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABG_AuJo"] + 1, :] = 100

    x_bounds[3].min[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, :] = -100
    x_bounds[3].max[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, :] = 100

    # du carpe
    x_bounds[3].min[fancy_names_index["vXrotC_AuJo"], :] = -100
    x_bounds[3].max[fancy_names_index["vXrotC_AuJo"], :] = 100

    x_bounds[3].min[fancy_names_index["vXrotC_JeCh"], :] = -100
    x_bounds[3].max[fancy_names_index["vXrotC_JeCh"], :] = 100

    # du dehanchement
    x_bounds[3].min[fancy_names_index["vYrotC_AuJo"], :] = -100
    x_bounds[3].max[fancy_names_index["vYrotC_AuJo"], :] = 100

    x_bounds[3].min[fancy_names_index["vYrotC_JeCh"], :] = -100
    x_bounds[3].max[fancy_names_index["vYrotC_JeCh"], :] = 100

    #
    # Contraintes de position: PHASE 4 la reception
    #

    # deplacement
    x_bounds[4].min[fancy_names_index["X_AuJo"], :] = -.1
    x_bounds[4].max[fancy_names_index["X_AuJo"], :] = .1
    x_bounds[4].min[fancy_names_index["Y_AuJo"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["Y_AuJo"], FIN] = .1
    x_bounds[4].min[fancy_names_index["Z_AuJo"], :] = 0
    x_bounds[4].max[fancy_names_index["Z_AuJo"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    x_bounds[4].min[fancy_names_index["Z_AuJo"], FIN] = 0
    x_bounds[4].max[fancy_names_index["Z_AuJo"], FIN] = .1

    x_bounds[4].min[fancy_names_index["X_JeCh"], :] = -.1
    x_bounds[4].max[fancy_names_index["X_JeCh"], :] = .1
    x_bounds[4].min[fancy_names_index["Y_JeCh"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["Y_JeCh"], FIN] = .1
    x_bounds[4].min[fancy_names_index["Z_JeCh"], :] = 0
    x_bounds[4].max[fancy_names_index["Z_JeCh"], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
    x_bounds[4].min[fancy_names_index["Z_JeCh"], FIN] = 0
    x_bounds[4].max[fancy_names_index["Z_JeCh"], FIN] = .1

    # le salto autour de x
    x_bounds[4].min[fancy_names_index["Xrot_AuJo"], :] = 2 * 3.14 + 3 / 2 * 3.14 - .2  # penche vers avant -> moins de salto
    x_bounds[4].max[fancy_names_index["Xrot_AuJo"], :] = -.50 + 4 * 3.14  # un peu carpe a la fin
    x_bounds[4].min[fancy_names_index["Xrot_AuJo"], FIN] = -.50 + 4 * 3.14 - .1
    x_bounds[4].max[fancy_names_index["Xrot_AuJo"], FIN] = -.50 + 4 * 3.14 + .1  # 2 salto fin un peu carpe

    x_bounds[4].min[fancy_names_index["Xrot_JeCh"], :] = 2 * 3.14 + 3 / 2 * 3.14 - .2  # penche vers avant -> moins de salto
    x_bounds[4].max[fancy_names_index["Xrot_JeCh"], :] = -.50 + 4 * 3.14  # un peu carpe a la fin
    x_bounds[4].min[fancy_names_index["Xrot_JeCh"], FIN] = -.50 + 4 * 3.14 - .1
    x_bounds[4].max[fancy_names_index["Xrot_JeCh"], FIN] = -.50 + 4 * 3.14 + .1  # 2 salto fin un peu carpe

    # limitation du tilt autour de y
    x_bounds[4].min[fancy_names_index["Yrot_AuJo"], :] = - 3.14 / 16
    x_bounds[4].max[fancy_names_index["Yrot_AuJo"], :] = 3.14 / 16

    x_bounds[4].min[fancy_names_index["Yrot_JeCh"], :] = - 3.14 / 16
    x_bounds[4].max[fancy_names_index["Yrot_JeCh"], :] = 3.14 / 16

    # la vrille autour de z
    x_bounds[4].min[fancy_names_index["Zrot_AuJo"], :] = 3 * 3.14 - .1  # complete la vrille
    x_bounds[4].max[fancy_names_index["Zrot_AuJo"], :] = 3 * 3.14 + .1

    x_bounds[4].min[fancy_names_index["Zrot_JeCh"], :] = 3 * 3.14 - .1  # complete la vrille
    x_bounds[4].max[fancy_names_index["Zrot_JeCh"], :] = 3 * 3.14 + .1

    # bras droit
    x_bounds[4].min[fancy_names_index["YrotBD_AuJo"], FIN] = 2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[fancy_names_index["YrotBD_AuJo"], FIN] = 2.9 + .1
    x_bounds[4].min[fancy_names_index["ZrotBD_AuJo"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotBD_AuJo"], FIN] = .1

    x_bounds[4].min[fancy_names_index["YrotBD_JeCh"], FIN] = 2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[fancy_names_index["YrotBD_JeCh"], FIN] = 2.9 + .1
    x_bounds[4].min[fancy_names_index["ZrotBD_JeCh"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotBD_JeCh"], FIN] = .1

    # bras gauche
    x_bounds[4].min[fancy_names_index["YrotBG_AuJo"], FIN] = -2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[fancy_names_index["YrotBG_AuJo"], FIN] = -2.9 + .1
    x_bounds[4].min[fancy_names_index["ZrotBG_AuJo"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotBG_AuJo"], FIN] = .1

    x_bounds[4].min[fancy_names_index["YrotBG_JeCh"], FIN] = -2.9 - .1  # debut bras aux oreilles
    x_bounds[4].max[fancy_names_index["YrotBG_JeCh"], FIN] = -2.9 + .1
    x_bounds[4].min[fancy_names_index["ZrotBG_JeCh"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotBG_JeCh"], FIN] = .1

    # coude droit
    x_bounds[4].min[fancy_names_index["ZrotABD_AuJo"]:fancy_names_index["XrotABD_AuJo"] + 1, FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotABD_AuJo"]:fancy_names_index["XrotABD_AuJo"] + 1, FIN] = .1

    x_bounds[4].min[fancy_names_index["ZrotABD_JeCh"]:fancy_names_index["XrotABD_JeCh"] + 1, FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotABD_JeCh"]:fancy_names_index["XrotABD_JeCh"] + 1, FIN] = .1
    # coude gauche
    x_bounds[4].min[fancy_names_index["ZrotABG_AuJo"]:fancy_names_index["XrotABG_AuJo"] + 1, FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotABG_AuJo"]:fancy_names_index["XrotABG_AuJo"] + 1, FIN] = .1

    x_bounds[4].min[fancy_names_index["ZrotABG_JeCh"]:fancy_names_index["XrotABG_JeCh"] + 1, FIN] = -.1
    x_bounds[4].max[fancy_names_index["ZrotABG_JeCh"]:fancy_names_index["XrotABG_JeCh"] + 1, FIN] = .1

    # le carpe
    x_bounds[4].min[fancy_names_index["XrotC_AuJo"], :] = -.4
    x_bounds[4].min[fancy_names_index["XrotC_AuJo"], FIN] = -.60
    x_bounds[4].max[fancy_names_index["XrotC_AuJo"], FIN] = -.40  # fin un peu carpe

    x_bounds[4].min[fancy_names_index["XrotC_JeCh"], :] = -.4
    x_bounds[4].min[fancy_names_index["XrotC_JeCh"], FIN] = -.60
    x_bounds[4].max[fancy_names_index["XrotC_JeCh"], FIN] = -.40  # fin un peu carpe

    # le dehanchement
    x_bounds[4].min[fancy_names_index["YrotC_AuJo"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["YrotC_AuJo"], FIN] = .1

    x_bounds[4].min[fancy_names_index["YrotC_JeCh"], FIN] = -.1
    x_bounds[4].max[fancy_names_index["YrotC_JeCh"], FIN] = .1

    # Contraintes de vitesse: PHASE 4 la reception

    # en xy bassin
    x_bounds[4].min[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"] + 1, :] = -10
    x_bounds[4].max[fancy_names_index["vX_AuJo"]:fancy_names_index["vY_AuJo"] + 1, :] = 10

    x_bounds[4].min[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, :] = -10
    x_bounds[4].max[fancy_names_index["vX_JeCh"]:fancy_names_index["vY_JeCh"] + 1, :] = 10

    # z bassin
    x_bounds[4].min[fancy_names_index["vZ_AuJo"], :] = -100
    x_bounds[4].max[fancy_names_index["vZ_AuJo"], :] = 100

    x_bounds[4].min[fancy_names_index["vZ_JeCh"], :] = -100
    x_bounds[4].max[fancy_names_index["vZ_JeCh"], :] = 100

    # autour de x
    x_bounds[4].min[fancy_names_index["vXrot_AuJo"], :] = -100
    x_bounds[4].max[fancy_names_index["vXrot_AuJo"], :] = 100

    x_bounds[4].min[fancy_names_index["vXrot_JeCh"], :] = -100
    x_bounds[4].max[fancy_names_index["vXrot_JeCh"], :] = 100

    # autour de y
    x_bounds[4].min[fancy_names_index["vYrot_AuJo"], :] = -100
    x_bounds[4].max[fancy_names_index["vYrot_AuJo"], :] = 100

    x_bounds[4].min[fancy_names_index["vYrot_JeCh"], :] = -100
    x_bounds[4].max[fancy_names_index["vYrot_JeCh"], :] = 100

    # autour de z
    x_bounds[4].min[fancy_names_index["vZrot_AuJo"], :] = -100
    x_bounds[4].max[fancy_names_index["vZrot_AuJo"], :] = 100

    x_bounds[4].min[fancy_names_index["vZrot_JeCh"], :] = -100
    x_bounds[4].max[fancy_names_index["vZrot_JeCh"], :] = 100

    # bras droit
    x_bounds[4].min[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotBD_AuJo"]:fancy_names_index["vYrotBD_AuJo"] + 1, :] = 100

    x_bounds[4].min[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotBD_JeCh"]:fancy_names_index["vYrotBD_JeCh"] + 1, :] = 100

    # bras droit
    x_bounds[4].min[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotBG_AuJo"]:fancy_names_index["vYrotBG_AuJo"] + 1, :] = 100

    x_bounds[4].min[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotBG_JeCh"]:fancy_names_index["vYrotBG_JeCh"] + 1, :] = 100

    # coude droit
    x_bounds[4].min[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABD_AuJo"] + 1, :] = 100

    x_bounds[4].min[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABD_JeCh"] + 1, :] = 100

    # coude gauche
    x_bounds[4].min[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABG_AuJo"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotABD_AuJo"]:fancy_names_index["vYrotABG_AuJo"] + 1, :] = 100

    x_bounds[4].min[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotABD_JeCh"]:fancy_names_index["vYrotABG_JeCh"] + 1, :] = 100

    # du carpe
    x_bounds[4].min[fancy_names_index["vXrotC_AuJo"], :] = -100
    x_bounds[4].max[fancy_names_index["vXrotC_AuJo"], :] = 100

    x_bounds[4].min[fancy_names_index["vXrotC_JeCh"], :] = -100
    x_bounds[4].max[fancy_names_index["vXrotC_JeCh"], :] = 100

    # du dehanchement
    x_bounds[4].min[fancy_names_index["vYrotC_AuJo"], :] = -100
    x_bounds[4].max[fancy_names_index["vYrotC_AuJo"], :] = 100

    x_bounds[4].min[fancy_names_index["vYrotC_JeCh"], :] = -100
    x_bounds[4].max[fancy_names_index["vYrotC_JeCh"], :] = 100

    return x_bounds

def set_x_init(biorbd_model, fancy_names_index):

    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    x0 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x1 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x2 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x3 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x4 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))

    x0[fancy_names_index["Xrot_AuJo"], 0] = .50
    x0[fancy_names_index["ZrotBG_AuJo"]] = -.75
    x0[fancy_names_index["ZrotBD_AuJo"]] = .75
    x0[fancy_names_index["YrotBG_AuJo"], 0] = -2.9
    x0[fancy_names_index["YrotBD_AuJo"], 0] = 2.9
    x0[fancy_names_index["YrotBG_AuJo"], 1] = -1.35
    x0[fancy_names_index["YrotBD_AuJo"], 1] = 1.35
    x0[fancy_names_index["XrotC_AuJo"], 0] = -.5
    x0[fancy_names_index["XrotC_AuJo"], 1] = -2.6

    x0[fancy_names_index["Xrot_JeCh"], 0] = .50
    x0[fancy_names_index["ZrotBG_JeCh"]] = -.75
    x0[fancy_names_index["ZrotBD_JeCh"]] = .75
    x0[fancy_names_index["YrotBG_JeCh"], 0] = -2.9
    x0[fancy_names_index["YrotBD_JeCh"], 0] = 2.9
    x0[fancy_names_index["YrotBG_JeCh"], 1] = -1.35
    x0[fancy_names_index["YrotBD_JeCh"], 1] = 1.35
    x0[fancy_names_index["XrotC_JeCh"], 0] = -.5
    x0[fancy_names_index["XrotC_JeCh"], 1] = -2.6

    x1[fancy_names_index["ZrotBG_AuJo"]] = -.75
    x1[fancy_names_index["ZrotBD_AuJo"]] = .75
    x1[fancy_names_index["Xrot_AuJo"], 1] = 2 * 3.14
    x1[fancy_names_index["YrotBG_AuJo"]] = -1.35
    x1[fancy_names_index["YrotBD_AuJo"]] = 1.35
    x1[fancy_names_index["XrotC_AuJo"]] = -2.6

    x1[fancy_names_index["ZrotBG_JeCh"]] = -.75
    x1[fancy_names_index["ZrotBD_JeCh"]] = .75
    x1[fancy_names_index["Xrot_JeCh"], 1] = 2 * 3.14
    x1[fancy_names_index["YrotBG_JeCh"]] = -1.35
    x1[fancy_names_index["YrotBD_JeCh"]] = 1.35
    x1[fancy_names_index["XrotC_JeCh"]] = -2.6

    x2[fancy_names_index["Xrot_AuJo"]] = 2 * 3.14
    x2[fancy_names_index["Zrot_AuJo"], 1] = 3.14
    x2[fancy_names_index["ZrotBG_AuJo"], 0] = -.75
    x2[fancy_names_index["ZrotBD_AuJo"], 0] = .75
    x2[fancy_names_index["YrotBG_AuJo"], 0] = -1.35
    x2[fancy_names_index["YrotBD_AuJo"], 0] = 1.35
    x2[fancy_names_index["XrotC_AuJo"], 0] = -2.6

    x2[fancy_names_index["Xrot_JeCh"]] = 2 * 3.14
    x2[fancy_names_index["Zrot_JeCh"], 1] = 3.14
    x2[fancy_names_index["ZrotBG_JeCh"], 0] = -.75
    x2[fancy_names_index["ZrotBD_JeCh"], 0] = .75
    x2[fancy_names_index["YrotBG_JeCh"], 0] = -1.35
    x2[fancy_names_index["YrotBD_JeCh"], 0] = 1.35
    x2[fancy_names_index["XrotC_JeCh"], 0] = -2.6

    x3[fancy_names_index["Xrot_AuJo"], 0] = 2 * 3.14
    x3[fancy_names_index["Xrot_AuJo"], 1] = 2 * 3.14 + 3/2 * 3.14
    x3[fancy_names_index["Zrot_AuJo"], 0] = 3.14
    x3[fancy_names_index["Zrot_AuJo"], 1] = 3 * 3.14

    x3[fancy_names_index["Xrot_JeCh"], 0] = 2 * 3.14
    x3[fancy_names_index["Xrot_JeCh"], 1] = 2 * 3.14 + 3 / 2 * 3.14
    x3[fancy_names_index["Zrot_JeCh"], 0] = 3.14
    x3[fancy_names_index["Zrot_JeCh"], 1] = 3 * 3.14

    x4[fancy_names_index["Xrot_AuJo"], 0] = 2 * 3.14 + 3/2 * 3.14
    x4[fancy_names_index["Xrot_AuJo"], 1] = 4 * 3.14
    x4[fancy_names_index["Zrot_AuJo"]] = 3 * 3.14
    x4[fancy_names_index["XrotC_AuJo"], 1] = -.5

    x4[fancy_names_index["Xrot_JeCh"], 0] = 2 * 3.14 + 3 / 2 * 3.14
    x4[fancy_names_index["Xrot_JeCh"], 1] = 4 * 3.14
    x4[fancy_names_index["Zrot_JeCh"]] = 3 * 3.14
    x4[fancy_names_index["XrotC_JeCh"], 1] = -.5

    x_init = InitialGuessList()
    x_init.add(x0, interpolation=InterpolationType.LINEAR)
    x_init.add(x1, interpolation=InterpolationType.LINEAR)
    x_init.add(x2, interpolation=InterpolationType.LINEAR)
    x_init.add(x3, interpolation=InterpolationType.LINEAR)
    x_init.add(x4, interpolation=InterpolationType.LINEAR)

    return x_init

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
    nb_qddot_joints = (nb_q / 2) - biorbd_model[0].nbRoot()

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, marker_index=1, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=4)

    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=final_time, weight=100000, phase=0)

    # Les hanches sont fixes a +-0.2 en bounds, mais les mains doivent quand meme être proches des jambes
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

    fancy_names_index = set_fancy_names_index()
    set_x_bounds(biorbd_model, fancy_names_index)

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

    set_x_init(biorbd_model, fancy_names_index)

    constraints = ConstraintList()
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.05, max_bound=.05, first_marker='MidMainGAuJo', second_marker='CibleMainGAuJo', phase=1)
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.05, max_bound=.05, first_marker='MidMainDAuJo', second_marker='CibleMainDAuJo', phase=1)
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.05, max_bound=.05, first_marker='MidMainGJeCh', second_marker='CibleMainGJeCh', phase=1)
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL_SHOOTING, min_bound=-.05, max_bound=.05, first_marker='MidMainDJeCh', second_marker='CibleMainDJeCh', phase=1)
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

    model_path = "Models/AuJo_JeCh.bioMod"
    n_threads = 4
    print_ocp_FLAG = True
    show_online_FLAG = True
    HSL_FLAG = True
    save_sol_FLAG = True

    n_shooting = (40, 100, 100, 100, 40)
    ocp = prepare_ocp(model_path, n_shooting=n_shooting, n_threads=n_threads, final_time=1.87)
    ocp.add_plot_penalty(CostType.ALL)
    if print_ocp_FLAG:
        ocp.print(to_graph=True)
    solver = Solver.IPOPT(show_online_optim=show_online_FLAG, show_options=dict(show_bounds=True))
    if HSL_FLAG:
        solver.set_linear_solver('ma57')
    else:
        print("Not using ma57")
    solver.set_maximum_iterations(10000)
    solver.set_convergence_tolerance(1e-4)
    sol = ocp.solve(solver)

    temps = time.strftime("%Y-%m-%d-%H%M")
    nom = model_path.split('/')[-1].removesuffix('.bioMod')
    qs = sol.states[0]['q']
    qdots = sol.states[0]['qdot']
    for i in range(1, len(sol.states)):
        qs = np.hstack((qs, sol.states[i]['q']))
        qdots = np.hstack((qdots, sol.states[i]['qdot']))
    if save_sol_FLAG:  # switch manuelle
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
