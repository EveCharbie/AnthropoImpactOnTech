
"""
The goal of this program is to optimize the movement to achieve a rudi out pike (803<).
Simultaneously for two anthropometric models.
"""
import numpy as np
import biorbd_casadi as biorbd
from typing import Union
import casadi as cas
import sys
import argparse
import os


from bioptim import (
    MultiBiorbdModel,
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    Bounds,
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
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
    PhaseTransitionList,
    PhaseTransitionFcn,
    NodeMappingList,
)


import time

try:
    import IPython

    IPYTHON = True
except ImportError:
    print("No IPython.")
    IPYTHON = False


def superimpose_markers_constraint(
        controller: PenaltyController,
):
    first_marker_D = 1
    second_marker_D = 5
    first_marker_G = 3
    second_marker_G = 6

    total_diff = []
    for index_model, biorbd_model in enumerate(controller.model.models):

        q = controller.states["q"].mx[controller.model.variable_index('q', index_model)]
        diff_markers_D = controller.model.models[index_model].marker(q, second_marker_D) - controller.model.models[
            index_model].marker(q, first_marker_D)
        diff_markers_G = controller.model.models[index_model].marker(q, second_marker_G) - controller.model.models[
            index_model].marker(q, first_marker_G)
        sum_diff_D = 0
        sum_diff_G = 0
        for i in range(3):
            sum_diff_D += (diff_markers_D[i]) ** 2
            sum_diff_G += (diff_markers_G[i]) ** 2
        total_diff += [sum_diff_D, sum_diff_G]
    return controller.mx_to_cx(
        f"diff_markers",
        cas.vertcat(*total_diff),
        controller.states["q"],
    )

def superimpose_markers(
        controller: PenaltyController,
):
    first_marker_D = 1
    second_marker_D = 5
    first_marker_G = 3
    second_marker_G = 6

    total_diff = 0
    for index_model, biorbd_model in enumerate(controller.model.models):
        q = controller.states["q"].mx[controller.model.variable_index('q', index_model)]
        diff_markers_D = controller.model.models[index_model].marker(q, second_marker_D) - controller.model.models[index_model].marker(q, first_marker_D)
        diff_markers_G = controller.model.models[index_model].marker(q, second_marker_G) - controller.model.models[index_model].marker(q, first_marker_G)
        for i in range(3):
            total_diff += (diff_markers_D[i])**2
            total_diff += (diff_markers_G[i])**2


    return controller.mx_to_cx(
        f"diff_markers",
        total_diff,
        controller.states["q"],
    )


def minimize_dofs(controller: PenaltyController, dofs: list, targets: list) -> cas.MX:
    diff = 0
    if isinstance(dofs, int):
        dofs = [dofs]
    for i, dof in enumerate(dofs):
        diff += (controller.states["q"].mx[dof] - targets[i]) ** 2
    return controller.mx_to_cx("minimize_dofs", diff, controller.states["q"])

def set_fancy_names_index(biorbd_models):
    """
    For readability
    """
    nb_model = len(biorbd_models[0].models)

    nb_q = biorbd_models[0][0].nb_q
    fancy_names_index = {}
    fancy_names_index["X"] = [0+i*nb_q for i in range(nb_model)]
    fancy_names_index["Y"] = [1+i*nb_q for i in range(nb_model)]
    fancy_names_index["Z"] = [2+i*nb_q for i in range(nb_model)]
    fancy_names_index["Xrot"] = [3 + i*nb_q for i in range(nb_model)]
    fancy_names_index["Yrot"] = [4 + i*nb_q for i in range(nb_model)]
    fancy_names_index["Zrot"] = [5 + i*nb_q for i in range(nb_model)]
    fancy_names_index["ZrotBD"] = [6 + i*nb_q for i in range(nb_model)]
    fancy_names_index["YrotBD"] = [7 + i*nb_q for i in range(nb_model)]
    fancy_names_index["ZrotABD"] = [8 + i*nb_q for i in range(nb_model)]
    fancy_names_index["XrotABD"] = [9 + i*nb_q for i in range(nb_model)]
    fancy_names_index["ZrotBG"] = [10 + i*nb_q for i in range(nb_model)]
    fancy_names_index["YrotBG"] = [11 + i*nb_q for i in range(nb_model)]
    fancy_names_index["ZrotABG"] = [12 + i*nb_q for i in range(nb_model)]
    fancy_names_index["XrotABG"] = [13 + i*nb_q for i in range(nb_model)]
    fancy_names_index["XrotC"] = [14 + i*nb_q for i in range(nb_model)]
    fancy_names_index["YrotC"] = [15 + i*nb_q for i in range(nb_model)]
    fancy_names_index["vX"] = [0+nb_q*nb_model+nb_q*i for i in range(nb_model)]
    fancy_names_index["vY"] = [1+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vZ"] = [2+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vXrot"] = [3+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vYrot"] = [4+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vZrot"] = [5+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vZrotBD"] = [6+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vYrotBD"] = [7+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vZrotABD"] = [8+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vYrotABD"] = [9+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vZrotBG"] = [10+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vYrotBG"] = [11+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vZrotABG"] = [12+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vYrotABG"] = [13+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vXrotC"] = [14+nb_q*nb_model+i*nb_q for i in range(nb_model)]
    fancy_names_index["vYrotC"] = [15+nb_q*nb_model+i*nb_q for i in range(nb_model)]

    return fancy_names_index


def set_x_bounds(biorbd_models, fancy_names_index, final_time, mappings):
    # for
    nb_q = biorbd_models[0].nb_q
    nb_qdot = biorbd_models[0].nb_qdot
    nb_models = len(biorbd_models[0].models)
    nb_q_per_model = nb_q//nb_models
    nb_qdot_per_model = nb_qdot//nb_models
    x_bounds = BoundsList()
    x_bounds.add(bounds=biorbd_models[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_models[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_models[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_models[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_models[0].bounds_from_ranges(["q", "qdot"]))


    #
    # Path constraint

    # Pour la lisibilite
    DEBUT, MILIEU, FIN = 0, 1, 2

    #
    # Contraintes de position: PHASE 0 la montee en carpe
    #

    zmax = 8  # 9.81 / 8 * final_time**2 + 1  # une petite marge

    # for i, model in enumerate(biorbd_models[0]):
    # deplacement
    for i in range(nb_models):
        x_bounds[0].min[fancy_names_index["X"][i], :] = -0.1
        x_bounds[0].max[fancy_names_index["X"][i], :] = 0.1
        x_bounds[0].min[fancy_names_index["Y"][i], :] = -1.0
        x_bounds[0].max[fancy_names_index["Y"][i], :] = 1.0
        x_bounds[0].min[:fancy_names_index["Z"][i]+1, DEBUT] = 0
        x_bounds[0].max[:fancy_names_index["Z"][i]+1, DEBUT] = 0
        x_bounds[0].min[fancy_names_index["Z"][i], MILIEU:] = 0

        x_bounds[0].max[
            fancy_names_index["Z"][i], MILIEU:
        ] = zmax
             # beaucoup plus que necessaire, juste pour que la parabole fonctionne

            # le salto autour de x
    for i in range(nb_models):
        x_bounds[0].min[fancy_names_index["Xrot"][i], :] = 0
        # 2 * 3.14 + 3 / 2 * 3.14 - .2
        x_bounds[0].max[fancy_names_index["Xrot"][i], :] = -.50 + 3.14

        x_bounds[0].min[fancy_names_index["Xrot"][i], DEBUT] = 0.50  # penche vers l'avant un peu carpe
        x_bounds[0].max[fancy_names_index["Xrot"][i], DEBUT] = 0.50

        x_bounds[0].min[fancy_names_index["Xrot"][i], MILIEU:] = 0

        x_bounds[0].max[fancy_names_index["Xrot"][i], MILIEU:] = 4 * 3.14 + 0.1  # salto


        # limitation du tilt autour de y
        x_bounds[0].min[fancy_names_index["Yrot"][i], DEBUT] = 0
        x_bounds[0].max[fancy_names_index["Yrot"][i], DEBUT] = 0

        x_bounds[0].min[fancy_names_index["Yrot"][i], MILIEU:] = -3.14 / 16  # vraiment pas suppose tilte

        x_bounds[0].max[fancy_names_index["Yrot"][i], MILIEU:] = 3.14 / 16


        # la vrille autour de z
        x_bounds[0].min[fancy_names_index["Zrot"][i], DEBUT] = 0

        x_bounds[0].max[fancy_names_index["Zrot"][i], DEBUT] = 0

        x_bounds[0].min[fancy_names_index["Zrot"][i], MILIEU:] = -0.1  # pas de vrille dans cette phase

        x_bounds[0].max[fancy_names_index["Zrot"][i], MILIEU:] = 0.1


    # bras droit
    x_bounds[0].min[fancy_names_index["YrotBD"], DEBUT] = 2.9  # debut bras aux oreilles

    x_bounds[0].max[fancy_names_index["YrotBD"], DEBUT] = 2.9

    x_bounds[0].min[fancy_names_index["ZrotBD"], DEBUT] = 0

    x_bounds[0].max[fancy_names_index["ZrotBD"], DEBUT] = 0


    # bras gauche
    x_bounds[0].min[fancy_names_index["YrotBG"], DEBUT] = -2.9  # debut bras aux oreilles

    x_bounds[0].max[fancy_names_index["YrotBG"], DEBUT] = -2.9

    x_bounds[0].min[fancy_names_index["ZrotBG"], DEBUT] = 0

    x_bounds[0].max[fancy_names_index["ZrotBG"], DEBUT] = 0


    # coude droit

    x_bounds[0].min[fancy_names_index["ZrotABD"] + fancy_names_index["XrotABD"], DEBUT] = 0

    x_bounds[0].max[fancy_names_index["ZrotABD"] + fancy_names_index["XrotABD"], DEBUT] = 0


    # coude gauche
    x_bounds[0].min[fancy_names_index["ZrotABG"] + fancy_names_index["XrotABG"], DEBUT] = 0

    x_bounds[0].max[fancy_names_index["ZrotABG"] + fancy_names_index["XrotABG"], DEBUT] = 0


    # le carpe
    x_bounds[0].min[fancy_names_index["XrotC"], DEBUT] = -0.50  # depart un peu ferme aux hanches

    x_bounds[0].max[fancy_names_index["XrotC"], DEBUT] = -0.50

    x_bounds[0].min[fancy_names_index["XrotC"], FIN] = -2.5      #-2.35

    # x_bounds[0].max[fancy_names_index["XrotC"], FIN] = -2.35


    # le dehanchement
    x_bounds[0].min[fancy_names_index["YrotC"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["YrotC"], DEBUT] = 0
    x_bounds[0].min[fancy_names_index["YrotC"], MILIEU:] = -0.1

    x_bounds[0].max[fancy_names_index["YrotC"], MILIEU:] = 0.1

    # Contraintes de vitesse: PHASE 0 la montee en carpe

    vzinit = 9.81 * final_time / 2  # vitesse initiale en z du CoM pour revenir a terre au temps final

    # en xy bassin
    for i in range(nb_models):
        x_bounds[0].min[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, :] =-10

        x_bounds[0].max[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, :] = 10
        x_bounds[0].min[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, DEBUT] = -0.5

        x_bounds[0].max[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, DEBUT] = 0.5


        # z bassin
        x_bounds[0].min[fancy_names_index["vZ"][i], :] = -50
        x_bounds[0].max[fancy_names_index["vZ"][i], :] = 50

        x_bounds[0].min[fancy_names_index["vZ"][i], DEBUT] = vzinit - 0.5

        x_bounds[0].max[fancy_names_index["vZ"][i], DEBUT] = vzinit + 0.5


        # autour de x
        x_bounds[0].min[fancy_names_index["vXrot"][i], :] = 0.5  # d'apres une observation video

        x_bounds[0].max[fancy_names_index["vXrot"][i], :] = 20  # aussi vite que nécessaire, mais ne devrait pas atteindre cette vitesse



        # autour de y
        x_bounds[0].min[fancy_names_index["vYrot"][i], :] = -50
        x_bounds[0].max[fancy_names_index["vYrot"][i], :] = 50

        x_bounds[0].min[fancy_names_index["vYrot"][i], DEBUT] = 0

        x_bounds[0].max[fancy_names_index["vYrot"][i], DEBUT] = 0


        # autour de z
        x_bounds[0].min[fancy_names_index["vZrot"][i], :] = -50

        x_bounds[0].max[fancy_names_index["vZrot"][i], :] = 50

        x_bounds[0].min[fancy_names_index["vZrot"][i], DEBUT] = 0

        x_bounds[0].max[fancy_names_index["vZrot"][i], DEBUT] = 0

        # decalage entre le bassin et le CoM
        CoM_Q_sym = cas.MX.sym("CoM", nb_q_per_model)
        CoM_Q_init = x_bounds[0].min[
                     i*nb_q_per_model: (i+1)*nb_q_per_model, DEBUT
                     ]  # min ou max ne change rien a priori, au DEBUT ils sont egaux normalement
        CoM_Q_func = cas.Function("CoM_Q_func", [CoM_Q_sym], [biorbd_models[0].models[i].model.CoM(CoM_Q_sym).to_mx()])
        bassin_Q_func = cas.Function(
            "bassin_Q_func", [CoM_Q_sym], [biorbd_models[0].models[i].model.globalJCS(CoM_Q_sym, 0).to_mx()]
        )  # retourne la RT du bassin

        r = (
                np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1, :3]
        )  # selectionne seulement la translation de la RT
        # tenir compte du decalage entre bassin et CoM avec la rotation
        # Qtransdot = Qtransdot + v cross Qrotdot

        borne_inf = (
            x_bounds[0].min[fancy_names_index["vX"][i] : fancy_names_index["vZ"][i]+1, DEBUT]
            + np.cross(r, x_bounds[0].min[fancy_names_index["vXrot"][i]: fancy_names_index["vZrot"][i]+1 , DEBUT])
        )[0]
        borne_sup = (
            x_bounds[0].max[fancy_names_index["vX"][i]: fancy_names_index["vZ"][i] +1, DEBUT]
            + np.cross(r, x_bounds[0].max[fancy_names_index["vXrot"][i]: fancy_names_index["vZrot"][i]+1, DEBUT])
        )[0]
        x_bounds[0].min[fancy_names_index["vX"][i]:fancy_names_index["vZ"][i]+1, DEBUT] = (
            min(borne_sup[0], borne_inf[0]),
            min(borne_sup[1], borne_inf[1]),
            min(borne_sup[2], borne_inf[2]),
        )
        x_bounds[0].max[fancy_names_index["vX"][i] : fancy_names_index["vZ"][i]+1 , DEBUT] = (
            max(borne_sup[0], borne_inf[0]),
            max(borne_sup[1], borne_inf[1]),
            max(borne_sup[2], borne_inf[2]),
        )

    # bras droit
    x_bounds[0].min[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], :] = -50
    x_bounds[0].max[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], :] = 50
    x_bounds[0].min[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], DEBUT] = 0


    # bras droit
    x_bounds[0].min[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], :] = -50
    x_bounds[0].max[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], :] = 50
    x_bounds[0].min[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], DEBUT] = 0

    # coude droit
    x_bounds[0].min[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], :] = -50
    x_bounds[0].max[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], :] = 50
    x_bounds[0].min[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], DEBUT] = 0

    # coude gauche
    x_bounds[0].min[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], :] = -50
    x_bounds[0].max[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], :] = 50
    x_bounds[0].min[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], DEBUT] = 0


    # du carpe
    x_bounds[0].min[fancy_names_index["vXrotC"], :] = -50
    x_bounds[0].max[fancy_names_index["vXrotC"], :] = 50
    x_bounds[0].min[fancy_names_index["vXrotC"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vXrotC"], DEBUT] = 0

    # du dehanchement
    x_bounds[0].min[fancy_names_index["vYrotC"], :] = -50
    x_bounds[0].max[fancy_names_index["vYrotC"], :] = 50
    x_bounds[0].min[fancy_names_index["vYrotC"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vYrotC"], DEBUT] = 0

    #
    # Contraintes de position: PHASE 1 le salto carpe
    #

    # deplacement
    for i in range(nb_models) :
        x_bounds[1].min[fancy_names_index["X"][i], :] = -0.1
        x_bounds[1].max[fancy_names_index["X"][i], :] = 0.1
        x_bounds[1].min[fancy_names_index["Y"][i], :] = -1.0
        x_bounds[1].max[fancy_names_index["Y"][i], :] = 1.0
        x_bounds[1].min[fancy_names_index["Z"][i], :] = 0
        x_bounds[1].max[
            fancy_names_index["Z"][i] , :
        ] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

        # le salto autour de x
        x_bounds[1].min[fancy_names_index["Xrot"][i], :] = 0
        x_bounds[1].max[fancy_names_index["Xrot"][i], :] = 4 * 3.14 -0.5
        x_bounds[1].min[fancy_names_index["Xrot"][i], FIN] = 2 * 3.14 - 0.1

        # limitation du tilt autour de y
        x_bounds[1].min[fancy_names_index["Yrot"][i], :] = -3.14 / 16
        x_bounds[1].max[fancy_names_index["Yrot"][i], :] = 3.14 / 16

        # la vrille autour de z
        x_bounds[1].min[fancy_names_index["Zrot"][i], :] = -0.1
        x_bounds[1].max[fancy_names_index["Zrot"][i], :] = 0.1

    # bras f4a a l'ouverture

    # le carpe
    x_bounds[1].min[fancy_names_index["XrotC"], :] = -2.5
    # x_bounds[1].max[fancy_names_index["XrotC"]  :] = -2.35 + 0.1

    # le dehanchement
    x_bounds[1].min[fancy_names_index["YrotC"], DEBUT] = -0.1
    x_bounds[1].max[fancy_names_index["YrotC"], DEBUT] = 0.1

    # Contraintes de vitesse: PHASE 1 le salto carpe

    for i in range(nb_models):
        # en xy bassin
        x_bounds[1].min[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, :] = -10
        x_bounds[1].max[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, :] = 10

        # z bassin
        x_bounds[1].min[fancy_names_index["vZ"][i], :] = -50
        x_bounds[1].max[fancy_names_index["vZ"][i], :] = 50

        # autour de x
        x_bounds[1].min[fancy_names_index["vXrot"][i], :] = -50
        x_bounds[1].max[fancy_names_index["vXrot"][i], :] = 50

        # autour de y
        x_bounds[1].min[fancy_names_index["vYrot"][i], :] = -50
        x_bounds[1].max[fancy_names_index["vYrot"][i], :] = 50

        # autour de z
        x_bounds[1].min[fancy_names_index["vZrot"][i], :] = -50
        x_bounds[1].max[fancy_names_index["vZrot"][i], :] = 50

    # bras droit
    x_bounds[1].min[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], :] = -50
    x_bounds[1].max[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], :] = 50

    # bras droit
    x_bounds[1].min[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], :] = -50
    x_bounds[1].max[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], :] = 50

    # coude droit
    x_bounds[1].min[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], :] = -50
    x_bounds[1].max[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], :] = 50
    # coude gauche
    x_bounds[1].min[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], :] = -50
    x_bounds[1].max[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], :] = 50

    # du carpe
    x_bounds[1].min[fancy_names_index["vXrotC"], :] = -50
    x_bounds[1].max[fancy_names_index["vXrotC"], :] = 50

    # du dehanchement
    x_bounds[1].min[fancy_names_index["vYrotC"], :] = -50
    x_bounds[1].max[fancy_names_index["vYrotC"], :] = 50

    #
    # Contraintes de position: PHASE 2 l'ouverture
    #

    for i in range(nb_models):
    # deplacement
        x_bounds[2].min[fancy_names_index["X"][i], :] = -0.2
        x_bounds[2].max[fancy_names_index["X"][i], :] = 0.2
        x_bounds[2].min[fancy_names_index["Y"][i], :] = -1.0
        x_bounds[2].max[fancy_names_index["Y"][i], :] = 1.0
        x_bounds[2].min[fancy_names_index["Z"][i], :] = 0
        x_bounds[2].max[fancy_names_index["Z"][i], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

        # le salto autour de x
        x_bounds[2].min[fancy_names_index["Xrot"][i], :] = 2 * 3.14 -1 #0.1  # 1 salto 3/4
        x_bounds[2].max[fancy_names_index["Xrot"][i], :] = 4 * 3.14

        # limitation du tilt autour de y
        x_bounds[2].min[fancy_names_index["Yrot"][i], :] = -3.14 / 4
        x_bounds[2].max[fancy_names_index["Yrot"][i], :] = 3.14 / 4

        # la vrille autour de z
        x_bounds[2].min[fancy_names_index["Zrot"][i], :] = 0
        x_bounds[2].max[fancy_names_index["Zrot"][i], :] = 3 * 3.14

    # bras f4a a l'ouverture

    # le carpe
    x_bounds[2].min[fancy_names_index["XrotC"], FIN] = -0.4

    # le dehanchement f4a a l'ouverture

    # Contraintes de vitesse: PHASE 2 l'ouverture
    for i in range(nb_models):
        # en xy bassin
        x_bounds[2].min[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, :] = -10
        x_bounds[2].max[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, :] = 10

        # z bassin
        x_bounds[2].min[fancy_names_index["vZ"][i], :] = -50
        x_bounds[2].max[fancy_names_index["vZ"][i], :] = 50

        # autour de x
        x_bounds[2].min[fancy_names_index["vXrot"][i], :] = -50
        x_bounds[2].max[fancy_names_index["vXrot"][i], :] = 50

        # autour de y
        x_bounds[2].min[fancy_names_index["vYrot"][i], :] = -50
        x_bounds[2].max[fancy_names_index["vYrot"][i], :] = 50

        # autour de z
        x_bounds[2].min[fancy_names_index["vZrot"][i], :] = -50
        x_bounds[2].max[fancy_names_index["vZrot"][i], :] = 50

    # bras droit
    x_bounds[2].min[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], :] = -50
    x_bounds[2].max[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], :] = 50

    # bras droit
    x_bounds[2].min[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], :] = -50
    x_bounds[2].max[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], :] = 50

    # coude droit
    x_bounds[2].min[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], :] = -50
    x_bounds[2].max[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], :] = 50

    # coude gauche
    x_bounds[2].min[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], :] = -50
    x_bounds[2].max[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], :] = 50

    # du carpe
    x_bounds[2].min[fancy_names_index["vXrotC"], :] = -50
    x_bounds[2].max[fancy_names_index["vXrotC"], :] = 50

    # du dehanchement
    x_bounds[2].min[fancy_names_index["vYrotC"], :] = -50
    x_bounds[2].max[fancy_names_index["vYrotC"], :] = 50

    # #
    # Contraintes de position: PHASE 3 la vrille et demie
    #
    for i in range(nb_models):
        # deplacement
        x_bounds[3].min[fancy_names_index["X"][i], :] = -0.2
        x_bounds[3].max[fancy_names_index["X"][i], :] = 0.2
        x_bounds[3].min[fancy_names_index["Y"][i], :] = -1.0
        x_bounds[3].max[fancy_names_index["Y"][i], :] = 1.0
        x_bounds[3].min[fancy_names_index["Z"][i], :] = 0
        x_bounds[3].max[fancy_names_index["Z"][i], :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

        # le salto autour de x
        x_bounds[3].min[fancy_names_index["Xrot"][i], :] = 2 * 3.14 - 0.1
        x_bounds[3].max[fancy_names_index["Xrot"][i], :] = 2 * 3.14 + 3 / 2 * 3.14 + 0.1  # 1 salto 3/4
        x_bounds[3].min[fancy_names_index["Xrot"][i], FIN] = 2 * 3.14 + 3 / 2 * 3.14 - 0.1
        x_bounds[3].max[fancy_names_index["Xrot"][i], FIN] = 2 * 3.14 + 3 / 2 * 3.14 + 0.1  # 1 salto 3/4

        # limitation du tilt autour de y
        x_bounds[3].min[fancy_names_index["Yrot"][i], :] = -3.14 / 4
        x_bounds[3].max[fancy_names_index["Yrot"][i], :] = 3.14 / 4
        x_bounds[3].min[fancy_names_index["Yrot"][i], FIN] = -3.14 / 8
        x_bounds[3].max[fancy_names_index["Yrot"][i], FIN] = 3.14 / 8

        # la vrille autour de z
        x_bounds[3].min[fancy_names_index["Zrot"][i], :] = 0
        x_bounds[3].max[fancy_names_index["Zrot"][i], :] = 3 * 3.14
        x_bounds[3].min[fancy_names_index["Zrot"][i], FIN] = 3 * 3.14 - 0.1  # complete la vrille
        x_bounds[3].max[fancy_names_index["Zrot"][i], FIN] = 3 * 3.14 + 0.1

    # bras f4a la vrille

    # le carpe
    x_bounds[3].min[fancy_names_index["XrotC"], :] = -0.4

    # le dehanchement f4a la vrille

    # Contraintes de vitesse: PHASE 3 la vrille et demie
    for i in range(nb_models):
        # en xy bassin
        x_bounds[3].min[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, :] = -10
        x_bounds[3].max[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, :] = 10

        # z bassin
        x_bounds[3].min[fancy_names_index["vZ"][i], :] = -50
        x_bounds[3].max[fancy_names_index["vZ"][i], :] = 50

        # autour de x
        x_bounds[3].min[fancy_names_index["vXrot"][i], :] = -50
        x_bounds[3].max[fancy_names_index["vXrot"][i], :] = 50

        # autour de y
        x_bounds[3].min[fancy_names_index["vYrot"][i], :] = -50
        x_bounds[3].max[fancy_names_index["vYrot"][i], :] = 50

        # autour de z
        x_bounds[3].min[fancy_names_index["vZrot"][i], :] = -50
        x_bounds[3].max[fancy_names_index["vZrot"][i], :] = 50

    # bras droit
    x_bounds[3].min[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], :] = -50
    x_bounds[3].max[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], :] = 50

    # bras droit
    x_bounds[3].min[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], :] = -50
    x_bounds[3].max[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], :] = 50

    # coude droit
    x_bounds[3].min[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], :] = -50
    x_bounds[3].max[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], :] = 50

    # coude gauche
    x_bounds[3].min[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], :] = -50
    x_bounds[3].max[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], :] = 50

    # du carpe
    x_bounds[3].min[fancy_names_index["vXrotC"], :] = -50
    x_bounds[3].max[fancy_names_index["vXrotC"], :] = 50

    # du dehanchement
    x_bounds[3].min[fancy_names_index["vYrotC"], :] = -50
    x_bounds[3].max[fancy_names_index["vYrotC"], :] = 50

    # #
    # Contraintes de position: PHASE 4 la reception
    #
    for i in range(nb_models) :
        # deplacement
        x_bounds[4].min[fancy_names_index["X"][i], :] = -0.1
        x_bounds[4].max[fancy_names_index["X"][i], :] = 0.1
        x_bounds[4].min[fancy_names_index["Y"][i], FIN] = -0.1
        x_bounds[4].max[fancy_names_index["Y"][i], FIN] = 0.1
        x_bounds[4].min[fancy_names_index["Z"][i], :] = 0
        x_bounds[4].max[
            fancy_names_index["Z"][i], :
        ] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne
        x_bounds[4].min[fancy_names_index["Z"][i], FIN] = 0
        x_bounds[4].max[fancy_names_index["Z"][i], FIN] = 0.1

        # le salto autour de x
        x_bounds[4].min[fancy_names_index["Xrot"][i], :] = 2 * 3.14 + 3 / 2 * 3.14 - 0.2  # penche vers avant -> moins de salto # -0.2
        x_bounds[4].max[fancy_names_index["Xrot"][i], :] = -0.50 + 4 * 3.14  # un peu carpe a la fin
        x_bounds[4].min[fancy_names_index["Xrot"][i], FIN] = -0.50 + 4 * 3.14 -0.1
        x_bounds[4].max[fancy_names_index["Xrot"][i], FIN] = -0.50 + 4 * 3.14 + 0.1  # 2 salto fin un peu carpe

        # limitation du tilt autour de y
        x_bounds[4].min[fancy_names_index["Yrot"][i], :] = -3.14 / 16    #-3.14/16
        x_bounds[4].max[fancy_names_index["Yrot"][i], :] = 3.14 / 16

        # la vrille autour de z
        x_bounds[4].min[fancy_names_index["Zrot"][i], :] = 3 * 3.14 - 0.1  # complete la vrille  #3 * 3.14 - 0.1
        x_bounds[4].max[fancy_names_index["Zrot"][i], :] = 3 * 3.14 + 0.1

    # bras droit
    x_bounds[4].min[fancy_names_index["YrotBD"], FIN] = 2.9 - 0.1  # debut bras aux oreilles
    x_bounds[4].max[fancy_names_index["YrotBD"], FIN] = 2.9 + 0.1
    x_bounds[4].min[fancy_names_index["ZrotBD"], FIN] = -0.1
    x_bounds[4].max[fancy_names_index["ZrotBD"], FIN] = 0.1

    # bras gauche
    x_bounds[4].min[fancy_names_index["YrotBG"], FIN] = -2.9 - 0.1  # debut bras aux oreilles
    x_bounds[4].max[fancy_names_index["YrotBG"], FIN] = -2.9 + 0.1
    x_bounds[4].min[fancy_names_index["ZrotBG"], FIN] = -0.1
    x_bounds[4].max[fancy_names_index["ZrotBG"], FIN] = 0.1

    # coude droit
    x_bounds[4].min[fancy_names_index["ZrotABD"] + fancy_names_index["XrotABD"], FIN] = -0.1
    x_bounds[4].max[fancy_names_index["ZrotABD"] + fancy_names_index["XrotABD"], FIN] = 0.1

    # coude gauche
    x_bounds[4].min[fancy_names_index["ZrotABG"] + fancy_names_index["XrotABG"], FIN] = -0.1
    x_bounds[4].max[fancy_names_index["ZrotABG"] + fancy_names_index["XrotABG"], FIN] = 0.1

    # le carpe
    x_bounds[4].min[fancy_names_index["XrotC"], :] = -0.4
    x_bounds[4].min[fancy_names_index["XrotC"], FIN] = -0.60
    x_bounds[4].max[fancy_names_index["XrotC"], FIN] = -0.40  # fin un peu carpe

    # le dehanchement
    x_bounds[4].min[fancy_names_index["YrotC"], FIN] = -0.1
    x_bounds[4].max[fancy_names_index["YrotC"], FIN] = 0.1

    # Contraintes de vitesse: PHASE 4 la reception

    for i in range(nb_models) :

        # en xy bassin
        x_bounds[4].min[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, :] = -10
        x_bounds[4].max[fancy_names_index["vX"][i]: fancy_names_index["vY"][i] +1, :] = 10

        # z bassin
        x_bounds[4].min[fancy_names_index["vZ"][i], :] = -50
        x_bounds[4].max[fancy_names_index["vZ"][i], :] = 50

        # autour de x
        x_bounds[4].min[fancy_names_index["vXrot"][i], :] = -50
        x_bounds[4].max[fancy_names_index["vXrot"][i], :] = 50

        # autour de y
        x_bounds[4].min[fancy_names_index["vYrot"][i], :] = -50
        x_bounds[4].max[fancy_names_index["vYrot"][i], :] = 50

        # autour de z
        x_bounds[4].min[fancy_names_index["vZrot"][i], :] = -50
        x_bounds[4].max[fancy_names_index["vZrot"][i], :] = 50

    # bras droit
    x_bounds[4].min[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], :] = -50
    x_bounds[4].max[fancy_names_index["vZrotBD"] + fancy_names_index["vYrotBD"], :] = 50

    # bras droit
    x_bounds[4].min[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], :] = -50
    x_bounds[4].max[fancy_names_index["vZrotBG"] + fancy_names_index["vYrotBG"], :] = 50

    # coude droit
    x_bounds[4].min[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], :] = -50
    x_bounds[4].max[fancy_names_index["vZrotABD"] + fancy_names_index["vYrotABD"], :] = 50

    # coude gauche
    x_bounds[4].min[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], :] = -50
    x_bounds[4].max[fancy_names_index["vZrotABG"] + fancy_names_index["vYrotABG"], :] = 50

    # du carpe
    x_bounds[4].min[fancy_names_index["vXrotC"], :] = -50
    x_bounds[4].max[fancy_names_index["vXrotC"], :] = 50

    # du dehanchement
    x_bounds[4].min[fancy_names_index["vYrotC"], :] = -50
    x_bounds[4].max[fancy_names_index["vYrotC"], :] = 50

    x_bounds_real = BoundsList()
    nb_q = biorbd_models[0].nb_q
    for phase in range(len(x_bounds.options)):
        # for idx_model in range(len(biorbd_models)):
        x_min = list(mappings['q'].to_first.map(x_bounds[phase].min[:nb_q]))
        x_min+= list(mappings['qdot'].to_first.map(x_bounds[phase].min[nb_q: ]))
        x_max = list(mappings['q'].to_first.map(x_bounds[phase].max[:nb_q]))
        x_max+= list(mappings['qdot'].to_first.map(x_bounds[phase].max[nb_q: ]))

        x_bounds_real.add(bounds=Bounds(np.array(x_min), np.array(x_max)))
    return x_bounds_real


def set_x_init( biorbd_models, fancy_names_index, mappings):
    nb_q = biorbd_models[0].nb_q
    nb_qdot = biorbd_models[0].nb_qdot
    nb_models = len(biorbd_models[0].models)
    x_init = InitialGuessList()
    x0 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x1 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x2 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x3 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x4 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))

    for i in range(nb_models):
        x0[fancy_names_index["Xrot"][i], 0] = 0.50
    x0[fancy_names_index["ZrotBG"]] = -0.75
    x0[fancy_names_index["ZrotBD"]] = 0.75
    x0[fancy_names_index["YrotBG"], 0] = -2.9
    x0[fancy_names_index["YrotBD"], 0] = 2.9
    x0[fancy_names_index["YrotBG"], 1] = -1.35
    x0[fancy_names_index["YrotBD"], 1] = 1.35
    x0[fancy_names_index["XrotC"], 0] = -0.5
    x0[fancy_names_index["XrotC"], 1] = -2.6



    x1[fancy_names_index["ZrotBG"]] = -0.75
    x1[fancy_names_index["ZrotBD"]] = 0.75
    for i in range(nb_models):
        x1[fancy_names_index["Xrot"][i], 1] = 2 * 3.14
    x1[fancy_names_index["YrotBG"]] = -1.35
    x1[fancy_names_index["YrotBD"]] = 1.35
    x1[fancy_names_index["XrotC"]] = -2.6

    for i in range(nb_models):
        x2[fancy_names_index["Xrot"][i]] = 2 * 3.14
        x2[fancy_names_index["Zrot"][i], 1] = 3.14
    x2[fancy_names_index["ZrotBG"], 0] = -0.75
    x2[fancy_names_index["ZrotBD"], 0] = 0.75
    x2[fancy_names_index["YrotBG"], 0] = -1.35
    x2[fancy_names_index["YrotBD"], 0] = 1.35
    x2[fancy_names_index["XrotC"], 0] = -2.6

    for i in range(nb_models):
        x3[fancy_names_index["Xrot"][i], 0] = 2 * 3.14
        x3[fancy_names_index["Xrot"][i], 1] = 2 * 3.14 + 3 / 2 * 3.14
        x3[fancy_names_index["Zrot"][i], 0] = 3.14
        x3[fancy_names_index["Zrot"][i], 1] = 3 * 3.14

        x4[fancy_names_index["Xrot"][i], 0] = 2 * 3.14 + 3 / 2 * 3.14
        x4[fancy_names_index["Xrot"][i] , 1] = 4 * 3.14
        x4[fancy_names_index["Zrot"][i]] = 3 * 3.14
    x4[fancy_names_index["XrotC"], 1] = -0.5

    nb_q = biorbd_models[0].nb_q

    x0_mapped = list(mappings['q'].to_first.map(x0[:nb_q]))
    x0_mapped += list(mappings['q'].to_first.map(x0[nb_q:]))
    x1_mapped = list(mappings['q'].to_first.map(x1[:nb_q]))
    x1_mapped += list(mappings['q'].to_first.map(x1[nb_q:]))
    x2_mapped = list(mappings['q'].to_first.map(x2[:nb_q]))
    x2_mapped += list(mappings['q'].to_first.map(x2[nb_q:]))
    x3_mapped = list(mappings['q'].to_first.map(x3[:nb_q]))
    x3_mapped += list(mappings['q'].to_first.map(x3[nb_q:]))
    x4_mapped = list(mappings['q'].to_first.map(x4[:nb_q]))
    x4_mapped += list(mappings['q'].to_first.map(x4[nb_q:]))

    x_init.add(np.array(x0_mapped), interpolation=InterpolationType.LINEAR)
    x_init.add(np.array(x1_mapped), interpolation=InterpolationType.LINEAR)
    x_init.add(np.array(x2_mapped), interpolation=InterpolationType.LINEAR)
    x_init.add(np.array(x3_mapped), interpolation=InterpolationType.LINEAR)
    x_init.add(np.array(x4_mapped), interpolation=InterpolationType.LINEAR)


    return x_init

def prepare_ocp(
    model_paths : tuple ,
    n_shooting: int,
    final_time: float,
    n_threads: int,
    ode_solver: OdeSolver = OdeSolver.RK4(),
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

    biorbd_models = [MultiBiorbdModel(model_paths), MultiBiorbdModel(model_paths), MultiBiorbdModel(model_paths), MultiBiorbdModel(model_paths), MultiBiorbdModel(model_paths)]

    #mapping partout sauf sur les racines
    nb_q = biorbd_models[0][0].nb_q
    nb_root = biorbd_models[0][0].nb_root
    nb_qddot_joints = nb_q - nb_root
    nb_models = len(biorbd_models[0].models)

    fancy_names_index = set_fancy_names_index(biorbd_models)

    global q_to_first
    global q_to_second
    q_to_first = []
    q_to_second = []
    current_index = 0
    for i in range(nb_models):
        q_to_first += list(range(current_index, current_index + nb_root))
        q_to_second += list(range(current_index, current_index + nb_root))
        current_index += nb_root
        if i == 0:
            q_to_first += list(range(current_index, current_index + nb_qddot_joints))
            current_index += nb_qddot_joints
        q_to_second += list(range(nb_root, nb_root+nb_qddot_joints))

    qddot_to_first = list(range(nb_qddot_joints))
    qddot_to_second = list(range(nb_qddot_joints)) * nb_models

    mappings = BiMappingList()
    mappings.add("q", to_first=q_to_first, to_second=q_to_second)
    mappings.add("qdot", to_first=q_to_first, to_second=q_to_second)
    mappings.add("qddot_joints", to_first=qddot_to_first, to_second=qddot_to_second)

    # Add objective functions
    objective_functions = ObjectiveList()
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


    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.0, max_bound=1.0, weight=1000000, phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.0, max_bound=1.0, weight=-100, phase=1  # 100000
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.0, max_bound=final_time, weight=1, phase=2
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.0, max_bound=final_time, weight=1, phase=3
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.0, max_bound=final_time, weight=1, phase=4
    )

    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key='q', index=[mappings['q'].to_second.map_idx[i] for i in fancy_names_index['Xrot']], node=Node.END, weight=1000, phase=4)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key='q', index=[mappings['q'].to_second.map_idx[i] for i in fancy_names_index['Yrot']],  node=Node.END, weight=1000, phase=4)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key='q', index=[mappings['q'].to_second.map_idx[i] for i in fancy_names_index['Zrot']], node=Node.END, weight=1000, phase=4)


    # Les hanches sont fixes a +-0.2 en bounds, mais les mains doivent quand meme être proches des jambes
    # for index_model, model in enumerate(biorbd_models[0].models):
    objective_functions.add(
        superimpose_markers,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        weight=1000,
        phase=0,
    )

    objective_functions.add(
        superimpose_markers,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.ALL,
        weight=1000,
        phase=1,
    )

    # arrete de gigoter les bras
    les_bras = [fancy_names_index["ZrotBD"][0],
                fancy_names_index["YrotBD"][0],
                fancy_names_index["ZrotABD"][0],
                fancy_names_index["XrotABD"][0],
                fancy_names_index["ZrotBG"][0],
                fancy_names_index["YrotBG"][0],
                fancy_names_index["ZrotABG"][0],
                fancy_names_index["XrotABG"][0]]

    les_coudes = [fancy_names_index["ZrotABD"][0],
                  fancy_names_index["XrotABD"][0],
                  fancy_names_index["ZrotABG"][0],
                  fancy_names_index["XrotABG"][0]]

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

    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        dofs=fancy_names_index["XrotC"][0],
        targets=[0],
        weight=10000,
        phase=3,
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)

    x_bounds = set_x_bounds(biorbd_models, fancy_names_index, final_time, mappings)

    qddot_joints_min, qddot_joints_max, qddot_joints_init = -500, 500, 0
    u_bounds = BoundsList()


    u_bounds.add(mappings['qddot_joints'].to_first.map([qddot_joints_min] * nb_qddot_joints), mappings['qddot_joints'].to_first.map([qddot_joints_max] * nb_qddot_joints))
    u_bounds.add(mappings['qddot_joints'].to_first.map([qddot_joints_min] * nb_qddot_joints), mappings['qddot_joints'].to_first.map([qddot_joints_max] * nb_qddot_joints))
    u_bounds.add(mappings['qddot_joints'].to_first.map([qddot_joints_min] * nb_qddot_joints), mappings['qddot_joints'].to_first.map([qddot_joints_max] * nb_qddot_joints))
    u_bounds.add(mappings['qddot_joints'].to_first.map([qddot_joints_min] * nb_qddot_joints), mappings['qddot_joints'].to_first.map([qddot_joints_max] * nb_qddot_joints))
    u_bounds.add(mappings['qddot_joints'].to_first.map([qddot_joints_min] * nb_qddot_joints), mappings['qddot_joints'].to_first.map([qddot_joints_max] * nb_qddot_joints))



    u_init = InitialGuessList()
    u_init.add(mappings['qddot_joints'].to_first.map([qddot_joints_init] * nb_qddot_joints))
    u_init.add(mappings['qddot_joints'].to_first.map([qddot_joints_init] * nb_qddot_joints))
    u_init.add(mappings['qddot_joints'].to_first.map([qddot_joints_init] * nb_qddot_joints))
    u_init.add(mappings['qddot_joints'].to_first.map([qddot_joints_init] * nb_qddot_joints))
    u_init.add(mappings['qddot_joints'].to_first.map([qddot_joints_init] * nb_qddot_joints))

    x_init = set_x_init(biorbd_models, fancy_names_index, mappings)


    constraints = ConstraintList()
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=1.5, phase=1)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=1.5, phase=2)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=0.7, phase=3)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=0.5, phase=4)

    constraints.add(superimpose_markers_constraint, node=Node.ALL_SHOOTING, min_bound=0, max_bound=0.15**2, phase=1)

    return OptimalControlProgram(
        biorbd_models,
        dynamics,
        n_shooting,
        [final_time /5 ] * 5,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        n_threads=n_threads,
        variable_mappings=mappings,
        assume_phase_dynamics=True
    )


def main():
    model_paths = ("Models/Models_Lisa/Athlete_03.bioMod",
                   "Models/Models_Lisa/Athlete_05.bioMod",
                   "Models/Models_Lisa/Athlete_18.bioMod",
                   "Models/Models_Lisa/Athlete_07.bioMod",
                   # "Models/Models_Lisa/Athlete_14.bioMod",
                   # "Models/Models_Lisa/Athlete_17.bioMod",
                   # "Models/Models_Lisa/Athlete_02.bioMod",
                   # "Models/Models_Lisa/Athlete_06.bioMod",
                   # "Models/Models_Lisa/Athlete_11.bioMod",
                   )

    n_threads = 28

    print_ocp_FLAG = False  # True.

    show_online_FLAG = False # True
    HSL_FLAG = True
    save_sol_FLAG = True

    n_shooting = (40, 100, 100, 100, 40)

    ocp = prepare_ocp(model_paths, n_shooting=n_shooting, n_threads=n_threads, final_time=1.87)
    ocp.add_plot_penalty(CostType.ALL)
    if print_ocp_FLAG:
        ocp.print(to_graph=True)
    solver = Solver.IPOPT(show_online_optim=show_online_FLAG, show_options=dict(show_bounds=True))
    if HSL_FLAG:
        solver.set_linear_solver("ma57")
    else:
        print("Not using ma57")
    solver.set_maximum_iterations(10000)
    solver.set_convergence_tolerance(1e-4)
    sol = ocp.solve(solver)

    temps = time.strftime("%Y-%m-%d-%H%M")
    nom = "MultiModel solutions "
    # il faut recuperer les Q de toutes les phases la onrecupere que lq premiere phase
    qs = sol.states[0]['q']
    qdots = sol.states[0]['qdot']
    q_mapped =[]
    for i in range(1, len(sol.states)):
        qs = np.hstack((qs, sol.states[i]["q"]))
        qdots = np.hstack((qdots, sol.states[i]["qdot"]))

    for i, index in enumerate(q_to_second):
        q_mapped.append(qs[index, :])

    dict_sol = {}
    dict_sol['q'] = qs
    dict_sol['qdot'] = qdots
    dict_sol['q_mapped'] = q_mapped
    dict_sol['mapping'] = {'to_first': q_to_first, 'to_second': q_to_second}
    del sol.ocp
    dict_sol['sol'] = sol

    import pickle
    save_name = ''
    for model_name in model_paths:
        save_name += model_name[-11:-7]
        save_name += '_'
    if sol.status == 0:
        save_name += 'CVG'
    else:
        save_name += 'DVG'
    save_name += '.pkl'

    path = 'Solutions_MultiModel'
    with open(f'{path}/{save_name}', 'wb') as f:
        pickle.dump(dict_sol, f)


    if IPYTHON:
        IPython.embed()  # afin de pouvoir explorer plus en details la solution

    # Print the last solution
    #sol.animate(n_frames=-1, show_floor=False)

    # sol.graphs(show_bounds=True)



if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    # main()
