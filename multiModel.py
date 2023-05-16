
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
    PenaltyNodeList,
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
        all_pn: PenaltyNodeList,
):
    first_marker_D = 1
    second_marker_D = 5
    first_marker_G = 3
    second_marker_G = 6

    nlp = all_pn.nlp

    total_diff = []
    for index_model, biorbd_model in enumerate(nlp.model.models):

        q = nlp.states[0]["q"].mx[nlp.model.variable_index('q', index_model)]
        diff_markers_D = nlp.model.models[index_model].marker(q, second_marker_D).to_mx() - nlp.model.models[
            index_model].marker(q, first_marker_D).to_mx()
        diff_markers_G = nlp.model.models[index_model].marker(q, second_marker_G).to_mx() - nlp.model.models[
            index_model].marker(q, first_marker_G).to_mx()
        sum_diff_D = 0
        sum_diff_G = 0
        for i in range(3):
            sum_diff_D += (diff_markers_D[i]) ** 2
            sum_diff_G += (diff_markers_G[i]) ** 2
        total_diff += [sum_diff_D, sum_diff_G]
    return nlp.mx_to_cx(
        f"diff_markers",
        cas.vertcat(*total_diff),
        nlp.states[0]["q"],
    )

def superimpose_markers(
        all_pn: PenaltyNodeList,
):
    first_marker_D = 1
    second_marker_D = 5
    first_marker_G = 3
    second_marker_G = 6

    nlp = all_pn.nlp

    total_diff = 0
    for index_model, biorbd_model in enumerate(nlp.model.models):
        q = nlp.states[0]["q"].mx[nlp.model.variable_index('q', index_model) ]
        diff_markers_D = nlp.model.models[index_model].marker(q, second_marker_D).to_mx() - nlp.model.models[index_model].marker(q, first_marker_D).to_mx()
        diff_markers_G = nlp.model.models[index_model].marker(q, second_marker_G).to_mx() - nlp.model.models[index_model].marker(q, first_marker_G).to_mx()
        for i in range(3):
            total_diff += (diff_markers_D[i])**2
            total_diff += (diff_markers_G[i])**2


    return nlp.mx_to_cx(
        f"diff_markers",
        total_diff,
        nlp.states[0]["q"],
    )


def minimize_dofs(all_pn: PenaltyNodeList, dofs: list, targets: list) -> cas.MX:
    diff = 0
    if isinstance(dofs, int):
        dofs = [dofs]
    for i, dof in enumerate(dofs):
        diff += (all_pn.nlp.states[0]["q"].mx[dof] - targets[i]) ** 2
    return all_pn.nlp.mx_to_cx("minimize_dofs", diff, all_pn.nlp.states[0]["q"])

def set_fancy_names_index(biorbd_models):
    """
    For readability
    """
    nb_model = len(biorbd_models[0].models)

    nb_q = biorbd_models[0].nb_q//nb_model
    fancy_names_index = {}
    fancy_names_index["X"] = [0+i*16 for i in range(nb_model)]
    fancy_names_index["Y"] = [1+i*16 for i in range(nb_model)]
    fancy_names_index["Z"] = [2+i*16 for i in range(nb_model)]
    fancy_names_index["Xrot"] = [3 + i*16 for i in range(nb_model)]
    fancy_names_index["Yrot"] = [4 + i*16 for i in range(nb_model)]
    fancy_names_index["Zrot"] = [5 + i*16 for i in range(nb_model)]
    fancy_names_index["ZrotBD"] = 6
    fancy_names_index["YrotBD"] = 7
    fancy_names_index["ZrotABD"] = 8
    fancy_names_index["XrotABD"] = 9
    fancy_names_index["ZrotBG"] = 10
    fancy_names_index["YrotBG"] = 11
    fancy_names_index["ZrotABG"] = 12
    fancy_names_index["XrotABG"] = 13
    fancy_names_index["XrotC"] = 14
    fancy_names_index["YrotC"] = 15
    fancy_names_index["vX"] = [0+nb_q*nb_model+16*i for i in range(nb_model)]
    fancy_names_index["vY"] = [1+nb_q*nb_model+i*16 for i in range(nb_model)]
    fancy_names_index["vZ"] = [2+nb_q*nb_model+i*16 for i in range(nb_model)]
    fancy_names_index["vXrot"] = [3+nb_q*nb_model+i*16 for i in range(nb_model)]
    fancy_names_index["vYrot"] = [4+nb_q*nb_model+i*16 for i in range(nb_model)]
    fancy_names_index["vZrot"] = [5+nb_q*nb_model+i*16 for i in range(nb_model)]
    fancy_names_index["vZrotBD"] = 6+nb_q*nb_model
    fancy_names_index["vYrotBD"] = 7+nb_q*nb_model
    fancy_names_index["vZrotABD"] = 8+nb_q*nb_model
    fancy_names_index["vYrotABD"] = 9+nb_q*nb_model
    fancy_names_index["vZrotBG"] = 10+nb_q*nb_model
    fancy_names_index["vYrotBG"] = 11+nb_q*nb_model
    fancy_names_index["vZrotABG"] = 12+nb_q*nb_model
    fancy_names_index["vYrotABG"] = 13+nb_q*nb_model
    fancy_names_index["vXrotC"] = 14+nb_q*nb_model
    fancy_names_index["vYrotC"] = 15+nb_q*nb_model

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
        x_bounds[0].min[: fancy_names_index["Z"][i]+1, DEBUT] = 0
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

    x_bounds[0].min[fancy_names_index["ZrotABD"]: fancy_names_index["XrotABD"]+1, DEBUT] = 0

    x_bounds[0].max[fancy_names_index["ZrotABD"]:fancy_names_index["XrotABD"]+1 , DEBUT] = 0


    # coude gauche
    x_bounds[0].min[fancy_names_index["ZrotABG"]: fancy_names_index["XrotABG"]+1, DEBUT] = 0

    x_bounds[0].max[fancy_names_index["ZrotABG"]: fancy_names_index["XrotABG"]+1, DEBUT] = 0


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

    vzinit = 9.81 / (2 * final_time)  # vitesse initiale en z du CoM pour revenir a terre au temps final

    # en xy bassin
    for i in range(nb_models):
        x_bounds[0].min[fancy_names_index["vX"][i]: fancy_names_index["vY"][i]+1 , :] =-10

        x_bounds[0].max[fancy_names_index["vX"][i]:fancy_names_index["vY"][i]+1, :] = 10
        x_bounds[0].min[fancy_names_index["vX"][i]: fancy_names_index["vY"][i]+1 , DEBUT] = -0.5

        x_bounds[0].max[fancy_names_index["vX"][i]: fancy_names_index["vY"][i]+1, DEBUT] = 0.5


    # z bassin
        x_bounds[0].min[fancy_names_index["vZ"][i], :] = -50
        x_bounds[0].max[fancy_names_index["vZ"][i], :] = 50

        x_bounds[0].min[fancy_names_index["vZ"][i], DEBUT] = vzinit - 0.5

        x_bounds[0].max[fancy_names_index["vZ"][i], DEBUT] = vzinit + 0.5


        # autour de x
        x_bounds[0].min[fancy_names_index["vXrot"][i], :] = 0.5  # d'apres une observation video

        x_bounds[0].max[
            fancy_names_index["vXrot"][i], :
        ] = 20  # aussi vite que nécessaire, mais ne devrait pas atteindre cette vitesse



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
        # A
        CoM_Q_sym = cas.MX.sym("CoM", nb_q_per_model)
        CoM_Q_init = x_bounds[0].min[
                     i*nb_q_per_model: (i+1)*nb_q_per_model, DEBUT
                     ]  # min ou max ne change rien a priori, au DEBUT ils sont egaux normalement
        CoM_Q_func = cas.Function("CoM_Q_func", [CoM_Q_sym], [biorbd_models[0].models[i].CoM(CoM_Q_sym).to_mx()])
        bassin_Q_func = cas.Function(
            "bassin_Q_func", [CoM_Q_sym], [biorbd_models[0].models[i].globalJCS(CoM_Q_sym, 0).to_mx()]
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
    x_bounds[0].min[fancy_names_index["vZrotBD"]: fancy_names_index["vYrotBD"] +1 , :] = -50
    x_bounds[0].max[fancy_names_index["vZrotBD"]: fancy_names_index["vYrotBD"]  +1, :] = 50
    x_bounds[0].min[fancy_names_index["vZrotBD"]: fancy_names_index["vYrotBD"]  +1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBD"]: fancy_names_index["vYrotBD"]  +1, DEBUT] = 0


    # bras droit
    x_bounds[0].min[fancy_names_index["vZrotBG"]: fancy_names_index["vYrotBG"] + 1 , :] = -50
    x_bounds[0].max[fancy_names_index["vZrotBG"]: fancy_names_index["vYrotBG"] + 1, :] = 50
    x_bounds[0].min[fancy_names_index["vZrotBG"]: fancy_names_index["vYrotBG"] + 1 , DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBG"]: fancy_names_index["vYrotBG"] + 1 , DEBUT] = 0

    # coude droit
    x_bounds[0].min[fancy_names_index["vZrotABD"]: fancy_names_index["vYrotABD"] + 1 , :] = -50
    x_bounds[0].max[fancy_names_index["vZrotABD"]: fancy_names_index["vYrotABD"] + 1 , :] = 50
    x_bounds[0].min[fancy_names_index["vZrotABD"]: fancy_names_index["vYrotABD"] + 1 , DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABD"]: fancy_names_index["vYrotABD"] + 1 , DEBUT] = 0

    # coude gauche
    x_bounds[0].min[fancy_names_index["vZrotABD"]: fancy_names_index["vYrotABG"] + 1, :] = -50
    x_bounds[0].max[fancy_names_index["vZrotABD"]: fancy_names_index["vYrotABG"] + 1 , :] = 50
    x_bounds[0].min[fancy_names_index["vZrotABG"]: fancy_names_index["vYrotABG"] + 1 , DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABG"]: fancy_names_index["vYrotABG"] + 1 , DEBUT] = 0


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
        x_bounds[1].min[fancy_names_index["Xrot"][i] , FIN] = 2 * 3.14 - 0.1

        # limitation du tilt autour de y
        x_bounds[1].min[fancy_names_index["Yrot"][i] , :] = -3.14 / 16
        x_bounds[1].max[fancy_names_index["Yrot"][i] , :] = 3.14 / 16

        # la vrille autour de z
        x_bounds[1].min[fancy_names_index["Zrot"][i] , :] = -0.1
        x_bounds[1].max[fancy_names_index["Zrot"][i] , :] = 0.1

    # bras f4a a l'ouverture

    # le carpe
    x_bounds[1].min[fancy_names_index["XrotC"] , :] = -2.5
    # x_bounds[1].max[fancy_names_index["XrotC"]  :] = -2.35 + 0.1

    # le dehanchement
    x_bounds[1].min[fancy_names_index["YrotC"] , DEBUT] = -0.1
    x_bounds[1].max[fancy_names_index["YrotC"] , DEBUT] = 0.1

    # Contraintes de vitesse: PHASE 1 le salto carpe

    for i in range(nb_models):
        # en xy bassin
        x_bounds[1].min[fancy_names_index["vX"][i] : fancy_names_index["vY"][i]+1 , :] = -10
        x_bounds[1].max[fancy_names_index["vX"][i] : fancy_names_index["vY"][i]+1 , :] = 10

        # z bassin
        x_bounds[1].min[fancy_names_index["vZ"][i], :] = -50
        x_bounds[1].max[fancy_names_index["vZ"][i], :] = 50

        # autour de x
        x_bounds[1].min[fancy_names_index["vXrot"][i] , :] = -50
        x_bounds[1].max[fancy_names_index["vXrot"][i] , :] = 50

        # autour de y
        x_bounds[1].min[fancy_names_index["vYrot"][i] , :] = -50
        x_bounds[1].max[fancy_names_index["vYrot"][i] , :] = 50

        # autour de z
        x_bounds[1].min[fancy_names_index["vZrot"][i] , :] = -50
        x_bounds[1].max[fancy_names_index["vZrot"][i] , :] = 50

    # bras droit
    x_bounds[1].min[fancy_names_index["vZrotBD"]  : fancy_names_index["vYrotBD"] + 1 , :] = -50
    x_bounds[1].max[fancy_names_index["vZrotBD"]  : fancy_names_index["vYrotBD"] + 1, :] = 50

    # bras droit
    x_bounds[1].min[fancy_names_index["vZrotBG"]  : fancy_names_index["vYrotBG"] + 1, :] = -50
    x_bounds[1].max[fancy_names_index["vZrotBG"]  : fancy_names_index["vYrotBG"] + 1 , :] = 50

    # coude droit
    x_bounds[1].min[fancy_names_index["vZrotABD"]  : fancy_names_index["vYrotABD"] + 1 , :] = -50
    x_bounds[1].max[fancy_names_index["vZrotABD"]  : fancy_names_index["vYrotABD"] + 1 , :] = 50
    # coude gauche
    x_bounds[1].min[fancy_names_index["vZrotABD"]  : fancy_names_index["vYrotABG"] + 1 , :] = -50
    x_bounds[1].max[fancy_names_index["vZrotABD"]  : fancy_names_index["vYrotABG"] + 1 , :] = 50

    # du carpe
    x_bounds[1].min[fancy_names_index["vXrotC"] , :] = -50
    x_bounds[1].max[fancy_names_index["vXrotC"] , :] = 50

    # du dehanchement
    x_bounds[1].min[fancy_names_index["vYrotC"] , :] = -50
    x_bounds[1].max[fancy_names_index["vYrotC"] , :] = 50

    #
    # Contraintes de position: PHASE 2 l'ouverture
    #

    for i in range(nb_models):
    # deplacement
        x_bounds[2].min[fancy_names_index["X"][i] , :] = -0.2
        x_bounds[2].max[fancy_names_index["X"][i] , :] = 0.2
        x_bounds[2].min[fancy_names_index["Y"][i] , :] = -1.0
        x_bounds[2].max[fancy_names_index["Y"][i] , :] = 1.0
        x_bounds[2].min[fancy_names_index["Z"][i] , :] = 0
        x_bounds[2].max[
            fancy_names_index["Z"][i] , :
        ] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

        # le salto autour de x
        x_bounds[2].min[fancy_names_index["Xrot"][i] , :] = 2 * 3.14 -1 #0.1  # 1 salto 3/4
        x_bounds[2].max[fancy_names_index["Xrot"][i] , :] = 4 * 3.14

        # limitation du tilt autour de y
        x_bounds[2].min[fancy_names_index["Yrot"][i] , :] = -3.14 / 4
        x_bounds[2].max[fancy_names_index["Yrot"][i] , :] = 3.14 / 4

        # la vrille autour de z
        x_bounds[2].min[fancy_names_index["Zrot"][i] , :] = 0
        x_bounds[2].max[fancy_names_index["Zrot"][i] , :] = 3 * 3.14

    # bras f4a a l'ouverture

    # le carpe
    x_bounds[2].min[fancy_names_index["XrotC"] , FIN] = -0.4

    # le dehanchement f4a a l'ouverture

    # Contraintes de vitesse: PHASE 2 l'ouverture
    for i in range(nb_models):
        # en xy bassin
        x_bounds[2].min[fancy_names_index["vX"][i] : fancy_names_index["vY"][i]+1 , :] = -10
        x_bounds[2].max[fancy_names_index["vX"][i] : fancy_names_index["vY"][i]+1, :] = 10

        # z bassin
        x_bounds[2].min[fancy_names_index["vZ"][i] , :] = -50
        x_bounds[2].max[fancy_names_index["vZ"][i] , :] = 50

        # autour de x
        x_bounds[2].min[fancy_names_index["vXrot"][i] , :] = -50
        x_bounds[2].max[fancy_names_index["vXrot"][i] , :] = 50

        # autour de y
        x_bounds[2].min[fancy_names_index["vYrot"][i] , :] = -50
        x_bounds[2].max[fancy_names_index["vYrot"][i] , :] = 50

        # autour de z
        x_bounds[2].min[fancy_names_index["vZrot"][i] , :] = -50
        x_bounds[2].max[fancy_names_index["vZrot"][i] , :] = 50

    # bras droit
    x_bounds[2].min[fancy_names_index["vZrotBD"]  : fancy_names_index["vYrotBD"] + 1 , :] = -50
    x_bounds[2].max[fancy_names_index["vZrotBD"]  : fancy_names_index["vYrotBD"] + 1 , :] = 50

    # bras droit
    x_bounds[2].min[fancy_names_index["vZrotBG"]  : fancy_names_index["vYrotBG"] + 1 , :] = -50
    x_bounds[2].max[fancy_names_index["vZrotBG"] : fancy_names_index["vYrotBG"] + 1 , :] = 50

    # coude droit
    x_bounds[2].min[fancy_names_index["vZrotABD"] : fancy_names_index["vYrotABD"] + 1 , :] = -50
    x_bounds[2].max[fancy_names_index["vZrotABD"] : fancy_names_index["vYrotABD"] + 1 , :] = 50

    # coude gauche
    x_bounds[2].min[fancy_names_index["vZrotABD"]  : fancy_names_index["vYrotABG"] + 1 , :] = -50
    x_bounds[2].max[fancy_names_index["vZrotABD"]  : fancy_names_index["vYrotABG"] + 1 , :] = 50

    # du carpe
    x_bounds[2].min[fancy_names_index["vXrotC"] , :] = -50
    x_bounds[2].max[fancy_names_index["vXrotC"] , :] = 50

    # du dehanchement
    x_bounds[2].min[fancy_names_index["vYrotC"] , :] = -50
    x_bounds[2].max[fancy_names_index["vYrotC"] , :] = 50

    # #
    # Contraintes de position: PHASE 3 la vrille et demie
    #
    for i in range(nb_models):
        # deplacement
        x_bounds[3].min[fancy_names_index["X"][i] , :] = -0.2
        x_bounds[3].max[fancy_names_index["X"][i] , :] = 0.2
        x_bounds[3].min[fancy_names_index["Y"][i] , :] = -1.0
        x_bounds[3].max[fancy_names_index["Y"][i] , :] = 1.0
        x_bounds[3].min[fancy_names_index["Z"][i] , :] = 0
        x_bounds[3].max[
            fancy_names_index["Z"][i] , :
        ] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

        # le salto autour de x
        x_bounds[3].min[fancy_names_index["Xrot"][i] , :] = 2 * 3.14 - 0.1
        x_bounds[3].max[fancy_names_index["Xrot"][i] , :] = 2 * 3.14 + 3 / 2 * 3.14 + 0.1  # 1 salto 3/4
        x_bounds[3].min[fancy_names_index["Xrot"][i] , FIN] = 2 * 3.14 + 3 / 2 * 3.14 - 0.1
        x_bounds[3].max[fancy_names_index["Xrot"][i] , FIN] = 2 * 3.14 + 3 / 2 * 3.14 + 0.1  # 1 salto 3/4

        # limitation du tilt autour de y
        x_bounds[3].min[fancy_names_index["Yrot"][i] , :] = -3.14 / 4
        x_bounds[3].max[fancy_names_index["Yrot"][i] , :] = 3.14 / 4
        x_bounds[3].min[fancy_names_index["Yrot"][i] , FIN] = -3.14 / 8
        x_bounds[3].max[fancy_names_index["Yrot"][i] , FIN] = 3.14 / 8

        # la vrille autour de z
        x_bounds[3].min[fancy_names_index["Zrot"][i] , :] = 0
        x_bounds[3].max[fancy_names_index["Zrot"][i] , :] = 3 * 3.14
        x_bounds[3].min[fancy_names_index["Zrot"][i] , FIN] = 3 * 3.14 - 0.1  # complete la vrille
        x_bounds[3].max[fancy_names_index["Zrot"][i] , FIN] = 3 * 3.14 + 0.1

    # bras f4a la vrille

    # le carpe
    x_bounds[3].min[fancy_names_index["XrotC"], :] = -0.4

    # le dehanchement f4a la vrille

    # Contraintes de vitesse: PHASE 3 la vrille et demie
    for i in range(nb_models):
        # en xy bassin
        x_bounds[3].min[fancy_names_index["vX"][i] :fancy_names_index["vY"][i]+1 , :] = -10
        x_bounds[3].max[fancy_names_index["vX"][i]: fancy_names_index["vY"][i]+1  , :] = 10

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
    x_bounds[3].min[fancy_names_index["vZrotBD"] : fancy_names_index["vYrotBD"]+ 1, :] = -50
    x_bounds[3].max[fancy_names_index["vZrotBD"] : fancy_names_index["vYrotBD"] + 1, :] = 50

    # bras droit
    x_bounds[3].min[fancy_names_index["vZrotBG"] : fancy_names_index["vYrotBG"] + 1, :] = -50
    x_bounds[3].max[fancy_names_index["vZrotBG"] : fancy_names_index["vYrotBG"] + 1, :] = 50

    # coude droit
    x_bounds[3].min[fancy_names_index["vZrotABD"]: fancy_names_index["vYrotABD"] + 1, :] = -50
    x_bounds[3].max[fancy_names_index["vZrotABD"] : fancy_names_index["vYrotABD"] + 1, :] = 50

    # coude gauche
    x_bounds[3].min[fancy_names_index["vZrotABD"] : fancy_names_index["vYrotABG"] + 1, :] = -50
    x_bounds[3].max[fancy_names_index["vZrotABD"] : fancy_names_index["vYrotABG"] + 1, :] = 50

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
    x_bounds[4].min[fancy_names_index["ZrotABD"] : fancy_names_index["XrotABD"] + 1, FIN] = -0.1
    x_bounds[4].max[fancy_names_index["ZrotABD"] : fancy_names_index["XrotABD"] + 1, FIN] = 0.1

    # coude gauche
    x_bounds[4].min[fancy_names_index["ZrotABG"] : fancy_names_index["XrotABG"] + 1, FIN] = -0.1
    x_bounds[4].max[fancy_names_index["ZrotABG"] : fancy_names_index["XrotABG"] + 1, FIN] = 0.1

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
        x_bounds[4].min[fancy_names_index["vX"][i] : fancy_names_index["vY"][i]+1 , :] = -10
        x_bounds[4].max[fancy_names_index["vX"][i] : fancy_names_index["vY"][i]+1 , :] = 10

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
    x_bounds[4].min[fancy_names_index["vZrotBD"] : fancy_names_index["vYrotBD"] + 1, :] = -50
    x_bounds[4].max[fancy_names_index["vZrotBD"] : fancy_names_index["vYrotBD"] + 1, :] = 50

    # bras droit
    x_bounds[4].min[fancy_names_index["vZrotBG"] : fancy_names_index["vYrotBG"] + 1, :] = -50
    x_bounds[4].max[fancy_names_index["vZrotBG"] : fancy_names_index["vYrotBG"] + 1, :] = 50

    # coude droit
    x_bounds[4].min[fancy_names_index["vZrotABD"]: fancy_names_index["vYrotABD"] + 1, :] = -50
    x_bounds[4].max[fancy_names_index["vZrotABD"] : fancy_names_index["vYrotABD"] + 1, :] = 50

    # coude gauche
    x_bounds[4].min[fancy_names_index["vZrotABD"]: fancy_names_index["vYrotABG"] + 1, :] = -50
    x_bounds[4].max[fancy_names_index["vZrotABD"] : fancy_names_index["vYrotABG"] + 1, :] = 50

    # du carpe
    x_bounds[4].min[fancy_names_index["vXrotC"], :] = -50
    x_bounds[4].max[fancy_names_index["vXrotC"], :] = 50

    # du dehanchement
    x_bounds[4].min[fancy_names_index["vYrotC"], :] = -50
    x_bounds[4].max[fancy_names_index["vYrotC"], :] = 50

    x_bounds_real = BoundsList()
    nb_q = biorbd_models[0].nb_q
    nb_qdot = biorbd_models[0].nb_qdot
    for phase in range(len(x_bounds.options)):
        # for idx_model in range(len(biorbd_models)):
        x_min = list(mappings['q'].to_first.map(x_bounds.options[phase][0].min[:nb_q]))
        x_min+= list(mappings['qdot'].to_first.map(x_bounds.options[phase][0].min[nb_q: ]))
        x_max = list(mappings['q'].to_first.map(x_bounds.options[phase][0].max[:nb_q]))
        x_max+= list(mappings['qdot'].to_first.map(x_bounds.options[phase][0].max[nb_q: ]))

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
    #biorbd_model_list = [BiorbdModel(model) for model in model_paths]

    #mapping partout sauf sur les racines
    nb_q = biorbd_models[0].nb_q
    nb_qdot = biorbd_models[0].nb_qdot
    nb_qddot_joints = nb_q - biorbd_models[0].nb_root

    fancy_names_index = set_fancy_names_index(biorbd_models)

    nb_models = len(biorbd_models[0].models)
    nb_freedom = nb_q // len(biorbd_models[0].models)
    global q_to_first
    global q_to_second
    q_to_first =list(range(nb_freedom))
    q_to_second = list(range(nb_freedom))


    roots = list(range(biorbd_models[0].nb_root//len(biorbd_models[0].models)))
    joints= list(range(roots[-1]+1, roots[-1]+nb_qddot_joints//len(biorbd_models[0].models)+1))

    for degree_of_freedom in range(q_to_first[-1]+1,nb_q):
        index = degree_of_freedom%nb_freedom
        if index in roots:
            q_to_first.append(degree_of_freedom)
            q_to_second.append(degree_of_freedom)
        if index in joints:
            q_to_second.append(index)

    qdot_to_first = q_to_first
    qdot_to_second = q_to_second
    qddot_to_first = joints
    qddot_to_second = [i for i in range(len(qddot_to_first))]*nb_models # for j in range(nb_models)]

    # qdot_to_first = [nb_q+i for i in q_to_first]
    # qdot_to_second = [nb_q+i for i in qdot_to_first]

    #
    mappings= BiMappingList()
    mappings.add("q", to_first=q_to_first, to_second=q_to_second)
    mappings.add("qdot", to_first=qdot_to_first, to_second=qdot_to_second)
    mappings.add("qddot_joints", to_first=qddot_to_first, to_second=qddot_to_second)

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, marker_index=1, weight=-1)
    ## AuJo
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
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.0, max_bound=1.0, weight=100000, phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.0, max_bound=1.0, weight=1, phase=1  # 100000
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
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.MINIMIZE_STATE, key='q', index=fancy_names_index['Xrot'], node=Node.END, weight=1000, phase=4)
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.MINIMIZE_STATE, key='q', index=fancy_names_index['Yrot'],  node=Node.END, weight=1000, phase=4)
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.MINIMIZE_STATE, key='q', index=fancy_names_index['Zrot'], node=Node.END, weight=1000, phase=4)


    # Les hanches sont fixes a +-0.2 en bounds, mais les mains doivent quand meme être proches des jambes
    # for index_model, model in enumerate(biorbd_models[0].models):
    model = biorbd_models[0].models
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
    les_bras = []
    # for i, model in enumerate(biorbd_models) :
    les_bras+=[fancy_names_index["ZrotBD"]]
    les_bras+=[fancy_names_index["YrotBD"]]
    les_bras+=[fancy_names_index["ZrotABD"]]
    les_bras+=[fancy_names_index["XrotABD"]]
    les_bras+=[fancy_names_index["ZrotBG"]]
    les_bras+=[fancy_names_index["YrotBG"]]
    les_bras+=[fancy_names_index["ZrotABG"]]
    les_bras+=[fancy_names_index["XrotABG"]]

    les_coudes = [ ]
    # for i, model in enumerate(biorbd_models):
    les_coudes+=[fancy_names_index["ZrotABD"]]
    les_coudes+=[fancy_names_index["XrotABD"]]
    les_coudes+=[fancy_names_index["ZrotABG"]]
    les_coudes+=[fancy_names_index["XrotABG"]]

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
        dofs=fancy_names_index["XrotC"],
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

    x_init = set_x_init(biorbd_models,fancy_names_index, mappings)


    # constraints = ConstraintList()
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=1.5, phase=1)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=1.5, phase=2)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=0.7, phase=3)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=0.5, phase=4)

    # constraints.add(superimpose_markers_constraint, node=Node.ALL_SHOOTING, min_bound=0, max_bound=0.30**2, phase=1)

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
        # constraints=constraints,
        ode_solver=ode_solver,
        n_threads=n_threads,
        variable_mappings=mappings,
        assume_phase_dynamics=True
    )


def main():
    model_paths = ("Models/Models_Lisa/AdCh.bioMod","Models/Models_Lisa/AlAd.bioMod")

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
    # name = input('what is the name of the file ?')
    name = 'test_test'
    path = 'Solutions_MultiModel'
    with open(f'{path}/{name}.pkl', 'wb') as f:
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
