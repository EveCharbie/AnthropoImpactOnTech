


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


# sys.path.append("/home/laseche/Documents/Stage_Lisa/AnthropoImpactOnTech/Models/")
#sys.path.append("/home/laseche/Documents/Projects_/bioptim-master/")
from bioptim import (
    MultiBiorbdModel,
    BiorbdModel,
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


def superimpose_markers(
        all_pn: PenaltyNodeList,
        first_marker: str | int,
        second_marker: str | int,
        #model_index : int,
      #  models_list : list ,
):
    if first_marker == "MidMainD":
        first_marker = 0
    elif first_marker == "MidMainG":
        first_marker = 1

    # second marker
    if second_marker == "CibleMainD":
        second_marker = 2
    if second_marker == "CibleMainG":
        second_marker = 3

    # boucle for
    nlp = all_pn.nlp
    total_diff = 0

    for index_model, biorbd_model in enumerate(nlp.model.models):

        q = nlp.states["q"].mx[nlp.model.variable_index('q', index_model) ] #[model_index: (model_index+1)*16, :].mx
        diff_markers = nlp.model.models[index_model].marker(q, second_marker).to_mx() - nlp.model.models[index_model].marker(q, first_marker).to_mx()
        sum_diff= 0
        for i in range(diff_markers.shape[0]):
            sum_diff  += (diff_markers[i])**2
        total_diff  += (sum_diff)**2

    return nlp.mx_to_cx(
        f"diff_markers",
        diff_markers,
        nlp.states["q"],
    )
def minimize_dofs(all_pn: PenaltyNodeList, dofs: list, targets: list) -> cas.MX:
    diff = 0
    if isinstance(dofs, int):
        dofs = [dofs]
    for i, dof in enumerate(dofs):
        diff += (all_pn.nlp.states["q"].mx[dof] - targets[i]) ** 2
    return all_pn.nlp.mx_to_cx("minimize_dofs", diff, all_pn.nlp.states["q"])

# def custom_func_track_markers(all_pn: PenaltyNodeList, first_marker: str, second_marker: str, method: int) -> MX:
#
#     # Get the index of the markers from their name
#     marker_0_idx = all_pn.nlp.model.marker_index(first_marker)
#     marker_1_idx = all_pn.nlp.model.marker_index(second_marker)
#
#     if method == 0:
#         # Convert the function to the required format and then subtract
#         markers = all_pn.nlp.mx_to_cx("markers", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
#         markers_diff = markers[:, marker_1_idx] - markers[:, marker_0_idx]
#
#     else:
#         # Do the calculation in biorbd API and then convert to the required format
#         markers = all_pn.nlp.model.markers(all_pn.nlp.states["q"].mx)
#         markers_diff = markers[marker_1_idx] - markers[marker_0_idx]
#         markers_diff = all_pn.nlp.mx_to_cx("markers", markers_diff, all_pn.nlp.states["q"])
#
#     return markers_diff

def set_fancy_names_index(biorbd_models):
    """
    For readability
    """
    nb_model = len(biorbd_models[0].models)

    nb_q = biorbd_models[0].nb_q//nb_model
    # for i inb_model):n range(
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
    fancy_names_index["vX"] = [0+nb_q+i*16 for i in range(nb_model)]
    fancy_names_index["vY"] = [1+nb_q+i*16 for i in range(nb_model)]
    fancy_names_index["vZ"] = [2+nb_q+i*16 for i in range(nb_model)]
    fancy_names_index["vXrot"] = [3+nb_q+i*16 for i in range(nb_model)]
    fancy_names_index["vYrot"] = [4+nb_q+i*16 for i in range(nb_model)]
    fancy_names_index["vZrot"] = [5+nb_q+i*16 for i in range(nb_model)]
    fancy_names_index["vZrotBD"] = 6
    fancy_names_index["vYrotBD"] = 7
    fancy_names_index["vZrotABD"] = 8
    fancy_names_index["vYrotABD"] = 9
    fancy_names_index["vZrotBG"] = 10
    fancy_names_index["vYrotBG"] = 11
    fancy_names_index["vZrotABG"] = 12
    fancy_names_index["vYrotABG"] = 13
    fancy_names_index["vXrotC"] = 14
    fancy_names_index["vYrotC"] = 15

    return fancy_names_index


def set_x_bounds(biorbd_models, fancy_names_index, final_time):
    # for
    nb_q = biorbd_models[0].nb_q
    nb_qdot = biorbd_models[0].nb_qdot
    nb_models = len(biorbd_models[0].models)
    nb_q_per_model = nb_q//nb_models
    nb_qdot_per_model= nb_qdot//nb_models
    x_bounds = BoundsList()
    x_bounds.add(bounds=biorbd_models[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_models[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_models[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_models[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_models[0].bounds_from_ranges(["q", "qdot"]))



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

    x_bounds[0].max[fancy_names_index["ZrotABD"]:fancy_names_index["XrotABD"]+ 1 , DEBUT] = 0


    # coude gauche
    x_bounds[0].min[fancy_names_index["ZrotABG"] : fancy_names_index["XrotABG"]+1 , DEBUT] = 0

    x_bounds[0].max[fancy_names_index["ZrotABG"] : fancy_names_index["XrotABG"] +1 , DEBUT] = 0


    # le carpe
    x_bounds[0].min[fancy_names_index["XrotC"], DEBUT] = -0.50  # depart un peu ferme aux hanches

    x_bounds[0].max[fancy_names_index["XrotC"], DEBUT] = -0.50

    x_bounds[0].min[fancy_names_index["XrotC"], FIN] = -2.35

    # x_bounds[0].max[fancy_names_index["XrotC"], FIN] = -2.35


    # le dehanchement
    x_bounds[0].min[fancy_names_index["YrotC"], DEBUT] = 0
    x_bounds[0].max[fancy_names_index["YrotC"], DEBUT] = 0
    x_bounds[0].min[fancy_names_index["YrotC"], MILIEU:] = -0.1

    x_bounds[0].max[fancy_names_index["YrotC"], MILIEU:] = 0.1

    # Contraintes de vitesse: PHASE 0 la montee en carpe

    vzinit = 9.81 / 2 * final_time  # vitesse initiale en z du CoM pour revenir a terre au temps final

    # en xy bassin
    for i in range(nb_models):
        x_bounds[0].min[fancy_names_index["vX"][i]: fancy_names_index["vY"][i]+1 , :]=-10

        x_bounds[0].max[fancy_names_index["vX"][i] :fancy_names_index["vY"][i]+1, :] = 10
        x_bounds[0].min[fancy_names_index["vX"][i]: fancy_names_index["vY"][i]+1 , DEBUT] = -0.5

        x_bounds[0].max[fancy_names_index["vX"][i] : fancy_names_index["vY"][i]+1, DEBUT] = 0.5


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
        # AUJO
        CoM_Q_sym = cas.MX.sym("CoM", nb_q_per_model)
        CoM_Q_init = x_bounds[0].min[
                     i*nb_q_per_model: (i+1)*nb_q_per_model, DEBUT
                     ]  # min ou max ne change rien a priori, au DEBUT ils sont egaux normalement
        CoM_Q_func = cas.Function("CoM_Q_func", [CoM_Q_sym], [biorbd_models[0].models[i].CoM(CoM_Q_sym).to_mx()])
        bassin_Q_func = cas.Function(
            "bassin_Q_func", [CoM_Q_sym], [biorbd_models[0].models[i].globalJCS(0).to_mx()]
        )  # retourne la RT du bassin

        r = (
                np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1, :3]
        )  # selectionne seulement la translation de la RT
    # tenir compte du decalage entre bassin et CoM avec la rotation
    # Qtransdot = Qtransdot + v cross Qrotdot

        borne_inf = (
            x_bounds[0].min[fancy_names_index["vX"][i] : fancy_names_index["vZ"][i]+1 , DEBUT]
            + np.cross(r, x_bounds[0].min[fancy_names_index["vXrot"][i]: fancy_names_index["vZrot"][i]+1 , DEBUT])
        )[0]
        borne_sup = (
            x_bounds[0].max[fancy_names_index["vX"][i]: fancy_names_index["vZ"][i] +1  , DEBUT]
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
    x_bounds[0].min[fancy_names_index["vZrotBD"]  : fancy_names_index["vYrotBD"] +1 , :] = -50
    x_bounds[0].max[fancy_names_index["vZrotBD"]  : fancy_names_index["vYrotBD"]  +1, :] = 50
    x_bounds[0].min[fancy_names_index["vZrotBD"]  : fancy_names_index["vYrotBD"]  +1, DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBD"]  : fancy_names_index["vYrotBD"]  +1, DEBUT] = 0


    # bras droit
    x_bounds[0].min[fancy_names_index["vZrotBG"] : fancy_names_index["vYrotBG"] + 1 , :] = -50
    x_bounds[0].max[fancy_names_index["vZrotBG"]: fancy_names_index["vYrotBG"] + 1, :] = 50
    x_bounds[0].min[fancy_names_index["vZrotBG"] : fancy_names_index["vYrotBG"] + 1 , DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotBG"] : fancy_names_index["vYrotBG"] + 1 , DEBUT] = 0

    # coude droit
    x_bounds[0].min[fancy_names_index["vZrotABD"] : fancy_names_index["vYrotABD"] + 1 , :] = -50
    x_bounds[0].max[fancy_names_index["vZrotABD"]  : fancy_names_index["vYrotABD"] + 1 , :] = 50
    x_bounds[0].min[fancy_names_index["vZrotABD"] : fancy_names_index["vYrotABD"] + 1 , DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABD"]  : fancy_names_index["vYrotABD"] + 1 , DEBUT] = 0

    # coude gauche
    x_bounds[0].min[fancy_names_index["vZrotABD"]  : fancy_names_index["vYrotABG"] + 1, :] = -50
    x_bounds[0].max[fancy_names_index["vZrotABD"]  : fancy_names_index["vYrotABG"] + 1 , :] = 50
    x_bounds[0].min[fancy_names_index["vZrotABG"]  : fancy_names_index["vYrotABG"] + 1 , DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vZrotABG"]  : fancy_names_index["vYrotABG"] + 1 , DEBUT] = 0


    # du carpe
    x_bounds[0].min[fancy_names_index["vXrotC"] , :] = -50
    x_bounds[0].max[fancy_names_index["vXrotC"] , :] = 50
    x_bounds[0].min[fancy_names_index["vXrotC"] , DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vXrotC"] , DEBUT] = 0

    # du dehanchement
    x_bounds[0].min[fancy_names_index["vYrotC"] , :] = -50
    x_bounds[0].max[fancy_names_index["vYrotC"] , :] = 50
    x_bounds[0].min[fancy_names_index["vYrotC"] , DEBUT] = 0
    x_bounds[0].max[fancy_names_index["vYrotC"] , DEBUT] = 0

    #
    # Contraintes de position: PHASE 1 le salto carpe
    #

    # deplacement
    for i in range(nb_models) :
        x_bounds[1].min[fancy_names_index["X"][i] , :] = -0.1
        x_bounds[1].max[fancy_names_index["X"][i] , :] = 0.1
        x_bounds[1].min[fancy_names_index["Y"][i] , :] = -1.0
        x_bounds[1].max[fancy_names_index["Y"][i] , :] = 1.0
        x_bounds[1].min[fancy_names_index["Z"][i] , :] = 0
        x_bounds[1].max[
            fancy_names_index["Z"][i] , :
        ] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

        # le salto autour de x
        x_bounds[1].min[fancy_names_index["Xrot"][i], :] = 0
        x_bounds[1].max[fancy_names_index["Xrot"][i], :] = 4 * 3.14
        x_bounds[1].min[fancy_names_index["Xrot"][i] , FIN] = 2 * 3.14 - 0.1

        # limitation du tilt autour de y
        x_bounds[1].min[fancy_names_index["Yrot"][i] , :] = -3.14 / 16
        x_bounds[1].max[fancy_names_index["Yrot"][i] , :] = 3.14 / 16

        # la vrille autour de z
        x_bounds[1].min[fancy_names_index["Zrot"][i] , :] = -0.1
        x_bounds[1].max[fancy_names_index["Zrot"][i] , :] = 0.1

    # bras f4a a l'ouverture

    # le carpe
    x_bounds[1].min[fancy_names_index["XrotC"] , :] = -2.35 - 0.1
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
        x_bounds[2].min[fancy_names_index["Xrot"][i] , :] = 2 * 3.14 + 0.1  # 1 salto 3/4
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
        x_bounds[4].min[fancy_names_index["Xrot"][i], :] = 2 * 3.14 + 3 / 2 * 3.14 - 0.2  # penche vers avant -> moins de salto
        x_bounds[4].max[fancy_names_index["Xrot"][i], :] = -0.50 + 4 * 3.14  # un peu carpe a la fin
        x_bounds[4].min[fancy_names_index["Xrot"][i], FIN] = -0.50 + 4 * 3.14 - 0.1
        x_bounds[4].max[fancy_names_index["Xrot"][i], FIN] = -0.50 + 4 * 3.14 + 0.1  # 2 salto fin un peu carpe

        # limitation du tilt autour de y
        x_bounds[4].min[fancy_names_index["Yrot"][i], :] = -3.14 / 16
        x_bounds[4].max[fancy_names_index["Yrot"][i], :] = 3.14 / 16

        # la vrille autour de z
        x_bounds[4].min[fancy_names_index["Zrot"][i], :] = 3 * 3.14 - 0.1  # complete la vrille
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
        x_bounds[4].min[fancy_names_index["vZ"][i], :] = -100
        x_bounds[4].max[fancy_names_index["vZ"][i], :] = 100

        # autour de x
        x_bounds[4].min[fancy_names_index["vXrot"][i], :] = -100
        x_bounds[4].max[fancy_names_index["vXrot"][i], :] = 100

        # autour de y
        x_bounds[4].min[fancy_names_index["vYrot"][i], :] = -100
        x_bounds[4].max[fancy_names_index["vYrot"][i], :] = 100

        # autour de z
        x_bounds[4].min[fancy_names_index["vZrot"][i], :] = -100
        x_bounds[4].max[fancy_names_index["vZrot"][i], :] = 100

    # bras droit
    x_bounds[4].min[fancy_names_index["vZrotBD"] : fancy_names_index["vYrotBD"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotBD"] : fancy_names_index["vYrotBD"] + 1, :] = 100

    # bras droit
    x_bounds[4].min[fancy_names_index["vZrotBG"] : fancy_names_index["vYrotBG"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotBG"] : fancy_names_index["vYrotBG"] + 1, :] = 100

    # coude droit
    x_bounds[4].min[fancy_names_index["vZrotABD"]: fancy_names_index["vYrotABD"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotABD"] : fancy_names_index["vYrotABD"] + 1, :] = 100

    # coude gauche
    x_bounds[4].min[fancy_names_index["vZrotABD"]: fancy_names_index["vYrotABG"] + 1, :] = -100
    x_bounds[4].max[fancy_names_index["vZrotABD"] : fancy_names_index["vYrotABG"] + 1, :] = 100

    # du carpe
    x_bounds[4].min[fancy_names_index["vXrotC"], :] = -100
    x_bounds[4].max[fancy_names_index["vXrotC"], :] = 100

    # du dehanchement
    x_bounds[4].min[fancy_names_index["vYrotC"], :] = -100
    x_bounds[4].max[fancy_names_index["vYrotC"], :] = 100



    return x_bounds


def set_x_init( biorbd_models, fancy_names_index):
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

    #x5[:] = x0[:]
    #x6[:] = x1[:]
    #x7[:] = x2[:]
    #x8[:] = x3[:]
    #x9[:] = x4[:]

        # x1[nb_q * i // len(biorbd_models):nb_q * 2 * i // len(biorbd_models), :] = x1[0:nb_q // len(biorbd_models)]
        # x2[nb_q * i // len(biorbd_models):nb_q * 2 * i // len(biorbd_models), :] = x2[0:nb_q // len(biorbd_models)]
        # x3[nb_q * i // len(biorbd_models):nb_q * 2 * i // len(biorbd_models), :] = x3[0:nb_q // len(biorbd_models)]
        # x4[nb_q * i // len(biorbd_models):nb_q * 2 * i // len(biorbd_models), :] = x4[0:nb_q // len(biorbd_models)]
        # X = [x0, x1, x2, x3, x4]
    x_init.add(x0, interpolation=InterpolationType.LINEAR)
    x_init.add(x1, interpolation=InterpolationType.LINEAR)
    x_init.add(x2, interpolation=InterpolationType.LINEAR)
    x_init.add(x3, interpolation=InterpolationType.LINEAR)
    x_init.add(x4, interpolation=InterpolationType.LINEAR)

    return x_init


# def root_explicit_dynamic(
#     states: Union[cas.MX, cas.SX],
#     controls: Union[cas.MX, cas.SX],
#     parameters: Union[cas.MX, cas.SX],
#     nlp: NonLinearProgram,
# ) -> tuple:
#
#     DynamicsFunctions.apply_parameters(parameters, nlp)
#     q = DynamicsFunctions.get(nlp.states["q"], states)
#     qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
#     nb_root = nlp.model.nbRoot()
#
#     qddot_joints = DynamicsFunctions.get(nlp.controls["qddot_joints"], controls)
#
#     mass_matrix_nl_effects = nlp.model.InverseDynamics(
#         q, qdot, cas.vertcat(cas.MX.zeros((nb_root, 1)), qddot_joints)
#     ).to_mx()[:nb_root]
#
#     mass_matrix = nlp.model.massMatrix(q).to_mx()
#     mass_matrix_nl_effects_func = Function(
#         "mass_matrix_nl_effects_func", [q, qdot, qddot_joints], [mass_matrix_nl_effects[:nb_root]]
#     ).expand()
#
#     M_66 = mass_matrix[:nb_root, :nb_root]
#     M_66_func = Function("M66_func", [q], [M_66]).expand()
#
#     qddot_root = solve(-M_66_func(q), mass_matrix_nl_effects_func(q, qdot, qddot_joints), "ldl")
#
#     return cas.vertcat(qdot, cas.vertcat(qddot_root, qddot_joints))


# def custom_configure_root_explicit(ocp: OptimalControlProgram, nlp: NonLinearProgram):
#     ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
#     ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
#     configure_qddot_joint(nlp, as_states=False, as_controls=True)
#     ConfigureProblem.configure_dynamics_function(ocp, nlp, root_explicit_dynamic, expand=False)


# def configure_qddot_joint(nlp, as_states: bool, as_controls: bool):
#     nb_root = nlp.model.nbRoot()
#     name_qddot_joint = [str(i + nb_root) for i in range(nlp.model.nbQddot() - nb_root)]
#     ConfigureProblem.configure_new_variable("qddot_joints", name_qddot_joint, nlp, as_states, as_controls)


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
    q_to_first =list(range(nb_freedom))
    q_to_second = list(range(nb_freedom))


    roots = list(range(biorbd_models[0].nb_root//len(biorbd_models[0].models)))
    joints= list(range(roots[-1]+1, roots[-1]+nb_qddot_joints//len(biorbd_models[0].models)+1))
    qddot_joints_to_first = list
    # qdddot_joints_to_second = []
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

    ## AuJo
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.0, max_bound=final_time, weight=100000, phase=0
    )

    # Les hanches sont fixes a +-0.2 en bounds, mais les mains doivent quand meme être proches des jambes
    ## AuJo
    # objective_functions.add(
#     ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
    #     node=Node.END,
    #     first_marker="MidMainG",
    #     second_marker="CibleMainG",
    #     weight=1000,
    #     phase=0,
    # )
    # for i, model in enumerate(biorbd_models[0]):
    objective_functions.add(
        superimpose_markers,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        first_marker="MidMainG",
        second_marker="CibleMainG",
        weight=10000,
        phase=0,
    )
    objective_functions.add(
        superimpose_markers,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        first_marker="MidMainD",
        second_marker="CibleMainD",
        weight=10000,
        phase=0,
    )
   # objective_functions.add(
   #     ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
   #     node=Node.END,
   #     first_marker="MidMainD",
  #      second_marker="CibleMainD",
   #     weight=1000,
   #     phase=0,
   # )


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
    # les_bras_copy= []
    # for i in range(len(les_bras)):
    #     les_bras_copy.append(les_bras[i][j] for j in range(len(les_bras[i])))

    les_coudes = [ ]
    # for i, model in enumerate(biorbd_models):
    les_coudes+=[fancy_names_index["ZrotABD"]]
    les_coudes+=[fancy_names_index["XrotABD"]]
    les_coudes+=[fancy_names_index["ZrotABG"]]
    les_coudes+=[fancy_names_index["XrotABG"]]

    ## AuJo
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        dofs=les_coudes,
        targets=np.zeros(len(les_coudes)),
        weight=10000,
        phase=0,
    )
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        dofs=les_bras,
        targets=np.zeros(len(les_bras)),
        weight=10000,
        phase=2,
    )
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        dofs=les_bras,
        targets=np.zeros(len(les_bras)),
        weight=10000,
        phase=3,
    )
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        dofs=les_coudes,
        targets=np.zeros(len(les_coudes)),
        weight=10000,
        phase=4,
    )


    # argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("model", type=str, help="the bioMod file")
    # parser.add_argument("--no-hsl", dest='with_hsl', action='store_false', help="do not use libhsl")
    # parser.add_argument("-j", default=1, dest='n_threads', type=int, help="number of threads in the solver")
    # parser.add_argument("--no-sol", action='store_false', dest='savesol', help="do not save the solution")
    # parser.add_argument("--no-show-online", action='store_false', dest='show_online', help="do not show graphs during optimization")
    # parser.add_argument("--print-ocp", action='store_true', dest='print_ocp', help="print the ocp")
    # args = parser.parse_args()
    # ouvre les hanches rapidement apres la vrille
    ## AuJo
    # dofs = []
    # for i, model in enumerate(biorbd_models) :
    # dofs.append(fancy_names_index["XrotC"])
    objective_functions.add(
        minimize_dofs,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        dofs=fancy_names_index["XrotC"],
        targets=[0, 0],
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

    x_bounds = set_x_bounds(biorbd_models, fancy_names_index, final_time)

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



    x_init = set_x_init(biorbd_models,fancy_names_index)


    constraints = ConstraintList()
    #    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0, max_bound=final_time, phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=1)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=2)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=3)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=4)



    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)  # 0-1
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)  # 1-2
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=2)  # 2-3
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=3)  # 3-4
    phase_transitions.add(
        PhaseTransitionFcn.DISCONTINUOUS,
        phase_pre_idx=4,
    )  # 4-5


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
        constraints,
        ode_solver=ode_solver,
        n_threads=n_threads,
        variable_mappings = mappings, # node_mappings=node_mappings, #### ajouter le mapping , la les bounds ne sotn pas de la bonnes tailles
        phase_transitions=phase_transitions,
    )


def main():
#    models =
# mettre tout les models
    os.listdir('Models/')
    model_paths = ("Models/AuJo_TechOpt83.bioMod","Models/AuJo_TechOpt83.bioMod")

    n_threads = 4

    print_ocp_FLAG = False  # True.

    show_online_FLAG = False  # True
    HSL_FLAG = True
    save_sol_FLAG = True
    # n_shooting = (40, 100, 100, 100, 40,
    #               40, 100, 100, 100, 40)
    n_shooting = (1, 2, 2, 2, 1)

    ocp = prepare_ocp(model_paths, n_shooting=n_shooting, n_threads=n_threads, final_time=1.87)
    # ocp.add_plot_penalty(CostType.ALL)
    if print_ocp_FLAG:
        ocp.print(to_graph=True)
    solver = Solver.IPOPT(show_online_optim=show_online_FLAG, show_options=dict(show_bounds=True))
    if HSL_FLAG:
        solver.set_linear_solver("ma57")
    else:
        print("Not using ma57")
    solver.set_maximum_iterations(1)
    solver.set_convergence_tolerance(1e-4)
    sol = ocp.solve(solver)

    temps = time.strftime("%Y-%m-%d-%H%M")
    nom = "MultiModel solutions "
    # il faut recuperer les Q de toutes les phases la onrecupere que lq premiere phase
    qs = sol.states[0]["q"]
    qdots = sol.states[0]["qdot"]

    for i in range(1, len(sol.states)):
        qs = np.hstack((qs, sol.states[i]["q"]))
        qdots = np.hstack((qdots, sol.states[i]["qdot"]))
    if save_sol_FLAG:  # switch manuelle
        np.save(f"/home/laseche/Documents/Stage_Lisa/AnthropoImpactOnTech/Solutions_MultiModel/{nom}-{str(n_shooting).replace(', ', '_')}-{temps}-q.npy", qs)
        np.save(f"/home/laseche/Documents/Stage_Lisa/AnthropoImpactOnTech/Solutions_MultiModel/{nom}-{str(n_shooting).replace(', ', '_')}-{temps}-qdot.npy", qdots)
        np.save(f"/home/laseche/Documents/Stage_Lisa/AnthropoImpactOnTech/Solutions_MultiModel/{nom}-{str(n_shooting).replace(', ', '_')}-{temps}-t.npy", sol.phase_time)

    if IPYTHON:
        IPython.embed()  # afin de pouvoir explorer plus en details la solution

    # Print the last solution
    #sol.animate(n_frames=-1, show_floor=False)

    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
    # main()
