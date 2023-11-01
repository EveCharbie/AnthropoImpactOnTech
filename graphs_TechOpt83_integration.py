import biorbd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from decimal import *
import bioviz
import bioptim

from IPython import embed
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os

import numpy as np
import biorbd_casadi as biorbd
from casadi import MX, Function

folder_per_athlete = {
    "Athlete_03": "Athlete_03/",
    "Athlete_05": "Athlete_05/",
    "Athlete_18": "Athlete_18/",
    "Athlete_07": "Athlete_07/",
    "Athlete_14": "Athlete_14/",
    "Athlete_17": "Athlete_17/",
    "Athlete_02": "Athlete_02/",
    "Athlete_06": "Athlete_06/",
    "Athlete_11": "Athlete_11/",
    "Athlete_13": "Athlete_13/",
    "Athlete_16": "Athlete_16/",
    "Athlete_12": "Athlete_12/",
    "Athlete_04": "Athlete_04/",
    "Athlete_10": "Athlete_10/",
    "Athlete_08": "Athlete_08/",
    "Athlete_09": "Athlete_09/",
    "Athlete_01": "Athlete_01/",
    "Athlete_15": "Athlete_15/",
}

folder_per_twist_nb = {"3": "Solutions_vrille_et_demi/"} # {"5": "Solutions_double_vrille_et_demi/"} #

results_path = "solutions_multi_start/"
folder_graphs = "kinematics_graphs"
model_path = "Models/Models_Lisa"
num_half_twist = "3"  # "5"
if num_half_twist == "3":
    folder_to_save = "vrille_et_demi"
elif num_half_twist == "5":
    folder_to_save = "double_vrille_et_demi"
folder = folder_per_twist_nb[num_half_twist].removeprefix('Solutions_').removesuffix('/')
athlete_done = []
for athlete in folder_per_athlete:
    results_path_this_time = results_path + folder_per_twist_nb[num_half_twist] + folder_per_athlete[athlete]

    if athlete in athlete_done:
        print(f'{athlete} for {folder} has already a graph')
        continue
    else:
        print(f'Building graph for {athlete} doing {folder}')

    nb_twists = int(num_half_twist)
    noise = []
    C = []
    Q = []
    Q_integrated = []
    Error = []
    nb = 0
    fig = None
    axs = None

    fig, axs = plt.subplots(4, 4, figsize=(18, 9))
    axs = axs.ravel()
    for filename in os.listdir(results_path_this_time):
        nb += 1
        athlete = filename.split("_")[0]
        if filename.removesuffix(".pkl")[-3] == "C":
            file_name = f'/kinematics_graph for {athlete}_{folder_per_twist_nb[num_half_twist].removesuffix("/")}.png'
            noise += filename.split("_")[-2]
            print(filename)
            f = os.path.join(results_path_this_time, filename)
            filename = results_path_this_time + filename
            model = biorbd.Model(f"{model_path}/{athlete}.bioMod")
            # ocp = prepare_ocp(
            #     biorbd_model_path=model, nb_twist=int(num_half_twist), seed=int(filename.split("_")[-2]), n_threads=3
            # )
            if os.path.isfile(f):
                if filename.endswith(".pkl"):
                    with open(filename, "rb") as f:
                        data = pickle.load(f)
                        q = data["q"]

                        C.append(data["sol"].cost.toarray()[0][0])

                        # # integrated
                        # sol = data["sol"]
                        # sol.ocp = ocp
                        #
                        # sol_integrated = sol.integrate(
                        #     shooting_type=Shooting.SINGLE, keep_intermediate_points=False, merge_phases=True
                        # )
                        # q_integrated = sol_integrated.states["q"]
                        # Q_integrated.append(q_integrated)

                        erreur = 0
                        # for degree in range(len(q[0])):
                        #
                        #     if degree not in [0, 1, 2]:
                        #         erreur += abs(q[-1][degree][-1] - q_integrated[degree][-1])
                        Q.append(q)
                        Error.append(erreur)
    if Error != []:
        min_error = np.array(Error).min()
        max_error = np.array(Error).max()

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

        for i in range(len(noise)):
            noise_i = noise[i]
            cost_i = COST[i]
            q = Q[i]
            # q_integrated = Q_integrated[i]  # pour chaque opti
            # error_i = Error[i]
            alpha = 1  # abs((error_i - max_error)/(min_error - max_error))
            alpha_decimal = Decimal(alpha)
            error_i_roundresult = None  # error_i.round(4)
            linewidth_max = 3
            linewidth_min = 0.4
            linewidth = lambda alpha : (linewidth_max-linewidth_min)*alpha +linewidth_min

            if min != max:
                ratio = (cost_i - min) / (max - min)
            else:
                ratio = 1

            color = cmap(ratio)

            print(f"alpha is {alpha}")
            for degree in range(len(q[0])):
                q_plot = []
                for phase in range(len(q)):
                    q_plot += q[phase][degree].tolist()[:]

                if degree == 0:
                    axs[degree].plot(q_plot, color=color, label=f'{noise_i}, {error_i_roundresult}', linewidth = linewidth(alpha))

                else:
                    axs[degree].plot(q_plot, color=color,  linewidth=linewidth(alpha))

                axs[degree].set_title(f"{model.nameDof()[degree].to_string()}")

        data_to_save = {'cost': COST,
                        'q': Q,
                        # 'q_integrated': Q_integrated,
                        # 'reintegration_error': Error,
                        'noise': noise,}
        with open(f"kinematics_graphs/{folder_to_save}/data_pickled/{athlete}.pkl", "wb") as f:
            pickle.dump(data_to_save, f)


    if Q!= []:
        axs[0].legend(bbox_to_anchor=(0.5, 1), loc="upper left", borderaxespad=-5, ncols=nb, fontsize=12)
        plt.subplots_adjust(left=0.05, right=0.8, hspace=0.4)
        plt.savefig(f"{folder_graphs}/{folder}/{file_name}", dpi=300)
        plt.show()
