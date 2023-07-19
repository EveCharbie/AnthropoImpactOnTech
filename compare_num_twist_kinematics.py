
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython import embed

import biorbd
import bioviz
import sys



model_path = "Models/JeCh_TechOpt83.bioMod"
model = biorbd.Model(model_path)

athletes_number = {"WeEm": 1,
                   "FeBl": 2,
                   "AdCh": 3,
                   "MaJa": 4,
                   "AlAd": 5,
                   "JeCh": 6,
                   "Benjamin": 7,
                   "Sarah": 8,
                   "SoMe": 9,
                   "OlGa": 10,
                   "KaFu": 11,
                   "MaCu": 12,
                   "KaMi": 13,
                   "ElMe": 14,
                   "ZoTs": 15,
                   "LaDe": 16,
                   "EvZl": 17,
                   "AuJo": 18}

nb_twists = {"1": 'vrille_et_demi', "2": 'double_vrille_et_demi'}
chosen_clusters_dict = {}
results_path = 'solutions_multi_start/'

cmap = cm.get_cmap('viridis')
fig, axs = plt.subplots(2, 3, figsize=(18, 9))

for idx, key in enumerate(nb_twists.keys()):
    results_path_this_time = results_path + f'Solutions_{nb_twists[key]}/'

    for i_name, name in enumerate(athletes_number.keys()):
        for i_sol in range(9):
            file_name = results_path_this_time + name + '/' + name + f'_{nb_twists[key]}_' + str(i_sol) + "_CVG.pkl"
            if not os.path.isfile(file_name):
                continue
            print(file_name)
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
            Q = data['q']
            time_parameters = data['sol'].parameters['time']
            time_vector = np.hstack((np.linspace(0, float(time_parameters[0]), 41)[:-1],
                                     np.linspace(float(time_parameters[0]), float(time_parameters[0]+time_parameters[1]), 101)[:-1]))
            time_vector = np.hstack((time_vector, np.linspace(float(time_parameters[0]+time_parameters[1]), float(time_parameters[0]+time_parameters[1]+time_parameters[2]), 101)[:-1]))
            time_vector = np.hstack((time_vector, np.linspace(float(time_parameters[0]+time_parameters[1]+time_parameters[2]), float(time_parameters[0]+time_parameters[1]+time_parameters[2]+time_parameters[3]), 101)[:-1]))
            time_vector = np.hstack((time_vector, np.linspace(float(time_parameters[0]+time_parameters[1]+time_parameters[2]+time_parameters[3]), float(time_parameters[0]+time_parameters[1]+time_parameters[2]+time_parameters[3]+time_parameters[4]), 41)))

            normalized_time_vector = time_vector / time_vector[-1]

            q = np.zeros(np.shape(Q[0][:, :-1]))
            q[:, :] = Q[0][:, :-1]
            for i in range(1, len(Q)):
                if i == len(Q) - 1:
                    q = np.hstack((q, Q[i]))
                else:
                    q = np.hstack((q, Q[i][:, :-1]))

            rgba = cmap(idx*0.8 + i_name * 1/18 * 0.2)
            axs[0, 0].plot(normalized_time_vector, q[6, :], color=rgba)
            axs[1, 0].plot(normalized_time_vector, q[7, :], color=rgba)

            axs[0, 1].plot(normalized_time_vector, -q[10, :], color=rgba)
            axs[1, 1].plot(normalized_time_vector, -q[11, :], color=rgba)

            axs[0, 2].plot(normalized_time_vector, q[14, :], color=rgba)
            axs[1, 2].plot(normalized_time_vector, q[15, :], color=rgba)

            if i_sol == 0:
                axs[0, 0].set_title(f"Change in elevation plane")  # Right arm
                axs[1, 0].set_title(f"Elevation")  # Right arm
                axs[0, 1].set_title(f"Change in elevation plane")  # Left arm
                axs[1, 1].set_title(f"Left arm elevation")  # Left arm
                axs[0, 2].set_title(f"Flexion")  # Hips
                axs[1, 2].set_title(f"Lateral flexion")  # Hips

                axs[1, 0].set_xlabel(f"Normalized time")
                axs[1, 1].set_xlabel(f"Normalized time")
                axs[1, 2].set_xlabel(f"Normalized time")

plt.savefig(f'cluster_graphs/comparison_graph_for_all_athletes.png', dpi=300)
# plt.show()

print("\n\n")
