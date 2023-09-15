
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

nb_twists = {"2": 'double_vrille_et_demi'}
chosen_clusters_dict = {}
results_path = 'solutions_multi_start/'

with open("q_bounds.pkl", 'rb') as f:
    q_bounds_min, q_bounds_max = pickle.load(f)

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

            if i_name == 0 and i_sol == 0:
                axs[0, 0].plot(normalized_time_vector, q_bounds_min[6, :] * 180 / np.pi, color='black', linewidth=0.5)
                axs[0, 0].plot(normalized_time_vector, q_bounds_max[6, :] * 180 / np.pi, color='black', linewidth=0.5)
                axs[1, 0].plot(normalized_time_vector, q_bounds_min[7, :] * 180 / np.pi, color='black', linewidth=0.5)
                axs[1, 0].plot(normalized_time_vector, q_bounds_max[7, :] * 180 / np.pi, color='black', linewidth=0.5)
                axs[0, 1].plot(normalized_time_vector, q_bounds_min[10, :] * 180 / np.pi, color='black', linewidth=0.5)
                axs[0, 1].plot(normalized_time_vector, q_bounds_max[10, :] * 180 / np.pi, color='black', linewidth=0.5)
                axs[1, 1].plot(normalized_time_vector, q_bounds_min[11, :] * 180 / np.pi, color='black', linewidth=0.5)
                axs[1, 1].plot(normalized_time_vector, q_bounds_max[11, :] * 180 / np.pi, color='black', linewidth=0.5)
                axs[0, 2].plot(normalized_time_vector, q_bounds_min[14, :] * 180 / np.pi, color='black', linewidth=0.5)
                axs[0, 2].plot(normalized_time_vector, q_bounds_max[14, :] * 180 / np.pi, color='black', linewidth=0.5)
                axs[1, 2].plot(normalized_time_vector, q_bounds_min[15, :] * 180 / np.pi, color='black', linewidth=0.5)
                axs[1, 2].plot(normalized_time_vector, q_bounds_max[15, :] * 180 / np.pi, color='black', linewidth=0.5)

            q = np.zeros(np.shape(Q[0][:, :-1]))
            q[:, :] = Q[0][:, :-1]
            for i in range(1, len(Q)):
                if i == len(Q) - 1:
                    q = np.hstack((q, Q[i]))
                else:
                    q = np.hstack((q, Q[i][:, :-1]))

            rgba = cmap(i_name * 1/18)
            axs[0, 0].plot(normalized_time_vector, q[6, :] * 180 / np.pi, color=rgba)
            axs[1, 0].plot(normalized_time_vector, q[7, :] * 180 / np.pi, color=rgba)

            axs[0, 1].plot(normalized_time_vector, -q[10, :] * 180 / np.pi, color=rgba)
            axs[1, 1].plot(normalized_time_vector, -q[11, :] * 180 / np.pi, color=rgba)

            axs[0, 2].plot(normalized_time_vector, q[14, :] * 180 / np.pi, color=rgba)
            axs[1, 2].plot(normalized_time_vector, q[15, :] * 180 / np.pi, color=rgba)

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

plt.savefig(f'cluster_graphs/comparison_graph_for_all_athletes_2.png', dpi=300)
# plt.show()

print("\n\n")


nb_twists = {"1": 'vrille_et_demi', "2": 'double_vrille_et_demi'}
chosen_clusters_dict = {}
results_path = 'solutions_multi_start/'

with open("q_bounds.pkl", 'rb') as f:
    q_bounds_min, q_bounds_max = pickle.load(f)

cmap = cm.get_cmap('viridis')
for i_name, name in enumerate(athletes_number.keys()):
    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    for idx, key in enumerate(nb_twists.keys()):
        bounds_plotted = False
        results_path_this_time = results_path + f'Solutions_{nb_twists[key]}/'
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

            # if bounds_plotted == False:
            #     axs[0, 0].plot(normalized_time_vector, q_bounds_min[6, :] * 180 / np.pi, color='black', linewidth=0.5)
            #     axs[0, 0].plot(normalized_time_vector, q_bounds_max[6, :] * 180 / np.pi, color='black', linewidth=0.5)
            #     axs[1, 0].plot(normalized_time_vector, q_bounds_min[7, :] * 180 / np.pi, color='black', linewidth=0.5)
            #     axs[1, 0].plot(normalized_time_vector, q_bounds_max[7, :] * 180 / np.pi, color='black', linewidth=0.5)
            #     axs[0, 1].plot(normalized_time_vector, q_bounds_min[10, :] * 180 / np.pi, color='black', linewidth=0.5)
            #     axs[0, 1].plot(normalized_time_vector, q_bounds_max[10, :] * 180 / np.pi, color='black', linewidth=0.5)
            #     axs[1, 1].plot(normalized_time_vector, q_bounds_min[11, :] * 180 / np.pi, color='black', linewidth=0.5)
            #     axs[1, 1].plot(normalized_time_vector, q_bounds_max[11, :] * 180 / np.pi, color='black', linewidth=0.5)
            #     axs[0, 2].plot(normalized_time_vector, q_bounds_min[14, :] * 180 / np.pi, color='black', linewidth=0.5)
            #     axs[0, 2].plot(normalized_time_vector, q_bounds_max[14, :] * 180 / np.pi, color='black', linewidth=0.5)
            #     axs[1, 2].plot(normalized_time_vector, q_bounds_min[15, :] * 180 / np.pi, color='black', linewidth=0.5)
            #     axs[1, 2].plot(normalized_time_vector, q_bounds_max[15, :] * 180 / np.pi, color='black', linewidth=0.5)

            q = np.zeros(np.shape(Q[0][:, :-1]))
            q[:, :] = Q[0][:, :-1]
            for i in range(1, len(Q)):
                if i == len(Q) - 1:
                    q = np.hstack((q, Q[i]))
                else:
                    q = np.hstack((q, Q[i][:, :-1]))

            rgba = cmap(idx*0.7 + 0.1)
            if bounds_plotted == False:
                axs[0, 0].plot(normalized_time_vector, q[6, :] * 180 / np.pi, color=rgba, label=f"{idx+1}.5 twists")
                bounds_plotted = True
            else:
                axs[0, 0].plot(normalized_time_vector, q[6, :] * 180 / np.pi, color=rgba)
            axs[1, 0].plot(normalized_time_vector, q[7, :] * 180 / np.pi, color=rgba)

            axs[0, 1].plot(normalized_time_vector, -q[10, :] * 180 / np.pi, color=rgba)
            axs[1, 1].plot(normalized_time_vector, -q[11, :] * 180 / np.pi, color=rgba)

            axs[0, 2].plot(normalized_time_vector, q[14, :] * 180 / np.pi, color=rgba)
            axs[1, 2].plot(normalized_time_vector, q[15, :] * 180 / np.pi, color=rgba)

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

    axs[0, 0].legend(bbox_to_anchor=(1.7, -1.35), loc="upper center", ncol=2)
    plt.suptitle(f"{name}", fontsize=16)
    plt.savefig(f'kinematics_graphs/comparison_graph_for_all_athletes_{name}.png', dpi=300)
plt.show()

print("\n\n")