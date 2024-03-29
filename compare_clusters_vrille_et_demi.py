
import numpy as np
import pickle
import os
import spm1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from IPython import embed

import biorbd
import bioviz
import sys
sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
import bioptim

# ---------------------------------------------------------------------------------
# To run this code, you should run the following line:
#             python compare_clusters.py > compare_clusters.txt
# ---------------------------------------------------------------------------------


model_path = "Models/Model_TechOpt83.bioMod"
model_path_right_arm = "Models/Model_TechOpt83_right_arm.bioMod"
model_path_left_arm = "Models/Model_TechOpt83_left_arm.bioMod"
model_path_hips = "Models/Model_TechOpt83_hips.bioMod"
model = biorbd.Model(model_path)
nb_twists = 1
chosen_clusters_dict = {}
results_path = 'solutions_multi_start/'
results_path_this_time = results_path + 'Solutions_vrille_et_demi/'
cmap_viridis = cm.get_cmap('viridis')
cmap_magma = cm.get_cmap('magma')

# Define the clusters of solutions per joint
cluster_right_arm = {
    "Athlete_03":     {"cluster_1": [1, 2, 5, 6, 7, 8],             "cluster_2": [],                             "cluster_3": [],           "cluster_4": [0, 4, 9], "cluster_5": [],                          "others": []},
    "Athlete_05":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [],                             "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "others": []},
    "Athlete_18":     {"cluster_1": [],                             "cluster_2": [1, 2, 3, 7],                   "cluster_3": [],           "cluster_4": [],        "cluster_5": [0, 4, 5, 6, 8, 9],          "others": []},
    "Athlete_07":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "others": [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]},  # 8 different from the others
    "Athlete_14":     {"cluster_1": [],                             "cluster_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "others": []},
    "Athlete_17":     {"cluster_1": [],                             "cluster_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "others": []},
    "Athlete_02":     {"cluster_1": [0, 1, 2, 3, 4, 6, 7, 8, 9],    "cluster_2": [],                             "cluster_3": [],           "cluster_4": [5],       "cluster_5": [],                          "others": []},
    "Athlete_06":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [],           "cluster_4": [],        "cluster_5": [0, 1, 2, 3, 4, 5, 7, 8, 9], "others": [6]},  # 6 is between cluster 1 and 2
    "Athlete_11":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [0, 3, 8],    "cluster_4": [],        "cluster_5": [1, 2, 4, 5, 6, 7, 9],       "others": []},
    "Athlete_13":     {"cluster_1": [],                             "cluster_2": [2, 3, 7],                      "cluster_3": [4, 9],       "cluster_4": [],        "cluster_5": [1, 5, 6, 8],                "others": []},
    "Athlete_16":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [2],          "cluster_4": [],        "cluster_5": [1, 3, 4, 6, 7, 8, 9],       "others": []},
    "Athlete_12":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [1, 3],       "cluster_4": [],        "cluster_5": [2, 4, 5, 6, 7, 8, 9],       "others": []},
    "Athlete_04":     {"cluster_1": [],                             "cluster_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "others": []},
    "Athlete_10":     {"cluster_1": [],                             "cluster_2": [0, 2, 3, 4, 5, 6, 7, 8, 9],    "cluster_3": [1],          "cluster_4": [],        "cluster_5": [],                          "others": []},
    "Athlete_08":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [3, 4, 8],    "cluster_4": [],        "cluster_5": [0, 1, 2, 5, 6, 7, 9],       "others": []},
    "Athlete_09":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [0, 1, 2, 8], "cluster_4": [],        "cluster_5": [3, 4, 5, 6, 7, 9],          "others": []},
    "Athlete_01":     {"cluster_1": [],                             "cluster_2": [2, 3],                         "cluster_3": [],           "cluster_4": [],        "cluster_5": [0, 1, 4, 5, 6, 7, 8, 9],    "others": []},
    "Athlete_15":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [3],          "cluster_4": [],        "cluster_5": [0, 1, 2, 4, 5, 6, 7, 8, 9], "others": []},
}

cluster_left_arm = {
    "Athlete_03":     {"cluster_1": [0, 1, 2, 4, 5, 6, 7, 8, 9],    "cluster_2": [],                             "cluster_3": []},
    "Athlete_05":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [],                             "cluster_3": []},
    "Athlete_18":     {"cluster_1": [4, 5, 6, 8, 9],                "cluster_2": [0, 2, 3, 7],                   "cluster_3": [1]},
    "Athlete_07":     {"cluster_1": [],                             "cluster_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_3": []},
    "Athlete_14":     {"cluster_1": [5],                            "cluster_2": [0, 1, 2, 3, 4, 6, 7, 8, 9],    "cluster_3": []},
    "Athlete_17":     {"cluster_1": [0, 1, 2, 4, 5, 6, 7, 8, 9],    "cluster_2": [],                             "cluster_3": [3]},
    "Athlete_02":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [],                             "cluster_3": []},
    "Athlete_06":     {"cluster_1": [0, 1, 2, 3, 4, 5, 7, 8, 9],    "cluster_2": [6],                            "cluster_3": []},
    "Athlete_11":     {"cluster_1": [],                             "cluster_2": [1, 2, 3, 4, 5, 6, 7, 8, 9],    "cluster_3": [0]},
    "Athlete_13":     {"cluster_1": [1, 4, 5, 6, 8, 9],             "cluster_2": [2, 3, 7],                      "cluster_3": []},
    "Athlete_16":     {"cluster_1": [],                             "cluster_2": [1, 3, 4, 6, 7, 8, 9],          "cluster_3": [2]},
    "Athlete_12":     {"cluster_1": [],                             "cluster_2": [1, 2, 3, 4, 6, 7, 8, 9],       "cluster_3": []},
    "Athlete_04":     {"cluster_1": [0, 2, 4, 5, 6, 9],             "cluster_2": [1, 3, 7, 8],                   "cluster_3": []},
    "Athlete_10":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [],                             "cluster_3": []},
    "Athlete_08":     {"cluster_1": [],                             "cluster_2": [0, 1, 2, 3, 5, 6, 7, 8, 9],    "cluster_3": [4]},
    "Athlete_09":     {"cluster_1": [],                             "cluster_2": [1, 2, 3, 4, 5, 6, 7, 8, 9],    "cluster_3": [0]},
    "Athlete_01":     {"cluster_1": [],                             "cluster_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_3": []},
    "Athlete_15":     {"cluster_1": [],                             "cluster_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_3": []},
}

cluster_thighs = {
    "Athlete_03":     {"cluster_1": [0, 1, 2, 4, 5, 6, 7, 8, 9],    "cluster_2": []},
    "Athlete_05":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Athlete_18":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Athlete_07":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Athlete_14":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Athlete_17":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Athlete_02":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Athlete_06":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Athlete_11":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Athlete_13":     {"cluster_1": [1, 2, 3, 4, 5, 6, 7, 8, 9],    "cluster_2": []},
    "Athlete_16":     {"cluster_1": [],                             "cluster_2": [1, 2, 3, 4, 6, 7, 8, 9]},
    "Athlete_12":     {"cluster_1": [1, 2, 3, 4, 6, 7, 8, 9],       "cluster_2": []},
    "Athlete_04":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Athlete_10":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Athlete_08":     {"cluster_1": [0, 1, 2, 4, 5, 6, 7, 8, 9],    "cluster_2": [3]},
    "Athlete_09":     {"cluster_1": [0, 3, 4, 5, 6, 7, 9],          "cluster_2": [1, 2, 8]},
    "Athlete_01":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Athlete_15":     {"cluster_1": [0, 1, 2, 4, 5, 6, 7, 8, 9],    "cluster_2": [3]},
}

with open("q_bounds.pkl", 'rb') as f:
    q_bounds_min, q_bounds_max = pickle.load(f)

# print all the solutions at once
fig, axs = plt.subplots(2, 3, figsize=(18, 9))
q_right_arm = {"q": {key: np.zeros((16, 381, 1)) for key in cluster_right_arm['Athlete_03'].keys()},
               "normalized_time_vector": {key: np.zeros((381, 1)) for key in
                                          cluster_right_arm['Athlete_03'].keys()}}
q_left_arm = {"q": {key: np.zeros((16, 381, 1)) for key in cluster_left_arm['Athlete_03'].keys()},
              "normalized_time_vector": {key: np.zeros((381, 1)) for key in
                                         cluster_left_arm['Athlete_03'].keys()}}
q_thighs = {"q": {key: np.zeros((16, 381, 1)) for key in cluster_thighs['Athlete_03'].keys()},
            "normalized_time_vector": {key: np.zeros((381, 1)) for key in cluster_thighs['Athlete_03'].keys()}}
names = cluster_right_arm.keys()

best_solution_per_athlete = {}
for i_name, name in enumerate(names):
    best_solution_per_athlete[name] = {"random_number": None, "cost": np.inf, "q": None, "right_arm_cluster": None, "left_arm_cluster": None, "thighs_cluster": None}
    for i_sol in range(9):
        file_name = results_path_this_time + name + '/' + name + '_vrille_et_demi_' + str(i_sol) + "_CVG.pkl"
        if not os.path.isfile(file_name):
            continue
        print(file_name)
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        q = data['q'][0][:, :-1]
        for i in range(4):
            if i == 3:
                q = np.hstack((q, data['q'][i + 1]))
            else:
                q = np.hstack((q, data['q'][i + 1][:, :-1]))

        if data['sol'].cost < best_solution_per_athlete[name]["cost"]:
            best_solution_per_athlete[name]["random_number"] = i_sol
            best_solution_per_athlete[name]["cost"] = data['sol'].cost
            best_solution_per_athlete[name]["q"] = q
            for i_clust, key in enumerate(cluster_right_arm[name].keys()):
                if i_sol in cluster_right_arm[name][key]:
                    best_solution_per_athlete[name]["right_arm_cluster"] = i_clust
            for i_clust, key in enumerate(cluster_left_arm[name].keys()):
                if i_sol in cluster_left_arm[name][key]:
                    best_solution_per_athlete[name]["left_arm_cluster"] = i_clust
            for i_clust, key in enumerate(cluster_thighs[name].keys()):
                if i_sol in cluster_thighs[name][key]:
                    best_solution_per_athlete[name]["thighs_cluster"] = i_clust

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

        for i_clust, key in enumerate(cluster_right_arm[name].keys()):
            if i_sol in cluster_right_arm[name][key]:
                q_right_arm['q'][key] = np.concatenate((q_right_arm['q'][key], q[:, :, np.newaxis]), axis=2)
                q_right_arm['normalized_time_vector'][key] = np.concatenate((q_right_arm['normalized_time_vector'][key], normalized_time_vector[:, np.newaxis]), axis=1)
                i_cluster_right_arm = i_clust
                rgba = cmap_magma(1 - i_cluster_right_arm * 1/6 - 1/6)
                axs[0, 0].plot(normalized_time_vector, q[6, :] * 180/np.pi, color=rgba)
                axs[1, 0].plot(normalized_time_vector, q[7, :] * 180/np.pi, color=rgba)

        for i_clust, key in enumerate(cluster_left_arm[name].keys()):
            if i_sol in cluster_left_arm[name][key]:
                q_left_arm['q'][key] = np.concatenate((q_left_arm['q'][key], q[:, :, np.newaxis]), axis=2)
                q_left_arm['normalized_time_vector'][key] = np.concatenate((q_left_arm['normalized_time_vector'][key], normalized_time_vector[:, np.newaxis]), axis=1)
                i_cluster_left_arm = i_clust
                rgba = cmap_viridis(i_cluster_left_arm * 1/3)
                axs[0, 1].plot(normalized_time_vector, -q[10, :] * 180/np.pi, color=rgba)
                axs[1, 1].plot(normalized_time_vector, -q[11, :] * 180/np.pi, color=rgba)

        for i_clust, key in enumerate(cluster_thighs[name].keys()):
            if i_sol in cluster_thighs[name][key]:
                q_thighs['q'][key] = np.concatenate((q_thighs['q'][key], q[:, :, np.newaxis]), axis=2)
                q_thighs['normalized_time_vector'][key] = np.concatenate((q_thighs['normalized_time_vector'][key], normalized_time_vector[:, np.newaxis]), axis=1)
                i_cluster_thighs = i_clust
                rgba = cmap_viridis(1 - i_cluster_thighs * 1/6)
                axs[0, 2].plot(normalized_time_vector, q[14, :] * 180/np.pi, color=rgba)
                axs[1, 2].plot(normalized_time_vector, q[15, :] * 180/np.pi, color=rgba)

        if i_sol == 0:
            axs[0, 0].set_title(f"Change in elevation plane")  # Right arm
            axs[1, 0].set_title(f"Elevation")  # Right arm
            axs[0, 1].set_title(f"Change in elevation plane")  # Left arm
            axs[1, 1].set_title(f"Elevation")  # Left arm
            axs[0, 2].set_title(f"Flexion")  # Hips
            axs[1, 2].set_title(f"Lateral flexion")  # Hips

            axs[1, 2].set_ylim(-25, 25)
            axs[1, 0].set_xlabel(f"Normalized time")
            axs[1, 1].set_xlabel(f"Normalized time")
            axs[1, 2].set_xlabel(f"Normalized time")

for i_clust, key in enumerate(cluster_right_arm[name].keys()):
    rgba = cmap_magma(1 - i_clust * 1/6 - 1/6)
    axs[1, 0].plot(normalized_time_vector[0], q[7, 0] * 180/np.pi, color=rgba, label="Cluster #" + str(i_clust + 1))
axs[1, 0].legend(bbox_to_anchor=(0.5, -0.17), loc='upper center')

for i_clust, key in enumerate(cluster_left_arm[name].keys()):
    rgba = cmap_viridis(i_clust * 1/3)
    axs[1, 1].plot(normalized_time_vector[0], q[7, 0] * 180/np.pi, color=rgba, label="Cluster #" + str(i_clust + 1))
axs[1, 1].legend(bbox_to_anchor=(0.5, -0.17), loc='upper center')

for i_clust, key in enumerate(cluster_thighs[name].keys()):
    rgba = cmap_viridis(1 - i_clust * 1/6)
    axs[1, 2].plot(normalized_time_vector[0], q[7, 0] * 180/np.pi, color=rgba, label="Cluster #" + str(i_clust + 1))
axs[1, 2].legend(bbox_to_anchor=(0.5, -0.17), loc='upper center')

plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9)
plt.suptitle(f"{nb_twists}.5 twists")
plt.savefig(f'cluster_graphs/clusters_graph_for_all_athletes_{nb_twists}.png', dpi=300)
# plt.show()

print("\n\n")

# Find the mean and std of each cluster
cluster_counter_right_arm = {key: 0 for key in cluster_right_arm["Athlete_05"].keys()}
optimal_counter_right_arm = {key: 0 for key in cluster_right_arm["Athlete_05"].keys()}
mean_std_per_cluster_right_arm = {key: np.zeros((16, )) for key in cluster_right_arm["Athlete_05"].keys()}
mean_q_per_cluster_right_arm = {'q': np.zeros((16, 381, 1)), 'normalized_time_vector': np.zeros((381, 1))}
std_q_per_cluster_right_arm = np.zeros((16, 381, 1))
range_q_per_cluster_right_arm = np.zeros((16, 381, 1))
for i_cluster, cluster_name in enumerate(cluster_right_arm['Athlete_05'].keys()):
    q_right_arm['q'][cluster_name] = q_right_arm['q'][cluster_name][:, :, 1:]
    q_right_arm['normalized_time_vector'][cluster_name] = q_right_arm['normalized_time_vector'][cluster_name][:, 1:]
    for i_name, name in enumerate(cluster_right_arm):
        if len(cluster_right_arm[name][cluster_name]) > 0:
            cluster_counter_right_arm[cluster_name] += 1
            if best_solution_per_athlete[name]["right_arm_cluster"] == i_cluster:
                optimal_counter_right_arm[cluster_name] += 1
    mean_std_per_cluster_right_arm[cluster_name] = np.mean(np.std(q_right_arm['q'][cluster_name], axis=2), axis=1)
    mean_q_per_cluster_right_arm['q'] = np.concatenate((mean_q_per_cluster_right_arm['q'], np.mean(q_right_arm['q'][cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
    mean_q_per_cluster_right_arm['normalized_time_vector'] = np.concatenate((mean_q_per_cluster_right_arm['normalized_time_vector'], np.mean(q_right_arm['normalized_time_vector'][cluster_name], axis=1)[:, np.newaxis]), axis=1)
    std_q_per_cluster_right_arm = np.concatenate((std_q_per_cluster_right_arm, np.std(q_right_arm['q'][cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
    min_curve = np.min(q_right_arm['q'][cluster_name], axis=2)[: , :, np.newaxis]
    max_curve = np.max(q_right_arm['q'][cluster_name], axis=2)[: , :, np.newaxis]
    complete_range_curves = max_curve - min_curve
    range_q_per_cluster_right_arm = np.concatenate((range_q_per_cluster_right_arm, complete_range_curves), axis=2)
mean_q_per_cluster_right_arm['q'] = mean_q_per_cluster_right_arm['q'][:, :, 1:]
mean_q_per_cluster_right_arm['normalized_time_vector'] = mean_q_per_cluster_right_arm['normalized_time_vector'][:, 1:]
std_q_per_cluster_right_arm = std_q_per_cluster_right_arm[:, :, 1:]
range_q_per_cluster_right_arm = range_q_per_cluster_right_arm[:, :, 1:]
mean_std_between_clusters_right_arm = np.mean(np.std(mean_q_per_cluster_right_arm['q'], axis=2), axis=1)


cluster_counter_left_arm = {key: 0 for key in cluster_left_arm['Athlete_05'].keys()}
optimal_counter_left_arm = {key: 0 for key in cluster_left_arm['Athlete_05'].keys()}
mean_std_per_cluster_left_arm = {key: np.zeros((16, )) for key in cluster_left_arm['Athlete_05'].keys()}
mean_q_per_cluster_left_arm = {'q': np.zeros((16, 381, 1)), 'normalized_time_vector': np.zeros((381, 1))}
std_q_per_cluster_left_arm = np.zeros((16, 381, 1))
range_q_per_cluster_left_arm = np.zeros((16, 381, 1))
for i_cluster, cluster_name in enumerate(cluster_left_arm['Athlete_05'].keys()):
    q_left_arm['q'][cluster_name] = q_left_arm['q'][cluster_name][:, :, 1:]
    for i_name, name in enumerate(cluster_left_arm):
        if len(cluster_left_arm[name][cluster_name]) > 0:
            cluster_counter_left_arm[cluster_name] += 1
            if best_solution_per_athlete[name]["left_arm_cluster"] == i_cluster:
                optimal_counter_left_arm[cluster_name] += 1
    mean_std_per_cluster_left_arm[cluster_name] = np.mean(np.std(q_left_arm['q'][cluster_name], axis=2), axis=1)
    mean_q_per_cluster_left_arm['q'] = np.concatenate((mean_q_per_cluster_left_arm['q'], np.mean(q_left_arm['q'][cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
    mean_q_per_cluster_left_arm['normalized_time_vector'] = np.concatenate((mean_q_per_cluster_left_arm['normalized_time_vector'], np.mean(q_left_arm['normalized_time_vector'][cluster_name], axis=1)[:, np.newaxis]), axis=1)
    std_q_per_cluster_left_arm = np.concatenate((std_q_per_cluster_left_arm, np.std(q_left_arm['q'][cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
    min_curve = np.min(q_left_arm['q'][cluster_name], axis=2)[: , :, np.newaxis]
    max_curve = np.max(q_left_arm['q'][cluster_name], axis=2)[: , :, np.newaxis]
    complete_range_curves = max_curve - min_curve
    range_q_per_cluster_left_arm = np.concatenate((range_q_per_cluster_left_arm, complete_range_curves), axis=2)
mean_q_per_cluster_left_arm['q'] = mean_q_per_cluster_left_arm['q'][:, :, 1:]
mean_q_per_cluster_left_arm['normalized_time_vector'] = mean_q_per_cluster_left_arm['normalized_time_vector'][:, 1:]
std_q_per_cluster_left_arm = std_q_per_cluster_left_arm[:, :, 1:]
range_q_per_cluster_left_arm = range_q_per_cluster_left_arm[:, :, 1:]
mean_std_between_clusters_left_arm = np.mean(np.std(mean_q_per_cluster_left_arm['q'], axis=2), axis=1)


cluster_counter_thighs = {key: 0 for key in cluster_thighs['Athlete_05'].keys()}
optimal_counter_thighs = {key: 0 for key in cluster_thighs['Athlete_05'].keys()}
mean_std_per_cluster_thighs = {key: np.zeros((16, )) for key in cluster_thighs['Athlete_05'].keys()}
mean_q_per_cluster_thighs = {'q': np.zeros((16, 381, 1)), 'normalized_time_vector': np.zeros((381, 1))}
std_q_per_cluster_thighs = np.zeros((16, 381, 1))
range_q_per_cluster_thighs = np.zeros((16, 381, 1))
for i_cluster, cluster_name in enumerate(cluster_thighs['Athlete_05'].keys()):
    q_thighs['q'][cluster_name] = q_thighs['q'][cluster_name][:, :, 1:]
    for i_name, name in enumerate(cluster_thighs):
        if len(cluster_thighs[name][cluster_name]) > 0:
            cluster_counter_thighs[cluster_name] += 1
            if best_solution_per_athlete[name]["thighs_cluster"] == i_cluster:
                optimal_counter_thighs[cluster_name] += 1
    mean_std_per_cluster_thighs[cluster_name] = np.mean(np.std(q_thighs['q'][cluster_name], axis=2), axis=1)
    mean_q_per_cluster_thighs['q'] = np.concatenate((mean_q_per_cluster_thighs['q'], np.mean(q_thighs['q'][cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
    mean_q_per_cluster_thighs['normalized_time_vector'] = np.concatenate((mean_q_per_cluster_thighs['normalized_time_vector'], np.mean(q_thighs['normalized_time_vector'][cluster_name], axis=1)[:, np.newaxis]), axis=1)
    std_q_per_cluster_thighs = np.concatenate((std_q_per_cluster_thighs, np.std(q_thighs['q'][cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
    min_curve = np.min(q_thighs['q'][cluster_name], axis=2)[: , :, np.newaxis]
    max_curve = np.max(q_thighs['q'][cluster_name], axis=2)[: , :, np.newaxis]
    complete_range_curves = max_curve - min_curve
    range_q_per_cluster_thighs = np.concatenate((range_q_per_cluster_thighs, complete_range_curves), axis=2)
mean_q_per_cluster_thighs['q'] = mean_q_per_cluster_thighs['q'][:, :, 1:]
mean_q_per_cluster_thighs['normalized_time_vector'] = mean_q_per_cluster_thighs['normalized_time_vector'][:, 1:]
std_q_per_cluster_thighs = std_q_per_cluster_thighs[:, :, 1:]
range_q_per_cluster_thighs = range_q_per_cluster_thighs[:, :, 1:]
mean_std_between_clusters_thighs = np.mean(np.std(mean_q_per_cluster_thighs['q'], axis=2), axis=1)

total_number_of_strategies_identified = q_thighs['q']["cluster_1"].shape[2] + q_thighs['q']["cluster_2"].shape[2]

print("Right arm clusters:")
# Plot the clusters with different colors
fig, axs = plt.subplots(2, 3, figsize=(18, 9))
for i_cluster, cluster_name in enumerate(cluster_right_arm['Athlete_05'].keys()):
    print(f"{cluster_name} was used by {cluster_counter_right_arm[cluster_name]} / {len(cluster_right_arm)} athletes and in {q_right_arm['q'][cluster_name].shape[2]/total_number_of_strategies_identified * 100}% of the cases")
    print(f"{cluster_name} was optimal for {optimal_counter_right_arm[cluster_name]} / 18 athletes = {optimal_counter_right_arm[cluster_name]/18 * 100}%")
    print(f"Sum of mean std on cluster {cluster_name} was {np.sum(mean_std_per_cluster_right_arm[cluster_name][3:])}")
    print(f"{cluster_name} has a right arm axial rotation range of {np.mean(range_q_per_cluster_right_arm[:, :, i_cluster][6, :]) / (np.max(mean_q_per_cluster_right_arm['q'][:, :, i_cluster][6, :]) - np.min(mean_q_per_cluster_right_arm['q'][:, :, i_cluster][6, :])) * 100}% of the average movement amplitude")
    print(f"{cluster_name} has a right arm elevation range of {np.mean(range_q_per_cluster_right_arm[:, :, i_cluster][7, :]) / (np.max(mean_q_per_cluster_right_arm['q'][:, :, i_cluster][7, :]) - np.min(mean_q_per_cluster_right_arm['q'][:, :, i_cluster][7, :])) * 100}% of the average max amplitude")

    if i_cluster == 0:
        axs[0, 0].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_min[6, :] * 180 / np.pi, color='black', linewidth=0.5)
        axs[0, 0].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_max[6, :] * 180 / np.pi, color='black', linewidth=0.5)
        axs[1, 0].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_min[7, :] * 180 / np.pi, color='black', linewidth=0.5)
        axs[1, 0].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_max[7, :] * 180 / np.pi, color='black', linewidth=0.5)

    rgba = cmap_magma(1 - i_cluster * 1/6 - 1/6)
    if i_cluster < 5:
        axs[0, 0].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], mean_q_per_cluster_right_arm['q'][6, :, i_cluster] * 180/np.pi - std_q_per_cluster_right_arm[6, :, i_cluster] * 180/np.pi,
                            mean_q_per_cluster_right_arm['q'][6, :, i_cluster] * 180/np.pi + std_q_per_cluster_right_arm[6, :,i_cluster] * 180/np.pi, color=rgba, alpha=0.2)
        axs[0, 0].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], mean_q_per_cluster_right_arm['q'][6, :, i_cluster] * 180/np.pi, color=rgba)
        axs[1, 0].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], mean_q_per_cluster_right_arm['q'][7, :, i_cluster] * 180/np.pi - std_q_per_cluster_right_arm[7, :, i_cluster] * 180/np.pi,
                            mean_q_per_cluster_right_arm['q'][7, :, i_cluster] * 180/np.pi + std_q_per_cluster_right_arm[7, :,i_cluster] * 180/np.pi, color=rgba, alpha=0.2)
        axs[1, 0].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], mean_q_per_cluster_right_arm['q'][7, :, i_cluster] * 180/np.pi, color=rgba, label="Cluster #" + str(i_cluster + 1))
    if i_cluster == 0:
        axs[0, 0].set_title(f"Change in elevation plane")  # Right arm
        axs[1, 0].set_title(f"Elevation")  # Right arm
    print('\n')
print('\n')
axs[1, 0].set_xlabel(f"Normalized time")
axs[1, 0].legend(bbox_to_anchor=(0.5, -0.17), loc='upper center')

print("Left arm clusters:")
for i_cluster, cluster_name in enumerate(cluster_left_arm['Athlete_05'].keys()):
    print(f"{cluster_name} was used by {cluster_counter_left_arm[cluster_name]} / {len(cluster_left_arm)} athletes and in {q_left_arm['q'][cluster_name].shape[2] / total_number_of_strategies_identified * 100}% of the cases")
    print(f"{cluster_name} was optimal for {optimal_counter_left_arm[cluster_name]} / 18 athletes = {optimal_counter_left_arm[cluster_name] / 18 * 100}%")
    print(f"Sum of mean std on cluster {cluster_name} was {np.sum(mean_std_per_cluster_left_arm[cluster_name][3:])}")
    print(f"{cluster_name} has a left arm axial rotation range of {np.mean(range_q_per_cluster_left_arm[:, :, i_cluster][10, :]) / (np.max(mean_q_per_cluster_left_arm['q'][:, :, i_cluster][10, :]) - np.min(mean_q_per_cluster_left_arm['q'][:, :, i_cluster][10, :])) * 100}% of the average movement amplitude")
    print(f"{cluster_name} has a left arm elevation range of {np.mean(range_q_per_cluster_left_arm[:, :, i_cluster][11, :]) / (np.max(mean_q_per_cluster_left_arm['q'][:, :, i_cluster][11, :]) - np.min(mean_q_per_cluster_left_arm['q'][:, :, i_cluster][11, :])) * 100}% of the average max amplitude")

    if i_cluster == 0:
        axs[0, 1].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_min[10, :] * 180 / np.pi, color='black', linewidth=0.5)
        axs[0, 1].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_max[10, :] * 180 / np.pi, color='black', linewidth=0.5)
        axs[1, 1].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_min[11, :] * 180 / np.pi, color='black', linewidth=0.5)
        axs[1, 1].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_max[11, :] * 180 / np.pi, color='black', linewidth=0.5)

    rgba = cmap_viridis(i_cluster * 1/3)
    axs[0, 1].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], -mean_q_per_cluster_left_arm['q'][10, :, i_cluster] * 180/np.pi - std_q_per_cluster_left_arm[10, :, i_cluster] * 180/np.pi,
                        -mean_q_per_cluster_left_arm['q'][10, :, i_cluster] * 180/np.pi + std_q_per_cluster_left_arm[10, :,i_cluster] * 180/np.pi, color=rgba, alpha=0.2)
    axs[0, 1].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], -mean_q_per_cluster_left_arm['q'][10, :, i_cluster] * 180/np.pi, color=rgba)
    axs[1, 1].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], -mean_q_per_cluster_left_arm['q'][11, :, i_cluster] * 180/np.pi - std_q_per_cluster_left_arm[11, :, i_cluster] * 180/np.pi,
                        -mean_q_per_cluster_left_arm['q'][11, :, i_cluster] * 180/np.pi + std_q_per_cluster_left_arm[11, :,i_cluster] * 180/np.pi, color=rgba, alpha=0.2)
    axs[1, 1].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], -mean_q_per_cluster_left_arm['q'][11, :, i_cluster] * 180/np.pi, color=rgba, label="Cluster #" + str(i_cluster + 1))
    if i_cluster == 0:
        axs[0, 1].set_title(f"Change in elevation plane")  # Left arm
        axs[1, 1].set_title(f"Elevation")  # Left arm
    print('\n')
print('\n')
axs[1, 1].set_xlabel(f"Normalized time")
axs[1, 1].legend(bbox_to_anchor=(0.5, -0.17), loc='upper center')

print("Thigh clusters:")
for i_cluster, cluster_name in enumerate(cluster_thighs['Athlete_05'].keys()):
    print(f"{cluster_name} was used by {cluster_counter_thighs[cluster_name]} / {len(cluster_thighs)} athletes  and in {q_thighs['q'][cluster_name].shape[2] / total_number_of_strategies_identified * 100}% of the cases")
    print(f"{cluster_name} was optimal for {optimal_counter_thighs[cluster_name]} / 18 athletes = {optimal_counter_thighs[cluster_name] / 18 * 100}%")
    print(f"Sum of mean std on cluster {cluster_name} was {np.sum(mean_std_per_cluster_thighs[cluster_name][3:])}")
    print(f"{cluster_name} has a hip flexion range of {np.mean(range_q_per_cluster_thighs[:, :, i_cluster][14, :]) / (np.max(mean_q_per_cluster_thighs['q'][:, :, i_cluster][14, :]) - np.min(mean_q_per_cluster_thighs['q'][:, :, i_cluster][14, :])) * 100}% of the average movement amplitude")
    print(f"{cluster_name} has a hip lateral flexion range of {np.mean(range_q_per_cluster_thighs[:, :, i_cluster][15, :]) / (np.max(mean_q_per_cluster_thighs['q'][:, :, i_cluster][15, :]) - np.min(mean_q_per_cluster_thighs['q'][:, :, i_cluster][15, :])) * 100}% of the average max amplitude")
    print(np.min(mean_q_per_cluster_thighs['q'][:, :, i_cluster][15, :]) * 180 / np.pi)
    print(np.max(mean_q_per_cluster_thighs['q'][:, :, i_cluster][15, :]) * 180 / np.pi)
    print(np.mean(range_q_per_cluster_thighs[:, :, i_cluster][15, :]) * 180 / np.pi)

    if i_cluster == 0:
        axs[0, 2].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_min[14, :] * 180 / np.pi, color='black', linewidth=0.5)
        axs[0, 2].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_max[14, :] * 180 / np.pi, color='black', linewidth=0.5)
        axs[1, 2].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_min[15, :] * 180 / np.pi, color='black', linewidth=0.5)
        axs[1, 2].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], q_bounds_max[15, :] * 180 / np.pi, color='black', linewidth=0.5)

    rgba = cmap_viridis(1 - i_cluster * 1/6)
    axs[0, 2].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], mean_q_per_cluster_thighs['q'][14, :, i_cluster] * 180/np.pi - std_q_per_cluster_thighs[14, :, i_cluster] * 180/np.pi,
                        mean_q_per_cluster_thighs['q'][14, :, i_cluster] * 180/np.pi + std_q_per_cluster_thighs[14, :, i_cluster] * 180/np.pi, color=rgba, alpha=0.2)
    axs[0, 2].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], mean_q_per_cluster_thighs['q'][14, :, i_cluster] * 180/np.pi, color=rgba)
    axs[1, 2].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], mean_q_per_cluster_thighs['q'][15, :, i_cluster] * 180/np.pi - std_q_per_cluster_thighs[15, :, i_cluster] * 180/np.pi,
                        mean_q_per_cluster_thighs['q'][15, :, i_cluster] * 180/np.pi + std_q_per_cluster_thighs[15, :, i_cluster] * 180/np.pi, color=rgba, alpha=0.2)
    axs[1, 2].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], mean_q_per_cluster_thighs['q'][15, :, i_cluster] * 180/np.pi, color=rgba, label="Cluster #" + str(i_cluster + 1))
    if i_cluster == 0:
        axs[0, 2].set_title(f"Flexion")  # Hips
        axs[1, 2].set_title(f"Lateral flexion")  # Hips
    print('\n')
print('\n')
axs[1, 2].set_ylim(-25, 25)
axs[1, 2].set_xlabel(f"Normalized time")
axs[1, 2].legend(bbox_to_anchor=(0.5, -0.17), loc='upper center')

plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9)
plt.suptitle(f"mean kinematics per cluster for {nb_twists}.5 twists")
plt.savefig(f'cluster_graphs/mean_clusters_graph_for_all_athletes_{nb_twists}.png', dpi=300)
# plt.show()

def find_significant_timing_blocks(dof_1, dof_2, q, cluster_name1, cluster_name2):
    significant_timings = []

    # Rot 1
    t = spm1d.stats.ttest2(q["q"][cluster_name1][dof_1, 1:-1, :].T,
                           q["q"][cluster_name2][dof_1, 1:-1, :].T)
    ti = t.inference(alpha=0.05, two_tailed=True)
    if ti.h0reject == True:
        if ti.clusters != []:
            significant_timing_list = ti.clusters
            for k in range(len(significant_timing_list)):
                significant_timing_1_x, _ = significant_timing_list[k].get_patch_vertices()
                significant_timings += list(range(int(significant_timing_1_x[1]+1), int(significant_timing_1_x[-2]+2)))
    # Rot 2
    t = spm1d.stats.ttest2(q["q"][cluster_name1][dof_2, 1:-1, :].T,
                           q["q"][cluster_name2][dof_2, 1:-1, :].T)
    ti = t.inference(alpha=0.05, two_tailed=True)
    if ti.h0reject == True:
        if ti.clusters != []:
            significant_timing_list = ti.clusters
            for k in range(len(significant_timing_list)):
                significant_timing_2_x, _ = significant_timing_list[k].get_patch_vertices()
                significant_timings += list(range(int(significant_timing_2_x[1]+1), int(significant_timing_2_x[-2]+2)))

    # Remove duplicates from list and sort
    significant_timings = list(set(significant_timings))
    # Find blocks of consecutive indices
    significant_timings_array = np.array(significant_timings)
    significant_timings_diff = significant_timings_array[1:] - significant_timings_array[:-1]
    significant_timings_index = np.hstack((-1, np.where(significant_timings_diff > 1)[0], len(significant_timings_array)-1))
    ranges = []
    for i in range(len(significant_timings_index)-1):
        ranges += [range(significant_timings_array[significant_timings_index[i]+1], significant_timings_array[significant_timings_index[i+1]])]

    return ranges, len(significant_timings)/381

# SPM1D between clusters
cluster_names_right_arm = list(cluster_right_arm['Athlete_05'].keys())[:-1]
fig, axs = plt.subplots(1, 3, figsize=(18, 4))
significant_timings = {"right_arm": {}, "left_arm": {}, "thighs": {}}
num_combinaition = 0
for i_cluster1, cluster_name1 in enumerate(cluster_names_right_arm):
    for i_cluster2, cluster_name2 in enumerate(cluster_names_right_arm[i_cluster1+1:]):
        timings, proportion = find_significant_timing_blocks(6, 7, q_right_arm, cluster_name1, cluster_name2)
        significant_timings["right_arm"][f"{cluster_name1}_vs_{cluster_name2}"] = timings
        rgba_1 = cmap_magma(1 - list(cluster_right_arm['Athlete_05'].keys()).index(cluster_name1) * 1 / 6 - 1 / 6)
        rgba_2 = cmap_magma(1 - list(cluster_right_arm['Athlete_05'].keys()).index(cluster_name2) * 1 / 6 - 1 / 6)
        for timing_this_time in timings:
            axs[0].fill_between([timing_this_time[0]/381, timing_this_time[-1]/381], [-num_combinaition, -num_combinaition], [-num_combinaition+0.25, -num_combinaition+0.25], color=rgba_1, alpha=0.8)
            axs[0].fill_between([timing_this_time[0]/381, timing_this_time[-1]/381], [-num_combinaition-0.25, -num_combinaition-0.25], [-num_combinaition, -num_combinaition], color=rgba_2, alpha=0.8)
        print(f"Right arm : {cluster_name1} vs {cluster_name2} has {proportion * 100}% of the time with significant differences")
        num_combinaition += 1

num_combinaition = 0
for i_cluster1, cluster_name1 in enumerate(cluster_left_arm['Athlete_05'].keys()):
    for i_cluster2, cluster_name2 in enumerate(list(cluster_left_arm['Athlete_05'].keys())[i_cluster1+1:]):
        timings, proportion = find_significant_timing_blocks(10, 11, q_left_arm, cluster_name1, cluster_name2)
        significant_timings["left_arm"][f"{cluster_name1}_vs_{cluster_name2}"] = timings
        rgba_1 = cmap_viridis(list(cluster_left_arm['Athlete_05'].keys()).index(cluster_name1) * 1/3)
        rgba_2 = cmap_viridis(list(cluster_left_arm['Athlete_05'].keys()).index(cluster_name2) * 1/3)
        for timing_this_time in timings:
            axs[1].fill_between([timing_this_time[0]/381, timing_this_time[-1]/381], [-num_combinaition, -num_combinaition], [-num_combinaition+0.25, -num_combinaition+0.25], color=rgba_1, alpha=0.8)
            axs[1].fill_between([timing_this_time[0]/381, timing_this_time[-1]/381], [-num_combinaition-0.25, -num_combinaition-0.25], [-num_combinaition, -num_combinaition], color=rgba_2, alpha=0.8)
        print(f"Left arm: {cluster_name1} vs {cluster_name2} has {proportion * 100}% of the time with significant differences")
        num_combinaition += 1

num_combinaition = 0
for i_cluster1, cluster_name1 in enumerate(cluster_thighs['Athlete_05'].keys()):
    for i_cluster2, cluster_name2 in enumerate(list(cluster_thighs['Athlete_05'].keys())[i_cluster1+1:]):
        timings, proportion = find_significant_timing_blocks(10, 11, q_left_arm, cluster_name1, cluster_name2)
        significant_timings["thighs"][f"{cluster_name1}_vs_{cluster_name2}"] = timings
        rgba_1 = cmap_viridis(1 - list(cluster_thighs['Athlete_05'].keys()).index(cluster_name1) * 1/6)
        rgba_2 = cmap_viridis(1 - list(cluster_thighs['Athlete_05'].keys()).index(cluster_name2) * 1/6)
        for timing_this_time in timings:
            axs[2].fill_between([timing_this_time[0]/381, timing_this_time[-1]/381], [-num_combinaition, -num_combinaition], [-num_combinaition+0.25, -num_combinaition+0.25], color=rgba_1, alpha=0.5)
            axs[2].fill_between([timing_this_time[0]/381, timing_this_time[-1]/381], [-num_combinaition-0.25, -num_combinaition-0.25], [-num_combinaition, -num_combinaition], color=rgba_2, alpha=0.5)
        print(f"Thighs: {cluster_name1} vs {cluster_name2} has {proportion * 100}% of the time with significant differences")
        num_combinaition += 1

for i in range(3):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].set_ylim(-10, 1)
    axs[i].get_yaxis().set_ticks([])

plt.savefig("cluster_graphs/significant_timing_blocks.png", dpi=300)
plt.show()

# Right arm : cluster_1 vs cluster_2 has 99.21259842519686% of the time with significant differences
# Right arm : cluster_1 vs cluster_3 has 99.21259842519686% of the time with significant differences
# Right arm : cluster_1 vs cluster_4 has 85.56430446194226% of the time with significant differences
# Right arm : cluster_1 vs cluster_5 has 99.21259842519686% of the time with significant differences
# Right arm : cluster_2 vs cluster_3 has 92.1259842519685% of the time with significant differences
# Right arm : cluster_2 vs cluster_4 has 95.8005249343832% of the time with significant differences
# Right arm : cluster_2 vs cluster_5 has 95.2755905511811% of the time with significant differences
# Right arm : cluster_3 vs cluster_4 has 97.63779527559055% of the time with significant differences
# Right arm : cluster_3 vs cluster_5 has 72.44094488188976% of the time with significant differences
# Right arm : cluster_4 vs cluster_5 has 96.8503937007874% of the time with significant differences
# Left arm: cluster_1 vs cluster_2 has 96.3254593175853% of the time with significant differences
# Left arm: cluster_1 vs cluster_3 has 97.11286089238845% of the time with significant differences
# Left arm: cluster_2 vs cluster_3 has 61.679790026246714% of the time with significant differences
# Thighs: cluster_1 vs cluster_2 has 96.3254593175853% of the time with significant differences

data_to_save = {"mean_q_per_cluster_right_arm": mean_q_per_cluster_right_arm,
                "mean_q_per_cluster_left_arm": mean_q_per_cluster_left_arm,
                "mean_q_per_cluster_thighs": mean_q_per_cluster_thighs,
                "std_q_per_cluster_right_arm": std_q_per_cluster_right_arm,
                "std_q_per_cluster_left_arm": std_q_per_cluster_left_arm,
                "std_q_per_cluster_thighs": std_q_per_cluster_thighs,
                "q_right_arm" : q_right_arm,
                "q_left_arm" : q_left_arm,
                "q_thighs" : q_thighs,
                "cluster_right_arm" : cluster_right_arm,
                "cluster_left_arm" : cluster_left_arm,
                "cluster_thighs": cluster_thighs,
                "best_solution_per_athlete": best_solution_per_athlete,
                "significant_timings": significant_timings,}


with open(f'overview_graphs/clusters_sol.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)


# Plot the clusters one by one to make sure they were correctly identified
var_name = ["right_arm", "left_arm", "thighs"]
var_list = [q_right_arm['q'], q_left_arm['q'], q_thighs['q']]
DoF_index = [[6, 7], [10, 11], [14, 15]]
for i_var in range(len(var_list)):
    for key in var_list[i_var].keys():
        fig, axs = plt.subplots(1, 2)
        axs = axs.ravel()
        for i in range(len(DoF_index[i_var])):
            axs[i].plot(var_list[i_var][key][DoF_index[i_var][i], :, :])
        plt.suptitle(key)
        plt.savefig(f'cluster_graphs/test_{var_name[i_var]}_{key}_graph_for_all_athletes_{nb_twists}.png', dpi=300)
# plt.show()


# Generate animations of the movements in the clusters
for i_cluster, cluster_name in enumerate(cluster_right_arm['Athlete_05'].keys()):
    if i_cluster < 5:
        print(f"right_arm_{cluster_name}")
        Q_to_animate = np.zeros((model.nbQ(), 381))
        Q_to_animate[5, :] = np.pi/2
        Q_to_animate[6, :] = mean_q_per_cluster_right_arm['q'][6, :, i_cluster]
        Q_to_animate[7, :] = mean_q_per_cluster_right_arm['q'][7, :, i_cluster]
        Q_to_animate[8, :] = mean_q_per_cluster_right_arm['q'][8, :, i_cluster]
        Q_to_animate[9, :] = mean_q_per_cluster_right_arm['q'][9, :, i_cluster]

        # print(f"Animating right arm Cluster #{i_cluster+1}")
        # b = bioviz.Viz(model_path_right_arm,
        #                mesh_opacity=1,
        #                show_global_center_of_mass=False,
        #                show_gravity_vector=False,
        #                show_segments_center_of_mass=False,
        #                show_global_ref_frame=False,
        #                show_local_ref_frame=False,
        #                show_markers=False,
        #                background_color=(1.0, 1.0, 1.0)
        #                )
        # b.set_camera_zoom(0.5)
        # b.load_movement(Q_to_animate)
        # b.exec()

for i_cluster, cluster_name in enumerate(cluster_left_arm['Athlete_05'].keys()):
    print(f"left_arm_{cluster_name}")
    Q_to_animate = np.zeros((model.nbQ(), 381))
    Q_to_animate[5, :] = np.pi/2
    Q_to_animate[10, :] = mean_q_per_cluster_left_arm['q'][10, :, i_cluster]
    Q_to_animate[11, :] = mean_q_per_cluster_left_arm['q'][11, :, i_cluster]
    Q_to_animate[12, :] = mean_q_per_cluster_left_arm['q'][12, :, i_cluster]
    Q_to_animate[13, :] = mean_q_per_cluster_left_arm['q'][13, :, i_cluster]

    print(f"Animating left arm Cluster #{i_cluster + 1}")
    b = bioviz.Viz(model_path_left_arm,
                   mesh_opacity=1,
                   show_global_center_of_mass=False,
                   show_gravity_vector=False,
                   show_segments_center_of_mass=False,
                   show_global_ref_frame=False,
                   show_local_ref_frame=False,
                   show_markers=False,
                   background_color=(1.0, 1.0, 1.0)
                   )
    b.set_camera_zoom(0.5)
    b.load_movement(Q_to_animate)
    b.exec()


for i_cluster, cluster_name in enumerate(cluster_thighs['Athlete_05'].keys()):
    print(f"thighs_{cluster_name}")
    Q_to_animate = np.zeros((model.nbQ(), 381))
    Q_to_animate[5, :] = np.pi/2
    Q_to_animate[14, :] = mean_q_per_cluster_thighs['q'][14, :, i_cluster]
    Q_to_animate[15, :] = mean_q_per_cluster_thighs['q'][15, :, i_cluster]

    print(f"Animating hips Cluster #{i_cluster + 1}")
    b = bioviz.Viz(model_path_hips,
                   mesh_opacity=1,
                   show_global_center_of_mass=False,
                   show_gravity_vector=False,
                   show_segments_center_of_mass=False,
                   show_global_ref_frame=False,
                   show_local_ref_frame=False,
                   show_markers=False,
                   background_color=(1.0, 1.0, 1.0)
                   )
    b.set_camera_zoom(0.5)
    b.load_movement(Q_to_animate)
    b.exec()


# Plot the proportion of solutions that are in each cluster
num_clusters_thighs_techniques = np.zeros((len(cluster_thighs['Athlete_05'].keys()), ))
num_clusters_left_arm_techniques = np.zeros((len(cluster_thighs['Athlete_05'].keys()), len(cluster_left_arm['Athlete_05'].keys())))
num_clusters_right_arm_techniques = np.zeros((len(cluster_thighs['Athlete_05'].keys()), len(cluster_left_arm['Athlete_05'].keys()), len(cluster_right_arm['Athlete_05'].keys())))
for i_cluster_thighs, cluster_thighs_name in enumerate(cluster_thighs['Athlete_05'].keys()):
    for i_name, name in enumerate(cluster_thighs.keys()):
        num_clusters_thighs_techniques[i_cluster_thighs] += len(cluster_thighs[name][cluster_thighs_name])

    for i_cluster_left_arm, cluster_left_arm_name in enumerate(cluster_left_arm['Athlete_05'].keys()):
        for i_name, name in enumerate(cluster_left_arm.keys()):
            for idx_tech in cluster_left_arm[name][cluster_left_arm_name]:
                if idx_tech in cluster_thighs[name][cluster_thighs_name]:
                    num_clusters_left_arm_techniques[i_cluster_thighs, i_cluster_left_arm] += 1

        for i_cluster_right_arm, cluster_right_arm_name in enumerate(cluster_right_arm['Athlete_05'].keys()):
            for i_name, name in enumerate(cluster_right_arm.keys()):
                for idx_tech in cluster_right_arm[name][cluster_right_arm_name]:
                    if idx_tech in cluster_thighs[name][cluster_thighs_name] and idx_tech in cluster_left_arm[name][cluster_left_arm_name]:
                        num_clusters_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, i_cluster_right_arm] += 1

pourcentage_clusters_thighs_techniques = num_clusters_thighs_techniques / np.sum(num_clusters_thighs_techniques) * 100
pourcentage_clusters_left_arm_techniques = np.zeros(num_clusters_left_arm_techniques.shape)
pourcentage_clusters_right_arm_techniques = np.zeros(num_clusters_right_arm_techniques.shape)
for i_cluster_thighs, cluster_thighs_name in enumerate(cluster_thighs['Athlete_05'].keys()):
    pourcentage_clusters_left_arm_techniques[i_cluster_thighs, :] = num_clusters_left_arm_techniques[i_cluster_thighs, :] / np.sum(num_clusters_thighs_techniques[i_cluster_thighs]) * pourcentage_clusters_thighs_techniques[i_cluster_thighs]
    for i_cluster_left_arm, cluster_left_arm_name in enumerate(cluster_left_arm['Athlete_05'].keys()):
        pourcentage_clusters_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, :] = num_clusters_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, :] / np.sum(num_clusters_left_arm_techniques[i_cluster_thighs, i_cluster_left_arm]) * pourcentage_clusters_left_arm_techniques[i_cluster_thighs, i_cluster_left_arm]

offset_left_arm = 0
offset_right_arm = 0
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for i_cluster_thighs in range(len(cluster_thighs['Athlete_05'].keys())):
    x_min_thighs = pourcentage_clusters_thighs_techniques[:i_cluster_thighs].sum()
    x_max_thighs = pourcentage_clusters_thighs_techniques[:i_cluster_thighs + 1].sum()
    ax.fill_between([x_min_thighs, x_max_thighs], [40, 40], [50, 50], color=cmap_viridis(1 - i_cluster_thighs * 1/6), alpha=0.8, linewidth=0.0)

    for i_cluster_left_arm in range(len(cluster_left_arm['Athlete_05'].keys())):
        x_min_left_arm = pourcentage_clusters_left_arm_techniques[i_cluster_thighs, :i_cluster_left_arm].sum()
        x_max_left_arm = pourcentage_clusters_left_arm_techniques[i_cluster_thighs, :i_cluster_left_arm + 1].sum()
        ax.fill_between([x_min_left_arm + offset_left_arm, x_max_left_arm + offset_left_arm], [20, 20], [30, 30], color=cmap_viridis(i_cluster_left_arm * 1/3), alpha=0.8, linewidth=0.0)

        for i_cluster_right_arm in range(len(cluster_right_arm['Athlete_05'].keys())):
            x_min_right_arm = np.nansum(pourcentage_clusters_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, :i_cluster_right_arm])
            x_max_right_arm = np.nansum(pourcentage_clusters_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, :i_cluster_right_arm + 1])
            ax.fill_between([x_min_right_arm + offset_right_arm, x_max_right_arm + offset_right_arm], [0, 0], [10, 10], color=cmap_magma(1 - i_cluster_right_arm * 1/6 - 1/6), alpha=0.8, linewidth=0.0)
            print(f"{pourcentage_clusters_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, i_cluster_right_arm]}% hip_{i_cluster_thighs+1} / left_arm_{i_cluster_left_arm+1} / right_arm_{i_cluster_right_arm+1}")

        offset_right_arm = x_max_left_arm + offset_left_arm
    offset_left_arm = x_max_thighs

ax.set_xlim([-10, 110])
ax.set_ylim([-5, 55])
plt.savefig(f'cluster_graphs/proportion_of_solutions_in_each_cluster_{nb_twists}.svg', dpi=300)
# plt.show()

# Count number of combinaition of strategies were used
num_combinations = np.zeros((len(cluster_thighs['Athlete_05'].keys()), len(cluster_left_arm['Athlete_05'].keys()), len(cluster_right_arm['Athlete_05'].keys())))
for i_cluster_thighs, cluster_thighs_name in enumerate(cluster_thighs['Athlete_05'].keys()):
    for i_cluster_left_arm, cluster_left_arm_name in enumerate(cluster_left_arm['Athlete_05'].keys()):
        for i_cluster_right_arm, cluster_right_arm_name in enumerate(cluster_right_arm['Athlete_05'].keys()):
            if pourcentage_clusters_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, i_cluster_right_arm] > 0:
                num_combinations[i_cluster_thighs, i_cluster_left_arm, i_cluster_right_arm] = 1


# Plot the proportion of solutions that are optimal in each cluster
num_opt_clusters_thighs_techniques = np.zeros((len(cluster_thighs['Athlete_05'].keys()), ))
num_opt_clusters_left_arm_techniques = np.zeros((len(cluster_thighs['Athlete_05'].keys()), len(cluster_left_arm['Athlete_05'].keys())))
num_opt_clusters_right_arm_techniques = np.zeros((len(cluster_thighs['Athlete_05'].keys()), len(cluster_left_arm['Athlete_05'].keys()), len(cluster_right_arm['Athlete_05'].keys())))
for i_name, name in enumerate(cluster_thighs.keys()):
    for i_cluster_thighs, cluster_thighs_name in enumerate(cluster_thighs['Athlete_05'].keys()):
        if i_cluster_thighs == best_solution_per_athlete[name]["thighs_cluster"]:
            num_opt_clusters_thighs_techniques[i_cluster_thighs] += 1

            for i_cluster_left_arm, cluster_left_arm_name in enumerate(cluster_left_arm['Athlete_05'].keys()):
                if i_cluster_left_arm == best_solution_per_athlete[name]["left_arm_cluster"]:
                    num_opt_clusters_left_arm_techniques[i_cluster_thighs, i_cluster_left_arm] += 1

                    for i_cluster_right_arm, cluster_right_arm_name in enumerate(cluster_right_arm['Athlete_05'].keys()):
                        if i_cluster_right_arm == best_solution_per_athlete[name]["right_arm_cluster"]:
                            num_opt_clusters_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, i_cluster_right_arm] += 1

pourcentage_clusters_opt_thighs_techniques = num_opt_clusters_thighs_techniques / np.sum(num_opt_clusters_thighs_techniques) * 100
pourcentage_clusters_opt_left_arm_techniques = np.zeros(num_opt_clusters_left_arm_techniques.shape)
pourcentage_clusters_opt_right_arm_techniques = np.zeros(num_opt_clusters_right_arm_techniques.shape)
for i_cluster_thighs, cluster_thighs_name in enumerate(cluster_thighs['Athlete_05'].keys()):
    pourcentage_clusters_opt_left_arm_techniques[i_cluster_thighs, :] = num_opt_clusters_left_arm_techniques[i_cluster_thighs, :] / np.sum(num_opt_clusters_thighs_techniques[i_cluster_thighs]) * pourcentage_clusters_opt_thighs_techniques[i_cluster_thighs]
    for i_cluster_left_arm, cluster_left_arm_name in enumerate(cluster_left_arm['Athlete_05'].keys()):
        pourcentage_clusters_opt_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, :] = num_opt_clusters_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, :] / np.sum(num_opt_clusters_left_arm_techniques[i_cluster_thighs, i_cluster_left_arm]) * pourcentage_clusters_opt_left_arm_techniques[i_cluster_thighs, i_cluster_left_arm]

offset_left_arm = 0
offset_right_arm = 0
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for i_cluster_thighs in range(len(cluster_thighs['Athlete_05'].keys())):
    x_min_thighs = pourcentage_clusters_opt_thighs_techniques[:i_cluster_thighs].sum()
    x_max_thighs = pourcentage_clusters_opt_thighs_techniques[:i_cluster_thighs + 1].sum()
    if x_max_thighs - x_min_thighs > 1e-8:
        ax.fill_between([x_min_thighs, x_max_thighs], [40, 40], [50, 50], color=cmap_viridis(1 - i_cluster_thighs * 1/6), alpha=0.8, linewidth=0.0)

    for i_cluster_left_arm in range(len(cluster_left_arm['Athlete_05'].keys())):
        x_min_left_arm = pourcentage_clusters_opt_left_arm_techniques[i_cluster_thighs, :i_cluster_left_arm].sum()
        x_max_left_arm = pourcentage_clusters_opt_left_arm_techniques[i_cluster_thighs, :i_cluster_left_arm + 1].sum()
        if x_max_left_arm - x_min_left_arm > 1e-8:
            ax.fill_between([x_min_left_arm + offset_left_arm, x_max_left_arm + offset_left_arm], [20, 20], [30, 30], color=cmap_viridis(i_cluster_left_arm * 1/3), alpha=0.8, linewidth=0.0)

        for i_cluster_right_arm in range(len(cluster_right_arm['Athlete_05'].keys())):
            x_min_right_arm = np.nansum(pourcentage_clusters_opt_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, :i_cluster_right_arm])
            x_max_right_arm = np.nansum(pourcentage_clusters_opt_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, :i_cluster_right_arm + 1])
            if x_max_right_arm - x_min_right_arm > 1e-8:
                ax.fill_between([x_min_right_arm + offset_right_arm, x_max_right_arm + offset_right_arm], [0, 0], [10, 10], color=cmap_magma(1 - i_cluster_right_arm * 1/6 - 1/6), alpha=0.8, linewidth=0.0)
                print(f"{pourcentage_clusters_opt_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, i_cluster_right_arm]}% hip_{i_cluster_thighs+1} / left_arm_{i_cluster_left_arm+1} / right_arm_{i_cluster_right_arm+1}")

        offset_right_arm = x_max_left_arm + offset_left_arm
    offset_left_arm = x_max_thighs

ax.set_xlim([-10, 110])
ax.set_ylim([-5, 55])
plt.savefig(f'cluster_graphs/proportion_of_solutions_optimal_for_each_cluster_{nb_twists}.svg', dpi=300)
plt.show()

# Count number of combinaition of strategies were used
num_combinations_opt = np.zeros((len(cluster_thighs['Athlete_05'].keys()), len(cluster_left_arm['Athlete_05'].keys()), len(cluster_right_arm['Athlete_05'].keys())))
for i_cluster_thighs, cluster_thighs_name in enumerate(cluster_thighs['Athlete_05'].keys()):
    for i_cluster_left_arm, cluster_left_arm_name in enumerate(cluster_left_arm['Athlete_05'].keys()):
        for i_cluster_right_arm, cluster_right_arm_name in enumerate(cluster_right_arm['Athlete_05'].keys()):
            if pourcentage_clusters_opt_right_arm_techniques[i_cluster_thighs, i_cluster_left_arm, i_cluster_right_arm] > 0:
                num_combinations_opt[i_cluster_thighs, i_cluster_left_arm, i_cluster_right_arm] = 1


print(num_combinations_opt.sum(), " combinations were optimal")

for i_name, name in enumerate(cluster_thighs.keys()):
    print(f"{name}: thigh cluster {best_solution_per_athlete[name]['thighs_cluster'] + 1}, left arm cluster: {best_solution_per_athlete[name]['left_arm_cluster'] + 1}, right arm cluster: {best_solution_per_athlete[name]['right_arm_cluster'] + 1}")
