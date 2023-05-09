
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


model_path = "Models/JeCh_TechOpt83.bioMod"
model = biorbd.Model(model_path)
nb_twists = 1
chosen_clusters_dict = {}
results_path = 'solutions_multi_start/'
results_path_this_time = results_path + 'Solutions_vrille_et_demi/'
cmap = cm.get_cmap('viridis')

# Define the clusters of solutions per joint
cluster_right_arm = {
    "AdCh":     {"cluster_1": [1, 2, 5, 6, 7, 8],             "cluster_2": [],                             "cluster_3": [],           "cluster_4": [0, 4, 9], "cluster_5": [],                          "cluster_6": []},
    "AlAd":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [],                             "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "cluster_6": []},
    "AuJo":     {"cluster_1": [],                             "cluster_2": [1, 2, 3, 7],                   "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "cluster_6": [0, 4, 5, 6, 8, 9]},
    "Benjamin": {"cluster_1": [8],                            "cluster_2": [],                             "cluster_3": [],           "cluster_4": [],        "cluster_5": [0, 1, 2, 3, 4, 5, 6, 7, 9], "cluster_6": []},
    "ElMe":     {"cluster_1": [],                             "cluster_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "cluster_6": []},
    "EvZl":     {"cluster_1": [],                             "cluster_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "cluster_6": []},
    "FeBl":     {"cluster_1": [0, 1, 2, 3, 4, 6, 7, 8, 9],    "cluster_2": [],                             "cluster_3": [],           "cluster_4": [5],       "cluster_5": [],                          "cluster_6": []},
    "JeCh":     {"cluster_1": [6],                            "cluster_2": [],                             "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "cluster_6": [0, 1, 2, 3, 4, 5, 7, 8, 9]},
    "KaFu":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [0, 3, 8],    "cluster_4": [],        "cluster_5": [],                          "cluster_6": [1, 2, 4, 5, 6, 9]},
    "KaMi":     {"cluster_1": [],                             "cluster_2": [0, 2, 3, 7],                   "cluster_3": [4, 9],       "cluster_4": [],        "cluster_5": [],                          "cluster_6": [1, 5, 6, 8]},
    "LaDe":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [2],          "cluster_4": [],        "cluster_5": [],                          "cluster_6": [1, 3, 4, 5, 6, 7, 8, 9]},
    "MaCu":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [1, 3],       "cluster_4": [],        "cluster_5": [],                          "cluster_6": [2, 4, 6, 7, 8, 9]},
    "MaJa":     {"cluster_1": [],                             "cluster_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "cluster_6": []},
    "OlGa":     {"cluster_1": [],                             "cluster_2": [0, 2, 3, 4, 5, 6, 7, 8, 9],    "cluster_3": [1],          "cluster_4": [],        "cluster_5": [],                          "cluster_6": []},
    "Sarah":    {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "cluster_6": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
    "SoMe":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [0, 1, 2, 8], "cluster_4": [],        "cluster_5": [],                          "cluster_6": [3, 4, 5, 6, 7, 9]},
    "WeEm":     {"cluster_1": [],                             "cluster_2": [2, 3],                         "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "cluster_6": [0, 1, 4, 5, 6, 7, 8, 9]},
    "ZoTs":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [3],          "cluster_4": [],        "cluster_5": [],                          "cluster_6": [0, 1, 2, 4, 5, 6, 7, 8, 9]},
}

cluster_left_arm = {
    "AdCh":     {"cluster_1": [0, 1, 2, 4, 5, 6, 7, 8, 9],    "cluster_2": [],                       "cluster_3": [],                             "cluster_4": []},
    "AlAd":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [],                       "cluster_3": [],                             "cluster_4": []},
    "AuJo":     {"cluster_1": [4, 5, 6, 8, 9],                "cluster_2": [],                       "cluster_3": [0, 2, 3, 7],                   "cluster_4": [1]},
    "Benjamin": {"cluster_1": [],                             "cluster_2": [],                       "cluster_3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_4": []},
    "ElMe":     {"cluster_1": [5],                            "cluster_2": [],                       "cluster_3": [0, 1, 2, 3, 4, 6, 7, 8, 9],    "cluster_4": []},
    "EvZl":     {"cluster_1": [0, 1, 2, 4, 5, 6, 7, 8, 9],    "cluster_2": [],                       "cluster_3": [],                             "cluster_4": [3]},
    "FeBl":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [],                       "cluster_3": [],                             "cluster_4": []},
    "JeCh":     {"cluster_1": [0, 1, 2, 3, 4, 5, 7, 8, 9],    "cluster_2": [],                       "cluster_3": [6],                            "cluster_4": []},
    "KaFu":     {"cluster_1": [],                             "cluster_2": [],                       "cluster_3": [1, 2, 3, 4, 5, 6, 7, 8, 9],    "cluster_4": [0]},
    "KaMi":     {"cluster_1": [0, 1, 4, 5, 6, 8, 9],          "cluster_2": [],                       "cluster_3": [2, 3, 7],                      "cluster_4": []},
    "LaDe":     {"cluster_1": [],                             "cluster_2": [1, 3, 4, 5, 6, 7, 8, 9], "cluster_3": [],                             "cluster_4": [2]},
    "MaCu":     {"cluster_1": [],                             "cluster_2": [1, 2, 3, 4, 6, 7, 8, 9], "cluster_3": [],                             "cluster_4": []},
    "MaJa":     {"cluster_1": [0, 2, 4, 5, 6, 9],             "cluster_2": [],                       "cluster_3": [1, 3, 7, 8],                   "cluster_4": []},
    "OlGa":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [],                       "cluster_3": [],                             "cluster_4": []},
    "Sarah":    {"cluster_1": [],                             "cluster_2": [],                       "cluster_3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_4": []},
    "SoMe":     {"cluster_1": [],                             "cluster_2": [],                       "cluster_3": [1, 2, 3, 4, 5, 6, 7, 8, 9],    "cluster_4": [0]},
    "WeEm":     {"cluster_1": [],                             "cluster_2": [],                       "cluster_3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_4": []},
    "ZoTs":     {"cluster_1": [],                             "cluster_2": [],                       "cluster_3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_4": []},
}

cluster_thighs = {
    "AdCh":     {"cluster_1": [0, 1, 2, 4, 5, 6, 7, 8, 9],    "cluster_2": []},
    "AlAd":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "AuJo":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Benjamin": {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "ElMe":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "EvZl":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "FeBl":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "JeCh":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "KaFu":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "KaMi":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "LaDe":     {"cluster_1": [],                             "cluster_2": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
    "MaCu":     {"cluster_1": [1, 2, 3, 4, 6, 7, 8, 9],       "cluster_2": []},
    "MaJa":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "OlGa":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "Sarah":    {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "SoMe":     {"cluster_1": [0, 3, 4, 5, 6, 7, 9],          "cluster_2": [1, 2, 8]},
    "WeEm":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
    "ZoTs":     {"cluster_1": [0, 1, 2, 4, 5, 6, 7, 8, 9],    "cluster_2": [3]},
}

# cluster_tighs_rotY = {
#     "AdCh":     {"cluster_1": [0, 4, 9], "cluster_2": [1, 2, 5, 6, 7, 8], "cluster_3": [], "cluster_4": []},
#     "AlAd":     {"cluster_1": [], "cluster_2": [0, 1, 2, 4, 5, 6, 7, 8, 9], "cluster_3": [3], "cluster_4": []},
#     "AuJo":     {"cluster_1": [0, 1, 2, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "Benjamin": {"cluster_1": [], "cluster_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_3": [], "cluster_4": []},
#     "ElMe":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "EvZl":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "FeBl":     {"cluster_1": [5], "cluster_2": [0, 1, 2, 3, 4, 6, 7, 8, 9], "cluster_3": [], "cluster_4": []},
#     "JeCh":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "KaFu":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "KaMi":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "LaDe":     {"cluster_1": [], "cluster_2": [], "cluster_3": [], "cluster_4": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
#     "MaCu":     {"cluster_1": [1, 2, 3, 4, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "MaJa":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "OlGa":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "Sarah":    {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "SoMe":     {"cluster_1": [], "cluster_2": [], "cluster_3": [], "cluster_4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
#     "WeEm":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "ZoTs":     {"cluster_1": [], "cluster_2": [], "cluster_3": [], "cluster_4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
# }


# print all the solutions at once
fig, axs = plt.subplots(2, 3, figsize=(18, 9))

q_right_arm = {"q": {key: np.zeros((16, 381, 1)) for key in cluster_right_arm['AdCh'].keys()}, "normalized_time_vector": {key: np.zeros((381, 1)) for key in cluster_right_arm['AdCh'].keys()}}
q_left_arm = {"q": {key: np.zeros((16, 381, 1)) for key in cluster_left_arm['AdCh'].keys()}, "normalized_time_vector": {key: np.zeros((381, 1)) for key in cluster_left_arm['AdCh'].keys()}}
q_thighs = {"q": {key: np.zeros((16, 381, 1)) for key in cluster_thighs['AdCh'].keys()}, "normalized_time_vector": {key: np.zeros((381, 1)) for key in cluster_thighs['AdCh'].keys()}}
names = cluster_right_arm.keys()
for i_name, name in enumerate(names):
    for i_sol in range(9):
        file_name = results_path_this_time + name + '/' + name + '_vrille_et_demi_' + str(i_sol) + "_CVG.pkl"
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

        for i_clust, key in enumerate(cluster_right_arm[name].keys()):
            if i_sol in cluster_right_arm[name][key]:
                q_right_arm['q'][key] = np.concatenate((q_right_arm['q'][key], q[:, :, np.newaxis]), axis=2)
                q_right_arm['normalized_time_vector'][key] = np.concatenate((q_right_arm['normalized_time_vector'][key], normalized_time_vector[:, np.newaxis]), axis=1)
                i_cluster_right_arm = i_clust
                rgba = cmap(i_cluster_right_arm * 1/6)
                axs[0, 0].plot(normalized_time_vector, q[6, :], color=rgba)
                axs[1, 0].plot(normalized_time_vector, q[7, :], color=rgba)

        for i_clust, key in enumerate(cluster_left_arm[name].keys()):
            if i_sol in cluster_left_arm[name][key]:
                q_left_arm['q'][key] = np.concatenate((q_left_arm['q'][key], q[:, :, np.newaxis]), axis=2)
                q_left_arm['normalized_time_vector'][key] = np.concatenate((q_left_arm['normalized_time_vector'][key], normalized_time_vector[:, np.newaxis]), axis=1)
                i_cluster_left_arm = i_clust
                rgba = cmap(i_cluster_left_arm * 1/6)
                axs[0, 1].plot(normalized_time_vector, -q[10, :], color=rgba)
                axs[1, 1].plot(normalized_time_vector, -q[11, :], color=rgba)

        for i_clust, key in enumerate(cluster_thighs[name].keys()):
            if i_sol in cluster_thighs[name][key]:
                q_thighs['q'][key] = np.concatenate((q_thighs['q'][key], q[:, :, np.newaxis]), axis=2)
                q_thighs['normalized_time_vector'][key] = np.concatenate((q_thighs['normalized_time_vector'][key], normalized_time_vector[:, np.newaxis]), axis=1)
                i_cluster_thighs = i_clust
                rgba = cmap(i_cluster_thighs * 1/6)
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

for i_clust, key in enumerate(cluster_right_arm[name].keys()):
    rgba = cmap(i_clust * 1/6)
    axs[1, 0].plot(normalized_time_vector[0], q[7, 0], color=rgba, label="Cluster #" + str(i_clust + 1))
axs[1, 0].legend(ncol=6, bbox_to_anchor=(1.7, -0.2), loc='center')

plt.suptitle(f"{nb_twists}.5 twists")
plt.savefig(f'cluster_graphs/clusters_graph_for_all_athletes_{nb_twists}.png', dpi=300)
# plt.show()

print("\n\n")

# Find the mean and std of each cluster
cluster_counter_right_arm = {key: 0 for key in cluster_right_arm["AlAd"].keys()}
mean_std_per_cluster_right_arm = {key: np.zeros((16, )) for key in cluster_right_arm["AlAd"].keys()}
mean_q_per_cluster_right_arm = {'q': np.zeros((16, 381, 1)), 'normalized_time_vector': np.zeros((381, 1))}
std_q_per_cluster_right_arm = np.zeros((16, 381, 1))
range_q_per_cluster_right_arm = np.zeros((16, 381, 1))
for i_cluster, cluster_name in enumerate(cluster_right_arm['AlAd'].keys()):
    q_right_arm['q'][cluster_name] = q_right_arm['q'][cluster_name][:, :, 1:]
    q_right_arm['normalized_time_vector'][cluster_name] = q_right_arm['normalized_time_vector'][cluster_name][:, 1:]
    for i_name, name in enumerate(cluster_right_arm):
        if len(cluster_right_arm[name][cluster_name]) > 0:
            cluster_counter_right_arm[cluster_name] += 1
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


cluster_counter_left_arm = {key: 0 for key in cluster_left_arm['AlAd'].keys()}
mean_std_per_cluster_left_arm = {key: np.zeros((16, )) for key in cluster_left_arm['AlAd'].keys()}
mean_q_per_cluster_left_arm = {'q': np.zeros((16, 381, 1)), 'normalized_time_vector': np.zeros((381, 1))}
std_q_per_cluster_left_arm = np.zeros((16, 381, 1))
range_q_per_cluster_left_arm = np.zeros((16, 381, 1))
for i_cluster, cluster_name in enumerate(cluster_left_arm['AlAd'].keys()):
    q_left_arm['q'][cluster_name] = q_left_arm['q'][cluster_name][:, :, 1:]
    for i_name, name in enumerate(cluster_left_arm):
        if len(cluster_left_arm[name][cluster_name]) > 0:
            cluster_counter_left_arm[cluster_name] += 1
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


cluster_counter_thighs = {key: 0 for key in cluster_thighs['AlAd'].keys()}
mean_std_per_cluster_thighs = {key: np.zeros((16, )) for key in cluster_thighs['AlAd'].keys()}
mean_q_per_cluster_thighs = {'q': np.zeros((16, 381, 1)), 'normalized_time_vector': np.zeros((381, 1))}
std_q_per_cluster_thighs = np.zeros((16, 381, 1))
range_q_per_cluster_thighs = np.zeros((16, 381, 1))
for i_cluster, cluster_name in enumerate(cluster_thighs['AlAd'].keys()):
    q_thighs['q'][cluster_name] = q_thighs['q'][cluster_name][:, :, 1:]
    for i_name, name in enumerate(cluster_thighs):
        if len(cluster_thighs[name][cluster_name]) > 0:
            cluster_counter_thighs[cluster_name] += 1
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

print("Right arm clusters:")
# Plot the clusters with different colors
fig, axs = plt.subplots(2, 3, figsize=(18, 9))
for i_cluster, cluster_name in enumerate(cluster_right_arm['AlAd'].keys()):
    print(f"{cluster_name} was used by {cluster_counter_right_arm[cluster_name]} / {len(cluster_right_arm)} athletes")
    print(f"Sum of mean std on cluster {cluster_name} was {np.sum(mean_std_per_cluster_right_arm[cluster_name][3:])}")
    print(f"{cluster_name} has a right arm axial rotation range of {np.mean(range_q_per_cluster_right_arm[:, :, i_cluster][6, :]) / (np.max(mean_q_per_cluster_right_arm['q'][:, :, i_cluster][6, :]) - np.min(mean_q_per_cluster_right_arm['q'][:, :, i_cluster][6, :])) * 100}% of the average movement amplitude")
    print(f"{cluster_name} has a right arm elevation range of {np.mean(range_q_per_cluster_right_arm[:, :, i_cluster][7, :]) / (np.max(mean_q_per_cluster_right_arm['q'][:, :, i_cluster][7, :]) - np.min(mean_q_per_cluster_right_arm['q'][:, :, i_cluster][7, :])) * 100}% of the average max amplitude")

    rgba = cmap(i_cluster * 1/6)
    axs[0, 0].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], mean_q_per_cluster_right_arm['q'][6, :, i_cluster] - std_q_per_cluster_right_arm[6, :, i_cluster],
                        mean_q_per_cluster_right_arm['q'][6, :, i_cluster] + std_q_per_cluster_right_arm[6, :,i_cluster], color=rgba, alpha=0.2)
    axs[0, 0].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], mean_q_per_cluster_right_arm['q'][6, :, i_cluster], color=rgba)
    axs[1, 0].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], mean_q_per_cluster_right_arm['q'][7, :, i_cluster] - std_q_per_cluster_right_arm[7, :, i_cluster],
                        mean_q_per_cluster_right_arm['q'][7, :, i_cluster] + std_q_per_cluster_right_arm[7, :,i_cluster], color=rgba, alpha=0.2)
    axs[1, 0].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, i_cluster], mean_q_per_cluster_right_arm['q'][7, :, i_cluster], color=rgba, label="Cluster #" + str(i_cluster + 1))
    if i_cluster == 0:
        axs[0, 0].set_title(f"Change in elevation plane")  # Right arm
        axs[1, 0].set_title(f"Elevation")  # Right arm
    print('\n')
print('\n')
axs[1, 0].set_xlabel(f"Normalized time")
axs[1, 0].legend(ncol=6, bbox_to_anchor=(1.7, -0.2), loc='center')

print("Left arm clusters:")
for i_cluster, cluster_name in enumerate(cluster_left_arm['AlAd'].keys()):
    print(f"{cluster_name} was used by {cluster_counter_left_arm[cluster_name]} / {len(cluster_left_arm)} athletes")
    print(f"Sum of mean std on cluster {cluster_name} was {np.sum(mean_std_per_cluster_left_arm[cluster_name][3:])}")
    print(f"{cluster_name} has a left arm axial rotation range of {np.mean(range_q_per_cluster_left_arm[:, :, i_cluster][10, :]) / (np.max(mean_q_per_cluster_left_arm['q'][:, :, i_cluster][10, :]) - np.min(mean_q_per_cluster_left_arm['q'][:, :, i_cluster][10, :])) * 100}% of the average movement amplitude")
    print(f"{cluster_name} has a left arm elevation range of {np.mean(range_q_per_cluster_left_arm[:, :, i_cluster][11, :]) / (np.max(mean_q_per_cluster_left_arm['q'][:, :, i_cluster][11, :]) - np.min(mean_q_per_cluster_left_arm['q'][:, :, i_cluster][11, :])) * 100}% of the average max amplitude")

    rgba = cmap(i_cluster * 1/6)
    axs[0, 1].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], -mean_q_per_cluster_left_arm['q'][10, :, i_cluster] - std_q_per_cluster_left_arm[10, :, i_cluster],
                        -mean_q_per_cluster_left_arm['q'][10, :, i_cluster] + std_q_per_cluster_left_arm[10, :,i_cluster], color=rgba, alpha=0.2)
    axs[0, 1].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], -mean_q_per_cluster_left_arm['q'][10, :, i_cluster], color=rgba)
    axs[1, 1].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], -mean_q_per_cluster_left_arm['q'][11, :, i_cluster] - std_q_per_cluster_left_arm[11, :, i_cluster],
                        -mean_q_per_cluster_left_arm['q'][11, :, i_cluster] + std_q_per_cluster_left_arm[11, :,i_cluster], color=rgba, alpha=0.2)
    axs[1, 1].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], -mean_q_per_cluster_left_arm['q'][11, :, i_cluster], color=rgba)
    if i_cluster == 0:
        axs[0, 1].set_title(f"Change in elevation plane")  # Left arm
        axs[1, 1].set_title(f"Elevation")  # Left arm
    print('\n')
print('\n')
axs[1, 1].set_xlabel(f"Normalized time")

print("Thigh clusters:")
for i_cluster, cluster_name in enumerate(cluster_thighs['AlAd'].keys()):
    print(f"{cluster_name} was used by {cluster_counter_thighs[cluster_name]} / {len(cluster_thighs)} athletes")
    print(f"Sum of mean std on cluster {cluster_name} was {np.sum(mean_std_per_cluster_thighs[cluster_name][3:])}")
    print(f"{cluster_name} has a hip flexion range of {np.mean(range_q_per_cluster_thighs[:, :, i_cluster][14, :]) / (np.max(mean_q_per_cluster_thighs['q'][:, :, i_cluster][14, :]) - np.min(mean_q_per_cluster_thighs['q'][:, :, i_cluster][14, :])) * 100}% of the average movement amplitude")
    print(f"{cluster_name} has a hip lateral flexion range of {np.mean(range_q_per_cluster_thighs[:, :, i_cluster][15, :]) / (np.max(mean_q_per_cluster_thighs['q'][:, :, i_cluster][15, :]) - np.min(mean_q_per_cluster_thighs['q'][:, :, i_cluster][15, :])) * 100}% of the average max amplitude")

    rgba = cmap(i_cluster * 1/6)
    axs[0, 2].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], mean_q_per_cluster_thighs['q'][14, :, i_cluster] - std_q_per_cluster_thighs[14, :, i_cluster],
                        mean_q_per_cluster_thighs['q'][14, :, i_cluster] + std_q_per_cluster_thighs[14, :, i_cluster], color=rgba, alpha=0.2)
    axs[0, 2].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], mean_q_per_cluster_thighs['q'][14, :, i_cluster], color=rgba)
    axs[1, 2].fill_between(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], mean_q_per_cluster_thighs['q'][15, :, i_cluster] - std_q_per_cluster_thighs[15, :, i_cluster],
                        mean_q_per_cluster_thighs['q'][15, :, i_cluster] + std_q_per_cluster_thighs[15, :, i_cluster], color=rgba, alpha=0.2)
    axs[1, 2].plot(mean_q_per_cluster_right_arm['normalized_time_vector'][:, 0], mean_q_per_cluster_thighs['q'][15, :, i_cluster], color=rgba)
    if i_cluster == 0:
        axs[0, 2].set_title(f"Flexion")  # Hips
        axs[1, 2].set_title(f"Lateral flexion")  # Hips
    print('\n')
print('\n')
axs[1, 2].set_xlabel(f"Normalized time")

plt.suptitle(f"mean kinematics per cluster for {nb_twists}.5 twists")
plt.savefig(f'cluster_graphs/mean_clusters_graph_for_all_athletes_{nb_twists}.png', dpi=300)
# plt.show()

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
                "cluster_thighs": cluster_thighs,}


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
plt.show()


# Generate animations of the movements in the clusters
for i_cluster, cluster_name in enumerate(cluster_right_arm['AlAd'].keys()):
    print(f"right_arm_{cluster_name}")
    Q_to_animate = np.zeros((model.nbQ(), 381))
    Q_to_animate[5, :] = np.pi/2
    Q_to_animate[6, :] = mean_q_per_cluster_right_arm['q'][6, :, i_cluster]
    Q_to_animate[7, :] = mean_q_per_cluster_right_arm['q'][7, :, i_cluster]
    Q_to_animate[8, :] = mean_q_per_cluster_right_arm['q'][8, :, i_cluster]
    Q_to_animate[9, :] = mean_q_per_cluster_right_arm['q'][9, :, i_cluster]
    b = bioviz.Viz(model_path)
    b.set_camera_zoom(0.5)
    b.load_movement(Q_to_animate)
    b.exec()
    # b.start_recording(f"videos_clusters/right_arm_{cluster_name}.ogv")
    # b.load_movement(Q_to_animate)
    # for f in range(Q_to_animate.shape[1] + 1):
    #     b.movement_slider[0].setValue(f)
    # b.add_frame()
    # b.stop_recording()
    # b.quit()

for i_cluster, cluster_name in enumerate(cluster_left_arm['AlAd'].keys()):
    print(f"left_arm_{cluster_name}")
    Q_to_animate = np.zeros((model.nbQ(), 381))
    Q_to_animate[5, :] = np.pi/2
    Q_to_animate[10, :] = mean_q_per_cluster_left_arm['q'][10, :, i_cluster]
    Q_to_animate[11, :] = mean_q_per_cluster_left_arm['q'][11, :, i_cluster]
    Q_to_animate[12, :] = mean_q_per_cluster_left_arm['q'][12, :, i_cluster]
    Q_to_animate[13, :] = mean_q_per_cluster_left_arm['q'][13, :, i_cluster]
    b = bioviz.Viz(model_path)
    b.set_camera_zoom(0.5)
    b.load_movement(Q_to_animate)
    b.exec()
    # b.start_recording(f"videos_clusters/left_arm_{cluster_name}.ogv")
    # b.load_movement(Q_to_animate)
    # for f in range(Q_to_animate.shape[1] + 1):
    #     b.movement_slider[0].setValue(f)
    #     b.add_frame()
    # b.stop_recording()
    # b.quit()

for i_cluster, cluster_name in enumerate(cluster_thighs['AlAd'].keys()):
    print(f"thighs_{cluster_name}")
    Q_to_animate = np.zeros((model.nbQ(), 381))
    Q_to_animate[5, :] = np.pi/2
    Q_to_animate[14, :] = mean_q_per_cluster_thighs['q'][14, :, i_cluster]
    Q_to_animate[15, :] = mean_q_per_cluster_thighs['q'][15, :, i_cluster]
    b = bioviz.Viz(model_path)
    b.set_camera_zoom(0.5)
    b.load_movement(Q_to_animate)
    b.exec()
    # b.start_recording(f"videos_clusters/thighs_{cluster_name}.ogv")
    # b.load_movement(Q_to_animate)
    # for f in range(Q_to_animate.shape[1] + 1):
    #     b.movement_slider[0].setValue(f)
    #     b.add_frame()
    # b.stop_recording()
    # b.quit()

