
import biorbd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import bioviz
import os
from IPython import embed

model_path = "Models/JeCh_TechOpt83.bioMod"
model = biorbd.Model(model_path)
nb_twists = 1
chosen_clusters_dict = {}
results_path = 'solutions_multi_start/'
results_path_this_time = results_path + 'Solutions_vrille_et_demi/'
cmap = cm.get_cmap('magma')

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
    "KaMi":     {"cluster_1": [],                             "cluster_2": [0, 2, 3, 7, 9],                "cluster_3": [4],          "cluster_4": [],        "cluster_5": [],                          "cluster_6": [1, 5, 6, 8]},
    "LaDe":     {"cluster_1": [],                             "cluster_2": [2],                            "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "cluster_6": [1, 3, 4, 5, 6, 7, 8, 9]},
    "MaCu":     {"cluster_1": [],                             "cluster_2": [],                             "cluster_3": [1, 3],       "cluster_4": [],        "cluster_5": [],                          "cluster_6": [2, 4, 6, 7, 8, 9]},
    "MaJa":     {"cluster_1": [],                             "cluster_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "cluster_6": []},
    "MeVa":     {"cluster_1": [],                             "cluster_2": [2, 7],                         "cluster_3": [],           "cluster_4": [],        "cluster_5": [],                          "cluster_6": [0, 1, 3, 4, 5, 6, 8, 9]},
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
    "MeVa":     {"cluster_1": [],                             "cluster_2": [],                       "cluster_3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_4": []},
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
    "MeVa":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": []},
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
#     "MeVa":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "OlGa":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "Sarah":    {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "SoMe":     {"cluster_1": [], "cluster_2": [], "cluster_3": [], "cluster_4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
#     "WeEm":     {"cluster_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "cluster_2": [], "cluster_3": [], "cluster_4": []},
#     "ZoTs":     {"cluster_1": [], "cluster_2": [], "cluster_3": [], "cluster_4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
# }

fig, axs = plt.subplots(2, 3, figsize=(18, 9))
axs = axs.ravel()

q_right_arm = {key: np.zeros((16, 381, 1)) for key in cluster_right_arm['AdCh'].keys()}
q_left_arm = {key: np.zeros((16, 381, 1)) for key in cluster_left_arm['AdCh'].keys()}
q_thighs = {key: np.zeros((16, 381, 1)) for key in cluster_thighs['AdCh'].keys()}
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

        q = np.zeros(np.shape(Q[0][:, :-1]))
        q[:, :] = Q[0][:, :-1]
        for i in range(1, len(Q)):
            if i == len(Q) - 1:
                q = np.hstack((q, Q[i]))
            else:
                q = np.hstack((q, Q[i][:, :-1]))

        for i_clust, key in enumerate(cluster_right_arm[name].keys()):
            if i_sol in cluster_right_arm[name][key]:
                q_right_arm[key] = np.concatenate((q_right_arm[key], q[:, :, np.newaxis]), axis=2)
                i_cluster_right_arm = i_clust
                rgba = cmap(i_cluster_right_arm * 1 / 7)
                axs[0].plot(q[6, :], color=rgba)
                axs[1].plot(q[7, :], color=rgba)

        for i_clust, key in enumerate(cluster_left_arm[name].keys()):
            if i_sol in cluster_left_arm[name][key]:
                q_left_arm[key] = np.concatenate((q_left_arm[key], q[:, :, np.newaxis]), axis=2)
                i_cluster_left_arm = i_clust
                rgba = cmap(i_cluster_left_arm * 1 / 7)
                axs[2].plot(q[10, :], color=rgba)
                axs[3].plot(q[11, :], color=rgba)

        for i_clust, key in enumerate(cluster_thighs[name].keys()):
            if i_sol in cluster_thighs[name][key]:
                q_thighs[key] = np.concatenate((q_thighs[key], q[:, :, np.newaxis]), axis=2)
                i_cluster_thighs = i_clust
                rgba = cmap(i_cluster_thighs * 1 / 7)
                axs[4].plot(q[14, :], color=rgba)
                axs[5].plot(q[15, :], color=rgba)

        if i_sol == 0:
            for i, DoF in enumerate([6, 7, 10, 11, 14, 15]):
                axs[i].set_title(f"{model.nameDof()[DoF].to_string()}")

plt.suptitle(f"{nb_twists}.5 twists")
plt.savefig(f'cluster_graphs/clusters_graph_for_all_athletes_{nb_twists}.png', dpi=300)
plt.show()

cluster_counter_right_arm = {key: 0 for key in cluster_right_arm["AlAd"].keys()}
mean_std_per_cluster_right_arm = {key: np.zeros((16, )) for key in cluster_right_arm["AlAd"].keys()}
mean_q_per_cluster_right_arm = np.zeros((16, 381, 1))
std_q_per_cluster_right_arm = np.zeros((16, 381, 1))
for i_cluster, cluster_name in enumerate(cluster_right_arm['AlAd'].keys()):
    q_right_arm[cluster_name] = q_right_arm[cluster_name][:, :, 1:]
    for i_name, name in enumerate(cluster_right_arm):
        if len(cluster_right_arm[name][cluster_name]) > 0:
            cluster_counter_right_arm[cluster_name] += 1
    mean_std_per_cluster_right_arm[cluster_name] = np.mean(np.std(q_right_arm[cluster_name], axis=2), axis=1)
    mean_q_per_cluster_right_arm = np.concatenate((mean_q_per_cluster_right_arm, np.mean(q_right_arm[cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
    std_q_per_cluster_right_arm = np.concatenate((std_q_per_cluster_right_arm, np.std(q_right_arm[cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
mean_q_per_cluster_right_arm = mean_q_per_cluster_right_arm[:, :, 1:]
std_q_per_cluster_right_arm = std_q_per_cluster_right_arm[:, :, 1:]
mean_std_between_clusters_right_arm = np.mean(np.std(mean_q_per_cluster_right_arm, axis=2), axis=1)


cluster_counter_left_arm = {key: 0 for key in cluster_left_arm['AlAd'].keys()}
mean_std_per_cluster_left_arm = {key: np.zeros((16, )) for key in cluster_left_arm['AlAd'].keys()}
mean_q_per_cluster_left_arm = np.zeros((16, 381, 1))
std_q_per_cluster_left_arm = np.zeros((16, 381, 1))
for i_cluster, cluster_name in enumerate(cluster_left_arm['AlAd'].keys()):
    q_left_arm[cluster_name] = q_left_arm[cluster_name][:, :, 1:]
    for i_name, name in enumerate(cluster_left_arm):
        if len(cluster_left_arm[name][cluster_name]) > 0:
            cluster_counter_left_arm[cluster_name] += 1
    mean_std_per_cluster_left_arm[cluster_name] = np.mean(np.std(q_left_arm[cluster_name], axis=2), axis=1)
    mean_q_per_cluster_left_arm = np.concatenate((mean_q_per_cluster_left_arm, np.mean(q_left_arm[cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
    std_q_per_cluster_left_arm = np.concatenate((std_q_per_cluster_left_arm, np.std(q_left_arm[cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
mean_q_per_cluster_left_arm = mean_q_per_cluster_left_arm[:, :, 1:]
std_q_per_cluster_left_arm = std_q_per_cluster_left_arm[:, :, 1:]
mean_std_between_clusters_left_arm = np.mean(np.std(mean_q_per_cluster_left_arm, axis=2), axis=1)


cluster_counter_thighs = {key: 0 for key in cluster_thighs['AlAd'].keys()}
mean_std_per_cluster_thighs = {key: np.zeros((16, )) for key in cluster_thighs['AlAd'].keys()}
mean_q_per_cluster_thighs = np.zeros((16, 381, 1))
std_q_per_cluster_thighs = np.zeros((16, 381, 1))
for i_cluster, cluster_name in enumerate(cluster_thighs['AlAd'].keys()):
    q_thighs[cluster_name] = q_thighs[cluster_name][:, :, 1:]
    for i_name, name in enumerate(cluster_thighs):
        if len(cluster_thighs[name][cluster_name]) > 0:
            cluster_counter_thighs[cluster_name] += 1
    mean_std_per_cluster_thighs[cluster_name] = np.mean(np.std(q_thighs[cluster_name], axis=2), axis=1)
    mean_q_per_cluster_thighs = np.concatenate((mean_q_per_cluster_thighs, np.mean(q_thighs[cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
    std_q_per_cluster_thighs = np.concatenate((std_q_per_cluster_thighs, np.std(q_thighs[cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
mean_q_per_cluster_thighs = mean_q_per_cluster_thighs[:, :, 1:]
std_q_per_cluster_thighs = std_q_per_cluster_thighs[:, :, 1:]
mean_std_between_clusters_thighs = np.mean(np.std(mean_q_per_cluster_thighs, axis=2), axis=1)

fig, axs = plt.subplots(2, 3, figsize=(18, 9))
axs = axs.ravel()
for i_cluster, cluster_name in enumerate(cluster_right_arm['AlAd'].keys()):
    print(f"{cluster_name} was used by {cluster_counter_right_arm[cluster_name]} / {len(cluster_right_arm)} athletes")
    print(f"Sum of mean std on cluster {cluster_name} was {np.sum(mean_std_per_cluster_right_arm[cluster_name][3:])}")
    rgba = cmap(i_cluster * 1/8)
    axs[0].fill_between(np.arange(381), mean_q_per_cluster_right_arm[6, :, i_cluster] - std_q_per_cluster_right_arm[6, :, i_cluster],
                        mean_q_per_cluster_right_arm[6, :, i_cluster] + std_q_per_cluster_right_arm[6, :,i_cluster], color=rgba, alpha=0.2)
    axs[0].plot(mean_q_per_cluster_right_arm[6, :, i_cluster], color=rgba)
    axs[1].fill_between(np.arange(381), mean_q_per_cluster_right_arm[7, :, i_cluster] - std_q_per_cluster_right_arm[7, :, i_cluster],
                        mean_q_per_cluster_right_arm[7, :, i_cluster] + std_q_per_cluster_right_arm[7, :,i_cluster], color=rgba, alpha=0.2)
    axs[1].plot(mean_q_per_cluster_right_arm[7, :, i_cluster], color=rgba)
    if i_cluster == 0:
        axs[0].set_title(f"{model.nameDof()[6].to_string()}")
        axs[1].set_title(f"{model.nameDof()[7].to_string()}")

for i_cluster, cluster_name in enumerate(cluster_left_arm['AlAd'].keys()):
    print(f"{cluster_name} was used by {cluster_counter_left_arm[cluster_name]} / {len(cluster_left_arm)} athletes")
    print(f"Sum of mean std on cluster {cluster_name} was {np.sum(mean_std_per_cluster_left_arm[cluster_name][3:])}")
    rgba = cmap(i_cluster * 1/8)
    axs[2].fill_between(np.arange(381), mean_q_per_cluster_left_arm[10, :, i_cluster] - std_q_per_cluster_left_arm[10, :, i_cluster],
                        mean_q_per_cluster_left_arm[10, :, i_cluster] + std_q_per_cluster_left_arm[10, :,i_cluster], color=rgba, alpha=0.2)
    axs[2].plot(mean_q_per_cluster_left_arm[10, :, i_cluster], color=rgba)
    axs[3].fill_between(np.arange(381), mean_q_per_cluster_left_arm[11, :, i_cluster] - std_q_per_cluster_left_arm[11, :, i_cluster],
                        mean_q_per_cluster_left_arm[11, :, i_cluster] + std_q_per_cluster_left_arm[11, :,i_cluster], color=rgba, alpha=0.2)
    axs[3].plot(mean_q_per_cluster_left_arm[11, :, i_cluster], color=rgba)
    if i_cluster == 0:
        axs[2].set_title(f"{model.nameDof()[10].to_string()}")
        axs[3].set_title(f"{model.nameDof()[11].to_string()}")

for i_cluster, cluster_name in enumerate(cluster_thighs['AlAd'].keys()):
    print(
        f"{cluster_name} was used by {cluster_counter_thighs[cluster_name]} / {len(cluster_thighs)} athletes")
    print(
        f"Sum of mean std on cluster {cluster_name} was {np.sum(mean_std_per_cluster_thighs[cluster_name][3:])}")
    rgba = cmap(i_cluster * 1 / 8)
    axs[4].fill_between(np.arange(381),
                        mean_q_per_cluster_thighs[14, :, i_cluster] - std_q_per_cluster_thighs[14, :, i_cluster],
                        mean_q_per_cluster_thighs[14, :, i_cluster] + std_q_per_cluster_thighs[14, :, i_cluster], color=rgba, alpha=0.2)
    axs[4].plot(mean_q_per_cluster_thighs[14, :, i_cluster], color=rgba)
    axs[5].fill_between(np.arange(381),
                        mean_q_per_cluster_thighs[15, :, i_cluster] - std_q_per_cluster_thighs[15, :, i_cluster],
                        mean_q_per_cluster_thighs[15, :, i_cluster] + std_q_per_cluster_thighs[15, :, i_cluster], color=rgba, alpha=0.2)
    axs[5].plot(mean_q_per_cluster_thighs[15, :, i_cluster], color=rgba)
    if i_cluster == 0:
        axs[4].set_title(f"{model.nameDof()[14].to_string()}")
        axs[5].set_title(f"{model.nameDof()[15].to_string()}")

plt.suptitle(f"mean kinematics per cluster for {nb_twists}.5 twists")
plt.savefig(f'cluster_graphs/mean_clusters_graph_for_all_athletes_{nb_twists}.png', dpi=300)
# plt.show()


# plot clusters to make sure they were correctly identified
var_name = ["right_arm", "left_arm", "thighs"]
var_list = [q_right_arm, q_left_arm, q_thighs]
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