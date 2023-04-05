
import biorbd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import bioviz
import os
from IPython import embed

# Compare the clusters for amplitude
model_path = "Models/JeCh_TechOpt83.bioMod"
model = biorbd.Model(model_path)
nb_twists = 1
chosen_clusters_dict = {}
results_path = 'solutions_multi_start/'
results_path_this_time = results_path + 'Solutions_vrille_et_demi/'
cmap = cm.get_cmap('magma')

# good_sols_per_athlete = {
#     "AuJo": {"cluster_1": [0], "cluster_2": [3], "cluster_3": [2]}, # , "cluster_4": [8]},
#     "ElMe": {"cluster_1": [], "cluster_2": [], "cluster_3": []}, # , "cluster_4": []},
#     "EvZl": {"cluster_1": [1], "cluster_2": [], "cluster_3": []}, # , "cluster_4": [5]},
#     "FeBl": {"cluster_1": [0], "cluster_2": [3], "cluster_3": [7]}, # , "cluster_4": []},
#     "JeCh_2": {"cluster_1": [], "cluster_2": [3], "cluster_3": [2]}, # , "cluster_4": [6, 7]},
#     "KaFu": {"cluster_1": [4], "cluster_2": [], "cluster_3": []}, # , "cluster_4": [9]},
#     "KaMi": {"cluster_1": [0], "cluster_2": [3], "cluster_3": [2]}, # , "cluster_4": [9]},
#     "LaDe": {"cluster_1": [0], "cluster_2": [], "cluster_3": []}, # , "cluster_4": [9]},
#     "MaCu": {"cluster_1": [0, 4], "cluster_2": [1], "cluster_3": [3]}, # , "cluster_4": [9]},
#     "MaJa": {"cluster_1": [], "cluster_2": [8], "cluster_3": [9]}, # , "cluster_4": [4]},
#     "OlGa": {"cluster_1": [4], "cluster_2": [], "cluster_3": []}, # , "cluster_4": [1]},
#     "Sarah":{"cluster_1": [0], "cluster_2": [], "cluster_3": []}, # , "cluster_4": [1]},
#     "SoMe": {"cluster_1": [], "cluster_2": [2], "cluster_3": [4]}, # , "cluster_4": []},
# }

good_sols_per_athlete = {
    "AuJo": {"cluster_2": [3], "cluster_3": [2]}, # , "cluster_4": [8]},
    "ElMe": {"cluster_2": [], "cluster_3": []}, # , "cluster_4": []},
    "EvZl": {"cluster_2": [], "cluster_3": []}, # , "cluster_4": [5]},
    "FeBl": {"cluster_2": [3], "cluster_3": [7]}, # , "cluster_4": []},
    "JeCh_2": {"cluster_2": [3], "cluster_3": [2]}, # , "cluster_4": [6, 7]},
    "KaFu": {"cluster_2": [], "cluster_3": []}, # , "cluster_4": [9]},
    "KaMi": {"cluster_2": [3], "cluster_3": [2]}, # , "cluster_4": [9]},
    "LaDe": {"cluster_2": [], "cluster_3": []}, # , "cluster_4": [9]},
    "MaCu": {"cluster_2": [1], "cluster_3": [3]}, # , "cluster_4": [9]},
    "MaJa": {"cluster_2": [8], "cluster_3": [9]}, # , "cluster_4": [4]},
    "OlGa": {"cluster_2": [], "cluster_3": []}, # , "cluster_4": [1]},
    "Sarah":{"cluster_2": [], "cluster_3": []}, # , "cluster_4": [1]},
    "SoMe": {"cluster_2": [2], "cluster_3": [4]}, # , "cluster_4": []},
}


fig, axs = plt.subplots(4, 4, figsize=(18, 9))
axs = axs.ravel()

cluster_arrays = {key: np.zeros((16, 381, 1)) for key in good_sols_per_athlete['AuJo'].keys()}
for i_name, name in enumerate(good_sols_per_athlete):
    for i_cluster, cluster_name in enumerate(good_sols_per_athlete['AuJo'].keys()):
        for i_sol in good_sols_per_athlete[name][cluster_name]:
            file_name = results_path_this_time + name + '/' + name + '_vrille_et_demi_' + str(i_sol) + "_CVG.pkl"
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

            rgba = cmap(i_cluster * 0.25 + 0.25)
            # Create a graph of the temporal evolution of Q (DoFs)
            for i in range(q.shape[0]):
                if i == 0:
                    axs[i].plot(q[i, :], color=rgba, label=name)
                else:
                    axs[i].plot(q[i, :], color=rgba)

                # if name == 'AuJo' and i_cluster == 0:
                #     axs[i].set_title(f"{model.nameDof()[i].to_string()}")

            cluster_arrays[cluster_name] = np.concatenate((cluster_arrays[cluster_name], q[:, :, np.newaxis]), axis=2)

axs[0].legend(bbox_to_anchor=(4.8, 1), loc='upper left', borderaxespad=0., ncols=2, fontsize=12)
plt.subplots_adjust(left=0.05, right=0.8, hspace=0.4)
plt.suptitle(f"{nb_twists}.5 twists")
plt.savefig(f'clusters_graph_for_all_athletes_{nb_twists}.png', dpi=300)
plt.show()

cluster_counter = {key: 0 for key in good_sols_per_athlete['AuJo'].keys()}
mean_std_per_cluster = {key: np.zeros((16, )) for key in good_sols_per_athlete['AuJo'].keys()}
mean_q_per_cluster = np.zeros((16, 381, 1))
std_q_per_cluster = np.zeros((16, 381, 1))
for i_cluster, cluster_name in enumerate(good_sols_per_athlete['AuJo'].keys()):
    cluster_arrays[cluster_name] = cluster_arrays[cluster_name][:, :, 1:]
    for i_name, name in enumerate(good_sols_per_athlete):
        if len(good_sols_per_athlete[name][cluster_name]) > 0:
            cluster_counter[cluster_name] += 1

    mean_std_per_cluster[cluster_name] = np.mean(np.std(cluster_arrays[cluster_name], axis=2), axis=1)
    mean_q_per_cluster = np.concatenate((mean_q_per_cluster, np.mean(cluster_arrays[cluster_name], axis=2)[: , :, np.newaxis]), axis=2)
    std_q_per_cluster = np.concatenate((std_q_per_cluster, np.std(cluster_arrays[cluster_name], axis=2)[: , :, np.newaxis]), axis=2)

mean_q_per_cluster = mean_q_per_cluster[:, :, 1:]
std_q_per_cluster = std_q_per_cluster[:, :, 1:]
mean_std_between_clusters = np.mean(np.std(mean_q_per_cluster, axis=2), axis=1)

fig, axs = plt.subplots(4, 4, figsize=(18, 9))
axs = axs.ravel()
for i_cluster, cluster_name in enumerate(good_sols_per_athlete['AuJo'].keys()):
    print(f"{cluster_name} was used by {cluster_counter[cluster_name]} / {len(good_sols_per_athlete)} athletes")
    print(f"Sum of mean std on cluster {cluster_name} was {np.sum(mean_std_per_cluster[cluster_name][3:])}")

    rgba = cmap(i_cluster * 0.25 + 0.25)
    for i in range(mean_q_per_cluster.shape[0]):
        axs[i].fill_between(np.arange(381), mean_q_per_cluster[i, :, i_cluster] - std_q_per_cluster[i, :, i_cluster],
                            mean_q_per_cluster[i, :, i_cluster] + std_q_per_cluster[i, :,i_cluster], color=rgba, alpha=0.2)
        axs[i].plot(mean_q_per_cluster[i, :, i_cluster], color=rgba)
    if i_cluster == 0:
        axs[i].set_title(f"{model.nameDof()[i].to_string()}")

plt.suptitle(f"mean kinematics per cluster for {nb_twists}.5 twists")
plt.savefig(f'mean_clusters_graph_for_all_athletes_{nb_twists}.png', dpi=300)
# plt.show()




# # fig, axs = plt.subplots(4, 4, figsize=(18, 9))
# # axs = axs.ravel()
# plt.figure()
# for i_cluster, cluster_name in enumerate(good_sols_per_athlete['AuJo'].keys()):
#     print(f"{cluster_name} was used by {cluster_counter[cluster_name]} / {len(good_sols_per_athlete)} athletes")
#     print(f"Sum of mean std on cluster {cluster_name} was {np.sum(mean_std_per_cluster[cluster_name][3:])}")
#
#     rgba = cmap(i_cluster * 0.25 + 0.25)
#     # for i_trial in range(len(good_sols_per_athlete['AuJo'][cluster_name])):
#     #     for i in range(mean_q_per_cluster.shape[0]):
#     #         axs[i].plot(cluster_arrays[cluster_name][i, :, i_trial], color=rgba)
#     #     if i_cluster == 0:
#     #         axs[i].set_title(f"{model.nameDof()[i].to_string()}")
#
#     for i_trial in range(cluster_arrays[cluster_name].shape[2]):
#         plt.plot(cluster_arrays[cluster_name][0, :, i_trial], color=rgba)
#
# plt.suptitle(f"Cluster for {nb_twists}.5 twists")
# plt.savefig(f'test_clusters_graph_for_all_athletes_{nb_twists}.png', dpi=300)
# # plt.show()


plt.figure()
plt.plot(cluster_arrays["cluster_2"][0, :, :])
plt.savefig(f'test_clusters_graph_for_all_athletes_{nb_twists}.png', dpi=300)

plt.figure()
plt.plot(cluster_arrays["cluster_3"][0, :, :])
plt.savefig(f'test_clusters_graph_for_all_athletes_{nb_twists}.png', dpi=300)




fig, axs = plt.subplots(4, 4, figsize=(18, 9))
axs = axs.ravel()
rgba = cmap(0)
for i in range(mean_q_per_cluster.shape[0]):
    axs[i].fill_between(np.arange(381), np.mean(mean_q_per_cluster, axis=2)[i, :] - np.std(mean_q_per_cluster, axis=2)[i, :],
                        np.mean(mean_q_per_cluster, axis=2)[i, :] + np.std(mean_q_per_cluster, axis=2)[i, :], color=rgba, alpha=0.2)
    for i_cluster, cluster_name in enumerate(good_sols_per_athlete['AuJo'].keys()):
        axs[i].plot(mean_q_per_cluster[i, :, i_cluster], color=rgba)
    axs[i].set_title(f"{model.nameDof()[i].to_string()}")
np.std(mean_q_per_cluster, axis=2)
plt.show()
print(f"Whereas the weighted mean std between clusters was {np.sum(mean_std_between_clusters[3:])}")
