
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
from IPython import embed
import scipy

import biorbd
import bioviz
import bioviz



def compute_normalized_trajectory_length(model, q, marker_name):
    marker_names = [model.markerNames()[j].to_string() for j in range(model.nbMarkers())]
    marker_idx = marker_names.index(marker_name)
    marker = np.zeros((3, q.shape[1]))
    for i in range(q.shape[1]):
        marker[:, i] = model.markers(q[:, i])[marker_idx].to_array()
    marker_diff = np.diff(marker, axis=1)
    norm_marker_diff = np.linalg.norm(marker_diff, axis=0)
    return np.sum(norm_marker_diff)


nb_twists = 1
save_path = "overview_graphs/"
models_path = "Models/Models_Lisa/"
saved_data_path = "kinematics_graphs/vrille_et_demi/data_pickled/"

cmap = cm.get_cmap('viridis')
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

with open(save_path + "clusters_sol.pkl", "rb") as f:
    data = pickle.load(f)
    mean_q_per_cluster_right_arm = data["mean_q_per_cluster_right_arm"]
    mean_q_per_cluster_left_arm = data["mean_q_per_cluster_left_arm"]
    mean_q_per_cluster_thighs = data["mean_q_per_cluster_thighs"]
    std_q_per_cluster_right_arm = data["std_q_per_cluster_right_arm"]
    std_q_per_cluster_left_arm = data["std_q_per_cluster_left_arm"]
    std_q_per_cluster_thighs = data["std_q_per_cluster_thighs"]
    q_right_arm = data["q_right_arm"]
    q_left_arm = data["q_left_arm"]
    q_thighs = data["q_thighs"]
    cluster_right_arm = data["cluster_right_arm"]
    cluster_left_arm = data["cluster_left_arm"]
    cluster_thighs = data["cluster_thighs"]

    path_to_degree_of_liberty = "Passive_rotations/passive rotations results/"
    excel = pd.read_excel(f"{path_to_degree_of_liberty}degrees_of_liberty.xlsx", index_col=None, header=None).to_numpy()

data_to_graph = {name: {"noise_idx": None, "noise_index_to_keep": None, "arms_twist_potential": None, "hips_twist_potential":None, "right_arm_trajectory": None, "left_arm_trajectory": None, "legs_trajectory": None} for name in athletes_number.keys()}
markers = ['o', 'x', '^', '*', 's', 'p']
i_athlete = 0
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for name in athletes_number.keys():
    filename = name + ".pkl"
    with open(saved_data_path + filename, "rb") as f:
        data = pickle.load(f)
        cost = data['cost']
        q = data['q']
        q_integrated = data['q_integrated']
        reintegration_error = data['reintegration_error']
        noise_idx = data['noise']

    model = biorbd.Model(models_path + name + ".bioMod")
    right_arm_trajectory = np.zeros(len(noise_idx))
    left_arm_trajectory = np.zeros(len(noise_idx))
    legs_trajectory = np.zeros(len(noise_idx))
    for noise_index_this_time in range(len(noise_idx)):
        q_joined = np.zeros(np.shape(q[noise_index_this_time][0][:, :-1]))
        q_joined[:, :] = q[noise_index_this_time][0][:, :-1]
        for i in range(1, len(q[noise_index_this_time])):
            if i == len(q[noise_index_this_time]) - 1:
                q_joined = np.hstack((q_joined, q[noise_index_this_time][i]))
            else:
                q_joined = np.hstack((q_joined, q[noise_index_this_time][i][:, :-1]))

        q_joined[:6, :] = 0
        right_arm_trajectory[noise_index_this_time] = compute_normalized_trajectory_length(model, q_joined, "RightArmNormalized")
        left_arm_trajectory[noise_index_this_time] = compute_normalized_trajectory_length(model, q_joined, "LeftArmNormalized")
        legs_trajectory[noise_index_this_time] = compute_normalized_trajectory_length(model, q_joined, "LegsNormalized")


    index_arms = np.where(excel[:, 0] == name + " bras gauche bas, droit descend")[0][0]
    arms_twist_potential = abs(excel[index_arms, 4]) * 360
    index_hips = np.where(excel[:, 0] == name + " bras en  bas, jambes tilt")[0][0]
    hips_twist_potential = abs(excel[index_hips, 4]) * 360

    best_cost = np.min(cost)
    noise_index_to_keep = np.where(cost <= 1.01*best_cost)[0]

    color = cmap(i_athlete / 18)
    for i, idx in enumerate(noise_index_to_keep):
        for j, key in enumerate(cluster_right_arm[name].keys()):
            if int(noise_idx[idx]) in cluster_right_arm[name][key]:
                axs[0].plot(arms_twist_potential, right_arm_trajectory[idx], marker=markers[j], color=color)
        for j, key in enumerate(cluster_left_arm[name].keys()):
            if int(noise_idx[idx]) in cluster_left_arm[name][key]:
                axs[1].plot(arms_twist_potential, left_arm_trajectory[idx], marker=markers[j], color=color)
        for j, key in enumerate(cluster_thighs[name].keys()):
            if int(noise_idx[idx]) in cluster_thighs[name][key]:
                axs[2].plot(hips_twist_potential, legs_trajectory[idx], marker=markers[j], color=color)

    data_to_graph[name]["noise_idx"] = noise_idx
    data_to_graph[name]["noise_index_to_keep"] = noise_index_to_keep
    data_to_graph[name]["arms_twist_potential"] = arms_twist_potential
    data_to_graph[name]["hips_twist_potential"] = hips_twist_potential
    data_to_graph[name]["right_arm_trajectory"] = right_arm_trajectory
    data_to_graph[name]["left_arm_trajectory"] = left_arm_trajectory
    data_to_graph[name]["legs_trajectory"] = legs_trajectory


    i_athlete += 1
    axs[0].plot(0, 0, '-', color=color, label="Athlete #" + str(athletes_number[name]))

for j, key in enumerate(cluster_right_arm[name].keys()):
    axs[1].plot(0, 0, marker=markers[j], linestyle='None', color='black', label="Cluster #" + key[-1])

fig.subplots_adjust(right=0.8)
axs[0].legend(loc='center left', bbox_to_anchor=(3.5, 0.3))
axs[1].legend(loc='center left', bbox_to_anchor=(2.3, 0.9))
axs[0].set_xlabel("Arm twist potential [$^\circ$]")
axs[0].set_ylabel("Normalized right arm trajectory length [m]")
axs[1].set_xlabel("Arm twist potential [$^\circ$]")
axs[1].set_ylabel("Normalized left arm trajectory length [m]")
axs[2].set_xlabel("Hips twist potential [$^\circ$]")
axs[2].set_ylabel("Normalized legs trajectory length [m]")
axs[0].set_xlim(425, 775)
axs[0].set_ylim(3.8, 10.15)
axs[1].set_xlim(425, 775)
axs[1].set_ylim(3.8, 10.15)
axs[2].set_xlim(0.27*360, 0.42*360)
axs[2].set_ylim(3.8, 10.15)
# plt.show()
plt.savefig(save_path + "clusters_length_path_for_all_athletes.png", dpi=300)




fig, axs = plt.subplots(6, 3, figsize=(15, 15))
for j, key in enumerate(cluster_right_arm[name].keys()):
    i_athlete = 0
    for name in athletes_number.keys():

        color = cmap(i_athlete / 18)
        for i, idx in enumerate(data_to_graph[name]["noise_index_to_keep"]):
            if key in cluster_right_arm[name].keys():
                if int(noise_idx[idx]) in cluster_right_arm[name][key]:
                    axs[j, 0].plot(data_to_graph[name]["arms_twist_potential"], data_to_graph[name]["right_arm_trajectory"][idx], marker=markers[j], color=color)
            if key in cluster_left_arm[name].keys():
                if int(noise_idx[idx]) in cluster_left_arm[name][key]:
                    axs[j, 1].plot(data_to_graph[name]["arms_twist_potential"], data_to_graph[name]["left_arm_trajectory"][idx], marker=markers[j], color=color)
            if key in cluster_thighs[name].keys():
                if int(noise_idx[idx]) in cluster_thighs[name][key]:
                    axs[j, 2].plot(data_to_graph[name]["hips_twist_potential"], data_to_graph[name]["legs_trajectory"][idx], marker=markers[j], color=color)

        i_athlete += 1
        if j == 0:
            axs[0, 0].plot(0, 0, '-', color=color, label="Athlete #" + str(athletes_number[name]))

for j, key in enumerate(cluster_right_arm[name].keys()):
    axs[0, 1].plot(0, 0, marker=markers[j], linestyle='None', color='black', label="Cluster #" + key[-1])

fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.2)
# axs[0, 0].legend(loc='center left', bbox_to_anchor=(3.2, -3))  # Athlete number
# axs[0, 1].legend(loc='center left', bbox_to_anchor=(2.1, 0.5))  # Cluster marker shape
for i in range(6):
    if "cluster_" + str(i+1) not in cluster_right_arm[name].keys():
        axs[i, 0].set_axis_off()
    if "cluster_" + str(i+1) not in cluster_left_arm[name].keys():
        axs[i, 1].set_axis_off()
    if "cluster_" + str(i+1) not in cluster_thighs[name].keys():
        axs[i, 2].set_axis_off()
    axs[i, 0].set_ylabel("Cluster #" + str(i + 1))
    # axs[i, 0].set_xlabel("Arm twist potential [$^\circ$]")
    # axs[i, 0].set_ylabel("Normalized right arm trajectory length [m]")
    # axs[i, 1].set_xlabel("Arm twist potential [$^\circ$]")
    # axs[i, 1].set_ylabel("Normalized left arm trajectory length [m]")
    # axs[i, 2].set_xlabel("Hips twist potential [$^\circ$]")
    # axs[i, 2].set_ylabel("Normalized legs trajectory length [m]")
    axs[i, 0].set_xlim(425, 775)
    axs[i, 0].set_ylim(3, 10.5)
    axs[i, 1].set_xlim(425, 775)
    axs[i, 1].set_ylim(3, 10.5)
    # axs[i, 2].set_xlim(0.27*360, 0.42*360)
    # axs[i, 2].set_ylim(6.1, 8.1)

axs[0, 0].set_title("Right Arm")
axs[0, 1].set_title("Left Arm")
axs[0, 2].set_title("Hips")

# plt.show()
plt.savefig(save_path + "length_path_for_all_athletes_per_clusters.png", dpi=300)


def plot_length_path_for_all_solutions_all_joints(data_to_graph, graph_type="arm_arm_hips"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    i_athlete = 0
    i_trajectory = 0
    twist_potential = []
    joints_trajectories = []
    twist_potential_per_athlete = {}
    for name in athletes_number.keys():
        color = cmap(i_athlete / 18)
        for i, idx in enumerate(data_to_graph[name]["noise_index_to_keep"]):
            if graph_type == "arm_arm_hips":
                twist_potential += [data_to_graph[name]["arms_twist_potential"] + data_to_graph[name]["hips_twist_potential"]]
                joints_trajectories += [(data_to_graph[name]["right_arm_trajectory"][idx] + data_to_graph[name]["left_arm_trajectory"][idx])/2 + data_to_graph[name]["legs_trajectory"][idx]]
            elif graph_type == "arm_arm":
                twist_potential += [data_to_graph[name]["arms_twist_potential"]]
                joints_trajectories += [(data_to_graph[name]["right_arm_trajectory"][idx] + data_to_graph[name]["left_arm_trajectory"][idx])/2]
            else:
                raise ValueError("graph_type must be either 'arm_arm_hips' or 'arm_arm'")

            ax.plot(twist_potential[i_trajectory], joints_trajectories[i_trajectory], 'o', color=color)
            i_trajectory += 1
        if graph_type == "arm_arm_hips":
            twist_potential_per_athlete[name] = data_to_graph[name]["arms_twist_potential"] + data_to_graph[name]["hips_twist_potential"]
        else:
            twist_potential_per_athlete[name] = data_to_graph[name]["arms_twist_potential"]
        i_athlete += 1
        plt.plot(0, 0, 'o', color=color, label="Athlete #" + str(athletes_number[name]))

    fig.subplots_adjust(right=0.8)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Athlete number
    if graph_type == "arm_arm_hips":   # Arm + Hips
        ax.set_xlabel(r"arm + hips twist potential [$^\circ$]")
        ax.set_ylabel("Normalized combined arms and hips trajectories length [m]")
    else:
        ax.set_xlabel(r"arm twist potential [$^\circ$]")
        ax.set_ylabel("Normalized combined arms trajectories length [m]")
    if graph_type == "arm_arm_hips":
        # ax.set_xlim(435, 725)
        # ax.set_ylim(15.45, 29.15)
        ax.set_xlim(525, 900)
        ax.set_ylim(10, 20)
    else:
        ax.set_xlim(400, 800)
        ax.set_ylim(4, 11)

    return twist_potential, twist_potential_per_athlete, joints_trajectories

twist_potential, twist_potential_per_athlete, joints_trajectories = plot_length_path_for_all_solutions_all_joints(data_to_graph)
# plt.show()
plt.savefig(save_path + "length_path_for_all_solutions_all_joints.png", dpi=300)

_, _, _ = plot_length_path_for_all_solutions_all_joints(data_to_graph, graph_type="arm_arm")
# plt.show()
plt.savefig(save_path + "length_path_for_all_solutions_arms.png", dpi=300)

_, _, _ = plot_length_path_for_all_solutions_all_joints(data_to_graph)
correlation = scipy.stats.spearmanr(np.array(twist_potential), np.array(joints_trajectories))[0]
lin_regress = scipy.stats.linregress(np.array(twist_potential), np.array(joints_trajectories))
slope = lin_regress.slope
intercept = lin_regress.intercept

x_lin_regress = np.linspace(500, 900, 10)
y_lin_regress = slope*x_lin_regress + intercept
plt.plot(x_lin_regress, y_lin_regress, '-k', linewidth=0.5)
plt.text(860, 19.5, "S=" + str(round(correlation, 2)), fontsize=10)
plt.savefig(save_path + "length_path_for_all_solutions_all_joints_with_correlation.png", dpi=300)
# plt.show()

"""
Spearman associsation:
0-0.19 : very weak
0.2-0.39 : weak
0.4-0.59 : moderate
0.6-0.79 : strong
0.8-1 : very strong
"""


# plot the correlation between the twist potential and the anthropometry
athletes_reduced_anthropo = {name: None for name in athletes_number.keys()}
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
for i, name in enumerate(athletes_number.keys()):
    ax.plot(data_to_graph[name]["arms_twist_potential"], data_to_graph[name]["hips_twist_potential"], '.k')
    ax.text(data_to_graph[name]["arms_twist_potential"]+0.5, data_to_graph[name]["hips_twist_potential"]+0.5, str(athletes_number[name]), fontsize=10)
ax.set_xlabel("Arm twist potential [$\circ$]")
ax.set_ylabel("Hips twist potential [$\circ$]")
plt.savefig("overview_graphs/arm_vs_hips_twist_potential.png", dpi=300)
# plt.show()


# plot the correlation between the arm twist potential and hip twist potential
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for i, name in enumerate(athletes_number.keys()):
    model_anthropo_file_name = f'Models/text_files/{name}.txt'
    with open(model_anthropo_file_name) as f:
        model_anthropo = f.readlines()

    right_arm_perimeter = float(model_anthropo[24][6:-1])
    left_arm_perimeter = float(model_anthropo[25][6:-1])
    arm_perimeter = (right_arm_perimeter + left_arm_perimeter) / 2
    height = float(model_anthropo[-1][14:-1])
    athletes_reduced_anthropo[name] = {"arm_perimeter": arm_perimeter, "height": height}

min_twist_potential = np.min(twist_potential)
max_twist_potential = np.max(twist_potential)

ax.scatter(athletes_reduced_anthropo["WeEm"]["arm_perimeter"], athletes_reduced_anthropo["WeEm"]["height"], c=twist_potential_per_athlete["WeEm"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[0], label="Athlete #1 (MAG)")
ax.scatter(athletes_reduced_anthropo["FeBl"]["arm_perimeter"], athletes_reduced_anthropo["FeBl"]["height"], c=twist_potential_per_athlete["FeBl"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[0], label="Athlete #2 (MAG)")
ax.scatter(athletes_reduced_anthropo["AdCh"]["arm_perimeter"], athletes_reduced_anthropo["AdCh"]["height"], c=twist_potential_per_athlete["AdCh"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[0], label="Athlete #3 (MAG)")
ax.scatter(athletes_reduced_anthropo["MaJa"]["arm_perimeter"], athletes_reduced_anthropo["MaJa"]["height"], c=twist_potential_per_athlete["MaJa"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[0], label="Athlete #4 (MAG)")

ax.scatter(athletes_reduced_anthropo["AlAd"]["arm_perimeter"], athletes_reduced_anthropo["AlAd"]["height"], c=twist_potential_per_athlete["AlAd"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[1], label="Athlete #5 (MT)")
ax.scatter(athletes_reduced_anthropo["JeCh"]["arm_perimeter"], athletes_reduced_anthropo["JeCh"]["height"], c=twist_potential_per_athlete["JeCh"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[1], label="Athlete #6 (MT)")
ax.scatter(athletes_reduced_anthropo["Benjamin"]["arm_perimeter"], athletes_reduced_anthropo["Benjamin"]["height"], c=twist_potential_per_athlete["Benjamin"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[1], label="Athlete #7 (MT)")
ax.scatter(athletes_reduced_anthropo["Sarah"]["arm_perimeter"], athletes_reduced_anthropo["Sarah"]["height"], c=twist_potential_per_athlete["Sarah"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[1], label="Athlete #8 (WT)")
ax.scatter(athletes_reduced_anthropo["SoMe"]["arm_perimeter"], athletes_reduced_anthropo["SoMe"]["height"], c=twist_potential_per_athlete["SoMe"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[1], label="Athlete #9 (WT)")

ax.scatter(athletes_reduced_anthropo["OlGa"]["arm_perimeter"], athletes_reduced_anthropo["OlGa"]["height"], c=twist_potential_per_athlete["OlGa"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[2], label="Athlete #10 (MD)")
ax.scatter(athletes_reduced_anthropo["KaFu"]["arm_perimeter"], athletes_reduced_anthropo["KaFu"]["height"], c=twist_potential_per_athlete["KaFu"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[2], label="Athlete #11 (WD)")
ax.scatter(athletes_reduced_anthropo["MaCu"]["arm_perimeter"], athletes_reduced_anthropo["MaCu"]["height"], c=twist_potential_per_athlete["MaCu"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[2], label="Athlete #12 (MD)")
ax.scatter(athletes_reduced_anthropo["KaMi"]["arm_perimeter"], athletes_reduced_anthropo["KaMi"]["height"], c=twist_potential_per_athlete["KaMi"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[2], label="Athlete #13 (WD)")

ax.scatter(athletes_reduced_anthropo["ElMe"]["arm_perimeter"], athletes_reduced_anthropo["ElMe"]["height"], c=twist_potential_per_athlete["ElMe"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[3], label="Athlete #14 (WAG)")
ax.scatter(athletes_reduced_anthropo["ZoTs"]["arm_perimeter"], athletes_reduced_anthropo["ZoTs"]["height"], c=twist_potential_per_athlete["ZoTs"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[3], label="Athlete #15 (WAG)")
ax.scatter(athletes_reduced_anthropo["LaDe"]["arm_perimeter"], athletes_reduced_anthropo["LaDe"]["height"], c=twist_potential_per_athlete["LaDe"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[3], label="Athlete #16 (WAG)")
ax.scatter(athletes_reduced_anthropo["EvZl"]["arm_perimeter"], athletes_reduced_anthropo["EvZl"]["height"], c=twist_potential_per_athlete["EvZl"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[3], label="Athlete #17 (WAG)")

color_bar_handle = ax.scatter(athletes_reduced_anthropo["AuJo"]["arm_perimeter"], athletes_reduced_anthropo["AuJo"]["height"], c=twist_potential_per_athlete["AuJo"], vmin=min_twist_potential, vmax=max_twist_potential, marker=markers[4], label="Athlete #18 (WAS)")

fig.subplots_adjust(left=0.07, right=0.74, bottom=0.1, top=0.9)
ax.set_xlabel("Biceps perimeter [cm]")
ax.set_ylabel("Height [cm]")
ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), ncol=1)
cbar_ax = fig.add_axes([0.75, 0.1, 0.02, 0.8])
cbar = fig.colorbar(color_bar_handle, cax=cbar_ax)
cbar.ax.set_title('Combined\ntwist potential [$\circ$]')
plt.savefig("overview_graphs/musculature_height_twist_potential.png", dpi=300)
# plt.show()

# Create a figure showing the length of the trajectory for all clusters of solutions with STD and min-max range
right_arm_trajectory_per_cluster = {key: [] for key in cluster_right_arm[name].keys()}
left_arm_trajectory_per_cluster = {key: [] for key in cluster_left_arm[name].keys()}
legs_trajectory_per_cluster = {key: [] for key in cluster_thighs[name].keys()}
for i_athlete, name in enumerate(athletes_number.keys()):
    for key in cluster_right_arm[name].keys():
        idx_this_time = []
        for idx in cluster_right_arm[name][key]:
            idx_this_time += [data_to_graph[name]["noise_idx"].index(str(idx))]
        trajectory_cluster_to_add = data_to_graph[name]["right_arm_trajectory"][idx_this_time]
        right_arm_trajectory_per_cluster[key] += list(trajectory_cluster_to_add)

    for key in cluster_left_arm[name].keys():
        idx_this_time = []
        for idx in cluster_left_arm[name][key]:
            idx_this_time += [data_to_graph[name]["noise_idx"].index(str(idx))]
        trajectory_cluster_to_add = data_to_graph[name]["left_arm_trajectory"][idx_this_time]
        left_arm_trajectory_per_cluster[key] += list(trajectory_cluster_to_add)

    for key in cluster_thighs[name].keys():
        idx_this_time = []
        for idx in cluster_thighs[name][key]:
            idx_this_time += [data_to_graph[name]["noise_idx"].index(str(idx))]
        trajectory_cluster_to_add = data_to_graph[name]["legs_trajectory"][idx_this_time]
        legs_trajectory_per_cluster[key] += list(trajectory_cluster_to_add)

cmap_viridis = cm.get_cmap('viridis')
cmap_magma = cm.get_cmap('magma')
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for i_cluster, key in enumerate(cluster_right_arm[name].keys()):
    rgba = cmap_magma(i_cluster * 1 / 6)
    axs[0].plot(i_cluster + 0.25,
                np.mean(right_arm_trajectory_per_cluster[key]),
                "s",
                color=rgba)
    axs[0].fill_between(np.array([i_cluster, i_cluster + 0.5]),
                        np.array([np.mean(right_arm_trajectory_per_cluster[key]) - np.std(right_arm_trajectory_per_cluster[key]),
                                  np.mean(right_arm_trajectory_per_cluster[key]) - np.std(right_arm_trajectory_per_cluster[key])]),
                        np.array([np.mean(right_arm_trajectory_per_cluster[key]) + np.std(right_arm_trajectory_per_cluster[key]),
                                    np.mean(right_arm_trajectory_per_cluster[key]) + np.std(right_arm_trajectory_per_cluster[key])]),
                        color=rgba,
                        alpha=0.2)
    axs[0].plot(np.array([i_cluster, i_cluster + 0.5]),
                np.array([np.min(right_arm_trajectory_per_cluster[key]), np.min(right_arm_trajectory_per_cluster[key])]),
                "-",
                color=rgba)
    axs[0].plot(np.array([i_cluster, i_cluster + 0.5]),
                np.array([np.max(right_arm_trajectory_per_cluster[key]), np.max(right_arm_trajectory_per_cluster[key])]),
                "-",
                color=rgba)
    axs[0].plot(np.array([i_cluster + 0.25, i_cluster + 0.25]),
                np.array([np.min(right_arm_trajectory_per_cluster[key]), np.max(right_arm_trajectory_per_cluster[key])]),
                "-",
                color=rgba)

for i_cluster, key in enumerate(cluster_left_arm[name].keys()):
    rgba = cmap_viridis(i_cluster * 1/6)
    axs[1].plot(i_cluster + 0.25,
                np.mean(left_arm_trajectory_per_cluster[key]),
                "s",
                color=rgba)
    axs[1].fill_between(np.array([i_cluster, i_cluster + 0.5]),
                        np.array([np.mean(left_arm_trajectory_per_cluster[key]) - np.std(left_arm_trajectory_per_cluster[key]),
                                  np.mean(left_arm_trajectory_per_cluster[key]) - np.std(left_arm_trajectory_per_cluster[key])]),
                        np.array([np.mean(left_arm_trajectory_per_cluster[key]) + np.std(left_arm_trajectory_per_cluster[key]),
                                    np.mean(left_arm_trajectory_per_cluster[key]) + np.std(left_arm_trajectory_per_cluster[key])]),
                        color=rgba,
                        alpha=0.2)
    axs[1].plot(np.array([i_cluster, i_cluster + 0.5]),
                np.array([np.min(left_arm_trajectory_per_cluster[key]), np.min(left_arm_trajectory_per_cluster[key])]),
                "-",
                color=rgba)
    axs[1].plot(np.array([i_cluster, i_cluster + 0.5]),
                np.array([np.max(left_arm_trajectory_per_cluster[key]), np.max(left_arm_trajectory_per_cluster[key])]),
                "-",
                color=rgba)
    axs[1].plot(np.array([i_cluster + 0.25, i_cluster + 0.25]),
                np.array([np.min(left_arm_trajectory_per_cluster[key]), np.max(left_arm_trajectory_per_cluster[key])]),
                "-",
                color=rgba)

for i_cluster, key in enumerate(cluster_thighs[name].keys()):
    rgba = cmap_viridis(1 - i_cluster * 1/6)
    axs[2].plot(i_cluster + 0.25,
                np.mean(legs_trajectory_per_cluster[key]),
                "s",
                color=rgba)
    axs[2].fill_between(np.array([i_cluster, i_cluster + 0.5]),
                        np.array([np.mean(legs_trajectory_per_cluster[key]) - np.std(legs_trajectory_per_cluster[key]),
                                  np.mean(legs_trajectory_per_cluster[key]) - np.std(legs_trajectory_per_cluster[key])]),
                        np.array([np.mean(legs_trajectory_per_cluster[key]) + np.std(legs_trajectory_per_cluster[key]),
                                    np.mean(legs_trajectory_per_cluster[key]) + np.std(legs_trajectory_per_cluster[key])]),
                        color=rgba,
                        alpha=0.2)
    axs[2].plot(np.array([i_cluster, i_cluster + 0.5]),
                np.array([np.min(legs_trajectory_per_cluster[key]), np.min(legs_trajectory_per_cluster[key])]),
                "-",
                color=rgba)
    axs[2].plot(np.array([i_cluster, i_cluster + 0.5]),
                np.array([np.max(legs_trajectory_per_cluster[key]), np.max(legs_trajectory_per_cluster[key])]),
                "-",
                color=rgba)
    axs[2].plot(np.array([i_cluster + 0.25, i_cluster + 0.25]),
                np.array([np.min(legs_trajectory_per_cluster[key]), np.max(legs_trajectory_per_cluster[key])]),
                "-",
                color=rgba)

axs[0].set_xlim(-0.5, 6)
axs[1].set_xlim(-0.5, 6)
axs[2].set_xlim(-0.5, 6)
plt.savefig("cluster_graphs/mean_length_path_for_clusters.svg")
plt.show()






# 2.5 Twists -----------------------------------------------------------------------------------------------------------
nb_twists = 2

results_path = 'solutions_multi_start/'
results_path_this_time = results_path + f'Solutions_double_vrille_et_demi/'

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for i_name, name in enumerate(athletes_number.keys()):

    color = cmap(i_name / 18)
    model = biorbd.Model(models_path + name + ".bioMod")
    full_twist_potential = data_to_graph[name]["arms_twist_potential"] + data_to_graph[name]["hips_twist_potential"]

    for i_sol in range(9):
        file_name = results_path_this_time + name + '/' + name + f'_double_vrille_et_demi_' + str(i_sol) + "_CVG.pkl"
        if not os.path.isfile(file_name):
            continue

        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        Q = data['q']

        q_joined = np.zeros(np.shape(Q[0][:, :-1]))
        q_joined[:, :] = Q[0][:, :-1]
        for i in range(1, len(Q)):
            if i == len(Q) - 1:
                q_joined = np.hstack((q_joined, Q[i]))
            else:
                q_joined = np.hstack((q_joined, Q[i][:, :-1]))

        q_joined[:6, :] = 0
        length_right_arm_trajectory = compute_normalized_trajectory_length(model, q_joined, "RightArmNormalized")
        length_left_arm_trajectory = compute_normalized_trajectory_length(model, q_joined, "LeftArmNormalized")
        length_leg_trajectory = compute_normalized_trajectory_length(model, q_joined, "LegsNormalized")
        full_trajectory = (length_right_arm_trajectory + length_left_arm_trajectory)/2 + length_leg_trajectory

        ax.plot(full_twist_potential, full_trajectory, 'o', color=color)

        if i_sol == 0:
            plt.plot(0, 0, 'o', color=color, label="Athlete #" + str(athletes_number[name]))

fig.subplots_adjust(right=0.8)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Athlete number
ax.set_xlim(650, 900)
ax.set_ylim(12, 23)
ax.set_xlabel(r"arm + hips twist potential [$^\circ$]")
ax.set_ylabel("Normalized combined arms and hips trajectories length [m]")
plt.savefig(save_path + "length_path_for_all_solutions_all_joints_double_vrille.png", dpi=300)
plt.show()