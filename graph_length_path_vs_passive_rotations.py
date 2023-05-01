
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statsmodels.api as sm
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

if nb_twists == 1:
    saved_data_path = "kinematics_graphs/vrille_et_demi/data_pickled/"
elif nb_twists == 2:
    saved_data_path = "kinematics_graphs/double_vrille_et_demi/data_pickled/"
else:
    raise ValueError("Number of twists not implemented")

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

    excel = pd.read_excel("degrees_of_liberty.xlsx", index_col=None, header=None).to_numpy()

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
    noise_index_to_keep = np.where(cost <= 1.05*best_cost)[0]

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
axs[0].set_xlim(0.45*360, 0.8*360)
axs[0].set_ylim(3.8, 10.15)
axs[1].set_xlim(0.45*360, 0.8*360)
axs[1].set_ylim(5.9, 10.6)
axs[2].set_xlim(0.27*360, 0.42*360)
axs[2].set_ylim(6.1, 8.1)
plt.show()
# plt.savefig(save_path + "clusters_length_path_for_all_athletes.png", dpi=300)




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
    axs[i, 0].set_xlim(0.45*360, 0.8*360)
    axs[i, 0].set_ylim(3.8, 10.15)
    axs[i, 1].set_xlim(0.45*360, 0.8*360)
    axs[i, 1].set_ylim(5.9, 10.6)
    axs[i, 2].set_xlim(0.27*360, 0.42*360)
    axs[i, 2].set_ylim(6.1, 8.1)

axs[0, 0].set_title("Right Arm")
axs[0, 1].set_title("Left Arm")
axs[0, 2].set_title("Hips")

# plt.show()
plt.savefig(save_path + "length_path_for_all_athletes_per_clusters.png", dpi=300)



embed()
def plot_length_path_for_all_solutions_all_joints(data_to_graph):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    i_athlete = 0
    i_trajectory = 0
    twist_potential = []
    joints_trajectories = []
    for name in athletes_number.keys():
        color = cmap(i_athlete / 18)
        for i, idx in enumerate(data_to_graph[name]["noise_index_to_keep"]):
            twist_potential += [2*data_to_graph[name]["arms_twist_potential"] + data_to_graph[name]["hips_twist_potential"]]
            joints_trajectories += [data_to_graph[name]["right_arm_trajectory"][idx] + data_to_graph[name]["left_arm_trajectory"][idx] + data_to_graph[name]["legs_trajectory"][idx]]
            ax.plot(twist_potential[i_trajectory], joints_trajectories[i_trajectory], 'o', color=color)
            i_trajectory += 1
        i_athlete += 1
        plt.plot(0, 0, 'o', color=color, label="Athlete #" + str(athletes_number[name]))

    fig.subplots_adjust(right=0.8)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Athlete number
    ax.set_xlabel("Twist potential [$^\circ$]")  # 2*Arm + Hips
    ax.set_ylabel("Normalized summed joint trajectories length [m]")
    ax.set_xlim(1.235*360, 2*360)
    ax.set_ylim(15.45, 29.15)

    return twist_potential, joints_trajectories

twist_potential, joints_trajectories = plot_length_path_for_all_solutions_all_joints(data_to_graph)
# plt.show()
plt.savefig(save_path + "length_path_for_all_solutions_all_joints.png", dpi=300)

_, _ = plot_length_path_for_all_solutions_all_joints(data_to_graph)
correlation = scipy.stats.spearmanr(np.array(twist_potential), np.array(joints_trajectories))[0]
lin_regress = scipy.stats.linregress(np.array(twist_potential), np.array(joints_trajectories))
slope = lin_regress.slope
intercept = lin_regress.intercept

x_lin_regress = np.linspace(1.235*360, 2*360, 10)
y_lin_regress = slope*x_lin_regress + intercept
plt.plot(x_lin_regress, y_lin_regress, '-k', linewidth=0.5)
plt.text(2*360-25, 29.15-0.5, "S=" + str(round(correlation, 2)), fontsize=10)
plt.savefig(save_path + "length_path_for_all_solutions_all_joints_with_correlation.png", dpi=300)
plt.show()


