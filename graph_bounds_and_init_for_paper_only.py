
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython import embed

import biorbd
import bioviz
import bioptim

biorbd_model_path = "Models/JeCh_TechOpt83.bioMod"
model = biorbd.Model(biorbd_model_path)
nb_q = model.nbQ()
nb_joints = nb_q - model.nbRoot()

final_time = 1.87
n_shooting = (40, 100, 100, 100, 40)



q_bounds_min = np.zeros((nb_q, sum(n_shooting) + 1))
q_bounds_max = np.zeros((nb_q, sum(n_shooting) + 1))
qdot_bounds_min = np.zeros((nb_q, sum(n_shooting) + 1))
qdot_bounds_max = np.zeros((nb_q, sum(n_shooting) + 1))
tau_bounds_min = np.zeros((nb_joints, sum(n_shooting) + 1))
tau_bounds_max = np.zeros((nb_joints, sum(n_shooting) + 1))
q_init = np.zeros((nb_q, sum(n_shooting) + 1))
qdot_init = np.zeros((nb_q, sum(n_shooting) + 1))
tau_init = np.zeros((nb_joints, sum(n_shooting) + 1))

q_bounds_min[0, :] = model.segments()[0].QRanges()[0].min()
q_bounds_max[0, :] = model.segments()[0].QRanges()[0].max()
q_bounds_min[1, :] = model.segments()[0].QRanges()[1].min()
q_bounds_max[1, :] = model.segments()[0].QRanges()[1].max()
q_bounds_min[2, :] = model.segments()[0].QRanges()[2].min()
q_bounds_max[2, :] = model.segments()[0].QRanges()[2].max()
q_bounds_min[3, :] = model.segments()[0].QRanges()[3].min()
q_bounds_max[3, :] = model.segments()[0].QRanges()[3].max()
q_bounds_min[4, :] = model.segments()[0].QRanges()[4].min()
q_bounds_max[4, :] = model.segments()[0].QRanges()[4].max()
q_bounds_min[5, :] = model.segments()[0].QRanges()[5].min()
q_bounds_max[5, :] = model.segments()[0].QRanges()[5].max()
q_bounds_min[6, :] = model.segments()[3].QRanges()[0].min()
q_bounds_max[6, :] = model.segments()[3].QRanges()[0].max()
q_bounds_min[7, :] = model.segments()[3].QRanges()[1].min()
q_bounds_max[7, :] = model.segments()[3].QRanges()[1].max()
q_bounds_min[8, :] = model.segments()[4].QRanges()[0].min()
q_bounds_max[8, :] = model.segments()[4].QRanges()[0].max()
q_bounds_min[9, :] = model.segments()[4].QRanges()[1].min()
q_bounds_max[9, :] = model.segments()[4].QRanges()[1].max()
q_bounds_min[10, :] = model.segments()[6].QRanges()[0].min()
q_bounds_max[10, :] = model.segments()[6].QRanges()[0].max()
q_bounds_min[11, :] = model.segments()[6].QRanges()[1].min()
q_bounds_max[11, :] = model.segments()[6].QRanges()[1].max()
q_bounds_min[12, :] = model.segments()[7].QRanges()[0].min()
q_bounds_max[12, :] = model.segments()[7].QRanges()[0].max()
q_bounds_min[13, :] = model.segments()[7].QRanges()[1].min()
q_bounds_max[13, :] = model.segments()[7].QRanges()[1].max()
q_bounds_min[14, :] = model.segments()[9].QRanges()[0].min()
q_bounds_max[14, :] = model.segments()[9].QRanges()[0].max()
q_bounds_min[15, :] = model.segments()[9].QRanges()[1].min()
q_bounds_max[15, :] = model.segments()[9].QRanges()[1].max()

qdot_bounds_min[0, :] = model.segments()[0].QDotRanges()[0].min()
qdot_bounds_max[0, :] = model.segments()[0].QDotRanges()[0].max()
qdot_bounds_min[1, :] = model.segments()[0].QDotRanges()[1].min()
qdot_bounds_max[1, :] = model.segments()[0].QDotRanges()[1].max()
qdot_bounds_min[2, :] = model.segments()[0].QDotRanges()[2].min()
qdot_bounds_max[2, :] = model.segments()[0].QDotRanges()[2].max()
qdot_bounds_min[3, :] = model.segments()[0].QDotRanges()[3].min()
qdot_bounds_max[3, :] = model.segments()[0].QDotRanges()[3].max()
qdot_bounds_min[4, :] = model.segments()[0].QDotRanges()[4].min()
qdot_bounds_max[4, :] = model.segments()[0].QDotRanges()[4].max()
qdot_bounds_min[5, :] = model.segments()[0].QDotRanges()[5].min()
qdot_bounds_max[5, :] = model.segments()[0].QDotRanges()[5].max()
qdot_bounds_min[6, :] = model.segments()[3].QDotRanges()[0].min()
qdot_bounds_max[6, :] = model.segments()[3].QDotRanges()[0].max()
qdot_bounds_min[7, :] = model.segments()[3].QDotRanges()[1].min()
qdot_bounds_max[7, :] = model.segments()[3].QDotRanges()[1].max()
qdot_bounds_min[8, :] = model.segments()[4].QDotRanges()[0].min()
qdot_bounds_max[8, :] = model.segments()[4].QDotRanges()[0].max()
qdot_bounds_min[9, :] = model.segments()[4].QDotRanges()[1].min()
qdot_bounds_max[9, :] = model.segments()[4].QDotRanges()[1].max()
# qdot_bounds_min[10, :] = model.segments()[6].QDotRanges()[0].min()
# qdot_bounds_max[10, :] = model.segments()[6].QDotRanges()[0].max()
# qdot_bounds_min[11, :] = model.segments()[6].QDotRanges()[1].min()
# qdot_bounds_max[11, :] = model.segments()[6].QDotRanges()[1].max()
qdot_bounds_min[12, :] = model.segments()[7].QDotRanges()[0].min()
qdot_bounds_max[12, :] = model.segments()[7].QDotRanges()[0].max()
qdot_bounds_min[13, :] = model.segments()[7].QDotRanges()[1].min()
qdot_bounds_max[13, :] = model.segments()[7].QDotRanges()[1].max()
qdot_bounds_min[14, :] = model.segments()[9].QDotRanges()[0].min()
qdot_bounds_max[14, :] = model.segments()[9].QDotRanges()[0].max()
qdot_bounds_min[15, :] = model.segments()[9].QDotRanges()[1].min()
qdot_bounds_max[15, :] = model.segments()[9].QDotRanges()[1].max()


q_bounds_min[0, :140] = -0.1
q_bounds_max[0, :140] = 0.1
q_bounds_min[0, 140:340] = -0.2
q_bounds_max[0, 140:340] = 0.2
q_bounds_min[0, 340:] = -0.1
q_bounds_max[0, 340:] = 0.1
q_bounds_min[1, :] = -1
q_bounds_max[1, :] = 1
q_bounds_min[1, -1] = -0.1
q_bounds_max[1, -1] = 0.1
q_bounds_min[2, :] = 0
q_bounds_max[2, 0] = 0
q_bounds_max[2, 1:] = 8
q_bounds_min[2, -1] = 0
q_bounds_max[2, -1] = 0.1

q_bounds_min[3, 0] = 0.5
q_bounds_max[3, 0] = 0.5
q_bounds_min[3, 1:240] = 0
q_bounds_max[3, 1:40] = 4 * 3.14 + 0.1
q_bounds_max[3, 40:240] = -0.50 + 4 * 3.14
q_bounds_min[3, 140:240] = 2 * 3.14 - 0.1
q_bounds_min[3, 240:340] = 2 * 3.14 - 0.1
q_bounds_max[3, 240:340] = 2 * 3.14 + 3 / 2 * 3.14 + 0.1
q_bounds_min[3, 340:] = 2 * 3.14 + 3 / 2 * 3.14 - 0.2
q_bounds_max[3, 340:] = -0.5 + 4 * 3.14
q_bounds_min[3, -1] = -0.5 + 4 * 3.14 - 0.1
q_bounds_max[3, -1] = -0.5 + 4 * 3.14 + 0.1
q_bounds_min[4, 0] = 0
q_bounds_max[4, 0] = 0
q_bounds_min[4, 1:140] = - 3.14 / 16
q_bounds_max[4, 1:140] = 3.14 / 16
q_bounds_min[4, 140:340] = - 3.14 / 4
q_bounds_max[4, 140:340] = 3.14 / 4
q_bounds_min[4, 340:] = - 3.14 / 16
q_bounds_max[4, 340:] = 3.14 / 16
q_bounds_min[5, 0] = 0
q_bounds_max[5, 0] = 0
q_bounds_min[5, 1:140] = -0.1
q_bounds_max[5, 1:140] = 0.1
q_bounds_min[5, 140:] = 0
q_bounds_max[5, 140:340] = 3.14
q_bounds_max[5, 240:340] = 5 * 3.14
q_bounds_min[5, 340:] = 3 * 3.14 - 0.1
q_bounds_max[5, 340:] = 3 * 3.14 + 0.1

q_bounds_min[6, 0] = 2.9
q_bounds_max[6, 0] = 2.9
q_bounds_min[6, -1] = 2.9 - 0.1
q_bounds_max[6, -1] = 2.9 + 0.1
q_bounds_min[7, 0] = 0
q_bounds_max[7, 0] = 0
q_bounds_min[7, -1] = -0.1
q_bounds_max[7, -1] = 0.1


q_bounds_min[8, 0] = 0
q_bounds_max[8, 0] = 0
q_bounds_min[8, -1] = -0.1
q_bounds_max[8, -1] = 0.1
q_bounds_min[9, 0] = 0
q_bounds_max[9, 0] = 0
q_bounds_min[9, -1] = -0.1
q_bounds_max[9, -1] = 0.1

# q_bounds_min[10, 0] = -2.9
# q_bounds_max[10, 0] = -2.9
# q_bounds_min[10, -1] = -2.9 - 0.1
# q_bounds_max[10, -1] = -2.9 + 0.1
# q_bounds_min[11, 0] = 0
# q_bounds_max[11, 0] = 0
# q_bounds_min[11, -1] = -0.1
# q_bounds_max[11, -1] = 0.1

q_bounds_min[12, 0] = 0
q_bounds_max[12, 0] = 0
q_bounds_min[12, -1] = -0.1
q_bounds_max[12, -1] = 0.1
q_bounds_min[13, 0] = 0
q_bounds_max[13, 0] = 0
q_bounds_min[13, -1] = -0.1
q_bounds_max[13, -1] = 0.1

q_bounds_min[14, 0] = -0.5
q_bounds_max[14, 0] = -0.5
q_bounds_max[14, 40:140] = -2.5
q_bounds_min[14, 240:] = -0.4
q_bounds_min[14, -1] = -0.6
q_bounds_max[14, -1] = -0.4

q_bounds_min[15, 0] = 0
q_bounds_max[15, 0] = 0
q_bounds_min[15, 1:41] = -0.1
q_bounds_max[15, 1:41] = 0.1
q_bounds_min[15, -1] = -0.1
q_bounds_max[15, -1] = 0.1

q_bounds_min[10:12, :] = q_bounds_min[6:8, :]
q_bounds_max[10:12, :] = q_bounds_max[6:8, :]

vzinit = 9.81 / (2 * final_time)

qdot_bounds_min[0, :] = -10
qdot_bounds_max[0, :] = 10
qdot_bounds_min[0, 0] = -0.5
qdot_bounds_max[0, 0] = 0.5
qdot_bounds_min[1, :] = -10
qdot_bounds_max[1, :] = 10
qdot_bounds_min[1, 0] = -0.5
qdot_bounds_max[1, 0] = 0.5
qdot_bounds_min[2, :] = -50
qdot_bounds_max[2, :] = 50
qdot_bounds_min[2, 0] = vzinit - 0.5
qdot_bounds_max[2, 0] = vzinit + 0.5

qdot_bounds_min[3, :] = 0.5
qdot_bounds_max[3, :] = 20
qdot_bounds_min[4, :] = -50
qdot_bounds_max[4, :] = 50
qdot_bounds_min[4, 0] = 0
qdot_bounds_max[4, 0] = 0
qdot_bounds_min[5, :] = -50
qdot_bounds_max[5, :] = 50
qdot_bounds_min[5, 0] = 0
qdot_bounds_max[5, 0] = 0

CoM_init = model.CoM(q_bounds_min[:, 0]).to_array()
root_orientation = model.globalJCS(q_bounds_min[:, 0], 0).to_array()
r = CoM_init - root_orientation[-1, :3]

bound_inf = q_bounds_min[:3, 0] + np.cross(r, qdot_bounds_min[:3, 0])
bound_sup = q_bounds_max[:3, 0] + np.cross(r, qdot_bounds_max[:3, 0])
qdot_bounds_min[:3, 0] = min(bound_sup[0], bound_inf[0]), min(bound_sup[1], bound_inf[1]), min(bound_sup[2], bound_inf[2])
qdot_bounds_max[:3, 0] = max(bound_sup[0], bound_inf[0]), max(bound_sup[1], bound_inf[1]), max(bound_sup[2], bound_inf[2])

qdot_bounds_min[6:, :] = -50
qdot_bounds_max[6:, :] = 50
qdot_bounds_min[6:, 0] = 0
qdot_bounds_max[6:, 0] = 0

q_init[3, :40] = np.linspace(0.5, 0, 40)
q_init[3, 40:140] = np.linspace(0, 2 * 3.14, 100)
q_init[3, 140:240] = 2 * 3.14
q_init[3, 240:340] = np.linspace(2 * 3.14, 2 * 3.14 + 3 / 2 * 3.14, 100)
q_init[3, 340:] = np.linspace(2 * 3.14 + 3 / 2 * 3.14, 4 * 3.14, 41)
q_init[5, 140:240] = 0.2
q_init[5, 240:340] = np.linspace(0, 3 * 3.14, 100)
q_init[5, 340:] = 3 * 3.14
q_init[6, :140] = 0.75
q_init[6, 140:240] = np.linspace(0.75, 0, 100)
q_init[7, :40] = np.linspace(2.9, 1.35, 40)
q_init[7, 40:140] = 1.35
q_init[7, 140:240] = np.linspace(1.35, 0, 100)
# q_init[10, :140] = -0.75
# q_init[10, 140:240] = np.linspace(-0.75, 0, 100)
# q_init[11, :40] = np.linspace(-2.9, -1.35, 40)
# q_init[11, 40:140] = -1.35
# q_init[11, 140:240] = np.linspace(-1.35, 0, 100)
q_init[14, :40] = np.linspace(-0.5, -2.6, 40)
q_init[14, 40:140] = -2.6
q_init[14, 140:240] = np.linspace(-2.6, 0, 100)
q_init[14, 340:] = np.linspace(0, -0.5, 41)

q_init[10:12, :] = q_init[6:8, :]

np.random.seed(1)

q_noise_matrix = (np.random.random((nb_q, sum(n_shooting)+1)) * 2 - 1) * 0.2
qdot_noise_matrix = (np.random.random((nb_q, sum(n_shooting)+1)) * 2 - 1) * 0.2
q_noise_matrix = q_noise_matrix * (q_bounds_max - q_bounds_min)
qdot_noise_matrix = qdot_noise_matrix * (qdot_bounds_max - qdot_bounds_min)
q_init += q_noise_matrix
qdot_init += qdot_noise_matrix

for shooting_point in range(sum(n_shooting)+1):
    too_small_index = np.where(q_init[:, shooting_point] < q_bounds_min[:, shooting_point])
    too_big_index = np.where(q_init[:, shooting_point] > q_bounds_max[:, shooting_point])
    q_init[too_small_index, shooting_point] = q_bounds_min[too_small_index, shooting_point] + 0.1
    q_init[too_big_index, shooting_point] = q_bounds_max[too_big_index, shooting_point] - 0.1

    too_small_index = np.where(qdot_init[:, shooting_point] < qdot_bounds_min[:, shooting_point])
    too_big_index = np.where(qdot_init[:, shooting_point] > qdot_bounds_max[:, shooting_point])
    qdot_init[too_small_index, shooting_point] = qdot_bounds_min[too_small_index, shooting_point] + 0.1
    qdot_init[too_big_index, shooting_point] = qdot_bounds_max[too_big_index, shooting_point] - 0.1


name_dof = ["Translation X",
            "Translation Y",
            "Translation Z",
            "Somersault",
            "Tilt",
            "Twist",
            "Right arm change plane",
            "Right arm elevation",
            "Right forearm pronation",
            "Right forearm flexion",
            "Left arm change plane",
            "Left arm elevation",
            "Left forearm pronation",
            "Left forearm flexion",
            "Hips flexion",
            "Hips lateral flexion"]
cmap = cm.get_cmap('viridis')
time = np.linspace(0, final_time, sum(n_shooting)+1)

fig, axs = plt.subplots(4, 4, figsize=(15, 15))
axs = axs.ravel()
for i in range(nb_q):
    axs[i].plot(time, q_bounds_min[i, :] * 180 / np.pi, color='k')
    axs[i].fill_between(time, np.ones((sum(n_shooting)+1, )) * -1000, q_bounds_min[i, :] * 180 / np.pi, color='k', alpha=0.1)
    axs[i].plot(time, q_bounds_max[i, :] * 180 / np.pi, color='k')
    axs[i].fill_between(time, q_bounds_max[i, :] * 180 / np.pi, np.ones((sum(n_shooting)+1, )) * 1000, color='k', alpha=0.1)
    axs[i].plot(time, q_init[i, :] * 180 / np.pi, '.', color=cmap(1/3))
    axs[i].set_title(name_dof[i], fontsize=18)
    axs[i].set_xlim([0, final_time])
    axs[i].set_ylim([np.min(q_bounds_min[i, :]) * 180 / np.pi - 10, np.max(q_bounds_max[i, :]) * 180 / np.pi + 10])

axs[0].set_ylabel("Joint angles [$^\circ$]", fontsize=20)
axs[4].set_ylabel("Joint angles [$^\circ$]", fontsize=20)
axs[8].set_ylabel("Joint angles [$^\circ$]", fontsize=20)
axs[12].set_ylabel("Joint angles [$^\circ$]", fontsize=20)
axs[12].set_xlabel("Time [s]", fontsize=20)
axs[13].set_xlabel("Time [s]", fontsize=20)
axs[14].set_xlabel("Time [s]", fontsize=20)
axs[15].set_xlabel("Time [s]", fontsize=20)

fig.tight_layout()
fig.savefig(f'q_bounds_init.png', dpi=300)
# fig.show()



fig, axs = plt.subplots(4, 4, figsize=(15, 15))
axs = axs.ravel()
for i in range(nb_q):
    axs[i].plot(time, qdot_bounds_min[i, :] * 180 / np.pi, color='k')
    axs[i].fill_between(time, np.ones((sum(n_shooting)+1, )) * -10000, qdot_bounds_min[i, :] * 180 / np.pi, color='k', alpha=0.1)
    axs[i].plot(time, qdot_bounds_max[i, :] * 180 / np.pi, color='k')
    axs[i].fill_between(time, qdot_bounds_max[i, :] * 180 / np.pi, np.ones((sum(n_shooting)+1, )) * 10000, color='k', alpha=0.1)
    axs[i].plot(time, qdot_init[i, :] * 180 / np.pi, '.', color=cmap(2/3))
    axs[i].set_title(name_dof[i], fontsize=18)
    axs[i].set_xlim([0, final_time])
    axs[i].set_ylim([np.min(qdot_bounds_min[i, :]) * 180 / np.pi - 100, np.max(qdot_bounds_max[i, :]) * 180 / np.pi + 100])

axs[0].set_ylabel("Joint velocities [$^\circ$/s]", fontsize=20)
axs[4].set_ylabel("Joint velocities [$^\circ$/s]", fontsize=20)
axs[8].set_ylabel("Joint velocities [$^\circ$/s]", fontsize=20)
axs[12].set_ylabel("Joint velocities [$^\circ$/s]", fontsize=20)
axs[12].set_xlabel("Time [s]", fontsize=20)
axs[13].set_xlabel("Time [s]", fontsize=20)
axs[14].set_xlabel("Time [s]", fontsize=20)
axs[15].set_xlabel("Time [s]", fontsize=20)

fig.tight_layout()
fig.savefig(f'qdot_bounds_init.png', dpi=300)
# fig.show()


qddot_max = 500
qddot_min = -500
qddot_init = (np.random.random((nb_q, sum(n_shooting)+1)) * 2 - 1) * 0.2
qddot_init *= qddot_max - qddot_min
fig, axs = plt.subplots(4, 4, figsize=(15, 15))
axs = axs.ravel()
for i in range(nb_q):
    if i < 6:
        axs[i].axis('off')
    else:
        axs[i].plot(time, np.ones((sum(n_shooting)+1, )) * qddot_min * 180 / np.pi, color='k')
        axs[i].fill_between(time, np.ones((sum(n_shooting)+1, )) * -10000, np.ones((sum(n_shooting)+1, )) * qddot_min * 180 / np.pi, color='k', alpha=0.1)
        axs[i].plot(time, np.ones((sum(n_shooting)+1, )) * qddot_max * 180 / np.pi, color='k')
        axs[i].fill_between(time, np.ones((sum(n_shooting)+1, )) * qddot_max * 180 / np.pi, np.ones((sum(n_shooting)+1, )) * 10000, color='k', alpha=0.1)
        axs[i].plot(time, qddot_init[i, :], '.', color=cmap(3/3))
        axs[i].set_title(name_dof[i], fontsize=18)
        axs[i].set_xlim([0, final_time])
        axs[i].set_ylim([qddot_min - 100, qddot_max + 100])

axs[0].set_ylabel(r"Joint accelerations [$^\circ/s^2$]", fontsize=19)
axs[4].set_ylabel(r"Joint accelerations [$^\circ/s^2$]", fontsize=19)
axs[8].set_ylabel(r"Joint accelerations [$^\circ/s^2$]", fontsize=19)
axs[12].set_ylabel(r"Joint accelerations [$^\circ/s^2$]", fontsize=19)
axs[12].set_xlabel("Time [s]", fontsize=20)
axs[13].set_xlabel("Time [s]", fontsize=20)
axs[14].set_xlabel("Time [s]", fontsize=20)
axs[15].set_xlabel("Time [s]", fontsize=20)

fig.tight_layout()
fig.savefig(f'qddot_bounds_init.png', dpi=300)
fig.show()
