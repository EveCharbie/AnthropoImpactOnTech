
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import bioptim
import bioviz
import scipy.interpolate as interp


athletes_to_compare = ["Athlete_05", "Athlete_16"]

folder_per_twist_nb = {"3": "Solutions_vrille_et_demi/"}  # {"5": "Solutions_double_vrille_et_demi/"}

results_path = "solutions_multi_start/"
model_path = "Models"
num_half_twist = "3"  # 5
athlete_done = []

best_q = {key: [] for key in athletes_to_compare}
best_cost = {key: np.inf for key in athletes_to_compare}
best_time = {key: np.inf for key in athletes_to_compare}
best_sol = {key: np.inf for key in athletes_to_compare}
for athlete in athletes_to_compare:
    results_path_this_time = results_path + folder_per_twist_nb[num_half_twist] + athlete + "/"
    nb_twists = int(num_half_twist)
    for filename in os.listdir(results_path_this_time):
        if athlete[-2:] != filename.split("_")[1]:
            continue
        if filename.removesuffix(".pkl")[-3] == "C":
            f = os.path.join(results_path_this_time, filename)
            filename = results_path_this_time + filename
            if os.path.isfile(f):
                if filename.endswith(".pkl"):
                    with open(filename, "rb") as f:
                        data = pickle.load(f)
                        cost = data["sol"].cost
                        q = data["q"]
                        if cost < best_cost[athlete]:
                            best_cost[athlete] = cost
                            qs = np.hstack((q[0][:, :-1], q[1][:, :-1], q[2][:, :-1], q[3][:, :-1], q[4]))
                            best_q[athlete] = qs
                            best_time[athlete] = [float(data["sol"].parameters["time"][i]) for i in range(data["sol"].parameters["time"].shape[0])]
                            best_sol[athlete] = data["sol"]

q_to_compare = np.vstack((best_q["Athlete_05"], best_q["Athlete_16"]))
fps = 300
n_frames = [round(best_time[athlete][i] * fps) for i in range(len(best_time[athlete]))]
n_shooting = np.array([40, 100, 100, 100, 40])

q_to_compare[1, :] -= 1
q_to_compare[16+1, :] += 1

for i in range(len(n_shooting)):

    n_shooting_start_this_time = np.sum(n_shooting[:i])
    if i == len(n_shooting) - 1:
        n_shooting_end_this_time = np.sum(n_shooting)
    else:
        n_shooting_end_this_time = np.sum(n_shooting[:i+1])

    interpolation = interp.interp1d(np.linspace(0, best_time[athlete][i], n_shooting[i]), q_to_compare[:, n_shooting_start_this_time:n_shooting_end_this_time], kind="cubic")
    if i == 0:
        q_for_video = interpolation(np.linspace(0, best_time[athlete][i], n_frames[i]))
    else:
        q_for_video = np.hstack((q_for_video, interpolation(np.linspace(0, best_time[athlete][i], n_frames[i]))))

b = bioviz.Viz("Models/Athlete_05_Athlete_16.bioMod",
               mesh_opacity=0.8,
               show_global_center_of_mass=False,
               show_gravity_vector=False,
               show_segments_center_of_mass=False,
               show_global_ref_frame=False,
               show_local_ref_frame=False,
               experimental_markers_color=(1, 1, 1),
               background_color=(1.0, 1.0, 1.0),
               )
b.load_movement(q_for_video)
b.set_camera_zoom(0.5)
b.set_camera_focus_point(0, 0, 0)
b.maximize()
b.update()
b.start_recording(f"comp_05_16.ogv")
for frame in range(q_for_video.shape[1] + 1):
    b.movement_slider[0].setValue(frame)
    b.add_frame()
b.stop_recording()
b.quit()