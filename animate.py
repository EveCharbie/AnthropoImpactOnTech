import biorbd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import bioviz
import os
from IPython import embed

model_path = "Models/JeCh_TechOpt83.bioMod"
Folder_per_twist_nb = {
    "1": "Solutions_vrille_et_demi/",
    "2": "Solutions_double_vrille_et_demi/",
    "3": "Solutions_triple_vrille_et_demi/",
}
results_path = "solutions_techOPT83/"
cmap = cm.get_cmap("viridis")

FLAG_SAME_FIG = True

if FLAG_SAME_FIG:
    fig, axs = plt.subplots(4, 4, figsize=(18, 9))
    axs = axs.ravel()

for key in Folder_per_twist_nb:
    num_athlete = 0
    if not FLAG_SAME_FIG:
        fig, axs = plt.subplots(4, 4, figsize=(18, 9))
        axs = axs.ravel()
    nb_twists = int(key)
    results_path_this_time = results_path + Folder_per_twist_nb[key]
    for foldername in os.listdir(results_path_this_time):
        for filename in os.listdir(results_path_this_time + foldername + "/"):
            f = os.path.join(results_path_this_time + foldername + "/", filename)
            # checking if it is a file
            if os.path.isfile(f):
                if filename.endswith("-q.npy"):
                    num_athlete += 1
                    athlete_name = filename.split("-")[0]
                    print(athlete_name + " - " + str(num_athlete), "\n")
                    model = biorbd.Model(model_path)
                    if FLAG_SAME_FIG:
                        if nb_twists == 1:
                            rgba = cmap(num_athlete / 18 * 0.45)
                        elif nb_twists == 2:
                            rgba = cmap(num_athlete / 18 * 0.45 + 0.55)
                        # else:
                        #     rgba = cmap(num_athlete / 18 * 0.2 + 0.8)
                    else:
                        rgba = cmap(num_athlete / 18)
                    # rgba = cmap(nb_twists / 3)

                    # Load results
                    q = np.load(f)

                    # # if athlete_name not in done_athletes and athlete_name not in problematic_althetes:
                    # # Create an animation of the results
                    b = bioviz.Viz(model_path)
                    b.load_movement(q)
                    b.exec()

                    # Create a graph of the temporal evolution of Q (DoFs)
                    for i in range(q.shape[0]):
                        if i == 0:
                            axs[i].plot(q[i, :], color=rgba, label=athlete_name)
                        else:
                            axs[i].plot(q[i, :], color=rgba)

                        if num_athlete == 1:
                            axs[i].set_title(f"{model.nameDof()[i].to_string()}")
    if not FLAG_SAME_FIG:
        axs[0].legend(bbox_to_anchor=(4.8, 1), loc="upper left", borderaxespad=0.0, ncols=2, fontsize=12)
        plt.subplots_adjust(left=0.05, right=0.8, hspace=0.4)
        plt.suptitle(f"{nb_twists}.5 twists")
        plt.savefig(f"kinematics_graph_for_all_athletes_{nb_twists}.png", dpi=300)
        # plt.show()
if FLAG_SAME_FIG:
    axs[0].legend(bbox_to_anchor=(4.8, 1), loc="upper left", borderaxespad=0.0, ncols=2, fontsize=12)
    plt.subplots_adjust(left=0.05, right=0.8, hspace=0.4)
    plt.savefig("kinematics_graph_for_all_athletes.png", dpi=300)
    # plt.show()

print("Report the number of clusters of solutions per twist number + STD inside cluster?")
