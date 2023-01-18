
import biorbd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import bioviz
import os
from IPython import embed

model_path = "Models/JeCh_TechOpt83.bioMod"
Folder_per_twist_nb = ["Solutions_vrille_et_demi/", "Solutions_double_vrille_et_demi/", "Solutions_triple_vrille_et_demi/"]
results_path = 'solutions_techOPT83/'
cmap = cm.get_cmap('viridis')

num_athlete = 0

fig, axs = plt.subplots(4, 4)
axs = axs.ravel()

# problematic_althetes = ["EvZl", "ZoTs", "MaJa", "ElMe_TechOpt83"]
# done_athletes = ["AdCh", "AlAd", "AuJo_TechOpt83", "KaFu", "LaDe", "MaCu", "MaJa", "OlGa", "SoMe", "Benjamin", "KaMi", "FeBl"]
for nb_twists in [1, 2, 3]:
    results_path_this_time = results_path + Folder_per_twist_nb[nb_twists - 1]
    for foldername in os.listdir(results_path_this_time):
        for filename in os.listdir(results_path_this_time + foldername + '/'):
            f = os.path.join(results_path_this_time + foldername + '/', filename)
            # checking if it is a file
            if os.path.isfile(f):
                if filename.endswith("-q.npy"):

                    num_athlete += 1
                    athlete_name = filename.split("-")[0]
                    print(athlete_name + ' - ' + str(num_athlete), '\n')
                    model = biorbd.Model(model_path)
                    if nb_twists == 1:
                        rgba = cmap(num_athlete / 18 * 0.2)
                    elif nb_twists == 2:
                        rgba = cmap(num_athlete / 18 * 0.2 + 0.4)
                    else:
                        rgba = cmap(num_athlete / 18 * 0.2 + 0.8)
                    # rgba = cmap(nb_twists / 3)

                    # Load results
                    q = np.load(f)

                    # if athlete_name not in done_athletes and athlete_name not in problematic_althetes:
                    #     # Create an animation of the results
                    #     b = bioviz.Viz(model_path)
                    #     b.load_movement(q)
                    #     b.exec()

                    # Create a graph of the temporal evolution of Q (DoFs)
                    for i in range(q.shape[0]):
                        if i == 0:
                            axs[i].plot(q[i, :], color=rgba, label=athlete_name)
                        else:
                            axs[i].plot(q[i, :], color=rgba)

                        if num_athlete == 1:
                            axs[i].set_title(f"{model.nameDof()[i].to_string()}")
axs[0].legend(bbox_to_anchor=(5, 1), loc='upper left', borderaxespad=0.)
plt.subplots_adjust(right=0.7)
# plt.tight_layout()
plt.show()