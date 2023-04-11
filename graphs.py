import biorbd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# import Colormap as normalize
import bioviz
import os

# import LinearSegmentedColormap
# import Bou
from IPython import embed
import pickle


model_path = "Models/JeCh_TechOpt83.bioMod"
folder_per_athlete = {"KaFu": "KaFu/"}
folder_per_twist_nb = {"1": "Solutions_vrille_et_demi/"}
# , "2": "Solutions_double_vrille_et_demi/", "3": "Solutions_triple_vrille_et_demi/"}

results_path = "/home/lim/Documents/Stage_Lisa/Solutions_Tech_opt_83/Sol_with_noise/"
# a quoi ca sert

# cmap = cm.colors.Colormap
# pour  un athlete on veut toutes les ol avec les bruits et le gradient de couleurs sur la valeur de la fonction objc
folder_graphs = "/home/lim/Documents/Stage_Lisa/kinematics_graphs"
done_athlete = os.listdir(folder_graphs)

FLAG_SAME_FIG = True

if FLAG_SAME_FIG:
    fig, axs = plt.subplots(4, 4, figsize=(18, 9))
    #    fi
    axs = axs.ravel()

# choix de l'athlete ou boucle for :
# athlete = 'Kafu'
for athlete in folder_per_athlete:
    file_name = f"kinematics_graph for {athlete}.png"
    if file_name not in done_athlete:
        for key in folder_per_twist_nb:
            if not FLAG_SAME_FIG:
                fig, axs = plt.subplots(4, 4, figsize=(18, 9))
                axs = axs.ravel()
            nb_twists = int(key)
            results_path_this_time = results_path + folder_per_athlete[athlete] + folder_per_twist_nb[key]
            Bruit = []
            C = []
            Data = []
            for filename in os.listdir(results_path_this_time):
                Bruit += filename.split("_")[-2]
                # #for filename in os.listdir(results_path_this_time + foldername + '/'):
                f = os.path.join(results_path_this_time, filename)
                filename = results_path_this_time + filename

                if os.path.isfile(f):
                    if filename.endswith(".pkl"):
                        # checking if it is a file
                        with open(filename, "rb") as f:
                            data = pickle.load(f)
                            Data.append(data["q"])
                            C.append(data["sol"].cost.toarray()[0][0])
            C = np.array(C)
            # cmap = cm.colors.LinearSegmentedColormap('color_map', C).from_list('color_map',['b', 'g'])
            norm = cm.colors.Normalize()

            C = norm.__call__(C)
            cdict = {
                "red": [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
                "green": [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.75, 1.0, 1.0), (1.0, 1.0, 1.0)],
                "blue": [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)],
            }

            cmap = cm.colors.LinearSegmentedColormap.from_list("cmap", ["b", "g"], 256, 1.0)
            cmap = cm.get_cmap("magma")
            # cmap.__call__(C, None, False)
            max = C.max()
            min = C.min()

            cmap(0)

            for i in range(len(Bruit)):
                bruit = Bruit[i]
                cost = C[i]
                q = Data[i]
                model = biorbd.Model(model_path)
                if FLAG_SAME_FIG:
                    if nb_twists == 1:
                        ratio = (cost - min) / (max - min)
                        # BoundaryNorm(ratio, (min,max), 100)
                        rgba = cmap(cost)

                        print(f" cost is {cost}")
                        print(f" ratio is {ratio}")
                        # rgba = (0.9, 0.8, 0.6, 1)

                        print(f" color is {rgba}")
                    elif nb_twists == 2:
                        rgba = cmap(cost - min / (max - min))
                    else:
                        rgba = cmap(cost - min / (max - min))

                    for i in range(len(q[0])):
                        Q = []
                        for phase in range(len(q)):
                            Q += q[phase][i].tolist()[:]

                        # for n in range(16):
                        # axs[i].pcolor(Q, cost, cmap=cm, vmin=-160000, vmax=160000, label=bruit)
                        # axs[i].title(title)

                        if i == 0:
                            axs[i].plot(Q, color=rgba, label=bruit)
                        # plt.colorbar()
                        # axs[i].colorbar()
                        else:
                            axs[i].plot(Q, color=rgba)
                        # axs.col
                        axs[i].set_title(f"{model.nameDof()[i].to_string()}")

                        # plt.show()
                        # plt.savefig

        # if not FLAG_SAME_FIG:
        #     axs[0].legend(bbox_to_anchor=(4.8, 1), loc='upper left', borderaxespad=0., ncols=2, fontsize=12)
        #     plt.subplots_adjust(left=0.05, right=0.8, hspace=0.4)
        #     plt.suptitle(f"{nb_twists}.5 twists")
        #     plt.savefig(f'kinematics_graph_for_all_athletes_{nb_twists}.png', dpi=300)
        #     plt.show()

        if FLAG_SAME_FIG:
            axs[0].legend(bbox_to_anchor=(4.8, 1), loc="upper left", borderaxespad=0.0, ncols=2, fontsize=12)
            plt.subplots_adjust(left=0.05, right=0.8, hspace=0.4)
            plt.savefig(f"kinematics_graph for {athlete}.png", dpi=300)
            plt.show()

print("Report the number of clusters of solutions per twist number + STD inside cluster?")

# def do_plot(y,n, cost, bruit):


# plt.subplot(4, 4, i+1)
#                                    plt.pcolor(cost, cmap=cm, vmin=-160000, vmax=160000, label=bruit)
#                                    plt.title(f"{model.nameDof()[i].to_string()}")
#                                    plt.colorbar()

# def raimbow(x,y, ref) :
#   color_ref = (0.9, 0.8, 0.6, 1)
