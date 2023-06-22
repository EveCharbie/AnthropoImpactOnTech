
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


athletes_names_list = ["AdCh", "AlAd" ,"AuJo" ,"Benjamin","ElMe" ,"EvZl" ,"FeBl" ,"JeCh" ,"KaFu" ,"KaMi" ,"LaDe" ,"MaCu" ,"MaJa" ,"OlGa" ,"Sarah","SoMe" ,"WeEm" ,"ZoTs"]

folder_per_twist_nb = {"5": "Multistart_double_vrille/"}  # {"3": "Solutions_vrille_et_demi/"}

results_path = "solutions_multi_start/"
model_path = "Models"
num_half_twist = "5"  # "3"
athlete_done = []
nb_iter = {name: [] for name in athletes_names_list}
results_path_this_time = folder_per_twist_nb[num_half_twist]

for athlete in athletes_names_list:
    nb_twists = int(num_half_twist)
    for filename in os.listdir(results_path_this_time):
        if athlete != filename.split("_")[0]:
            continue
        if filename.removesuffix(".pkl")[-3] == "C":
            f = os.path.join(results_path_this_time, filename)
            filename = results_path_this_time + filename
            if os.path.isfile(f):
                if filename.endswith(".pkl"):
                    with open(filename, "rb") as f:
                        data = pickle.load(f)
                        nb_iter[athlete].append(data["sol"].iterations)

fig, ax = plt.subplots(1, 1, figsize=(18, 9))
for i, athlete in enumerate(athletes_names_list):
    ax.plot(np.ones((len(nb_iter[athlete]), )) * i, nb_iter[athlete], '.')
ax.set_xticks([jj for jj in range(len(athletes_names_list))])
ax.set_xticklabels([name for name in athletes_names_list])
plt.savefig("number_of_iteration_so_far.png")
plt.show()
