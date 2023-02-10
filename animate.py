
import biorbd
import numpy as np
import matplotlib.pyplot as plt
import bioviz

#model_path = "/home/lim/Documents/Stage_Lisa/AnthropoImpactOnTech/Models_all_dofs/MaJa.bioMod"
model_path = "/home/lim/Documents/Stage_Lisa/AnthropoImpactOnTech/Models/AuJo.bioMod"
Q = np.load("/home/lim/Documents/Stage_Lisa/AnthropoImpactOnTech/Solutions_multiModel_constraints/AuJo_2-(4_10_10_10_4_4_10_10_10_4)-2023-02-09-1651-q.npy")
b = bioviz.Viz(model_path)
b.load_movement(Q)
b.exec()
#b.quit()
