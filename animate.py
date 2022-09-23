
import biorbd
import numpy as np
import matplotlib.pyplot as plt
import bioviz

model_path = "Models/AuJo_JeCh_withoutMesh.bioMod"

b = bioviz.Viz(model_path)
# b.load_movement(Q)
b.exec()
