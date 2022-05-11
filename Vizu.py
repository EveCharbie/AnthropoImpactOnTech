import numpy as np
import bioviz

q = np.load('Solutions/premiere-bonne-2022-05-11-q.npy')
Q = np.zeros(q.shape)
Q[6:] = q[6:]
#Q[2] = 1.  # au-dessus du plancher
Q[5] = -3.14/2

viz = bioviz.Viz('Models/JeCh_TechOpt83.bioMod')
viz.load_movement(Q)
viz.exec()
