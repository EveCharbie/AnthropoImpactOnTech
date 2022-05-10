import numpy as np
import bioviz

q = np.load('Solutions/sol-2022-05-10-164925-q.npy')
Q = np.zeros(q.shape)
Q[6:] = q[6:]

viz = bioviz.Viz('Models/JeCh_TechOpt83.bioMod')
viz.load_movement(Q)
viz.exec()
