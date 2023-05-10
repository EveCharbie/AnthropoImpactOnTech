'''
This code allows to animate de optimal solutions from the multi-model OCP.
'''

import bioviz
import pickle
import numpy as np

sol = "/home/mickaelbegon/Documents/Stage_Lisa/Anthropo Lisa/new_sol_double_vrille/AdCh_double_vrille_et_demi_0_CVG.pkl"


def animate(file, models, PKL_FLAG):
    model_path = 'Models/Models_Lisa/AdCh.bioMod'
    if PKL_FLAG:
        sol_file = open(file, "rb")
        dict_sol = pickle.load(sol_file)
        # q_mapped = dict_sol['q_mapped'][0]
        # sol = dict_sol['sol']
        q = dict_sol['q']
        # mapping = dict_sol['mapping']['to_second']

        b = bioviz.Viz(model_path)
        # q_mapped = []
        # for i in range(len(mapping)):
        #     q_mapped.append(q[mapping[i],:])
        # q_mapped = np.array(q_mapped)
        b.load_movement(q)
        b.exec()

    # q = np.load(sol)
animate(sol, '', True)
