'''
This code allows to animate de optimal solutions from the multi-model OCP.
'''

import bioviz
import biorbd
import pickle
import numpy as np
from bioptim import BiMappingList

sol = "/home/mickaelbegon/Documents/Eve/AnthropoImpactOnTech/Solutions_MultiModel/AdCh_AlAd_CVG.pkl"


def animate(file, models, PKL_FLAG):
    model_path = 'Models/AuJo_JeCh.bioMod'
    if PKL_FLAG:
        sol_file = open(file, "rb")
        dict_sol = pickle.load(sol_file)
        # sol = dict_sol['sol']
        q = dict_sol['q']
        q_to_first = dict_sol['mapping']['to_first']
        q_to_second = dict_sol['mapping']['to_second']

        mappings = BiMappingList()
        mappings.add("q", to_first=q_to_first, to_second=q_to_second)

        q_mapped = mappings['q'].to_second.map(q)

        b = bioviz.Viz(model_path)
        b.load_movement(q_mapped)
        b.exec()

    # q = np.load(sol)
animate(sol, '', True)
