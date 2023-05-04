import bioviz
import pickle
import numpy as np

sol = "/home/mickaelbegon/Documents/Stage_Lisa/AnthropoImpactOnTech/Solutions_MultiModel/AdCh_AlAd.pkl"


def animate(file, models, PKL_FLAG):
    model_path = '/home/mickaelbegon/Documents/Stage_Lisa/test_multimodel/double_model_AdCh'
    if PKL_FLAG:
        sol_file = open(file, "rb")
        dict_sol = pickle.load(sol_file)
        q_mapped = dict_sol['q_mapped'][0]
        # sol = dict_sol['sol']
        q = dict_sol['q']
        mapping = dict_sol['mapping']['to_second']

        b = bioviz.Viz(model_path)
        q_mapped = []
        for i in range(len(mapping)):
            q_mapped.append(q[mapping[i],:])
        q_mapped = np.array(q_mapped)
        b.load_movement(q_mapped)
        b.exec()

    # q = np.load(sol)
animate(sol, '', True)