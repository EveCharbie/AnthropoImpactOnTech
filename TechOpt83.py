"""
TODO: Create a more meaningful example (make sure to translate all the variables [they should correspond to the model])
This example uses a representation of a human body by a trunk_leg segment and two arms and has the objective to...
It is designed to show how to use a model that has quaternions in their degrees of freedom.
"""


import numpy as np
import biorbd_casadi as biorbd
from casadi import MX, Function
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    InterpolationType,
    OdeSolver,
    Node,
    Solver,
    BiMappingList,
    CostType,
    ConstraintList,
    ConstraintFcn,
)

import pickle
import time

def prepare_ocp(
    biorbd_model_path: str, n_shooting: int, final_time: float, ode_solver: OdeSolver = OdeSolver.RK4()
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    n_shooting: int
        The number of shooting points
    final_time: float
        The time at the final node
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, marker_index=1, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=100)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.9, max_bound=1.1, weight=0.01)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Define control path constraint
    #dof_mappings = BiMappingList()
    #dof_mappings.add("tau", to_second=[None, None, None, None, None, None, 0, 1, 2, 3], to_first=[6, 7, 8, 9])
    
    n_tau = 10#4
    tau_min, tau_max, tau_init = -500, 500, 0
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)

    # Initial guesses
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()

    x = np.vstack((np.zeros((nb_q, 2)), np.ones((nb_qdot, 2))))  # pourquoi 2 colonnes. rep: debut fin
    x_init = InitialGuessList()
    x_init.add(x, interpolation=InterpolationType.LINEAR)
    

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Pour plus tard ici
    # CoM_Q_sym = MX.sym('CoM', biorbd_model.nbQ()) # Q
    # CoM_Qdot_sym = MX.sym('CoMdot', biorbd_model.nbQ()) # Qdot
    # CoM_Q_init = np.zeros((biorbd_model.nbQ()))
    # CoM_Qdot_init = np.zeros((biorbd_model.nbQ()))
    #
    # CoM_func = Function('CoM_func', [CoM_Q_sym], [biorbd_model.CoM(CoM_Q_sym).to_mx()])
    # CoMdot_func = Function('CoMdot_func', [CoM_Q_sym, CoM_Qdot_sym], [biorbd_model.CoM(CoM_Q_sym, CoM_Qdot_sym).to_mx()])
    # bodyVelo_func = Function('bodyVelo_func', [CoM_Q_sym, CoM_Qdot_sym], [biorbd_model.bodyAngularVelocity(CoM_Q_sym, CoM_Qdot_sym).to_mx()])
    #
    # CoM_Q_int_DM = CoM_func(CoM_Q_init)
    # CoMdot_Q_int_DM = CoMdot_func(CoM_Q_init, CoM_Qdot_init)
    # bodyVelo__DM = bodyVelo_func(CoM_Q_init, CoM_Qdot_init)

    # Contraintes de position

    # deplacement
    x_bounds[0].min[:3, :] = -.1
    x_bounds[0].max[:3, :] = .1
    x_bounds[0].min[:3, 0] = 0
    x_bounds[0].max[:3, 0] = 0
    x_bounds[0].min[2, 1] = 0
    x_bounds[0].max[2, 1] = 20

    # le salto autour de x
    x_bounds[0].min[3, 0] = 0
    x_bounds[0].max[3, 0] = 0
    x_bounds[0].min[3, 1] = -2 * 3.14 - .1
    x_bounds[0].max[3, 1] = 0
    x_bounds[0].min[3, 2] = -2 * 3.14 - .1  # fin un tour vers l'avant
    x_bounds[0].max[3, 2] = -2 * 3.14 + .1
    # limitation du tilt autour de y
    x_bounds[0].min[4, 0] = 0
    x_bounds[0].max[4, 0] = 0
    x_bounds[0].min[4, 1] = - 3.14 / 4
    x_bounds[0].max[4, 1] = 3.14 / 4
    x_bounds[0].min[4, 2] = - 3.14 / 8
    x_bounds[0].max[4, 2] = 3.14 / 8
    # la vrille autour de z
    x_bounds[0].min[5, 0] = 0
    x_bounds[0].max[5, 0] = 0
    x_bounds[0].min[5, 1] = 0
    x_bounds[0].max[5, 1] = 3 * 3.14 + .1
    x_bounds[0].min[5, 2] = 3 * 3.14 - .1  # fin vrille et demi
    x_bounds[0].max[5, 2] = 3 * 3.14 + .1                            ##################

    # bras droit t=0
    # x_bounds[0].min[7, :] = -3.  # transfere dans bioMod
    # x_bounds[0].max[7, :] = .18
    x_bounds[0].min[7, 0] = -2.9  # debut bras aux oreilles
    x_bounds[0].max[7, 0] = -2.9
    # x_bounds[0].min[6, :] = -1.05  # transfere dans bioMod
    # x_bounds[0].max[6, :] = 1.5
    x_bounds[0].min[6, 0] = 0
    x_bounds[0].max[6, 0] = 0
    # bras gauche t=0
    # x_bounds[0].min[9, :] = -.18  # transfere dans bioMod
    # x_bounds[0].max[9, :] = 3.
    x_bounds[0].min[9, 0] = 2.9  # debut bras aux oreilles
    x_bounds[0].max[9, 0] = 2.9
    # x_bounds[0].min[8, :] = -1.5
    # x_bounds[0].max[8, :] = 1.05  # transfere dans bioMod
    x_bounds[0].min[8, 0] = 0
    x_bounds[0].max[8, 0] = 0

    # Contraintes de vitesse
    vzinit = 9.81 / 2 * final_time  # vitesse initiale en z du CoM pour revenir a terre au temps final
    vrotxinit = -2 * 3.14  # vitesse initiale en rot x du CoM. 2pi pour un salto

    # en xy bassin
    x_bounds[0].min[10:12, :] = -10
    x_bounds[0].max[10:12, :] = 10
    x_bounds[0].min[10:12, 0] = -.5
    x_bounds[0].max[10:12, 0] = .5
    # z bassin
    x_bounds[0].min[12, :] = -100
    x_bounds[0].max[12, :] = 100
    x_bounds[0].min[12, 0] = vzinit - .5
    x_bounds[0].max[12, 0] = vzinit + .5

    # autour de x
    x_bounds[0].min[13, :] = -100
    x_bounds[0].max[13, :] = 100
    x_bounds[0].min[13, 0] = vrotxinit - .5
    x_bounds[0].max[13, 0] = vrotxinit + .5
    # autour de y
    x_bounds[0].min[14, :] = -100
    x_bounds[0].max[14, :] = 100
    x_bounds[0].min[14, 0] = 0
    x_bounds[0].max[14, 0] = 0
    # autour de z
    x_bounds[0].min[15, :] = -100
    x_bounds[0].max[15, :] = 100
    x_bounds[0].min[15, 0] = 0
    x_bounds[0].max[15, 0] = 0

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)

    # constraints = ConstraintList()
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.9, max_bound=1.1, phase=0)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        # constraints,
        ode_solver=ode_solver,
        #variable_mappings=dof_mappings
    )


def main():
    """
    Prepares and solves an ocp that has quaternion in it. Animates the results
    """

    ocp = prepare_ocp("Models/JeCh_TechOpt83.bioMod", n_shooting=100, final_time=1.)
    ocp.add_plot_penalty(CostType.ALL)
    ocp.print(to_graph=True)
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(5000)
    sol = ocp.solve(solver)

    temps = time.strftime("%Y-%m-%d-%H%M%S")
    nom = 'sol'
    qs = sol.states['q']
    qdots = sol.states['qdot']
    if True:  # switch manuelle
        np.save(f'Solutions/{nom}-{temps}-q.npy', qs)
        np.save(f'Solutions/{nom}-{temps}-qdot.npy', qdots)

    # Print the last solution
    sol.animate(n_frames=-1)
    #sol.graphs()



if __name__ == "__main__":
    main()
