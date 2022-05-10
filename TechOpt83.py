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
)


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

    x_bounds[0].min[:3, :] = 0  # immobile en translation
    x_bounds[0].max[:3, :] = 0

    # le salto autour de x
    x_bounds[0].min[3, 0] = 0
    x_bounds[0].max[3, 0] = 0
    x_bounds[0].min[3, 1] = -2 * 3.14 - .1
    x_bounds[0].max[3, 1] = 0
    x_bounds[0].min[3, 2] = -2 * 3.14 - .1  # fin un tour vers l'avant
    x_bounds[0].max[3, 2] = -2 * 3.14 + .1
    # limitation du tilt autour de y
    x_bounds[0].min[4, :] = - 3.14 / 4
    x_bounds[0].max[4, :] = 3.14 / 4
    x_bounds[0].min[4, 0] = 0
    x_bounds[0].max[4, 0] = 0
    # la vrille autour de z
    x_bounds[0].min[5, 0] = 0
    x_bounds[0].max[5, 0] = 0
    x_bounds[0].min[5, 1] = 0
    x_bounds[0].max[5, 1] = 3 * 3.14 + .1
    x_bounds[0].min[5, 2] = 3 * 3.14 - .1  # fin vrille et demi
    x_bounds[0].max[5, 2] = 3 * 3.14 + .1
    # bras droit t=0
    x_bounds[0].min[7, :] = -3.
    x_bounds[0].max[7, :] = .18
    x_bounds[0].min[7, 0] = -2.9  # debut bras aux oreilles
    x_bounds[0].max[7, 0] = -2.9
    x_bounds[0].min[6, :] = -1.05
    x_bounds[0].max[6, :] = 1.5
    x_bounds[0].min[6, 0] = 0
    x_bounds[0].max[6, 0] = 0
    # bras gauche t=0
    x_bounds[0].min[9, :] = -.18
    x_bounds[0].max[9, :] = 3.
    x_bounds[0].min[9, 0] = 2.9  # debut bras aux oreilles
    x_bounds[0].max[9, 0] = 2.9
    x_bounds[0].min[8, :] = -1.5
    x_bounds[0].max[8, :] = 1.05
    x_bounds[0].min[8, 0] = 0
    x_bounds[0].max[8, 0] = 0

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)

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
        ode_solver=ode_solver,
        #variable_mappings=dof_mappings
    )


def main():
    """
    Prepares and solves an ocp that has quaternion in it. Animates the results
    """

    ocp = prepare_ocp("JeCh_10_OnDynamicsForSomer.bioMod", n_shooting=5, final_time=0.25)
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # Print the last solution
    sol.animate(n_frames=-1)
    #sol.graphs()
    # sol.states['q']

if __name__ == "__main__":
    main()
