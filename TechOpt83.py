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
    PenaltyNode,
)
import time

def minimize_difference(all_pn: PenaltyNode):
    return all_pn[0].nlp.controls.cx_end - all_pn[1].nlp.controls.cx

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
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=100, phase=0)  # pk Node.ALL_SHOOTING?
    #objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=100, phase=1)  # phases

    # objective_functions.add(  # oui? non? je sais pas  phases
    #     minimize_difference,
    #     custom_type=ObjectiveFcn.Mayer,
    #     node=Node.TRANSITION,
    #     weight=100,
    #     phase=1,
    #     quadratic=True,
    # )

    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.9, max_bound=1.1, weight=0.01)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    #dynamics.add(DynamicsFcn.TORQUE_DRIVEN) # phases

    # Constraints
    # constraints = ConstraintList()  # phases
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=0)
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=0)
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1", phase=1)

    # Define control path constraint
    dof_mappings = BiMappingList()
    dof_mappings.add("tau", to_second=[None, None, None, None, None, None, 0, 1, 2, 3, 4, 5], to_first=[6, 7, 8, 9, 10, 11])

    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    n_tau = nb_q - 6

    tau_min, tau_max, tau_init = -500, 500, 0
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)

    # Initial guesses
    x = np.vstack((np.random.random((nb_q, 2)), np.random.random((nb_qdot, 2))))  # pourquoi 2 colonnes. rep: debut fin
    x_init = InitialGuessList()
    x_init.add(x, interpolation=InterpolationType.LINEAR)

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Contraintes de position

    # deplacement
    x_bounds[0].min[:3, :] = -.1
    x_bounds[0].max[:3, :] = .1
    x_bounds[0].min[:3, 0] = 0
    x_bounds[0].max[:3, 0] = 0
    x_bounds[0].min[2, 1] = 0
    x_bounds[0].max[2, 1] = 20  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

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
    x_bounds[0].max[5, 2] = 3 * 3.14 + .1                            ################## ca d'l'air qu'il est capable!

    # bras droit t=0
    x_bounds[0].min[7, 0] = -2.9  # debut bras aux oreilles
    x_bounds[0].max[7, 0] = -2.9
    x_bounds[0].min[6, 0] = 0
    x_bounds[0].max[6, 0] = 0
    # bras gauche t=0
    x_bounds[0].min[9, 0] = 2.9  # debut bras aux oreilles
    x_bounds[0].max[9, 0] = 2.9
    x_bounds[0].min[8, 0] = 0
    x_bounds[0].max[8, 0] = 0

    # le carpe
    x_bounds[0].min[10, 0] = 0
    x_bounds[0].max[10, 0] = 0
    # le dehanchement
    x_bounds[0].min[11, 0] = 0
    x_bounds[0].max[11, 0] = 0

    # Contraintes de vitesse
    vzinit = 9.81 / 2 * final_time  # vitesse initiale en z du CoM pour revenir a terre au temps final
    vrotxinit = -2 * 3.14  # vitesse initiale en rot x du CoM. 2pi pour un salto

    # decalage entre le bassin et le CoM
    CoM_Q_sym = MX.sym('CoM', nb_q)
    CoM_Q_init = np.zeros(nb_q)
    CoM_Q_func = Function('CoM_Q_func', [CoM_Q_sym], [biorbd_model.CoM(CoM_Q_sym).to_mx()])
    bassin_Q_func = Function('bassin_Q_func', [CoM_Q_sym], [biorbd_model.globalJCS(0).to_mx()])  # retourne la RT du bassin

    v = np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1, :3]  # selectionne seulement la translation

    # en xy bassin
    x_bounds[0].min[12:14, :] = -10
    x_bounds[0].max[12:14, :] = 10
    x_bounds[0].min[12:14, 0] = -.5
    x_bounds[0].max[12:14, 0] = .5
    # z bassin
    x_bounds[0].min[14, :] = -100
    x_bounds[0].max[14, :] = 100
    x_bounds[0].min[14, 0] = vzinit - .5
    x_bounds[0].max[14, 0] = vzinit + .5

    # autour de x
    x_bounds[0].min[15, :] = -100
    x_bounds[0].max[15, :] = 100
    x_bounds[0].min[15, 0] = vrotxinit - .5
    x_bounds[0].max[15, 0] = vrotxinit + .5
    # autour de y
    x_bounds[0].min[16, :] = -100
    x_bounds[0].max[16, :] = 100
    x_bounds[0].min[16, 0] = 0
    x_bounds[0].max[16, 0] = 0
    # autour de z
    x_bounds[0].min[17, :] = -100
    x_bounds[0].max[17, :] = 100
    x_bounds[0].min[17, 0] = 0
    x_bounds[0].max[17, 0] = 0

    # tenir compte du decalage entre bassin et CoM avec la rotation
    x_bounds[0].min[12:15, 0] = x_bounds[0].min[12:15, 0] + np.cross(v, x_bounds[0].min[15:18, 0])
    x_bounds[0].max[12:15, 0] = x_bounds[0].max[12:15, 0] + np.cross(v, x_bounds[0].max[15:18, 0])

    # des bras
    x_bounds[0].min[18:22, :] = -100
    x_bounds[0].max[18:22, :] = 100
    x_bounds[0].min[18:22, 0] = 0
    x_bounds[0].max[18:22, 0] = 0

    # du carpe
    x_bounds[0].min[22, :] = -100
    x_bounds[0].max[22, :] = 100
    x_bounds[0].min[22, 0] = 0
    x_bounds[0].max[22, 0] = 0
    # du dehanchement
    x_bounds[0].min[23, :] = -100
    x_bounds[0].max[23, :] = 100
    x_bounds[0].min[23, 0] = 0
    x_bounds[0].max[23, 0] = 0

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
        variable_mappings=dof_mappings,
        n_threads=2
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
    solver.set_convergence_tolerance(1e-4)
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
