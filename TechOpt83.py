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

    biorbd_model = ( biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path) )

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, marker_index=1, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=100, phase=0)  # pk Node.ALL_SHOOTING?
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=100, phase=1)  # phases

    objective_functions.add(  # oui? non? je sais pas  phases
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.TRANSITION,
        weight=100,
        phase=1,
        quadratic=True,
    )

    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=.7, weight=.01, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=.0, max_bound=.7, weight=.01, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN) # phases

    # Define control path constraint
    dof_mappings = BiMappingList()
    dof_mappings.add("tau", to_second=[None, None, None, None, None, None, 0, 1, 2, 3, 4, 5], to_first=[6, 7, 8, 9, 10, 11], phase=0)
    dof_mappings.add("tau", to_second=[None, None, None, None, None, None, 0, 1, 2, 3, 4, 5], to_first=[6, 7, 8, 9, 10, 11], phase=1)

    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    n_tau = nb_q - 6

    tau_min, tau_max, tau_init = -500, 500, 0
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)

    # Initial guesses
    x = np.vstack((np.random.random((nb_q, 2)), np.random.random((nb_qdot, 2))))
    x_init = InitialGuessList()
    x_init.add(x, interpolation=InterpolationType.LINEAR)
    x_init.add(x, interpolation=InterpolationType.LINEAR)

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)
    u_init.add([tau_init] * n_tau)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))


    # Contraintes de position: PHASE 0 la montee vers le carpe

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
    x_bounds[0].min[3, 1:] = -2 * 3.14 - .1
    x_bounds[0].max[3, 1:] = 0
    # limitation du tilt autour de y
    x_bounds[0].min[4, 0] = 0
    x_bounds[0].max[4, 0] = 0
    x_bounds[0].min[4, 1:] = - 3.14 / 4
    x_bounds[0].max[4, 1:] = 3.14 / 4
    # la vrille autour de z
    x_bounds[0].min[5, 0] = 0
    x_bounds[0].max[5, 0] = 0
    x_bounds[0].min[5, 1:] = 0
    x_bounds[0].max[5, 1:] = 3 * 3.14 + .1

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
    x_bounds[0].min[10, 2] = 2.8
    x_bounds[0].max[10, 2] = 3.
    # le dehanchement
    x_bounds[0].min[11, 0] = 0
    x_bounds[0].max[11, 0] = 0

    # Contraintes de position: PHASE 1 l'ouverture et la vrille et demie

    # deplacement
    x_bounds[1].min[:3, :] = -.1
    x_bounds[1].max[:3, :] = .1
    x_bounds[1].min[2, :] = 0
    x_bounds[1].max[2, :] = 20  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # le salto autour de x
    x_bounds[1].min[3, :] = -2 * 3.14 - .1
    x_bounds[1].max[3, :] = 0
    x_bounds[1].min[3, 2] = -2 * 3.14 - .1  # fin un tour vers l'avant
    x_bounds[1].max[3, 2] = -2 * 3.14 + .1
    # limitation du tilt autour de y
    x_bounds[1].min[4, :] = - 3.14 / 4
    x_bounds[1].max[4, :] = 3.14 / 4
    x_bounds[1].min[4, 2] = - 3.14 / 8
    x_bounds[1].max[4, 2] = 3.14 / 8
    # la vrille autour de z
    x_bounds[1].min[5, :] = 0
    x_bounds[1].max[5, :] = 3 * 3.14 + .1
    x_bounds[1].min[5, 2] = 3 * 3.14 - .1  # fin vrille et demi
    x_bounds[1].max[5, 2] = 3 * 3.14 + .1                            ##################

    # bras droit t=0
    x_bounds[1].min[7, 2] = -2.9 - .1  # fini bras aux oreilles
    x_bounds[1].max[7, 2] = -2.9 + .1
    x_bounds[1].min[6, 2] = -.1
    x_bounds[1].max[6, 2] = .1
    # bras gauche t=0
    x_bounds[1].min[9, 2] = 2.9 - .1  # fini bras aux oreilles
    x_bounds[1].max[9, 2] = 2.9 + .1
    x_bounds[1].min[8, 2] = -.1
    x_bounds[1].max[8, 2] = .1

    # le carpe
    x_bounds[1].min[10, 0] = 2.8
    x_bounds[1].max[10, 0] = 3.
    x_bounds[1].min[10, 2] = -.1  # fini hanches ouvertes
    x_bounds[1].max[10, 2] = .1
    # le dehanchement
    x_bounds[1].min[11, 2] = -.1
    x_bounds[1].max[11, 2] = .1

    # Contraintes de vitesse: PHASE 0 la montee vers le carpe

    vzinit = 9.81 / 2 #* final_time  # vitesse initiale en z du CoM pour revenir a terre au temps final
    vrotxinit = -2 * 3.14  # vitesse initiale en rot x du CoM. 2pi pour un salto

    # decalage entre le bassin et le CoM
    CoM_Q_sym = MX.sym('CoM', nb_q)
    CoM_Q_init = np.zeros(nb_q)
    CoM_Q_func = Function('CoM_Q_func', [CoM_Q_sym], [biorbd_model[0].CoM(CoM_Q_sym).to_mx()])
    bassin_Q_func = Function('bassin_Q_func', [CoM_Q_sym], [biorbd_model[0].globalJCS(0).to_mx()])  # retourne la RT du bassin

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
    x_bounds[0].min[15, 0] = vrotxinit - 3.
    x_bounds[0].max[15, 0] = vrotxinit + 3.
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
    # Qtransdot = Qtransdot + v cross Qrotdot
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

    # Contraintes de vitesse: PHASE 1 l'ouverture et la vrille et demie

    # en xy bassin
    x_bounds[1].min[12:14, :] = -10
    x_bounds[1].max[12:14, :] = 10
    # z bassin
    x_bounds[1].min[14, :] = -100
    x_bounds[1].max[14, :] = 100

    # autour de x
    x_bounds[1].min[15, :] = -100
    x_bounds[1].max[15, :] = 100
    # x_bounds[1].min[15, 2] = vrotxinit - 3.  # peut-être trop contraignant
    # x_bounds[1].max[15, 2] = vrotxinit + 3.
    # autour de y
    x_bounds[1].min[16, :] = -100
    x_bounds[1].max[16, :] = 100
    # autour de z
    x_bounds[1].min[17, :] = -100
    x_bounds[1].max[17, :] = 100
    # x_bounds[1].min[17, 2] = -10  # peut-être trop contraignant
    # x_bounds[1].max[17, 2] = 10

    # des bras
    x_bounds[1].min[18:22, :] = -100
    x_bounds[1].max[18:22, :] = 100
    x_bounds[1].min[18:22, 2] = -10  # peut-être trop contraignant
    x_bounds[1].max[18:22, 2] = 10

    # du carpe
    x_bounds[1].min[22, :] = -100
    x_bounds[1].max[22, :] = 100
    # x_bounds[1].min[22, 0] = -10  # peut-être trop contraignant
    # x_bounds[1].max[22, 0] = 10
    # du dehanchement
    x_bounds[1].min[23, :] = -100
    x_bounds[1].max[23, :] = 100
    # x_bounds[1].min[23, 0] = -10  # peut-être trop contraignant
    # x_bounds[1].max[23, 0] = 10


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

    ocp = prepare_ocp("Models/JeCh_TechOpt83.bioMod", n_shooting=(50, 50), final_time=(.5, .5))
    ocp.add_plot_penalty(CostType.ALL)
    ocp.print(to_graph=True)
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(5000)
    solver.set_convergence_tolerance(1e-4)
    sol = ocp.solve(solver)

    temps = time.strftime("%Y-%m-%d-%H%M%S")
    nom = 'sol'
    qs = sol.states[0]['q']
    qdots = sol.states[0]['qdot']
    for i in range(1, len(sol.states)):
        qs = np.hstack((qs, sol.states[i]['q']))
        qdots = np.hstack((qdots, sol.states[i]['qdot']))
    if True:  # switch manuelle
        np.save(f'Solutions/{nom}-{temps}-q.npy', qs)
        np.save(f'Solutions/{nom}-{temps}-qdot.npy', qdots)

    # Print the last solution
    sol.animate(n_frames=-1)
    #sol.graphs()



if __name__ == "__main__":
    main()
