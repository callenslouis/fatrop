from trajectory_solver import TrajectorySolver
from trajectory_visualizer import TrajectoryVisualizer
from pendulum_3d_model import Pendulum3DModel
import numpy as np
import yaml

### Read config file
def read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
config = read_config('config.yaml')

### Set up the model
n = config['n']
model = Pendulum3DModel(nb_pendulums=n, m=[config['m']]*n, L=[config['L']]*n, g=config['g'])
if config['stabilizer']['use']:
    model.add_stabilizer(gamma_1=config['stabilizer']['gamma_1'], 
                         gamma_2=config['stabilizer']['gamma_2'])
if config['stiff_joints']['use']:
    model.set_stiff_joints(joint_stiffness=config['stiff_joints']['joint_stiffness'], 
                           joint_damping=config['stiff_joints']['joint_damping'])
model.set_model()
model.print_model_dimensions()
q0 = model.get_init_vector(randomize=False)

### Set up the solver
solver = TrajectorySolver(model)
if config['scenario']['tracking']:
    solver.set_path_tracking_scenario()

T = config['scenario']['T']
N = config['scenario']['N']
use_baked_expl = False
use_baked_impl = False

### Solve explicit
solver.implicit = False
nb_iter_expl = None
if use_baked_expl:
    exit_expl, q_sol_expl, v_sol_expl, F_sol_expl, z_sol_expl = \
        solver.solve_baked_trajectory(q0=q0, v0=np.zeros(3*n), T=T, N=N)
else:
    exit_expl, q_sol_expl, v_sol_expl, F_sol_expl, z_sol_expl = \
        solver.solve_trajectory(q0=q0, v0=np.zeros(3*n), T=T, N=N)
nb_iter_expl = solver.get_nb_iters()

### Solve implicit
nb_iter_impl = None
solver.implicit = True
if use_baked_impl:
    exit_impl, q_sol, v_sol, F_sol, z_sol = \
        solver.solve_baked_trajectory(q0=q0, v0=np.zeros(3*n), T=T, N=N)
else:
    exit_impl, q_sol, v_sol, F_sol, z_sol = solver.solve_trajectory(q0=q0, v0=np.zeros(3*n), T=T, N=N)
nb_iter_impl = solver.get_nb_iters()


### Print info
print(f"Explicit solver iterations: {nb_iter_expl if nb_iter_expl is not None else 'N/A'}")
print(f"Implicit solver iterations: {nb_iter_impl if nb_iter_impl is not None else 'N/A'}")


### Visualize
if config['visualize']:
    visualizer = TrajectoryVisualizer(model, solver)
    if exit_impl:
        visualizer.add_data(T=T, q_sol=q_sol, v_sol=v_sol, F_sol=F_sol, z_sol=z_sol)
        visualizer.visualize_all(appendix="impl")
    
    if exit_expl:
        visualizer.add_data(T=T, q_sol=q_sol_expl, v_sol=v_sol_expl, F_sol=F_sol_expl, z_sol=z_sol_expl)
        visualizer.visualize_all(appendix="expl")
