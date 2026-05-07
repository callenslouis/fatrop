from trajectory_solver import TrajectorySolver, SolveExplicit, SolveImplicit, SolveReformulated, SolveAccelerated 
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
v0 = 0*q0

### Set up the solver
solver = TrajectorySolver(model)
if config['scenario']['tracking']:
    solver.set_path_tracking_scenario()

T = config['scenario']['T']
N = config['scenario']['N']

### Solve methods
result_expl = SolveExplicit(solver, q0, v0, T, N)
result_reform = SolveReformulated(solver, q0, v0, T, N)
result_accel = SolveAccelerated(solver, q0, v0, T, N)


### Visualize
if config['visualize']:
    visualizer = TrajectoryVisualizer(model, solver)
    visualizer.visualize_result(T, result_expl, appendix="explicit")
    visualizer.visualize_result(T, result_reform, appendix="reformulated")
    visualizer.visualize_result(T, result_accel, appendix="accelerated")
    # visualizer.visualize_result(T, result_impl, appendix="implicit")
