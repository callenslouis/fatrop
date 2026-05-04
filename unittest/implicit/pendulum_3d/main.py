from trajectory_solver import TrajectorySolver
from trajectory_visualizer import TrajectoryVisualizer
from pendulum_3d_model import Pendulum3DModel
import numpy as np

n = 3
model = Pendulum3DModel(nb_pendulums=n, m=[1]*n, L=[1]*n, g=9.81)
# model.add_stabilizer(gamma_1=500, gamma_2=50000)
# model.set_stiff_joints(joint_stiffness=5, joint_damping=5)
model.add_stabilizer(gamma_1=20, gamma_2=2000)
model.set_stiff_joints(joint_stiffness=1, joint_damping=4)
model.set_model()

model.print_model_dimensions()
solver = TrajectorySolver(model)
solver.set_path_tracking_scenario()

T = 4.0
N = 200
use_backed = True

q0 = model.get_init_vector(randomize=False)
solver.implicit = False
nb_iter_expl = None
try:
    if use_backed:
        q_sol_expl, v_sol_expl, F_sol_expl, z_sol_expl = \
            solver.solve_backed_trajectory(q0=q0, v0=np.zeros(3*n), T=T, N=N)
    else:
        q_sol_expl, v_sol_expl, F_sol_expl, z_sol_expl = \
            solver.solve_trajectory(q0=q0, v0=np.zeros(3*n), T=T, N=N)
    nb_iter_expl = solver.get_nb_iters()
except RuntimeError as e:
    print("Explicit solver failed with error:", e)

solver.implicit = True
if use_backed:
    q_sol, v_sol, F_sol, z_sol = \
        solver.solve_backed_trajectory(q0=q0, v0=np.zeros(3*n), T=T, N=N)
else:
    q_sol, v_sol, F_sol, z_sol = solver.solve_trajectory(q0=q0, v0=np.zeros(3*n), T=T, N=N)
nb_iter_impl = solver.get_nb_iters()

print(f"Explicit solver iterations: {nb_iter_expl if nb_iter_expl is not None else 'N/A'}")
print(f"Implicit solver iterations: {nb_iter_impl}")

visualizer = TrajectoryVisualizer(model, solver)
visualizer.add_data(T=T, q_sol=q_sol, v_sol=v_sol, F_sol=F_sol, z_sol=z_sol)
visualizer.visualize_all(appendix="impl")
visualizer.add_data(T=T, q_sol=q_sol_expl, v_sol=v_sol_expl, F_sol=F_sol_expl, z_sol=z_sol_expl)
visualizer.visualize_all(appendix="expl")