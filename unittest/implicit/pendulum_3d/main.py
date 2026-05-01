from trajectory_solver import TrajectorySolver
from trajectory_visualizer import TrajectoryVisualizer
from pendulum_3d_model import Pendulum3DModel
import numpy as np

n = 3
model = Pendulum3DModel(nb_pendulums=n, m=[1]*n, L=[1]*n, g=9.81)
model.add_stabilizer(gamma_1=5000, gamma_2=5000)
model.print_model_dimensions()
solver = TrajectorySolver(model)

T = 2.0
q0 = model.get_init_vector(randomize=True)
solver.implicit = False
try:
    q_sol, v_sol, F_sol, z_sol = solver.solve_trajectory(q0=q0, v0=np.zeros(3*n), T=T, N=100)
except RuntimeError as e:
    print("Explicit solver failed with error:", e)
solver.implicit = True
q_sol, v_sol, F_sol, z_sol = solver.solve_trajectory(q0=q0, v0=np.zeros(3*n), T=T, N=100)

visualizer = TrajectoryVisualizer(model)
visualizer.add_data(T=T, q_sol=q_sol, v_sol=v_sol, F_sol=F_sol, z_sol=z_sol)
visualizer.visualize_all()
# visualizer.show()
# input("Press Enter to close visualizations...")
# visualizer.close()