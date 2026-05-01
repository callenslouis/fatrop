from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class TrajectoryVisualizer():
    def __init__(self, model):
        self.model = model

    def add_data(self, T, q_sol, v_sol, F_sol, z_sol):
        self.T = T
        self.N = q_sol.shape[1] - 1
        self.t_sol = np.linspace(0, T, self.N+1)
        self.q_sol = q_sol
        self.v_sol = v_sol
        self.F_sol = F_sol
        self.z_sol = z_sol

    def visualize_all(self):
        self.visualize_trajectory()
        self.visualize_forces()
        self.visualize_pendulum_length()
        self.animate_trajectory()

    def get_point(self, q, i, slice_min=0, slice_max=None):
        if slice_max is None:
            slice_max = q.shape[1]

        if i < 0:
            return np.zeros(slice_max - slice_min), np.zeros(slice_max - slice_min), np.zeros(slice_max - slice_min)

        return q[3*i, slice_min:slice_max], q[3*i+1, slice_min:slice_max], q[3*i+2, slice_min:slice_max]

    def visualize_trajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # plot anchor point
        ax.scatter(0, 0, 0, color='k', label='Anchor Point')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-self.model.nb_pendulums-0.5, 0.5)

        # plot trajectory
        for i in range(self.model.nb_pendulums):
            plt.plot(*self.get_point(self.q_sol, i), marker='o', label=f'Pendulum {i}')

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        plt.legend()

        plt.savefig("visualization_output/pendulum_trajectory.png", dpi=300)

    def show_pendulum_projection(self, ax, frame, plane_x, plane_y, plane_z):
        all_points = np.zeros((self.model.nb_pendulums + 1, 3))
        for i in range(-1, self.model.nb_pendulums):
            all_points[i+1, :] = np.array(self.get_point(self.q_sol, i, slice_min=frame, slice_max=frame+1)).T

        ax.plot(plane_x, all_points[:, 1], all_points[:,2], color='grey', marker='.', alpha=1.0)
        ax.plot(all_points[:, 0], plane_y, all_points[:,2], color='grey', marker='.', alpha=1.0)
        ax.plot(all_points[:, 0], all_points[:,1], plane_z, color='grey', marker='.', alpha=1.0)
        
    
    def animate_trajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # animation function
        def update(frame):
            ax.clear()
            ax.scatter(0, 0, 0, color='k', label='Anchor Point')
            for i in range(self.model.nb_pendulums):
                # show points
                point = self.get_point(self.q_sol, i, slice_min=frame, slice_max=frame+1)
                plt.plot(point[0], point[1], point[2], marker='.', color='r', label=f'Pendulum {i}')

                # show traced path
                # points = self.get_point(self.q_sol, i, slice_min=0, slice_max=frame)
                # plt.plot(self.q_sol[3*i, :frame], self.q_sol[3*i+1, :frame], self.q_sol[3*i+2, :frame], color='k', alpha=0.5)
                plt.plot(*self.get_point(self.q_sol, i, slice_min=0, slice_max=frame), color='k', alpha=0.5)

                # show pendulum itself
                # prev_point = [0, 0, 0] if i == 0 else [self.q_sol[3*(i-1), frame], self.q_sol[3*(i-1)+1, frame], self.q_sol[3*(i-1)+2, frame]]
                prev_point = self.get_point(self.q_sol, i-1, slice_min=frame, slice_max=frame+1)
                ax.plot([prev_point[0], point[0]], [prev_point[1], point[1]], [prev_point[2], point[2]], color='r')

            # show forces
            if frame < self.N:
                max_force = np.max(np.linalg.norm(self.F_sol, axis=0))
                force_scale = 0.5 / max_force if max_force > 0 else 1.0
                # point = [self.q_sol[-3, frame], self.q_sol[-2, frame], self.q_sol[-1, frame]]
                point = self.get_point(self.q_sol, self.model.nb_pendulums-1, slice_min=frame, slice_max=frame+1)
                force_vector = self.F_sol[:, frame] * force_scale
                ax.quiver(point[0], point[1], point[2], force_vector[0], force_vector[1], force_vector[2], color='b', length=0.5, normalize=True)

            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_zlim(-self.model.nb_pendulums-0.5, 0.5)

            # show pendulum projection on XZ, YZ and XY planes
            self.show_pendulum_projection(ax, frame, plane_x=-1.5, plane_y=1.5, plane_z=-self.model.nb_pendulums-0.5)
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_zlabel('Z Position (m)')
            # ax.legend(loc="upper right")

        interval_ms = int(1000 / 25)
        ani = animation.FuncAnimation(fig, update, frames=self.N+1, interval=interval_ms, blit=False)
        ani.save("visualization_output/pendulum_trajectory_animation.mp4", writer='ffmpeg', fps=25, dpi=150)

    def visualize_forces(self):
        fig = plt.figure()
        plt.plot(self.t_sol[:-1], self.F_sol.T)
        plt.xlabel("Time (s)")
        plt.ylabel("Control Forces (N)")
        plt.title("Control Forces over Time")
        plt.legend(['Fx', 'Fy', 'Fz'])
        plt.savefig("visualization_output/control_forces.png", dpi=300)

    def visualize_pendulum_length(self):
        lengths = np.zeros((self.model.nb_pendulums, self.N+1))
        for i in range(self.model.nb_pendulums):
            point = self.get_point(self.q_sol, i)
            prev_point = self.get_point(self.q_sol, i-1)
            lengths[i, :] = np.linalg.norm(np.array(point) - np.array(prev_point), axis=0)

        fig = plt.figure()
        plt.plot(self.t_sol, lengths.T)
        plt.xlabel("Time (s)")
        plt.ylabel("Pendulum Length (m)")
        plt.title("Pendulum Length over Time")
        plt.savefig("visualization_output/pendulum_length.png", dpi=300)

    def visualize_z_vars(self):
        fig = plt.figure()
        plt.plot(self.t_sol[:-1], self.z_sol.T)
        plt.xlabel("Time (s)")
        plt.ylabel("Slack Variables")
        plt.title("z Variables over Time")
        plt.savefig("visualization_output/z_variables.png", dpi=300)

    def show(self):
        plt.show()

    def close(self):
        plt.close('all')
