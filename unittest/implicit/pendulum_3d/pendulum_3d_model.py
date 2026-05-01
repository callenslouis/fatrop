import casadi as ca
import numpy as np

class Pendulum3DModel():
    def __init__(self, nb_pendulums=1, m=[1], L=[1], g=9.81):
        # define model parameters
        self.nb_pendulums = nb_pendulums
        assert len(m) == nb_pendulums, "Length of mass list must match number of pendulums"
        assert len(L) == nb_pendulums, "Length of length list must match number of pendulums"
        self.m = m
        self.L = L
        self.g = g

        self.set_model()
        self.stabilizer = None

    def set_model(self):
        # define q and v
        q = ca.SX.sym("q", 3*self.nb_pendulums)
        v = ca.SX.sym("v", 3*self.nb_pendulums)
        z = ca.SX.sym("z", 1*self.nb_pendulums)
        F = ca.SX.sym("F", 3)
        M = ca.SX.eye(3*self.nb_pendulums)
        for i in range(self.nb_pendulums):
            M[3*i:3*i+3, 3*i:3*i+3] = self.m[i] * ca.SX.eye(3)

        # define dynamics
        T = 0.5 * v.T @ M @ v
        V = 0
        c = ca.sumsqr(q[:3]) - self.L[0]**2
        for i in range(self.nb_pendulums):
            V += self.m[i] * self.g * (q[3*i + 2])
            if i > 0:
                c = ca.vertcat(c, ca.sumsqr(q[3*i:3*i+3] - q[3*(i-1):3*(i-1)+3]) - self.L[i]**2)
        Jc = ca.jacobian(c, q)

        qdd = ca.inv(M) @ (ca.vertcat(ca.SX.zeros(3*(self.nb_pendulums - 1),1), F) + ca.jacobian(T - V, q).T - ca.transpose(Jc) @ z) # assuming Mdot is 0
        self.f = ca.Function("f", [q, v, z, F], [v, qdd], ["q", "v", "z", "F"], ["qdot", "vdot"])

        eq = Jc @ qdd + ca.jacobian(Jc @ v, q) @ v
        self.g_eq = ca.Function("g_eq", [q, v, z, F], [eq], ["q", "v", "z", "F"], ["c"])
        self.g_eq0 = ca.Function("g_eq0", [q, v], [ca.vertcat(c, Jc @ v)], ["q", "v"], ["c0"])

        self.f.save("casadi_functions/pendulum_3d_dynamics.casadi")
        self.g_eq.save("casadi_functions/pendulum_3d_g_eq.casadi")
        self.g_eq0.save("casadi_functions/pendulum_3d_g_eq0.casadi")

    def add_stabilizer(self, gamma_1, gamma_2):
        q = ca.SX.sym("q", 3*self.nb_pendulums)
        v = ca.SX.sym("v", 3*self.nb_pendulums)

        stabilizer = ca.SX.zeros(0,1)
        for i in range(self.nb_pendulums):
            qi = q[3*i:3*i+3]
            vi = v[3*i:3*i+3]
            qi_prev = q[3*(i-1):3*(i-1)+3] if i > 0 else ca.SX.zeros(3)

            stabilizer = ca.vertcat(stabilizer,
                                -gamma_1*qi.T @ vi - gamma_2*(ca.sumsqr(qi - qi_prev) - self.L[i]**2)**2)

        self.stabilizer = ca.Function("stabilizer", [q, v], [-stabilizer], ["q", "v"], ["stabilizer"])
        self.stabilizer.save("casadi_functions/pendulum_3d_stabilizer.casadi")

    def print_model_dimensions(self):
        print(f"Number of pendulums: {self.nb_pendulums}")
        nx = self.f.sparsity_in(0).size1() + \
             self.f.sparsity_in(1).size1()
        nu = self.f.sparsity_in(2).size1() + \
             self.f.sparsity_in(3).size1()
        ng = self.g_eq.sparsity_out(0).size1()
        print(f"nx: {nx}\nnu: {nu}\nng: {ng}")

    def get_init_vector(self, randomize=False):
        def get_rotation_matrix(theta_x, theta_y, theta_z):
            Rx = ca.vertcat(
                ca.horzcat(1, 0, 0),
                ca.horzcat(0, ca.cos(theta_x), -ca.sin(theta_x)),
                ca.horzcat(0, ca.sin(theta_x), ca.cos(theta_x))
            )
            Ry = ca.vertcat(
                ca.horzcat(ca.cos(theta_y), 0, ca.sin(theta_y)),
                ca.horzcat(0, 1, 0),
                ca.horzcat(-ca.sin(theta_y), 0, ca.cos(theta_y))
            )
            Rz = ca.vertcat(
                ca.horzcat(ca.cos(theta_z), -ca.sin(theta_z), 0),
                ca.horzcat(ca.sin(theta_z), ca.cos(theta_z), 0),
                ca.horzcat(0, 0, 1)
            )
            return Rz @ Ry @ Rx
        
        q0 = np.zeros(3*self.nb_pendulums)
        p = q0[:3]
        for i in range(self.nb_pendulums):
            if randomize:
                theta_x = np.random.uniform(-0.3, 0.3)
                theta_y = np.random.uniform(-0.3, 0.3)
                theta_z = np.random.uniform(-0.3, 0.3)
            else:
                theta_x = 0.4 - 0.1*i
                theta_y = 0.2 - 0.1*i
                theta_z = 0.1 - 0.1*i
            R = get_rotation_matrix(theta_x, theta_y, theta_z)

            p += R @ ca.vertcat(0, 0, -self.L[i])    
            q0[3*i:3*(i+1)] = p.full().flatten()

        return q0