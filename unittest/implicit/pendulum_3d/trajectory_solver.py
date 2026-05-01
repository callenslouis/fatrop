from casadi import *
import numpy as np

class TrajectorySolver():
    def __init__(self, model, implicit=False):
        # load models
        self.model = model
        self.implicit = implicit

        # ocp scenario
        self.tracking = False

    def set_path_tracking_scenario(self):
        self.tracking = True
        # TODO

    def solve_trajectory(self, q0, v0, T, N):
        assert len(q0) == 3*self.model.nb_pendulums, "Length of q0 must match 3 times the number of pendulums"
        assert len(v0) == 3*self.model.nb_pendulums, "Length of v0 must match 3 times the number of pendulums"

        opti = Opti()
        q = opti.variable(3*self.model.nb_pendulums, N+1)
        v = opti.variable(3*self.model.nb_pendulums, N+1)
        z = opti.variable(1*self.model.nb_pendulums, N)
        F = opti.variable(3, N)

        dt = T/N

        # initial conditions
        # opti.subject_to(self.model.g_eq0(q[:, 0], v[:, 0]) == 0) --> enforced using initial conditions
        opti.subject_to(q[:, 0] == q0)
        opti.subject_to(v[:, 0] == v0)

        obj = 0.0
        for k in range(N):
            # dynamics constraints
            qdot, vdot = self.model.f(q[:,k], v[:,k], z[:,k], F[:,k])
            opti.subject_to(v[:,k+1] == v[:,k] + vdot*dt)
            idx = k+1 if self.implicit else k
            opti.subject_to(q[:,k+1] == q[:,k] + v[:,idx]*dt)
            eq = self.model.g_eq(q[:,k], v[:,k], z[:,k], F[:,k])
            if self.model.stabilizer is not None:
                eq += self.model.stabilizer(q[:,k], v[:,k])
            opti.subject_to(eq == 0)

            # inequalities
            opti.subject_to(-30 <= (F[:,k] <= 30))

            # objective
            # obj += 1e-2*sumsqr(v[:,k])
            obj += 1e-2*sumsqr(F[:,k])

        q_rest = np.zeros(3*self.model.nb_pendulums)
        q_rest[2] = -self.model.L[0]
        for i in range(1, self.model.nb_pendulums):
            q_rest[3*i + 2] = q_rest[3*(i-1) + 2] - self.model.L[i]
        obj += sumsqr(v[:,N]) + 1e1*sumsqr(q[:,N] - q_rest)

        # objective
        opti.minimize(obj)

        opti.set_initial(q, np.tile(q0, (N+1, 1)).T)
        opti.set_initial(v, np.tile(v0, (N+1, 1)).T)

        opti.solver('ipopt', {}, {'max_iter':400, 'tol':1e-4})
        sol = opti.solve()

        q_sol = sol.value(q)
        v_sol = sol.value(v)
        F_sol = sol.value(F)
        z_sol = sol.value(z)
        return q_sol, v_sol, F_sol, z_sol