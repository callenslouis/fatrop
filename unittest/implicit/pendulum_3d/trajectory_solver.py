from casadi import *
import numpy as np
import json
import yaml

class TrajectorySolver():
    def __init__(self, model, implicit=False):
        # load models
        self.model = model
        self.implicit = implicit

        # ocp params
        self.force_bounds = 50

        # ocp scenario
        self.tracking = False

    def set_path_tracking_scenario(self):
        self.tracking = True
        
        t = SX.sym("t")
        T = SX.sym("T")
        q0 = SX.sym("q0", 3*self.model.nb_pendulums)
        R_sq = 0.25*sumsqr(self.model.get_joint_pos(q0, 0)[:2])
        center = vertcat(0, 0, self.model.get_joint_pos(q0, 0)[2])
        phase_0 = atan2(self.model.get_joint_pos(q0, 0)[1], self.model.get_joint_pos(q0, 0)[0])

        nb_circles = 1.0
        ref_point = center + \
            vertcat(cos(2*np.pi*t/T*nb_circles*2 + phase_0), 
                       sin(2*np.pi*t/T*nb_circles + phase_0), 
                       0)*sqrt(R_sq)

        self.path_func = Function("path_func", [t, T, q0], [ref_point], ['t', 'T', 'q0'], ['ref_point'])

    def get_q_rest(self):
        q_rest = np.zeros(3*self.model.nb_pendulums)
        q_rest[2] = -self.model.L[0]
        for i in range(1, self.model.nb_pendulums):
            q_rest[3*i + 2] = q_rest[3*(i-1) + 2] - self.model.L[i]
        return q_rest
    
    def extract_solution(self, xx, uu):
        q_sol = xx[:3*self.model.nb_pendulums, :]
        v_sol = xx[3*self.model.nb_pendulums:6*self.model.nb_pendulums, :]
        p_sol = xx[6*self.model.nb_pendulums:, :]
        F_sol = uu[3*self.tracking + 1:, :]
        z_sol = uu[3*self.tracking:3*self.tracking + self.model.nb_pendulums, :]
        return q_sol, v_sol, F_sol, z_sol

    def bake(self, q0, v0, T, N):
        # xk
        q = SX.sym("q", 3*self.model.nb_pendulums)
        v = SX.sym("v", 3*self.model.nb_pendulums)
        p = SX.sym("p", 1)
        xk = vertcat(q, v, p)
        
        # xk_plus
        qp = SX.sym("qp", 3*self.model.nb_pendulums)
        vp = SX.sym("vp", 3*self.model.nb_pendulums)
        pp = SX.sym("pp", 1)
        xk_plus = vertcat(qp, vp, pp)

        # uk        
        s = SX.sym("s", 3*self.tracking)
        z = SX.sym("z", 1*self.model.nb_pendulums)
        F = SX.sym("F", 3*len(self.model.actuated_joint_idxs))
        uk = vertcat(s, z, F)

        dt = T/N

        ### define functions
        self.funcs = {}

        # objective
        objk = 1e-2*sumsqr(F) + (1e2*sumsqr(s) if self.tracking else 0) + 1e-2*sumsqr(xk) + 1e-2*sumsqr(uk)
        self.funcs['eval_objk'] = Function("eval_objk", [uk, xk], [objk], ['uk', 'xk'], ['objk'])

        objK = 0.0 if self.tracking else sumsqr(v) + 1e1*sumsqr(q - self.get_q_rest()) + 1e-2*sumsqr(xk)
        self.funcs['eval_objK'] = Function("eval_objK", [xk], [objK], ['xk'], ['objK'])

        # equality constraints
        self.funcs['eval_g0'] = Function("eval_g0", [uk, xk], [vertcat(q - q0, v - v0, p)], ['uk', 'xk'], ['g0'])

        gk = self.model.g_eq(q, v, z, F) + (self.model.stabilizer(q, v) if self.model.stabilizer is not None else 0)
        if self.tracking:
            ref_point = self.path_func(p, T, q0)
            gk = vertcat(gk, s - (q[:3] - ref_point))
        self.funcs['eval_gk'] = Function("eval_gk", [uk, xk], [gk], ['uk', 'xk'], ['gk'])
        self.funcs['eval_gK'] = Function("eval_gK", [xk], [vertcat(p - T)], ['xk'], ['gK'])
        # self.funcs['eval_gk'] = Function("eval_gk", [uk, xk], [vertcat()], ['uk', 'xk'], ['gk'])
        # self.funcs['eval_gK'] = Function("eval_gK", [xk], [vertcat()], ['xk'], ['gK'])

        # inequality constraints
        self.funcs['eval_gk_ineq'] = Function("eval_gk_ineq", [uk, xk], [F], ['uk', 'xk'], ['ineqk'])
        self.funcs['eval_gK_ineq'] = Function("eval_gK_ineq", [xk], [vertcat()], ['xk'], ['ineqK'])
        self.funcs["lb"] = Function("lb", [], [-self.force_bounds*SX.ones(F.size1(), 1)], [], ['lb'])
        self.funcs["ub"] = Function("ub", [], [self.force_bounds*SX.ones(F.size1(), 1)], [], ['ub'])
        self.funcs["lbK"] = Function("lbK", [], [vertcat()], [], ['lbK'])
        self.funcs["ubK"] = Function("ubK", [], [vertcat()], [], ['ubK'])

        # dynamics
        qdot, vdot = self.model.f(q, v, z, F)
        self.funcs['expl_dyn'] = Function("expl_dyn", [uk, xk], [vertcat(q + dt*v, v + dt*vdot, p + dt)], ['uk', 'xk'], ['xk_plus'])
        impl_dyn = vertcat(q + vp*dt - qp, v + vdot*dt - vp, p + dt - pp)
        self.funcs['impl_dyn'] = Function("impl_dyn", [uk, xk, xk_plus], [impl_dyn], ['uk', 'xk', 'xk_plus'], ['dyn_res'])

        # initialization
        k = SX.sym("k")
        self.funcs["x_init"] = Function("x_init", [k], [vertcat(q0, v0, k*dt)], [], ['x_init'])
        self.funcs["u_init"] = Function("u_init", [], [0*uk], [], ['u_init'])

        # metadata
        data = {
            "nx": xk.size1(),
            "nu": uk.size1(),
            "N": N,
            "T": T,
            "dt": dt,
            "x0": vertcat(q0, v0, 0).full().flatten().tolist(),
            "lb": self.funcs["lb"]()["lb"].full().flatten().tolist(),
            "ub": self.funcs["ub"]()["ub"].full().flatten().tolist(),
            "lbK": self.funcs["lbK"]()["lbK"].full().flatten().tolist(),
            "ubK": self.funcs["ubK"]()["ubK"].full().flatten().tolist(),
            "u_init": [self.funcs["u_init"]()["u_init"].full().flatten().tolist() for _ in range(N)],
            "x_init": [self.funcs["x_init"](k).full().flatten().tolist() for k in range(N+1)]
        }

        folder = "casadi_functions/"
        build_folder = "../../../build_docker/casadi_functions/"
        with open(f"{folder}metadata.json", "w") as f:
            json.dump(data, f, indent=4)
        with open(f"{build_folder}metadata.json", "w") as f:
            json.dump(data, f, indent=4)

        for name, func in self.funcs.items():
            func.save(f"{folder}{name}.casadi")
            func.save(f"{build_folder}{name}.casadi")

    def solve_baked_trajectory(self, q0, v0, T, N):
        self.bake(q0, v0, T, N)

        eval_objk = Function.load("casadi_functions/eval_objk.casadi")
        eval_objK = Function.load("casadi_functions/eval_objK.casadi")
        eval_g0 = Function.load("casadi_functions/eval_g0.casadi")
        eval_gk = Function.load("casadi_functions/eval_gk.casadi")
        eval_gK = Function.load("casadi_functions/eval_gK.casadi")
        eval_gk_ineq = Function.load("casadi_functions/eval_gk_ineq.casadi")
        eval_gK_ineq = Function.load("casadi_functions/eval_gK_ineq.casadi")
        lb = Function.load("casadi_functions/lb.casadi")
        ub = Function.load("casadi_functions/ub.casadi")
        lbK = Function.load("casadi_functions/lbK.casadi")
        ubK = Function.load("casadi_functions/ubK.casadi")

        expl_dyn = Function.load("casadi_functions/expl_dyn.casadi")
        impl_dyn = Function.load("casadi_functions/impl_dyn.casadi")
        x_init = Function.load("casadi_functions/x_init.casadi")
        u_init = Function.load("casadi_functions/u_init.casadi")
        
        nu = eval_objk.sparsity_in(0).size1()
        nx = eval_objk.sparsity_in(1).size1()

        opti = Opti()
        # xx = opti.variable(nx, N+1)
        # uu = opti.variable(nu, N)
        # define variables in a stage-wise way
        xx = []; uu = []
        for k in range(N):
            xx.append(opti.variable(nx))
            uu.append(opti.variable(nu))
        xx.append(opti.variable(nx))
        xx = hcat(xx)
        uu = hcat(uu)

        if eval_g0.sparsity_out(0).size1() > 0:
            opti.subject_to(eval_g0(uu[:, 0], xx[:, 0]) == 0)

        obj = 0
        for k in range(N):
            if self.implicit:
                opti.subject_to(impl_dyn(uu[:,k], xx[:,k], xx[:,k+1]) == 0)
            else:
                opti.subject_to(xx[:,k+1] == expl_dyn(uu[:,k], xx[:,k]))

            if eval_gk.sparsity_out(0).size1() > 0:
                opti.subject_to(eval_gk(uu[:,k], xx[:,k]) == 0)
            if eval_gk_ineq.sparsity_out(0).size1() > 0:
                opti.subject_to(lb()['lb'] <= (eval_gk_ineq(uu[:,k], xx[:,k]) <= ub()['ub']))

            obj += eval_objk(uu[:,k], xx[:,k])

        if eval_gK.sparsity_out(0).size1() > 0:
            opti.subject_to(eval_gK(xx[:,N]) == 0)
        if eval_gK_ineq.sparsity_out(0).size1() > 0:
            opti.subject_to(lbK()['lbK'] <= (eval_gK_ineq(xx[:,N]) <= ubK()['ubK']))
        obj += eval_objK(xx[:,N])

        opti.minimize(obj)

        for k in range(N):
            opti.set_initial(xx[:,k], x_init(k))
            opti.set_initial(uu[:,k], u_init()['u_init'])
        opti.set_initial(xx[:,N], x_init(N))

        opti.solver('ipopt', {}, {'max_iter':400, 'tol':1e-4})
        # opti.solver('fatrop', {'structure_detection':'auto', 'debug':True}, {'max_iter':400, 'tol':1e-4})
        try:
            self.sol = opti.solve()
        except RuntimeError as e:
            print("Solver failed with error:", e)
            return False, None, None, None, None

        xx_sol = self.sol.value(xx)
        uu_sol = self.sol.value(uu)
        q_sol = xx_sol[:3*self.model.nb_pendulums, :]
        v_sol = xx_sol[3*self.model.nb_pendulums:6*self.model.nb_pendulums, :]
        p_sol = xx_sol[6*self.model.nb_pendulums:, :]
        F_sol = uu_sol[3*self.tracking + 1:, :]
        z_sol = uu_sol[3*self.tracking:3*self.tracking + self.model.nb_pendulums, :]
        return True, q_sol, v_sol, F_sol, z_sol


    def solve_trajectory(self, q0, v0, T, N):
        assert len(q0) == 3*self.model.nb_pendulums, "Length of q0 must match 3 times the number of pendulums"
        assert len(v0) == 3*self.model.nb_pendulums, "Length of v0 must match 3 times the number of pendulums"

        opti = Opti()
        q = opti.variable(3*self.model.nb_pendulums, N+1)
        v = opti.variable(3*self.model.nb_pendulums, N+1)
        p = opti.variable(1, N+1)
        # dp = opti.variable(1, N)
        s = opti.variable(3*self.tracking, N+1)
        z = opti.variable(1*self.model.nb_pendulums, N)
        F = opti.variable(3*len(self.model.actuated_joint_idxs), N)

        dt = T/N

        # initial conditions
        # opti.subject_to(self.model.g_eq0(q[:, 0], v[:, 0]) == 0) --> enforced using initial conditions
        opti.subject_to(q[:, 0] == q0)
        opti.subject_to(v[:, 0] == v0)
        opti.subject_to(p[:, 0] == 0)

        obj = 0.0
        for k in range(N):
            # dynamics constraints
            qdot, vdot = self.model.f(q[:,k], v[:,k], z[:,k], F[:,k])
            opti.subject_to(v[:,k+1] == v[:,k] + vdot*dt)
            idx = k+1 if self.implicit else k
            opti.subject_to(q[:,k+1] == q[:,k] + v[:,idx]*dt)
            opti.subject_to(p[:,k+1] == p[:,k] + dt)

            # equalities
            eq = self.model.g_eq(q[:,k], v[:,k], z[:,k], F[:,k])
            if self.model.stabilizer is not None:
                eq += self.model.stabilizer(q[:,k], v[:,k])
            opti.subject_to(eq == 0)

            if self.tracking:
                ref_point = self.path_func(p[k], T, q0)
                opti.subject_to(s[:,k] == q[:3, k] - ref_point)

            # inequalities
            opti.subject_to(-self.force_bounds <= (F[:,k] <= self.force_bounds))

            # objective
            obj += 1e-2*sumsqr(F[:,k])*0
            if self.tracking:
                obj += 1e2*sumsqr(s[:,k])

        # terminal constraint
        opti.subject_to(p[:, -1] == T)

        if not self.tracking:
            obj += sumsqr(v[:,N]) + 1e1*sumsqr(q[:,N] - self.get_q_rest())

        # objective
        opti.minimize(obj)

        opti.set_initial(q, np.tile(q0, (N+1, 1)).T)
        opti.set_initial(v, np.tile(v0, (N+1, 1)).T)
        opti.set_initial(p, np.linspace(0, T, N+1))

        opti.solver('ipopt', {}, {'max_iter':400, 'tol':1e-4})
        try:
            self.sol = opti.solve()
        except RuntimeError as e:
            print("Solver failed with error:", e)
            return False, None, None, None, None

        q_sol = self.sol.value(q)
        v_sol = self.sol.value(v)
        F_sol = self.sol.value(F)
        z_sol = self.sol.value(z)
        return True, q_sol, v_sol, F_sol, z_sol

    def get_nb_iters(self):
        try:
            return self.sol.stats()['iter_count']
        except AttributeError:
            return None