import casadi
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import example_robot_data as erd
from pinocchio.visualize import MeshcatVisualizer
import json

def make_quadruped():
    robot = erd.load("anymal")
    model = robot.model

    return model, robot.collision_model, robot.visual_model

def sigmoid(x):
    try:
        if x >= 0:
            return 1 / (1 + casadi.exp(-x))
        else:
            ex = casadi.exp(x)
            return ex / (1 + ex)
    except:
        return 1 / (1 + np.exp(-x))
        
def ground_reaction_force(p, v):
    # return np.array([0, 0, 0])
    # Ground stiffness and damping parameters
    k_ground = 200  # N/m
    d_ground = 600  # N·s/m
    alpha_k = 40
    alpha_d = 40
    # k_ground = 0
    # d_ground = 0
    # alpha_k = 0
    # alpha_d = 0

    penetration = -p[2]
    fx = -d_ground * sigmoid(alpha_d*penetration)*v[0]
    fy = -d_ground * sigmoid(alpha_d*penetration)*v[1]
    fz = k_ground * np.exp(alpha_k*penetration) - d_ground * sigmoid(alpha_d*penetration)*v[2]
    return np.array([fx, fy, fz])

class PinocchioCasadi:
    """Take a Pinocchio model, turn it into a Casadi model
    and define the appropriate graphs.
    """

    def __init__(self, model: pin.Model, timestep=0.05):
        self.model = model
        self.cmodel = cpin.Model(model)  # cast to CasADi model
        self.cdata = self.cmodel.createData()
        self.timestep = timestep
        self.create_dynamics()
        self.create_discrete_dynamics()

    def create_dynamics(self):
        """Create the acceleration expression and acceleration function."""
        nq = self.model.nq
        nu = 12 # HAA, HFE, KFE for 4 legs (LF, LH, RF, RH)
        nv = self.model.nv
        q = casadi.SX.sym("q", nq)
        v = casadi.SX.sym("v", nv)
        u = casadi.SX.sym("u", nu)
        dq_ = casadi.SX.sym("dq_", nv)
        self.u_node = u
        self.q_node = q
        self.v_node = v
        self.dq_ = dq_

        # actuation
        B = np.zeros((18, 12))
        B[6:,:] = np.eye(12)
        tau = B @ u

        # get foot positions and velocities
        cpin.forwardKinematics(self.cmodel, self.cdata, q, v)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        foot_ids = [self.model.getFrameId(name) for name in foot_names]

        tau_contacts = 0
        temp = 0
        for fid in foot_ids:
            # Foot position and velocity (in world frame)
            p = self.cdata.oMf[fid].translation
            v_frame = cpin.getFrameVelocity(self.cmodel, self.cdata, fid, pin.LOCAL_WORLD_ALIGNED)
            v_foot = v_frame.linear

            f_contact = ground_reaction_force(p, v_foot)

            # Foot Jacobian (LOCAL_WORLD_ALIGNED)
            J = cpin.computeFrameJacobian(self.cmodel, self.cdata, q, fid, pin.LOCAL_WORLD_ALIGNED)
            Jlin = J[:3, :]  # linear part
            tau_contacts += Jlin.T @ f_contact

        tau += tau_contacts

        a = cpin.aba(self.cmodel, self.cdata, q, v, tau)
        self.acc = a
        self.acc_func = casadi.Function("acc", [q, v, u], [a, temp], ["q", "v", "u"], ["a", "temp"])

    def get_explicit_integrator(self):
        q = self.q_node
        v = self.v_node
        u = self.u_node
        a, temp = self.acc_func(q, v, u)
        # a = casadi.vertcat(casadi.SX.zeros(2,1), -9.81, casadi.SX.zeros(self.model.nv-3,1))

        dt = self.timestep
        vnext = v + a * dt
        qnext = cpin.integrate(self.cmodel, q, dt * vnext)
        # qnext = q + casadi.vertcat(0, dt * vnext)

        x = casadi.vertcat(q, v)
        xnext = casadi.vertcat(qnext, vnext)

        explicit_integrator = casadi.Function(
            "quadruped_explicit_integrator",
            [u, x], [xnext]
        )       

        return explicit_integrator 

    def get_implicit_integrator(self):
        q = self.q_node
        v = self.v_node
        u = self.u_node
        qnext = casadi.SX.sym("qnext", self.model.nq)
        vnext = casadi.SX.sym("vnext", self.model.nv)
        xnext = casadi.vertcat(qnext, vnext)
        a_start, _ = self.acc_func(q, v, u)
        a_end, _ = self.acc_func(qnext, vnext, u)
        # a = 0.5 * (a_start + a_end)
        a = a_end

        dt = self.timestep
        x = casadi.vertcat(q, v)
        dyn_equations = \
            casadi.vertcat(cpin.integrate(self.cmodel, q, dt * vnext) - qnext,
                           v + a * dt - vnext)

        implicit_integrator = casadi.Function(
            "quadruped_implicit_integrator",
            [u, x, xnext], [dyn_equations]
        )        

        return implicit_integrator

    def create_discrete_dynamics(self):
        """
        Create the map `(q,v) -> (qnext, vnext)` using semi-implicit Euler integration.
        """
        q = self.q_node
        v = self.v_node
        u = self.u_node
        dq_ = self.dq_
        # q' = q + dq
        # q_dq = cpin.integrate(self.cmodel, q, dq_)
        q_dq = q
        self.q_dq = q_dq
        # express acceleration using q' = q + dq
        a, temp = self.acc_func(q_dq, v, u)

        dt = self.timestep
        vnext = v + a * dt
        qnext = cpin.integrate(self.cmodel, self.q_dq, dt * vnext)

        self.dyn_qv_fn_ = casadi.Function(
            "discrete_dyn",
            [q, dq_, v, u],
            [qnext, vnext, temp],
            ["q", "dq_", "v", "u"],
            ["qnext", "vnext", "temp"],
        )

        explicit_integrator = self.get_explicit_integrator()
        implicit_integrator = self.get_implicit_integrator()

        return explicit_integrator, implicit_integrator

    def forward(self, x, u):
        nq = self.model.nq
        nv = self.model.nv
        q = x[:nq]
        v = x[nq:]
        dq_ = np.zeros(nv)
        qnext, vnext, temp = self.dyn_qv_fn_(q, dq_, v, u)
        # print("temp value:", temp)
        xnext = np.concatenate((qnext, vnext))
        return xnext

    def forward_casadi(self, x, u):
        nq = self.model.nq
        nv = self.model.nv
        q = x[:nq]
        v = x[nq:]
        dq_ = casadi.MX.zeros(nv)
        qnext, vnext, temp = self.dyn_qv_fn_(q, dq_, v, u)
        # print("temp value:", temp)
        xnext = casadi.vertcat(qnext, vnext)
        return xnext

    def residual_fwd(self, x, u, xnext):
        nv = self.model.nv
        dq = np.zeros(nv)
        dqn = dq
        res = self.dyn_residual(x, u, xnext, dq, dqn)
        return res

    def test_dynamics(self):
        q_test = [0, 0, 0.5292, 0, 0, 0, 1, -0.1, 0.7, -1, 0.1, 0.7, -1, -0.1, -0.7, -1, 0.1, -0.7, -1]
        v_test = np.zeros(self.model.nv)

        u = np.ones((12, 1))

        # evaluate forward
        x_test = np.concatenate((q_test, v_test))
        xnext = self.forward(x_test, u)

        print("xnext:", xnext)

class QuadrupedDynamics(PinocchioCasadi):
    def __init__(self, timestep=0.05):
        model, collision_model, visual_model = make_quadruped()
        self.collision_model = collision_model
        self.visual_model = visual_model
        super().__init__(model=model, timestep=timestep)