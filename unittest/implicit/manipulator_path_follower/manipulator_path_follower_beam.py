from casadi import *
import numpy as np


### define helper functions
def get_all_functions():
    functions = {"fk":None, "fd":None, "fk_ee":None, "fkpos_ee":None, "fkrot_ee":None, "id":None}
    for f in functions.keys():
        try:
            func = Function.load(f"panda_{f}.casadi")
        except:
            func = Function.load(f"unittest/implicit/manipulator_path_follower/panda_{f}.casadi")
        functions[f] = func

    try:
        functions["beam"] = Function.load("beam_model.casadi")
    except:
        functions["beam"] = Function.load("unittest/implicit/manipulator_path_follower/beam_model.casadi")

    return functions

def add_full_model(functions, wn, zeta, L):
    q = MX.sym("q", 7)
    qd = MX.sym("qd", 7)
    tau = MX.sym("tau", 7)
    qdd = MX(7)
    th = MX.sym("th", 1)
    thd = MX.sym("thd", 1)
    thdd = MX(1)

    qdd = functions['fd'](q, qd, tau)
    thdd = functions['beam'](q, qd, qdd, wn, zeta, L, th, thd)

    full_model = Function("full_model", [q, qd, tau, th, thd], [qdd, thdd])
    functions['full_model'] = full_model
    return functions

def get_full_equilibrium(functions, q0, wn, zeta, L):
    opti = Opti()
    tau_eq = opti.variable(7)
    th_eq = opti.variable(1)

    q_eq = DM(q0)
    qd_eq = DM.zeros(7)
    qdd_eq = DM.zeros(7)

    opti.subject_to(tau_eq == functions['id'](q_eq, qd_eq, qdd_eq))
    opti.subject_to(0 == functions['beam'](q_eq, qd_eq, qdd_eq, wn, zeta, L, th_eq, 0))

    opti.minimize(sumsqr(tau_eq))
    opti.solver('ipopt', {}, {})
    sol = opti.solve()
    print("Equilibrium tau:", sol.value(tau_eq))
    print("Equilibrium th:", sol.value(th_eq))
    return sol.value(tau_eq), sol.value(th_eq)

def get_path_function(functions, q0):
    t = MX.sym("t", 1)
    q = MX.zeros(7, 1)
    for i in range(7):
        q[i] = q0[i] + 0.5*(1-cos(t*(0.5*i+1)))
        # q[i] = q0[i]

    return Function("eval_path", [t], [functions['fkpos_ee'](q)])


### Main
# define parameters
wn = 18
zeta = 0.005
L = 0.3
functions = get_all_functions()
functions = add_full_model(functions, wn, zeta, L)
q0 = np.array([[-np.pi/2, -np.pi/6, 0., -2*np.pi/3, -np.pi/2, np.pi/2, np.pi/4]]).T
tau_eq, th_eq = get_full_equilibrium(functions, q0, wn, zeta, L)
path_func = get_path_function(functions, q0)

opti = Opti()
N = 70
T = 3.0
dt = T/N
tau_bound = [87, 87, 87, 87, 12, 12, 12]
path_tol = 0.05
control_penalty = 1.0e-4
n = 7

with_slack = 1
with_progress_variable = 1
with_accel_variables = 1
with_implicit_integrator = 0

# define variables
q = []; qd = []; qdd = []; tau = []; s = []
p = []; dp = []
th = []; thd = []; thdd = []
for k in range(N):
    # states
    q.append(opti.variable(n))
    th.append(opti.variable(1))
    qd.append(opti.variable(n))
    thd.append(opti.variable(1))
    if with_progress_variable:
        p.append(opti.variable(1))
    # controls
        dp.append(opti.variable(1))

    if with_accel_variables:
        qdd.append(opti.variable(n))
        thdd.append(opti.variable(1))
    tau.append(opti.variable(n))
    if with_slack:
        s.append(opti.variable(3))
q.append(opti.variable(n))
th.append(opti.variable(1))
qd.append(opti.variable(n))
thd.append(opti.variable(1))
p.append(opti.variable(1))
if with_accel_variables:
    qdd.append(opti.variable(n))

q = hcat(q); qd = hcat(qd); th = hcat(th); thd = hcat(thd)
if with_accel_variables:
    qdd = hcat(qdd); thdd = hcat(thdd)
tau = hcat(tau)
if with_slack:
    s = hcat(s)
else:
    s = MX.zeros(3, N)
if with_progress_variable:
    p = hcat(p)
    dp = hcat(dp)
else:
    p = transpose(linspace(0, T, N+1))
    dp = MX.ones(1, N)

# initial state
opti.subject_to(q[:, 0] == q0)
opti.subject_to(th[:, 0] == th_eq)
opti.subject_to(qd[:, 0] == 0)
opti.subject_to(thd[:, 0] == 0)
if with_progress_variable:
    opti.subject_to(p[:,0] == 0)

obj = 0.0
for k in range(N):
    # dymamics
    rhs_idx = k if not with_implicit_integrator else k+1
    opti.subject_to(q[:,k+1] == q[:,k] + qd[:,rhs_idx]*dp[:,k])
    opti.subject_to(th[:,k+1] == th[:,k] + thd[:,rhs_idx]*dp[:,k])
    if with_accel_variables:
        opti.subject_to(qd[:,k+1] == qd[:,k] + qdd[:,k]*dp[:,k])
        opti.subject_to(thd[:,k+1] == thd[:,k] + thdd[:,k]*dp[:,k])
    else:
        qdd, thdd = functions['full_model'](q[:,rhs_idx], qd[:,rhs_idx], tau[:,k], th[:,rhs_idx], thd[:,rhs_idx])
        opti.subject_to(qd[:,k+1] == qd[:,k] + qdd*dp[:,k])
        opti.subject_to(thd[:,k+1] == thd[:,k] + thdd*dp[:,k])
    
    if with_progress_variable:
        opti.subject_to(p[:,k+1] == p[:,k] + dp[:,k])


    # slack constraints
    ee_pos_k = path_func(p[:,k])
    if with_slack:
        opti.subject_to(functions['fkpos_ee'](q[:,k]) - ee_pos_k - s[:,k] == 0)
    else:
        s[:,k] = functions['fkpos_ee'](q[:,k]) - ee_pos_k
    
    # true dynamics
    if with_accel_variables:
        opti.subject_to(tau[:,k] == functions['id'](q[:,k], qd[:,k], qdd[:,k]))
        opti.subject_to(thdd[:,k] == functions['beam'](q[:,k], qd[:,k], qdd[:,k], wn, zeta, L, th[:,k], thd[:,k]))

    # inequalities
    opti.subject_to(-1 <= (sumsqr(s[:,k]) <= path_tol**2))
    for i in range(n):
        opti.subject_to(-tau_bound[i] <= (tau[i,k] <= tau_bound[i]))

    if with_progress_variable:
        opti.subject_to(0.2*dt <= (dp[:,k] <= 2.0*dt))
        # opti.subject_to(dp[:,k] == dt)

    obj += 0*sumsqr(s[:,k]) + control_penalty*sumsqr(tau[:,k]) + 1e-4*sumsqr(dp[:,k] - dt)
    if with_accel_variables:
        # obj += sumsqr(qdd[:,k])*dp[:,k]
        obj += sumsqr(qdd[:,k])

if with_progress_variable:
    opti.subject_to(p[-1,-1] == T)
    opti.set_initial(p, transpose(linspace(0, T, N+1)))
    opti.set_initial(dp, dt)

for k in range(N):
    opti.set_initial(q[:,k], q0)
    opti.set_initial(th[:,k], th_eq)
    opti.set_initial(tau[:,k], tau_eq)

opti.minimize(obj)
# opti.solver('fatrop', {'structure_detection':'auto', 'debug':True}, {'tolerance':1e-4})
opti.solver('ipopt', {}, {'tol':1e-4})

sol = opti.solve()