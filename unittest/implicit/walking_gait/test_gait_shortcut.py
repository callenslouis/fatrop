from casadi import *

# load functions
folder = "casadi_funcs/n_coll_3"
f_objk = Function.load(f"{folder}/f_objk.casadi")
f_g0 = Function.load(f"{folder}/f_g0.casadi")
f_gk = Function.load(f"{folder}/f_gk.casadi")
f_gK = Function.load(f"{folder}/f_gN.casadi")
f_gk_ineq = Function.load(f"{folder}/f_gk_ineq.casadi")
f_gap = Function.load(f"{folder}/f_gap.casadi")

f_lb = Function.load(f"{folder}/f_lb.casadi")
f_ub = Function.load(f"{folder}/f_ub.casadi")
f_x_init = Function.load(f"{folder}/f_init_x.casadi")
f_u_init = Function.load(f"{folder}/f_init_u.casadi")

# derive dimensions
xx_init = f_x_init()['x_init']
uu_init = f_u_init()['u_init']
lb = f_lb()['lb']
ub = f_ub()['ub']
nx = xx_init.shape[0]
nu = uu_init.shape[0]
N = xx_init.shape[1] - 1

# setup opti instance
opti = Opti()
xx = []
uu = []
for k in range(N):
    xx.append(opti.variable(nx, 1))
    uu.append(opti.variable(nu, 1))
xx.append(opti.variable(nx, 1))  # add state at mesh point N

obj = 0
for k in range(N):
    # gap-closing constraint
    opti.subject_to(xx[k+1] == f_gap(xx[k], uu[k]))
    
    # # equality constraint
    if k == 0:
        opti.subject_to(f_g0(xx[k], uu[k]) == 0)
    else:
        opti.subject_to(f_gk(xx[k], uu[k]) == 0)
        
    # inequality constraint
    opti.subject_to(lb <= (f_gk_ineq(xx[k], uu[k]) <= ub))
    
    # initial guess
    opti.set_initial(xx[k], xx_init[:, k])
    opti.set_initial(uu[k], uu_init[:, k])
    
    # final constraints
    if k == N-1:
        opti.subject_to(f_gK(xx[k+1]) == 0)
        opti.set_initial(xx[k+1], xx_init[:, k+1])
        
    # objective
    obj += f_objk(xx[k], uu[k])
    
opti.minimize(obj)

opti.solver('fatrop', 
            {'expand': True, 
             'detect_simple_bounds': True,
             'structure_detection': 'auto'},
            {'tol': 1e-4,
             'mu_init': 0.1})

# opti_f = opti.to_function('opti_f', [], [hcat(xx), hcat(uu)], [], ['xx', 'uu'])
# opti_f.save('casadi_funcs/test_gait_shortcut_python.casadi')
sol = opti.solve()

