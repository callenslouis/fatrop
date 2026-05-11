from casadi import *
from collocation_scheme import collocation_scheme
from reformulator import Reformulator

######################
### load functions ###
######################
folder = "casadi_funcs"
f_objk_orig = Function.load(f"{folder}/f_objk.casadi")
f_g0_orig = Function.load(f"{folder}/f_g0.casadi")
f_gk_orig = Function.load(f"{folder}/f_gk.casadi")
f_gK_orig = Function.load(f"{folder}/f_gN.casadi")
f_gk_ineq_orig = Function.load(f"{folder}/f_gk_ineq.casadi")
f_gap_orig = Function.load(f"{folder}/f_gap.casadi")
f_lb_orig = Function.load(f"{folder}/f_lb.casadi")
f_ub_orig = Function.load(f"{folder}/f_ub.casadi")
f_x_init_orig = Function.load(f"{folder}/f_init_x.casadi")
f_u_init_orig = Function.load(f"{folder}/f_init_u.casadi")
f_g0_orig_unique = Function.load(f"{folder}/f_g0_unique.casadi")

# derive dimensions
xx_init = f_x_init_orig()['x_init']
uu_init = f_u_init_orig()['u_init']
lb = f_lb_orig()['lb']
ub = f_ub_orig()['ub']
nx = xx_init.shape[0]
nu = uu_init.shape[0]
N = xx_init.shape[1] - 1
xk = SX.sym('xk', nx)


#############################
### change ordering of uk ###
#############################
# from
# [act_mesh_k_SX, q_coll(1)_k_SX, ..., q_coll(3)_k_SX, qdot_coll(1)_k_SX, ..., qdot_coll(3)_k_SX, qddot_coll(1)_k_SX, ..., qddot_coll(3)_k_SX]
#                                      --------------                          ------------------                           
# to
# [act_mesh_k_SX, q_coll(1)_k_SX, ..., qddot_coll(1)_k_SX, q_coll(2)_k_SX, ..., qddot_coll(2)_k_SX, qddot_coll(3)_k_SX, q_coll(3)_k_SX, ...]
#                                                                                                                       --------------------
# such that zk variables are at the back
n_act_mesh = 18
n_coll = 3
n_coords = 9
dt = (1.0/0.9/2.0)/N
act_mesh = SX.sym('act_mesh_k', n_act_mesh)
q_coll = SX.sym('q_coll_k', n_coll * n_coords)
qdot_coll = SX.sym('qdot_coll_k', n_coll * n_coords)
qddot_coll = SX.sym('qddot_coll_k', n_coll * n_coords)
uk_orig = vertcat(act_mesh, q_coll, qdot_coll, qddot_coll)
assert uk_orig.shape[0] == nu

introduce_periodicity_vars = True
periodicity_vars = SX.sym('periodicity_vars', introduce_periodicity_vars*(n_coords * 2 - 1)) # all q and qdot variables except for the forward foot x position
zk = vertcat(q_coll[2*n_coords:3*n_coords], qdot_coll[2*n_coords:3*n_coords], periodicity_vars)
print(f"zk: {zk.shape}")
if introduce_periodicity_vars:
    assert zk.shape[0] == nx
    periodicity_constraint = periodicity_vars - xk[n_coords * 2:]
else:
    periodicity_constraint = periodicity_vars

uk  = vertcat(act_mesh, 
              q_coll[0:n_coords], qdot_coll[0:n_coords], qddot_coll[0:n_coords],
              q_coll[n_coords:2*n_coords], qdot_coll[n_coords:2*n_coords], qddot_coll[n_coords:2*n_coords],
              qddot_coll[2*n_coords:3*n_coords], # accelerations are still controls
              zk)
f_objk = Function('f_objk', [xk, uk], [f_objk_orig(xk, uk_orig)])
f_g0 = Function('f_g0', [xk, uk], [vertcat(f_g0_orig(xk, uk_orig), periodicity_constraint)])
f_gk = Function('f_gk', [xk, uk], [vertcat(f_gk_orig(xk, uk_orig), periodicity_constraint)])
f_gK = Function('f_gK', [xk], [f_gK_orig(xk)])
f_gk_ineq = Function('f_gk_ineq', [xk, uk], [f_gk_ineq_orig(xk, uk_orig)])
f_gap = Function('f_gap', [xk, uk], [zk])
f_g0_unqique = Function('f_g0_unique', [xk, uk], [f_g0_orig_unique(xk, uk_orig)])

###############################
### Update path constraints ###
###############################
reformulator = Reformulator(n_coll, n_coords, n_act_mesh, dt, xk, uk, uk_orig, q_coll, qdot_coll, qddot_coll)
f_gk_reformulated = reformulator.get_f_gk_reformulated(f_gk)
f_g0_reformulated = Function('f_g0_reformulated', [xk, uk], [vertcat(f_g0_unqique(xk, uk), f_gk_reformulated(xk, uk))])
reformulator.show_gk_jacobian_structure_for_debugging(f_gk)
reformulator.show_gk_jacobian_structure_for_debugging(f_gk_reformulated)

########################################
### update initial guess of controls ###
########################################
act_init = uu_init[:n_act_mesh, :]
q_coll_init = uu_init[n_act_mesh:n_act_mesh+n_coll*n_coords, :]
qdot_coll_init = uu_init[n_act_mesh+n_coll*n_coords:n_act_mesh+2*n_coll*n_coords, :]
qddot_coll_init = uu_init[n_act_mesh+2*n_coll*n_coords:n_act_mesh+3*n_coll*n_coords, :]

uu_init_reordered = np.vstack([
    act_init,
    q_coll_init[0:n_coords, :], qdot_coll_init[0:n_coords, :], qddot_coll_init[0:n_coords, :],
    q_coll_init[n_coords:2*n_coords, :], qdot_coll_init[n_coords:2*n_coords, :], qddot_coll_init[n_coords:2*n_coords, :],
    qddot_coll_init[2*n_coords:3*n_coords, :], # accelerations are still controls
    q_coll_init[2*n_coords:3*n_coords, :], qdot_coll_init[2*n_coords:3*n_coords, :],
    xx_init[n_coords * 2:, :-1]
])
assert uu_init_reordered.shape[0] == nu + 2*n_coords - 1


###########################
### setup opti instance ###
###########################
opti = Opti()
xx = []
uu = []
for k in range(N):
    xx.append(opti.variable(nx, 1))
    uu.append(opti.variable(nu + 2*n_coords - 1, 1))
xx.append(opti.variable(nx, 1))  # add state at mesh point N

obj = 0
for k in range(N):
    # gap-closing constraint
    opti.subject_to(xx[k+1] == f_gap(xx[k], uu[k]))
    
    # # equality constraint
    if k == 0:
        # opti.subject_to(f_g0(xx[k], uu[k]) == 0)
        opti.subject_to(f_g0_reformulated(xx[k], uu[k]) == 0)
    else:
        # opti.subject_to(f_gk(xx[k], uu[k]) == 0)
        opti.subject_to(f_gk_reformulated(xx[k], uu[k]) == 0)

    # inequality constraint
    opti.subject_to(lb <= (f_gk_ineq(xx[k], uu[k]) <= ub))
    
    # initial guess
    opti.set_initial(xx[k], xx_init[:, k])
    opti.set_initial(uu[k], uu_init_reordered[:, k])
    
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
             'mu_init': 0.1,
             'max_iter': 300})

# print(f"creating opti to function object")
# opti_f = opti.to_function('opti_f', [], [hcat(xx), hcat(uu)], [], ['xx', 'uu'])
# print(f"saving opti to function object")
# opti_f.save('casadi_funcs/test_gait_shortcut_python_reformulated.casadi')
print(f"solving")
sol = opti.solve()

