import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "build_docker"))
from preprocess_info import *
from test_debug_helper import *

BAbt_expected = [m.copy() for m in BAbt_original]
GuGx_expected = [m.copy() for m in GuGx_original]
FuFx_expected = [m.copy() for m in FuFx_original]
RSQrqt_expected = [m.copy() for m in RSQrqt_original]
Gg_eqt_expected = [m.copy() for m in Gg_eqt_original]
Gg_ineqt_expected = [m.copy() for m in Gg_ineqt_original]

# add regularization to hessian
for k in range(K):
    for i in range(nu[k] + nx[k]):
        RSQrqt_expected[k][i,i] += D_x[k][i]

# pre-processing code
Dl_list = []
Dr_list = []
Dl_inv_list = []
Dr_inv_list = []
for k in range(K-1):
    # construct Dl and Dr
    U1 = U[k][:r[k], :r[k]]
    U2 = U[k][:r[k], r[k]:]
    J = Jt[k].T
    if Pl[k].shape[0] == 0 or Pl[k].shape[1] == 0:
        Pl[k] = np.zeros((0,0))
    if Pr[k].shape[0] == 0 or Pr[k].shape[1] == 0:
        Pr[k] = np.zeros((0,0))
    if L[k].shape[0] == 0 or L[k].shape[1] == 0:
        L[k] = np.zeros((0,0))

    Dl = Pl[k] @ L[k] @ np.block([[-U1, np.zeros((r[k],nx[k+1]-r[k]))], [np.zeros((nx[k+1]-r[k],r[k])), np.eye(nx[k+1]-r[k])]])
    Dr = np.block([[np.eye(r[k]), np.linalg.inv(U1) @ U2], [np.zeros((nx[k+1]-r[k],r[k])), np.eye(nx[k+1]-r[k])]]) @ Pr[k].T
    Dl_inv = np.linalg.inv(Dl)
    Dr_inv = np.linalg.inv(Dr)
    Dl_list.append(Dl)
    Dr_list.append(Dr)
    Dl_inv_list.append(Dl_inv)
    Dr_inv_list.append(Dr_inv)
    norm = np.linalg.norm(J - Dl @ np.block([[-np.eye(r[k]), np.zeros((r[k],nx[k+1]-r[k]))], [np.zeros((nx[k+1]-r[k],nx[k+1]))]]) @ Dr)
    assert norm < 1e-6, f"Decomposition error at stage {k}: {norm}"

    # construct W
    W = np.block([[np.eye(nu[k+1]), np.zeros((nu[k+1], nx[k+1]))],
                  [np.zeros((nx[k+1] - r[k], nu[k+1] + r[k])), np.eye(nx[k+1] - r[k])],
                  [np.zeros((r[k], nu[k+1])), np.eye(r[k]), np.zeros((r[k], nx[k+1] - r[k]))]])
    Wp = np.block([[W, np.zeros((W.shape[0], 1))],
                   [np.zeros((1, W.shape[1])), 1]])
        
    BAbt_expected[k] = BAbt_expected[k] @ Dl_inv.T
    
    temp = np.zeros((nu[k] + nx[k], nu[k+1] + nx[k+1]))
    temp[:, nu[k+1]:] = FuFx_expected[k] @ Dr_inv
    GuGx_expected[k] = np.block([temp[:, :nu[k+1]], temp[:, nu[k+1]+modified_nx[k+1]:nu[k+1]+nx[k+1]]])
    FuFx_expected[k] = temp[:, nu[k+1]:nu[k+1]+modified_nx[k+1]]

    RSQrqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ RSQrqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    RSQrqt_expected[k+1][:,nu[k+1]:] = RSQrqt_expected[k+1][:,nu[k+1]:] @ np.linalg.inv(Dr)
    RSQrqt_expected[k+1] = Wp @ RSQrqt_expected[k+1]
    RSQrqt_expected[k+1] = RSQrqt_expected[k+1] @ W.T

    Gg_eqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ Gg_eqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    Gg_eqt_expected[k+1] = Wp @ Gg_eqt_expected[k+1]
    Gg_ineqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ Gg_ineqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    Gg_ineqt_expected[k+1] = Wp @ Gg_ineqt_expected[k+1]

    if k < K-2:
        BAbt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ BAbt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
        FuFx_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ FuFx_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1],:]

        FuFx_expected[k+1] = W @ FuFx_expected[k+1]
        BAbt_expected[k+1] = Wp @ BAbt_expected[k+1]

    Gg_eqt_expected[k] = np.block([Gg_eqt_expected[k], BAbt_expected[k][:, r[k]:]])
    BAbt_expected[k] = BAbt_expected[k][:, :r[k]]

# check pre-processing results
for k in range(K):
    if k < K-1:
        n = np.linalg.norm(BAbt[k] - BAbt_expected[k])
        if n > 1e-6: 
            print(f"BAbt[{k}] error: {n}")
        n = np.linalg.norm(GuGx[k] - GuGx_expected[k])
        if n > 1e-6: 
            print(f"GuGx[{k}] error: {n}")
        n = np.linalg.norm(FuFx[k] - FuFx_expected[k])
        if n > 1e-6: 
            print(f"FuFx[{k}] error: {n}")
        n = np.linalg.norm(RSQrqt[k] - RSQrqt_expected[k])
        if n > 1e-6: 
            print(f"RSQrqt[{k}] error: {n}")
            # print(f"RSQrqt original\n{RSQrqt_original[k]}")
            # print(f"RSQrqt expected\n{RSQrqt_expected[k]}")
            # print(f"RSQrqt:\n{RSQrqt[k]}")
        # print(f"GuGx original\n{GuGx_original[k]}")
        # print(f"GuGx expected\n{GuGx_expected[k]}")
        # print(f"GuGx:\n{GuGx[k]}")
        # print(f"FuFx original\n{FuFx_original[k]}")
        # print(f"FuFx expected\n{FuFx_expected[k]}")
        # print(f"FuFx:\n{FuFx[k]}")

    n = np.linalg.norm(Gg_eqt[k] - Gg_eqt_expected[k])
    if n > 1e-6:
        print(f"Gg_eqt[{k}] error: {n}")
    n = np.linalg.norm(Gg_ineqt[k] - Gg_ineqt_expected[k])
    if n > 1e-6:
        print(f"Gg_ineqt[{k}] error: {n}")

# post-processing code
primal_x_offset = 0
for k in range(K):
    if k > 0:
        ukxk = x[primal_x_offset:primal_x_offset+nu[k]+nx[k]]
        uk = ukxk[:nu[k]].copy()
        sk = ukxk[nu[k]:modified_nu[k]].copy()
        xk = ukxk[modified_nu[k]:].copy()
        x[primal_x_offset+nu[k]:primal_x_offset + nu[k] + r[k-1]] = xk
        x[primal_x_offset+nu[k]+r[k-1]:primal_x_offset+nu[k]+nx[k]] = sk
        x[primal_x_offset+nu[k]:primal_x_offset+nu[k]+nx[k]] = \
            Dr_inv_list[k-1] @ x[primal_x_offset+nu[k]:primal_x_offset+nu[k]+nx[k]] 
    
    primal_x_offset += nu[k] + nx[k]
x_expected = x

original_eq_path_offset = 0
original_eq_dyn_offset = 0
new_eq_path_offset = 0
new_eq_dyn_offset = 0
original_total_nb_eq_path = sum(modified_ng_eq)
original_total_nb_dyn_eq = sum(modified_nx[1:])
original_eq_path = eq_mult[:original_total_nb_eq_path].copy()
original_eq_dyn = eq_mult[original_total_nb_eq_path:original_total_nb_eq_path+original_total_nb_dyn_eq].copy()
new_total_nb_eq_path = sum(ng_eq)
new_total_nb_dyn_eq = sum(nx[1:])
new_eq_path = np.zeros((new_total_nb_eq_path,))
new_eq_dyn = np.zeros((new_total_nb_dyn_eq,))
for k in range(K-1):
    lmbd = original_eq_path[original_eq_path_offset:original_eq_path_offset+modified_ng_eq[k]].copy()
    pi = original_eq_dyn[original_eq_dyn_offset:original_eq_dyn_offset+modified_nx[k+1]].copy()

    true_pi = np.block([pi, lmbd[ng_eq[k]:]])
    lmbd = lmbd[:ng_eq[k]].copy()
    
    print(f"\n[{k}] eq_mult before: {true_pi}")
    true_pi = Dl_inv_list[k].T @ true_pi
    print(f"[{k}] eq_mult after: {true_pi}")

    new_eq_path[new_eq_path_offset:new_eq_path_offset+ng_eq[k]] = lmbd
    new_eq_dyn[new_eq_dyn_offset:new_eq_dyn_offset+nx[k+1]] = true_pi

    original_eq_path_offset += modified_ng_eq[k]
    original_eq_dyn_offset += modified_nx[k+1]
    new_eq_path_offset += ng_eq[k]
    new_eq_dyn_offset += nx[k+1]
new_eq_path[new_eq_path_offset:] = original_eq_path[original_eq_path_offset:].copy()

eq_mult_expected = np.block([new_eq_path, new_eq_dyn])

from final_solution import *
x = x[:,0]
eq_mult = eq_mult[:,0]

print(f"x error norm: {np.linalg.norm(x - x_expected)}")
print(f"eq_mult error norm: {np.linalg.norm(eq_mult - eq_mult_expected)}")

for i in range(len(x)):
    if abs(x[i] - x_expected[i]) > 1e-6:
        print(f"x[{i}] error: {x[i] - x_expected[i]}")

for i in range(eq_mult.shape[0]):
    if i == new_total_nb_eq_path:
        print(f"--- eq_mult[{i}] is the first multiplier for dynamics ---")
    if abs(eq_mult[i] - eq_mult_expected[i]) > 1e-5:
        # print(f"eq_mult[{i}] error: {eq_mult[i] - eq_mult_expected[i]}")
        print(f"eq_mult[{i}] error: {eq_mult[i]} - {eq_mult_expected[i]}")


### Solution to preprocessed problem
blocks = GetBlockMatrices(K, modified_nu, modified_nx, modified_ng_eq, RSQrqt, GuGx, FuFx, Gg_eqt, BAbt)
KKT, rhs = GetKKT(K, modified_nu, modified_nx, modified_ng_eq, blocks['R'], blocks['S'], blocks['Q'], blocks['Gu'], blocks['Gx'], blocks['Fu'], blocks['Fx'], blocks['Hu'], blocks['Hx'], blocks['B'], blocks['A'], blocks['r'], blocks['q'], blocks['h'], blocks['b'])
solution_vector = np.linalg.solve(KKT, -rhs)
solution = extract_solultion(K, modified_nu, modified_nx, modified_ng_eq, solution_vector)
# print_solution(solution)

### Solution to original problem
blocks = GetBlockMatrices(K, nu, nx, ng_eq, RSQrqt_original, GuGx_original, FuFx_original, Gg_eqt_original, BAbt_original)
KKT, rhs = GetKKT(K, nu, nx, ng_eq, blocks['R'], blocks['S'], blocks['Q'], blocks['Gu'], blocks['Gx'], blocks['Fu'], blocks['Fx'], blocks['Hu'], blocks['Hx'], blocks['B'], blocks['A'], blocks['r'], blocks['q'], blocks['h'], blocks['b'], Jt=Jt)
solution_vector = np.linalg.solve(KKT, -rhs)
solution = extract_solultion(K, nu, nx, ng_eq, solution_vector)
# print_solution(solution)