import numpy as np

K = 3
nu = [2, 2, 1]
nx = [2, 3, 2]
ng_ineq = [0, 0, 0]
ng_eq = [2, 2, 1]
r = [2, 1]
modified_K = 3
modified_nu = [2, 3, 2]
modified_nx = [2, 2, 1]
modified_ng_ineq = [0, 0, 0]
modified_ng_eq = [3, 3, 1]
RSQrqt = [
np.array([
	[   1.05892,    1.25762,    1.55044,   0.647349 ],
	[   1.25762,    1.94663,    1.78034,   0.748745 ],
	[   1.55044,    1.78034,    2.55766,   0.811005 ],
	[  0.647349,   0.748745,   0.811005,   0.466765 ],
	[         0,          1,          2,          3 ],
	])
,
np.array([
	[   1.98941,     2.2339, 0.00856321,    1.05523,    1.64883 ],
	[    2.2339,    2.73325,  0.0614126,    1.22935,    1.62659 ],
	[0.00856321,  0.0614126,   0.774716, -0.0275171,  -0.326046 ],
	[   1.05523,    1.22935, -0.0275171,   0.649282,   0.926051 ],
	[   1.64883,    1.62659,  -0.326046,   0.926051,    1.90647 ],
	[         4,          5,   -2.26666,          8,          7 ],
	])
,
np.array([
	[   0.90293,   0.768566,    1.21722 ],
	[  0.768566,    0.68665,    1.01912 ],
	[   1.21722,    1.01912,    2.00597 ],
	[         9,    8.87392,         11 ],
	])
]
RSQrqt_original = [
np.array([
	[   1.05892,    1.25762,    1.55044,   0.647349 ],
	[   1.25762,    1.94663,    1.78034,   0.748745 ],
	[   1.55044,    1.78034,    2.55766,   0.811005 ],
	[  0.647349,   0.748745,   0.811005,   0.466765 ],
	[         0,          1,          2,          3 ],
	])
,
np.array([
	[   1.98941,     2.2339,    1.68515,    1.64883,    1.05523 ],
	[    2.2339,    2.73325,    1.77686,    1.62659,    1.22935 ],
	[   1.68515,    1.77686,      2.032,    1.51656,   0.932604 ],
	[   1.64883,    1.62659,    1.51656,    1.90647,   0.926051 ],
	[   1.05523,    1.22935,   0.932604,   0.926051,   0.649282 ],
	[         4,          5,          6,          7,          8 ],
	])
,
np.array([
	[   0.90293,   0.893174,    1.21722 ],
	[  0.893174,    0.91633,    1.22448 ],
	[   1.21722,    1.22448,    2.00597 ],
	[         9,         10,         11 ],
	])
]
FuFx = [
np.array([
	[ 0.0394869,  0.0230533 ],
	[ 0.0470132,  0.0474868 ],
	[ 0.0383462,  0.0287991 ],
	[ 0.0102863,  0.0878452 ],
	])
,
np.array([
	[ 0.0537023 ],
	[0.00995691 ],
	[ 0.0469249 ],
	[ 0.0508315 ],
	[ 0.0904648 ],
	])
]
FuFx_original = [
np.array([
	[ 0.0533206,  0.0230533,  0.0394869 ],
	[ 0.0618809,  0.0474868,  0.0470132 ],
	[ 0.0716075,  0.0287991,  0.0383462 ],
	[  0.074917,  0.0878452,  0.0102863 ],
	])
,
np.array([
	[ 0.0260341,  0.0537023 ],
	[ 0.0447926, 0.00995691 ],
	[ 0.0352312,  0.0469249 ],
	[  0.084114,  0.0904648 ],
	[0.00375594,  0.0508315 ],
	])
]
GuGx = [
np.array([
	[         0,          0,  0.0218074 ],
	[         0,          0,  0.0081698 ],
	[         0,          0,   0.035824 ],
	[         0,          0, 0.000583462 ],
	])
,
np.array([
	[         0,  0.0205365 ],
	[         0,  0.0437733 ],
	[         0,  0.0304274 ],
	[         0, -0.00144774 ],
	[         0,   0.074853 ],
	])
]
GuGx_original = [
np.array([
	[         0,          0 ],
	[         0,          0 ],
	[         0,          0 ],
	[         0,          0 ],
	])
,
np.array([
	[         0 ],
	[         0 ],
	[         0 ],
	[         0 ],
	[         0 ],
	])
]
Gg_eqt = [
np.array([
	[  0.720633,    0.58202,  -0.265408 ],
	[  0.537373,   0.758616,   0.665175 ],
	[  0.105908,     0.4736,  -0.149839 ],
	[  0.186332,   0.736918,   0.277973 ],
	[         0,          1,   -2.73565 ],
	])
,
np.array([
	[  0.358152,   0.750686,  0.0785526 ],
	[  0.607831,   0.325047,  0.0648557 ],
	[ -0.943635,  -0.217985, -0.0865342 ],
	[  0.635059,     0.9953,   0.741926 ],
	[  0.958949,    0.65279,   0.213764 ],
	[         2,          3,    6.90661 ],
	])
,
np.array([
	[   0.58185 ],
	[  0.365773 ],
	[  0.474698 ],
	[         4 ],
	])
]
Gg_eqt_original = [
np.array([
	[  0.720633,    0.58202 ],
	[  0.537373,   0.758616 ],
	[  0.105908,     0.4736 ],
	[  0.186332,   0.736918 ],
	[         0,          1 ],
	])
,
np.array([
	[  0.358152,   0.750686 ],
	[  0.607831,   0.325047 ],
	[ 0.0384254,   0.634274 ],
	[  0.958949,    0.65279 ],
	[  0.635059,     0.9953 ],
	[         2,          3 ],
	])
,
np.array([
	[   0.58185 ],
	[  0.414369 ],
	[  0.474698 ],
	[         4 ],
	])
]
Gg_ineqt = [
np.array([
	[],
	[],
	[],
	[],
	[],
	])
,
np.array([
	[],
	[],
	[],
	[],
	[],
	[],
	])
,
np.array([
	[],
	[],
	[],
	[],
	])
]
Gg_ineqt_original = [
np.array([
	[],
	[],
	[],
	[],
	[],
	])
,
np.array([
	[],
	[],
	[],
	[],
	[],
	[],
	])
,
np.array([
	[],
	[],
	[],
	[],
	])
]
BAbt = [
np.array([
	[    -1.203,   0.222669 ],
	[     4.535,   -6.87219 ],
	[  -4.75893,     6.0845 ],
	[   6.35628,   -9.53411 ],
	[  -28.1906,    27.0655 ],
	])
,
np.array([
	[ -0.341874 ],
	[ -0.511731 ],
	[   1.11621 ],
	[ -0.967861 ],
	[  -1.42496 ],
	[  -12.6298 ],
	])
]
BAbt_original = [
np.array([
	[  0.592845,   0.844266,   0.857946 ],
	[  0.847252,   0.623564,   0.384382 ],
	[  0.297535,   0.056713,   0.272656 ],
	[  0.477665,   0.812169,   0.479977 ],
	[         5,          6,          7 ],
	])
,
np.array([
	[   0.21655,   0.135218 ],
	[  0.324141,   0.149675 ],
	[  0.222321,   0.386489 ],
	[  0.902598,    0.44995 ],
	[  0.613063,   0.902349 ],
	[         8,          9 ],
	])
]
Jt = [
np.array([
	[  0.724752,   0.776917,   0.753623 ],
	[  0.566482,   0.630923,   0.599824 ],
	[  0.818279,    0.81858,   0.824195 ],
	])
,
np.array([
	[  0.064844,   0.016968 ],
	[  0.633421,    0.16575 ],
	])
]
L = [
np.array([[1, 0, 0],
[0.993187, 1, 0],
[0.992822, -0.825251, 1],
])
,
np.array([[1, 0],
[0.261674, 1],
])
]
U = [
np.array([[0.824195, 0.599824, 0.753623],
[0, 0.0351854, 0.0284288],
[0, 0, -6.93889e-17],
])
,
np.array([[0.633421, 0.064844],
[0, 0],
])
]
Pl = [
np.array([
	[         0,          0,          1 ],
	[         0,          1,          0 ],
	[         1,          0,          0 ],
	])
,
np.array([
	[         1,          0 ],
	[         0,          1 ],
	])
]
Pr = [
np.array([
	[         0,          0,          1 ],
	[         0,          1,          0 ],
	[         1,          0,          0 ],
	])
,
np.array([
	[         0,          1 ],
	[         1,          0 ],
	])
]
x = np.array([-1.87676, 3.75106, -2.37757, -2.20823, 7.47255, -4.68363, 1.68072, -11.6432, 7.45674, 13.6013, -19.2458, -10.2683])
eq_mult = np.array([10.8484, 6.50868, -7.8914, -12.6774, -8.84262, 15.5921, 10.3281, 10.6416, 7.80735, -7.23691])

BAbt_expected = [m.copy() for m in BAbt_original]
GuGx_expected = [m.copy() for m in GuGx_original]
FuFx_expected = [m.copy() for m in FuFx_original]
RSQrqt_expected = [m.copy() for m in RSQrqt_original]
Gg_eqt_expected = [m.copy() for m in Gg_eqt_original]
Gg_ineqt_expected = [m.copy() for m in Gg_ineqt_original]

# number_of_primal_vars = sum(nu) + sum(nx)
# number_of_eqs = sum(ng_eq) + sum(nx[1:])
# KKT = np.zeros((number_of_primal_vars + number_of_eqs, number_of_primal_vars + number_of_eqs))
# rhs = np.zeros((number_of_primal_vars + number_of_eqs, 1))
# ptr = 0
# for k in range(K-1, -1, -1):
# 	KKT[ptr:ptr+nu[k]+nx[k], ptr:ptr+nu[k]+nx[k]] = RSQrqt_original[k][:nu[k]+nx[k], :nu[k]+nx[k]]
# 	rhs[ptr:ptr+nu[k]+nx[k]] = -RSQrqt_original[k][nu[k]+nx[k]:nu[k]+nx[k]+1, :nu[k]+nx[k]].T
    
# 	KKT[ptr:ptr+nu[k]+nx[k], ptr:ptr+ng_eq[k]] = Gg_eqt_original[k][:nu[k]+nx[k], :]
# 	KKT[ptr:ptr+ng_eq[k], ptr:ptr+nu[k]+nx[k]] = Gg_eqt_original[k][:nu[k]+nx[k], :].T
# 	rhs[ptr+nu[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]] = -Gg_eqt_original[k][nu[k]+nx[k]:nu[k]+nx[k]+1, :ng_eq[k]].T
      
# 	if k > 0:
# 		KKT[ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1], ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = BAbt_original[k-1][:nu[k-1]+nx[k-1]]
# 		KKT[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k], ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1]] = BAbt_original[k-1][:nu[k-1]+nx[k-1]].T
# 		rhs[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = -BAbt_original[k-1][nu[k-1]+nx[k-1]:nu[k-1]+nx[k-1]+1, :].T

# 		KKT[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k], ptr+nu[k]:ptr+nu[k]+nx[k]] = Jt[k-1][:nx[k], :nx[k]].T
# 		KKT[ptr+nu[k]:ptr+nu[k]+nx[k], ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = Jt[k-1][:nx[k], :nx[k]]
# 		KKT[ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1], ptr+nu[k]:ptr+nu[k]+nx[k]] = FuFx_original[k-1][:nx[k-1]+nu[k-1], :nx[k]]
# 		KKT[ptr+nu[k]:ptr+nu[k]+nx[k], ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1]] = FuFx_original[k-1][:nx[k-1]+nu[k-1], :nx[k]].T
# 		ptr += nu[k] + nx[k] + ng_eq[k] + nx[k]
# print(KKT)
# solution = np.linalg.solve(KKT, rhs)
# # print(rhs)
# print(f"true solution:\n{solution}")

# number_of_primal_vars = sum(nu) + sum(nx)
# number_of_eqs = sum(ng_eq) + sum(nx[1:])
# KKT_preprocessed = np.zeros((number_of_primal_vars + number_of_eqs, number_of_primal_vars + number_of_eqs))
# rhs_preprocessed = np.zeros((number_of_primal_vars + number_of_eqs, 1))
# ptr = 0
# for k in range(K-1, -1, -1):
# 	KKT_preprocessed[ptr:ptr+modified_nu[k]+modified_nx[k], ptr:ptr+modified_nu[k]+modified_nx[k]] = RSQrqt[k][:modified_nu[k]+modified_nx[k], :modified_nu[k]+modified_nx[k]]
# 	rhs_preprocessed[ptr:ptr+modified_nu[k]+modified_nx[k]] = -RSQrqt[k][modified_nu[k]+modified_nx[k]:modified_nu[k]+modified_nx[k]+1, :modified_nu[k]+modified_nx[k]].T
    
# 	KKT_preprocessed[ptr:ptr+modified_nu[k]+modified_nx[k], ptr+modified_nu[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]] = Gg_eqt[k][:modified_nu[k]+modified_nx[k], :]
# 	KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k], ptr:ptr+modified_nu[k]+modified_nx[k]] = Gg_eqt[k][:modified_nu[k]+modified_nx[k], :].T
# 	rhs_preprocessed[ptr+modified_nu[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]] = -Gg_eqt[k][modified_nu[k]+modified_nx[k]:modified_nu[k]+modified_nx[k]+1, :modified_ng_eq[k]].T
      
# 	if k > 0:
# 		KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1], 
#             ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]] = BAbt[k-1][:modified_nu[k-1]+modified_nx[k-1]]
# 		KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k], 
# 			ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1]] = BAbt[k-1][:modified_nu[k-1]+modified_nx[k-1]].T
# 		rhs_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]] = -BAbt[k-1][modified_nu[k-1]+modified_nx[k-1]:modified_nu[k-1]+modified_nx[k-1]+1, :].T

# 		KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k],
#             ptr+modified_nu[k]:ptr+modified_nu[k]+modified_nx[k]] = -np.eye(modified_nx[k])
# 		KKT_preprocessed[ptr+modified_nu[k]:ptr+modified_nu[k]+modified_nx[k],
#       		ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]] = -np.eye(modified_nx[k])
# 		KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1], 
#             ptr+modified_nu[k]:ptr+modified_nu[k]+modified_nx[k]] = FuFx[k-1][:modified_nx[k-1]+modified_nu[k-1], :modified_nx[k]]
# 		KKT_preprocessed[ptr+modified_nu[k]:ptr+modified_nu[k]+modified_nx[k], 
#       		ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1]] = FuFx[k-1][:modified_nx[k-1]+modified_nu[k-1], :modified_nx[k]].T
# 		KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1], 
#             ptr:ptr+modified_nu[k]] = GuGx[k-1][:modified_nx[k-1]+modified_nu[k-1], :modified_nu[k]]
# 		KKT_preprocessed[ptr:ptr+modified_nu[k], 
#       		ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1]] = GuGx[k-1][:modified_nx[k-1]+modified_nu[k-1], :modified_nu[k]].T
# 		ptr += modified_nu[k] + modified_nx[k] + modified_ng_eq[k] + modified_nx[k]
# # print(KKT_preprocessed)
# solution_preprocessed = np.linalg.solve(KKT_preprocessed, rhs_preprocessed)
# # print(rhs_preprocessed)
# print(f"solution of preprocessed system:\n{solution_preprocessed}")

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

    Dl = Pl[k].T @ L[k] @ np.block([[-U1, np.zeros((r[k],nx[k+1]-r[k]))], [np.zeros((nx[k+1]-r[k],r[k])), np.eye(nx[k+1]-r[k])]])
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
    if k == 1: print(f"Starting with FuFx expected:\n{FuFx_expected[k]}")
    temp[:, nu[k+1]:] = FuFx_expected[k] @ Dr_inv
    GuGx_expected[k] = np.block([temp[:, :nu[k+1]], temp[:, nu[k+1]+modified_nx[k+1]:nu[k+1]+nx[k+1]]])
    FuFx_expected[k] = temp[:, nu[k+1]:nu[k+1]+modified_nx[k+1]]
    if k == 1: print(f"after scaling:\n{temp}")

    RSQrqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ RSQrqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    RSQrqt_expected[k+1][:,nu[k+1]:] = RSQrqt_expected[k+1][:,nu[k+1]:] @ np.linalg.inv(Dr)
    RSQrqt_expected[k+1] = Wp @ RSQrqt_expected[k+1] @ W

    Gg_eqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ Gg_eqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    Gg_eqt_expected[k+1] = Wp @ Gg_eqt_expected[k+1]
    Gg_ineqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ Gg_ineqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    Gg_ineqt_expected[k+1] = Wp @ Gg_ineqt_expected[k+1]

    if k < K-2:
        BAbt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ BAbt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
        if k == 0: print(f"FuFx before scaling:\n{FuFx_expected[k+1]}")
        FuFx_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ FuFx_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1],:]
        if k == 0: print(f"FuFx after scaling:\n{FuFx_expected[k+1]}")

        FuFx_expected[k+1] = W @ FuFx_expected[k+1]
        if k == 0: print(f"FuFx after W:\n{FuFx_expected[k+1]}")
        BAbt_expected[k+1] = Wp @ BAbt_expected[k+1]

    Gg_eqt_expected[k] = np.block([Gg_eqt_expected[k], BAbt_expected[k][:, r[k]:]])
    BAbt_expected[k] = BAbt_expected[k][:, :r[k]]

# check pre-processing results
for k in range(K):
    if k < K-1:
        print(f"BAbt error: {np.linalg.norm(BAbt[k] - BAbt_expected[k])}")
        print(f"GuGx error: {np.linalg.norm(GuGx[k] - GuGx_expected[k])}")
        print(f"GuGx original\n{GuGx_original[k]}")
        print(f"GuGx expected\n{GuGx_expected[k]}")
        print(f"GuGx:\n{GuGx[k]}")
        print(f"FuFx original\n{FuFx_original[k]}")
        print(f"FuFx expected\n{FuFx_expected[k]}")
        print(f"FuFx:\n{FuFx[k]}")
        print(f"FuFx error: {np.linalg.norm(FuFx[k] - FuFx_expected[k])}")
        print(f"RSQrqt error: {np.linalg.norm(RSQrqt[k] - RSQrqt_expected[k])}")

    print(f"Gg_eqt error: {np.linalg.norm(Gg_eqt[k] - Gg_eqt_expected[k])}")
    print(f"Gg_ineqt error: {np.linalg.norm(Gg_ineqt[k] - Gg_ineqt_expected[k])}")

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
        x[primal_x_offset+nu[k]:primal_x_offset+nu[k]+nx[k]] = Dr_inv_list[k-1] @ x[primal_x_offset+nu[k]:primal_x_offset+nu[k]+nx[k]] 
    primal_x_offset += nu[k] + nx[k]

# check post-processing results
print("expected x:", x)

original_eq_path_offset = 0
original_eq_dyn_offset = 0
new_eq_path_offset = 0
new_eq_dyn_offset = 0
total_nb_eq_path = sum(ng_eq)
total_nb_dyn_eq = sum(nx[1:])
original_eq_path = eq_mult[:total_nb_eq_path].copy()
original_eq_dyn = eq_mult[total_nb_eq_path:total_nb_eq_path+total_nb_dyn_eq].copy()
new_eq_path = np.zeros_like(original_eq_path)
new_eq_dyn = np.zeros_like(original_eq_dyn)
for k in range(K):
    if k < K-1:
        lmbd = original_eq_path[original_eq_path_offset:original_eq_path_offset+modified_ng_eq[k]].copy()
        pi = original_eq_dyn[original_eq_dyn_offset:original_eq_dyn_offset+modified_nx[k+1]].copy()

        true_pi = np.block([pi, lmbd[r[k]:]])
        lmbd = lmbd[:r[k]].copy()

        true_pi = Dl_inv_list[k].T @ true_pi

        new_eq_path[new_eq_path_offset:new_eq_path_offset+ng_eq[k]] = lmbd
        new_eq_dyn[new_eq_dyn_offset:new_eq_dyn_offset+nx[k+1]] = true_pi

        original_eq_path_offset += modified_ng_eq[k]
        original_eq_dyn_offset += modified_nx[k+1]
        new_eq_path_offset += ng_eq[k]
        new_eq_dyn_offset += nx[k+1]


# eq_mult_offset = 0
# for k in range(K):
#     if K < K-1:
#         pi = eq_mult[eq_mult_offset:eq_mult_offset+modified_nx[k+1]].copy()
#         lmbd = eq_mult[eq_mult_offset+modified_nx[k+1]:eq_mult_offset+modified_nx[k+1]+modified_ng_eq[k]].copy()

#         true_pi = np.block([pi, lmbd[ng_eq[k]:]])
#         lmbd = lmbd[:ng_eq[k]].copy()

#         true_pi = Dl_inv_list[k].T @ true_pi

#         eq_mult[eq_mult_offset:eq_mult_offset+nx[k+1]] = true_pi
#         eq_mult[eq_mult_offset+nx[k+1]:eq_mult_offset+nx[k+1]+ng_eq[k]] = lmbd

#         eq_mult_offset += modified_nx[k+1] + modified_ng_eq[k]
print("expected eq_mult:", eq_mult)

# # post-process primals computed earlier
# ptr = 0
# for k in range(K-1, -1, -1):
#     if k > 0:		
#         uk = solution_preprocessed[ptr:ptr+nu[k]].copy()
#         sk = solution_preprocessed[ptr+nu[k]:ptr+modified_nu[k]].copy()
#         xk = solution_preprocessed[ptr+modified_nu[k]:ptr+modified_nu[k]+modified_nx[k]].copy()
#         solution_preprocessed[ptr+nu[k]:ptr + nu[k] + r[k-1]] = xk
#         solution_preprocessed[ptr+nu[k]+r[k-1]:ptr+nu[k]+nx[k]] = sk
#         solution_preprocessed[ptr:ptr+nx[k]+nu[k]] = np.array([Dr_inv_list[k-1] @ solution_preprocessed[ptr:ptr+nx[k]+nu[k]]]).T
#     ptr += nu[k] + nx[k] + ng_eq[k] + nx[k]
# print("expected solution:\n", solution_preprocessed)


# ### manual transformation of KKT system
# print(f"KKT before manual transformation:\n{KKT}")
# KKT2 = KKT.copy()
# rhs2 = rhs.copy()
# ptr = KKT.shape[0] - ng_eq[0] - nx[0] - nu[0] - 1
# for k in range(K):
#     if k > 0:
#         # rows
#         KKT2[ptr+nu[k]:ptr+nu[k]+nx[k], :] = Dr_inv_list[k-1].T @ KKT2[ptr+nu[k]:ptr+nu[k]+nx[k], :]
#         rhs2[ptr+nu[k]:ptr+nu[k]+nx[k]] = Dr_inv_list[k-1].T @ rhs2[ptr+nu[k]:ptr+nu[k]+nx[k]]

#         KKT2[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k], :] = Dl_inv_list[k-1] @ KKT2[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k], :]
#         rhs2[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = Dl_inv_list[k-1] @ rhs2[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]]

#         # cols
#         KKT2[:, ptr+nu[k]:ptr+nu[k]+nx[k]] = KKT2[:, ptr+nu[k]:ptr+nu[k]+nx[k]] @ Dr_inv_list[k-1]

#         KKT2[:, ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = KKT2[:, ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] @ Dl_inv_list[k-1].T
#     ptr = ptr - (nu[k] + nx[k] + ng_eq[k] + nx[k])

# print(f"KKT after manual transformation:")
# for i in range(KKT2.shape[0]):
#     for j in range(KKT2.shape[1]):
#         print(f"{KKT2[i,j]:10.4f} ", end="")
#     print()

# solution2 = np.linalg.solve(KKT2, rhs2)
# print(f"solution of manually transformed system:\n{solution2}")

# KKT3 = KKT2.copy()
# rhs3 = rhs2.copy()
# KKT3 = np.block([KKT2[:,1:2], KKT2[:,0:1], KKT2[:,2:3], KKT2[:,4:6], KKT2[:,3:4]])
# KKT3 = np.block([[KKT3[1,:]], [KKT3[0,:]], [KKT3[2,:]], [KKT3[4:6,:]], [KKT3[3,:]]])
# rhs3 = np.block([[rhs2[1,:]], [rhs2[0,:]], [rhs2[2,:]], [rhs2[4:6,:]], [rhs2[3,:]]])
# print(f"KKT3:")
# for i in range(KKT3.shape[0]):
#     for j in range(KKT3.shape[1]):
#         print(f"{KKT3[i,j]:10.4f} ", end="")
#     print()
# solution3 = np.linalg.solve(KKT3, rhs3)
# print(f"solution of KKT3:\n{solution3}")

# print(f"KKT preprocessed:")
# for i in range(KKT_preprocessed.shape[0]):
#     for j in range(KKT_preprocessed.shape[1]):
#         print(f"{KKT_preprocessed[i,j]:10.4f} ", end="")
#     print()


