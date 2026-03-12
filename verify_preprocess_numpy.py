import numpy as np

K = 2
nu = [1, 0]
nx = [1, 2]
ng_ineq = [0, 0]
ng_eq = [0, 0]
r = [1]
modified_K = 2
modified_nu = [1, 1]
modified_nx = [1, 1]
modified_ng_ineq = [0, 0]
modified_ng_eq = [1, 0]
RSQrqt = [
np.array([
	[0.302505, 0.450711 ],
	[0.450711, 0.889996 ],
	[0, 1 ],
	])
,
np.array([
	[0.0267101, 0.148445 ],
	[0.148445, 0.853308 ],
	[1.76715, 2 ],
	])
]
RSQrqt_original = [
np.array([
	[0.302505, 0.450711 ],
	[0.450711, 0.889996 ],
	[0, 1 ],
	])
,
np.array([
	[0.853308, 0.674447 ],
	[0.674447, 0.533963 ],
	[2, 3 ],
	])
]
FuFx = [
np.array([
	[0.0473608 ],
	[0.0520477 ],
	])
]
FuFx_original = [
np.array([
	[0.0473608, 0.0800911 ],
	[0.0520477, 0.067888 ],
	])
]
GuGx = [
np.array([
	[0.0508966 ],
	[0.067888 ],
	])
]
GuGx_original = [
np.array([
	[],
	[],
	])
]
Gg_eqt = [
np.array([
	[0.731264 ],
	[0.683719 ],
	[1 ],
	])
,
np.array([
	[],
	[],
	[],
	])
]
Gg_eqt_original = [
np.array([
	[],
	[],
	[],
	])
,
np.array([
	[],
	[],
	[],
	])
]
Gg_ineqt = [
np.array([
	[],
	[],
	[],
	])
,
np.array([
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
	])
,
np.array([
	[],
	[],
	[],
	])
]
BAbt = [
np.array([
	[-3.19538 ],
	[-4.62425 ],
	[-0 ],
	])
]
BAbt_original = [
np.array([
	[0.592845, 0.844266 ],
	[0.857946, 0.847252 ],
	[0, 1 ],
	])
]
Jt = [
np.array([
	[0.185532, 0.0353642 ],
	[0.114367, 0.0217994 ],
	])
]
L = [
np.array([[1, 0],
[0.19061, 1],
])
]
U = [
np.array([[0.185532, 0.114367],
[0, 0],
])
]
Pl = [
np.array([
	[1, 0 ],
	[0, 1 ],
	])
]
Pr = [
np.array([
	[1, 0 ],
	[0, 1 ],
	])
]
x = np.array([13.1188, -15.4936, -216.99, 29.7269])
eq_mult = np.array([-4.67886, -5.02984])

BAbt_expected = BAbt_original.copy()
GuGx_expected = GuGx_original.copy()
FuFx_expected = FuFx_original.copy()
RSQrqt_expected = RSQrqt_original.copy()
Gg_eqt_expected = Gg_eqt_original.copy()
Gg_ineqt_expected = Gg_ineqt_original.copy()

number_of_primal_vars = sum(nu) + sum(nx)
number_of_eqs = sum(ng_eq) + sum(nx[1:])
KKT = np.zeros((number_of_primal_vars + number_of_eqs, number_of_primal_vars + number_of_eqs))
rhs = np.zeros((number_of_primal_vars + number_of_eqs, 1))
ptr = 0
for k in range(K-1, -1, -1):
	KKT[ptr:ptr+nu[k]+nx[k], ptr:ptr+nu[k]+nx[k]] = RSQrqt_original[k][:nu[k]+nx[k], :nu[k]+nx[k]]
	rhs[ptr:ptr+nu[k]+nx[k]] = -RSQrqt_original[k][nu[k]+nx[k]:nu[k]+nx[k]+1, :nu[k]+nx[k]].T
    
	KKT[ptr:ptr+nu[k]+nx[k], ptr:ptr+ng_eq[k]] = Gg_eqt_original[k][:nu[k]+nx[k], :]
	KKT[ptr:ptr+ng_eq[k], ptr:ptr+nu[k]+nx[k]] = Gg_eqt_original[k][:nu[k]+nx[k], :].T
	rhs[ptr+nu[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]] = -Gg_eqt_original[k][nu[k]+nx[k]:nu[k]+nx[k]+1, :ng_eq[k]].T
      
	if k > 0:
		KKT[ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1], ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = BAbt_original[k-1][:nu[k-1]+nx[k-1]]
		KKT[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k], ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1]] = BAbt_original[k-1][:nu[k-1]+nx[k-1]].T
		rhs[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = -BAbt_original[k-1][nu[k-1]+nx[k-1]:nu[k-1]+nx[k-1]+1, :].T

		KKT[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k], ptr+nu[k]:ptr+nu[k]+nx[k]] = Jt[k-1][:nx[k], :nx[k]].T
		KKT[ptr+nu[k]:ptr+nu[k]+nx[k], ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = Jt[k-1][:nx[k], :nx[k]]
		KKT[ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1], ptr+nu[k]:ptr+nu[k]+nx[k]] = FuFx_original[k-1][:nx[k-1]+nu[k-1], :nx[k]]
		KKT[ptr+nu[k]:ptr+nu[k]+nx[k], ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1]] = FuFx_original[k-1][:nx[k-1]+nu[k-1], :nx[k]].T
		ptr += nu[k] + nx[k] + ng_eq[k] + nx[k]
# print(KKT)
solution = np.linalg.solve(KKT, rhs)
# print(rhs)
print(f"true solution:\n{solution}")

number_of_primal_vars = sum(nu) + sum(nx)
number_of_eqs = sum(ng_eq) + sum(nx[1:])
KKT_preprocessed = np.zeros((number_of_primal_vars + number_of_eqs, number_of_primal_vars + number_of_eqs))
rhs_preprocessed = np.zeros((number_of_primal_vars + number_of_eqs, 1))
ptr = 0
for k in range(K-1, -1, -1):
	KKT_preprocessed[ptr:ptr+modified_nu[k]+modified_nx[k], ptr:ptr+modified_nu[k]+modified_nx[k]] = RSQrqt[k][:modified_nu[k]+modified_nx[k], :modified_nu[k]+modified_nx[k]]
	rhs_preprocessed[ptr:ptr+modified_nu[k]+modified_nx[k]] = -RSQrqt[k][modified_nu[k]+modified_nx[k]:modified_nu[k]+modified_nx[k]+1, :modified_nu[k]+modified_nx[k]].T
    
	KKT_preprocessed[ptr:ptr+modified_nu[k]+modified_nx[k], ptr+modified_nu[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]] = Gg_eqt[k][:modified_nu[k]+modified_nx[k], :]
	KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k], ptr:ptr+modified_nu[k]+modified_nx[k]] = Gg_eqt[k][:modified_nu[k]+modified_nx[k], :].T
	rhs_preprocessed[ptr+modified_nu[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]] = -Gg_eqt[k][modified_nu[k]+modified_nx[k]:modified_nu[k]+modified_nx[k]+1, :modified_ng_eq[k]].T
      
	if k > 0:
		KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1], 
            ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]] = BAbt[k-1][:modified_nu[k-1]+modified_nx[k-1]]
		KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k], 
			ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1]] = BAbt[k-1][:modified_nu[k-1]+modified_nx[k-1]].T
		rhs_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]] = -BAbt[k-1][modified_nu[k-1]+modified_nx[k-1]:modified_nu[k-1]+modified_nx[k-1]+1, :].T

		KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k],
            ptr+modified_nu[k]:ptr+modified_nu[k]+modified_nx[k]] = -np.eye(modified_nx[k])
		KKT_preprocessed[ptr+modified_nu[k]:ptr+modified_nu[k]+modified_nx[k],
      		ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]] = -np.eye(modified_nx[k])
		KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1], 
            ptr+modified_nu[k]:ptr+modified_nu[k]+modified_nx[k]] = FuFx[k-1][:modified_nx[k-1]+modified_nu[k-1], :modified_nx[k]]
		KKT_preprocessed[ptr+modified_nu[k]:ptr+modified_nu[k]+modified_nx[k], 
      		ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1]] = FuFx[k-1][:modified_nx[k-1]+modified_nu[k-1], :modified_nx[k]].T
		KKT_preprocessed[ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1], 
            ptr:ptr+modified_nu[k]] = GuGx[k-1][:modified_nx[k-1]+modified_nu[k-1], :modified_nu[k]]
		KKT_preprocessed[ptr:ptr+modified_nu[k], 
      		ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]:ptr+modified_nu[k]+modified_nx[k]+modified_ng_eq[k]+modified_nx[k]+modified_nu[k-1]+modified_nx[k-1]] = GuGx[k-1][:modified_nx[k-1]+modified_nu[k-1], :modified_nu[k]].T
		ptr += modified_nu[k] + modified_nx[k] + modified_ng_eq[k] + modified_nx[k]
# print(KKT_preprocessed)
solution_preprocessed = np.linalg.solve(KKT_preprocessed, rhs_preprocessed)
# print(rhs_preprocessed)
print(f"solution of preprocessed system:\n{solution_preprocessed}")


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
    print(f"norm: {norm}")

    # construct W
    W = np.block([[np.eye(nu[k+1]), np.zeros((nu[k+1], nx[k+1]))],
                  [np.zeros((nx[k+1] - r[k], nu[k+1] + r[k])), np.eye(nx[k+1] - r[k])],
                  [np.zeros((r[k], nu[k+1])), np.eye(r[k]), np.zeros((r[k], nx[k+1] - r[k]))]])
    Wp = np.block([[W, np.zeros((W.shape[0], 1))],
                   [np.zeros((1, W.shape[1])), 1]])
    
    BAbt_expected[k] = BAbt_expected[k] @ Dl_inv.T
    BAbt_expected[k] = BAbt_expected[k][:, :modified_nx[k+1]]
    
    temp = np.zeros((nu[k] + nx[k], nu[k+1] + nx[k+1]))
    temp[:, nu[k+1]:] = FuFx_expected[k] @ Dr_inv
    print(temp)
    GuGx_expected[k] = np.block([temp[:, :nu[k+1]], temp[:, nu[k+1]+modified_nx[k+1]:nx[k+1]]])
    FuFx_expected[k] = temp[:, nu[k+1]:nu[k+1]+modified_nx[k+1]]
    # FuFx_expected[k] = FuFx_expected[k][:, :modified_nx[k+1]]

    RSQrqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ RSQrqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    RSQrqt_expected[k+1][:,nu[k+1]:] = RSQrqt_expected[k+1][:,nu[k+1]:] @ np.linalg.inv(Dr)
    RSQrqt_expected[k+1] = Wp @ RSQrqt_expected[k+1] @ W

    Gg_eqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ Gg_eqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    Gg_eqt_expected[k+1] = Wp @ Gg_eqt_expected[k+1]
    Gg_ineqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ Gg_ineqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    Gg_ineqt_expected[k+1] = Wp @ Gg_ineqt_expected[k+1]

    if k < K-2:
        BAbt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ BAbt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
        FuFx_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = FuFx_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1],:] @ np.linalg.inv(Dr)
        BAbt_expected[k+1] = Wp @ BAbt_expected[k+1]

        Gg_eqt_expected[k] = np.block([Gg_eqt_expected, BAbt_expected[k][:r[k], :]])
        BAbt_expected[k] = BAbt_expected[k][:r[k], :]

# check pre-processing results
for k in range(K):
	if k < K-1:
		print(f"BAbt error: {np.linalg.norm(BAbt[k] - BAbt_expected[k])}")
		print(f"GuGx error: {np.linalg.norm(GuGx[k] - GuGx_expected[k])}")
		print(f"GuGx original\n{GuGx_original[k]}")
		print(f"GuGx expected\n{GuGx_expected[k]}")
		print(f"GuGx:\n{GuGx[k]}")
		print(f"FuFx error: {np.linalg.norm(FuFx[k] - FuFx_expected[k])}")
        # print(f"FuFx original\n{FuFx_original[k]}")
        # print(f"FuFx expected\n{FuFx_expected[k]}")
        # print(f"FuFx:\n{FuFx[k]}")
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
		x[primal_x_offset:primal_x_offset+nx[k]+nu[k]] = Dr_inv_list[k-1] @ x[primal_x_offset:primal_x_offset+nx[k]+nu[k]] 
	primal_x_offset += nu[k] + nx[k]

# check post-processing results
print("expected x:", x)


# post-process primals computed earlier
ptr = 0
for k in range(K-1, -1, -1):
	if k > 0:		
		uk = solution_preprocessed[ptr:ptr+nu[k]].copy()
		sk = solution_preprocessed[ptr+nu[k]:ptr+modified_nu[k]].copy()
		xk = solution_preprocessed[ptr+modified_nu[k]:ptr+modified_nu[k]+modified_nx[k]].copy()
		solution_preprocessed[ptr+nu[k]:ptr + nu[k] + r[k-1]] = xk
		solution_preprocessed[ptr+nu[k]+r[k-1]:ptr+nu[k]+nx[k]] = sk
		solution_preprocessed[ptr:ptr+nx[k]+nu[k]] = np.array([Dr_inv_list[k-1] @ solution_preprocessed[ptr:ptr+nx[k]+nu[k]]]).T
	ptr += nu[k] + nx[k] + ng_eq[k] + nx[k]
print("expected solution:\n", solution_preprocessed)


### manual transformation of KKT system
print(f"KKT before manual transformation:\n{KKT}")
KKT2 = KKT.copy()
rhs2 = rhs.copy()
ptr = KKT.shape[0] - ng_eq[0] - nx[0] - nu[0] - 1
for k in range(K):
	if k > 0:
		# rows
		KKT2[ptr+nu[k]:ptr+nu[k]+nx[k], :] = Dr_inv_list[k-1].T @ KKT2[ptr+nu[k]:ptr+nu[k]+nx[k], :]
		rhs2[ptr+nu[k]:ptr+nu[k]+nx[k]] = Dr_inv_list[k-1].T @ rhs2[ptr+nu[k]:ptr+nu[k]+nx[k]]

		KKT2[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k], :] = Dl_inv_list[k-1] @ KKT2[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k], :]
		rhs2[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = Dl_inv_list[k-1] @ rhs2[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]]

		# cols
		KKT2[:, ptr+nu[k]:ptr+nu[k]+nx[k]] = KKT2[:, ptr+nu[k]:ptr+nu[k]+nx[k]] @ Dr_inv_list[k-1]

		KKT2[:, ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = KKT2[:, ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] @ Dl_inv_list[k-1].T
	ptr = ptr - (nu[k] + nx[k] + ng_eq[k] + nx[k])

print(f"KKT after manual transformation:")
for i in range(KKT2.shape[0]):
	for j in range(KKT2.shape[1]):
		print(f"{KKT2[i,j]:10.4f} ", end="")
	print()

solution2 = np.linalg.solve(KKT2, rhs2)
print(f"solution of manually transformed system:\n{solution2}")

KKT3 = KKT2.copy()
rhs3 = rhs2.copy()
KKT3 = np.block([KKT2[:,1:2], KKT2[:,0:1], KKT2[:,2:3], KKT2[:,4:6], KKT2[:,3:4]])
KKT3 = np.block([[KKT3[1,:]], [KKT3[0,:]], [KKT3[2,:]], [KKT3[4:6,:]], [KKT3[3,:]]])
rhs3 = np.block([[rhs2[1,:]], [rhs2[0,:]], [rhs2[2,:]], [rhs2[4:6,:]], [rhs2[3,:]]])
print(f"KKT3:")
for i in range(KKT3.shape[0]):
	for j in range(KKT3.shape[1]):
		print(f"{KKT3[i,j]:10.4f} ", end="")
	print()
solution3 = np.linalg.solve(KKT3, rhs3)
print(f"solution of KKT3:\n{solution3}")

print(f"KKT preprocessed:")
for i in range(KKT_preprocessed.shape[0]):
	for j in range(KKT_preprocessed.shape[1]):
		print(f"{KKT_preprocessed[i,j]:10.4f} ", end="")
	print()


