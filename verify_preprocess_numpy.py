import numpy as np

K = 2
nu = [0, 0]
nx = [1, 1]
ng_ineq = [0, 0]
ng_eq = [0, 0]
r = [0]
modified_K = 2
modified_nu = [0, 1]
modified_nx = [1, 0]
modified_ng_ineq = [0, 0]
modified_ng_eq = [1, 0]
RSQrqt = [
np.array([
	[0.712785 ],
	[0 ],
	])
,
np.array([
	[1.83607 ],
	[1 ],
	])
]
RSQrqt_original = [
np.array([
	[0.712785 ],
	[0 ],
	])
,
np.array([
	[0.736071 ],
	[1 ],
	])
]
FuFx = [
np.array([
	[],
	])
]
FuFx_original = [
np.array([
	[0.0623564 ],
	])
]
GuGx = [
np.array([
	[0.0623564 ],
	])
]
GuGx_original = [
np.array([
	[],
	])
]
Gg_eqt = [
np.array([
	[0.592845 ],
	[0 ],
	])
,
np.array([
	[],
	[],
	])
]
Gg_eqt_original = [
np.array([
	[],
	[],
	])
,
np.array([
	[],
	[],
	])
]
Gg_ineqt = [
np.array([
	[],
	[],
	])
,
np.array([
	[],
	[],
	])
]
Gg_ineqt_original = [
np.array([
	[],
	[],
	])
,
np.array([
	[],
	[],
	])
]
BAbt = [
np.array([
	[],
	[],
	])
]
BAbt_original = [
np.array([
	[0.592845 ],
	[0 ],
	])
]
Jt = [
np.array([
	[0 ],
	])
]
L = [
np.array([[1],
])
]
U = [
np.array([[0],
])
]
Pl = [
np.array([
	[1 ],
	])
]
Pr = [
np.array([
	[1 ],
	])
]
x = np.array([0, -0.544641])
eq_mult = np.array([0])

BAbt_expected = BAbt_original.copy()
GuGx_expected = GuGx_original.copy()
FuFx_expected = FuFx_original.copy()
RSQrqt_expected = RSQrqt_original.copy()
Gg_eqt_expected = Gg_eqt_original.copy()
Gg_ineqt_expected = Gg_ineqt_original.copy()

# pre-processing code
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
    
    temp = np.zeros((nu[k+1] + nx[k+1], nu[k] + nx[k]))
    temp[nu[k+1]:, :] = Dr_inv.T @ FuFx_expected[k].T
    temp = W @ temp
    temp = temp.T
    GuGx_expected[k] = temp[:, :nu[k+1]]
    FuFx_expected[k] = temp[:, nu[k+1]:]

    RSQrqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ RSQrqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    RSQrqt_expected[k+1][:,nu[k+1]:] = RSQrqt_expected[k+1][:,nu[k+1]:] @ np.linalg.inv(Dr)
    RSQrqt_expected[k+1] = Wp @ RSQrqt_expected[k+1] @ W

    Gg_eqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ Gg_eqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    Gg_eqt_expected[k+1] = Wp @ Gg_eqt_expected[k+1]
    Gg_ineqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ Gg_ineqt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
    Gg_ineqt_expected[k+1] = Wp @ Gg_ineqt_expected[k+1]

    if k < K-2:
        BAbt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :] = np.linalg.inv(Dr).T @ BAbt_expected[k+1][nu[k+1]:nu[k+1]+nx[k+1], :]
        FuFx_expected[k+1][:,nu[k+1]:nu[k+1]+nx[k+1]] = FuFx_expected[k+1][:, nu[k+1]:nu[k+1]+nx[k+1]] @ np.linalg.inv(Dr)
        BAbt_expected[k+1] = Wp @ BAbt_expected[k+1]

        Gg_eqt_expected[k] = np.block([Gg_eqt_expected, BAbt_expected[k][:r[k], :]])
        BAbt_expected[k] = BAbt_expected[k][:r[k], :]

# check pre-processing results
for k in range(K):
    if k < K-1:
        print(f"BAbt error: {np.linalg.norm(BAbt[k] - BAbt_expected[k])}")
        try:
            print(f"GuGx error: {np.linalg.norm(GuGx[k] - GuGx_expected[k])}")
        except:
            print("GuGx error: unable to compute")
        print(f"FuFx error: {np.linalg.norm(FuFx[k] - FuFx_expected[k])}")
        print(f"RSQrqt error: {np.linalg.norm(RSQrqt[k] - RSQrqt_expected[k])}")

    print(f"Gg_eqt error: {np.linalg.norm(Gg_eqt[k] - Gg_eqt_expected[k])}")
    print(f"Gg_ineqt error: {np.linalg.norm(Gg_ineqt[k] - Gg_ineqt_expected[k])}")

# post-processing code
primal_x_offset = 0
for k in range(K):
    if k > 0:
        print(x[primal_x_offset:primal_x_offset+nx[k]+nu[k]])
        ukxk = x[primal_x_offset:primal_x_offset+nu[k]+nx[k]]
        uk = ukxk[:nu[k]]
        sk = ukxk[nu[k]:modified_nu[k]]
        xk = ukxk[modified_nu[k]:]
        x[primal_x_offset+nu[k]:primal_x_offset + nu[k] + r[k-1]] = ukxk[modified_nu[k]:]
        x[primal_x_offset+nu[k]+r[k-1]:primal_x_offset+nu[k]+nx[k]] = ukxk[:modified_nu[k]]
        print(x[primal_x_offset:primal_x_offset+nx[k]+nu[k]])

        x[primal_x_offset:primal_x_offset+nx[k]+nu[k]] = Dr_inv_list[k-1] @ x[primal_x_offset:primal_x_offset+nx[k]+nu[k]] 

    primal_x_offset += nu[k] + nx[k]

# check post-processing results
print("expected x:", x)