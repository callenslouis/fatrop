import numpy as np
RSQrq_hat = np.array([
	[1.32393, 0.546996, 0.681899, 0, 0, 0 ],
	[0.546996, 0.423284, -0.0392615, 0, 0, 0 ],
	[0.681899, -0.0392615, 1.07145, 0, 0, 0 ],
	[2.61878, 2.16723, 2, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0 ],
	])
Llt = np.array([
	[1.15062, 0, 0 ],
	[0.475391, 0.44417, 0 ],
	[0.592635, -0.722685, 0 ],
	[2.27597, 2.44333, 0 ],
	[0, 0, 0 ],
	[0, 0, 0 ],
	[0, 0, 0 ],
	])
GuGx_tilde = np.array([
	[0.00760866, -0.0540083, 0 ],
	[0.00878905, -0.00747166, 0 ],
	])
#-----------------------------------
#testing RSQrqt_bar update (also copy RSQrqt_hat, Llt and GuGxtilde definition)
nx = 1
nu = 2
nunxm1 = 2
rank_k = 0
RSQrqt_underbar_before = np.array([
	[0.853308, 0.674447 ],
	[0.674447, 0.533963 ],
	[0, 1 ],
	])
RSQrqt_tilde = np.array([
	[1.32393, 0.546996, 0.681899, 0, 0, 0 ],
	[0.546996, 0.423284, -0.0392615, 0, 0, 0 ],
	[0.681899, -0.0392615, 1.07145, 0, 0, 0 ],
	[2.61878, 2.16723, 2, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0 ],
	])
v_r_tilde = np.array([
	[2.27597, 2.44333, 0, 0, 0, 0 ],
	])
GuGx_hat = np.array([
	[0.00661264, -0.128671, 0 ],
	[0.00763851, -0.0249971, 0 ],
	])
RSQrqt_underbar_intermediate = np.array([
	[0.836708, 0.67118 ],
	[0.67118, 0.53328 ],
	[0, 1 ],
	])
temp = np.array([
	[0.299337, 0.0436912 ],
	])
RSQrqt_underbar_after = np.array([
	[0.836708, 0.67118 ],
	[0.67118, 0.53328 ],
	[0.299337, 1.04369 ],
	])
FuFx_underbar_before = np.array([
	[0.0736918, 0, 0 ],
	[0.0324141, 0, 0 ],
	])
Llt = np.array([
	[1.15062, 0, 0 ],
	[0.475391, 0.44417, 0 ],
	[0.592635, -0.722685, 0 ],
	[2.27597, 2.44333, 0 ],
	[0, 0, 0 ],
	[0, 0, 0 ],
	[0, 0, 0 ],
	])
FuFx_underbar_after = np.array([
	[0.362624, 0, 0 ],
	[0.0847798, 0, 0 ],
	])

RSQrq_hat = RSQrq_hat[:nu+nx+1, :nu+nx]
GuGx_tilde = GuGx_tilde[:nunxm1, :nu]
GuGx_hat = GuGx_hat[:nunxm1, :nu]
FuFx_underbar_before = FuFx_underbar_before[:nunxm1, :nx]
FuFx_underbar_after = FuFx_underbar_after[:nunxm1, :nx]

lmbd = Llt[:nu, :nu]
print(f"Lambda should be \n{lmbd}")
for i in range(nu):
    for j in range(i+1,nu):
        lmbd[i,j] = 0
R_hat = RSQrq_hat[:nu, :nu]
print(f"Check cholesky decomposition: {np.linalg.norm(R_hat - lmbd @ lmbd.T)}")

print(f"GuGx_tilde[:, rank_k:] is\n{GuGx_tilde[:, rank_k:]}")
print(f"np.linalg.inv(lmbd).T is\n{np.linalg.inv(lmbd).T}")
GuGx_hat_expected = GuGx_tilde[:, rank_k:] @ np.linalg.inv(lmbd).T
print(f"GuGx_hat should be\n{GuGx_hat_expected}")
print(f"GuGx_hat after:\n{GuGx_hat}")
print(f"GuGx_hat error: {np.linalg.norm(GuGx_hat - GuGx_hat_expected)}")

RSQrqt_underbar_expected = RSQrqt_underbar_before - \
	(GuGx_tilde[:, rank_k:] @ np.linalg.inv(lmbd).T @ np.linalg.inv(lmbd) @ np.block([[GuGx_tilde[:, rank_k:].T, RSQrqt_tilde[nu+nx:nu+nx+1, rank_k:nu].T]])).T
print(f"RSQrqt error: {np.linalg.norm(RSQrqt_underbar_after - RSQrqt_underbar_expected)}")

S_hat = RSQrq_hat[:nu, nu:nu+nx].T
L = np.linalg.inv(lmbd) @ S_hat.T
print(f"S_hat:\n{S_hat}")
print(f"L:\n{L}")
FuFx_expected = FuFx_underbar_before - GuGx_hat[:, rank_k:] @ L
print(f"FuFx before:\n{FuFx_underbar_before}")
print(f"FuFx expected:\n{FuFx_expected}")
print(f"FuFx after:\n{FuFx_underbar_after}")
print(f"FuFx error: {np.linalg.norm(FuFx_underbar_after - FuFx_expected)}")