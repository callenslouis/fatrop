import numpy as np
RSQrq_hat = np.array([
	[0.088527, 0.000000 ],
	[2.000000, 0.000000 ],
	[0.000000, 0.000000 ],
	])
Llt = np.array([
	[0.297535 ],
	[6.721907 ],
	[0.000000 ],
	])
GuGx_tilde = np.array([
	[0.238833 ],
	[0.406084 ],
	])
#-----------------------------------
#testing RSQrqt_bar update (also copy RSQrqt_hat, Llt and GuGxtilde definition)
nx = 0
nu = 1
nunxm1 = 2
rank_k = 0
RSQrqt_underbar_before = np.array([
	[1.453906, 0.860652 ],
	[0.860652, 0.536581 ],
	[0.000000, 1.000000 ],
	])
RSQrqt_tilde = np.array([
	[0.088527, 0.000000 ],
	[2.000000, 0.000000 ],
	[0.000000, 0.000000 ],
	])
v_r_tilde = np.array([
	[6.721907, 0.000000 ],
	])
GuGx_hat = np.array([
	[0.802705 ],
	[1.364831 ],
	])
RSQrqt_underbar_intermediate = np.array([
	[0.809571, -0.234905 ],
	[-0.234905, -1.326182 ],
	[0.000000, 1.000000 ],
	])
temp = np.array([
	[-5.395709, -9.174265 ],
	])
RSQrqt_underbar_after = np.array([
	[0.809571, -0.234905 ],
	[-0.234905, -1.326182 ],
	[-5.395709, -8.174265 ],
	])

lmbd = Llt[:nu, :nu]
print(f"Lambda should be \n{lmbd}")
for i in range(nu):
    for j in range(i+1,nu):
        lmbd[i,j] = 0
R_hat = RSQrq_hat[:nu, :nu]
print(f"Check cholesky decomposition: {np.linalg.norm(R_hat - lmbd @ lmbd.T)}")

GuGx_hat_expected = GuGx_tilde[:, rank_k:] @ np.linalg.inv(lmbd).T
print(f"GuGx_hat should be\n{GuGx_hat_expected}")
print(f"GuGx_hat after:\n{GuGx_hat}")
print(f"GuGx_hat error: {np.linalg.norm(GuGx_hat - GuGx_hat_expected)}")

# RSQrqt_intermediate = RSQrqt_underbar_before.copy()
# RSQrqt_intermediate[:nunxm1, :nunxm1] -= \
# 	(GuGx_tilde[:, rank_k:] @ np.linalg.inv(lmbd).T @ np.linalg.inv(lmbd) @ GuGx_tilde[:, rank_k:].T).T
# print(f"RSQrqt_intermediate should be\n{RSQrqt_intermediate}")

# print(f"v_r_tilde should be\n{np.linalg.inv(lmbd) @ RSQrqt_tilde[nu+nx:nu+nx+1, rank_k:nu].T}")

# print(f"additional term is:\n{GuGx_tilde[:, rank_k:] @ np.linalg.inv(lmbd).T @ np.linalg.inv(lmbd) @ RSQrqt_tilde[nu+nx:nu+nx+1, rank_k:nu]}")

RSQrqt_underbar_expected = RSQrqt_underbar_before - \
	(GuGx_tilde[:, rank_k:] @ np.linalg.inv(lmbd).T @ np.linalg.inv(lmbd) @ np.block([[GuGx_tilde[:, rank_k:].T, RSQrqt_tilde[nu+nx:nu+nx+1, rank_k:nu].T]])).T
# print(f"RSQrqt_underbar expected:\n{RSQrqt_underbar_expected}")
# print(f"RSQrqt_underbar after:\n{RSQrqt_underbar_after}")
print(f"RSQrqt error: {np.linalg.norm(RSQrqt_underbar_after - RSQrqt_underbar_expected)}")
