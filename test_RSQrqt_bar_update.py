import numpy as np
RSQrq_hat = np.array([
	[0.236276, 0.5112, 0, 0 ],
	[0.5112, 1.10667, 0, 0 ],
	[2, 1, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	])
Llt = np.array([
	[0.486082, 0 ],
	[1.05167, 0 ],
	[4.11453, 0 ],
	[0, 0 ],
	[0, 0 ],
	])
GuGx_tilde = np.array([
	[0.406084, 0 ],
	])
#-----------------------------------
#testing RSQrqt_bar update (also copy RSQrqt_hat, Llt and GuGxtilde definition)
nx = 1
nu = 1
nunxm1 = 1
rank_k = 0
RSQrqt_underbar_before = np.array([
	[0.736071 ],
	[0 ],
	])
RSQrqt_tilde = np.array([
	[0.236276, 0.5112, 0, 0 ],
	[0.5112, 1.10667, 0, 0 ],
	[2, 1, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	])
v_r_tilde = np.array([
	[4.11453, 0, 0, 0 ],
	])
GuGx_hat = np.array([
	[0.835423, 0 ],
	])
RSQrqt_underbar_intermediate = np.array([
	[0.0381394 ],
	[0 ],
	])
temp = np.array([
	[-3.43737 ],
	])
RSQrqt_underbar_after = np.array([
	[0.0381394 ],
	[-3.43737 ],
	])
FuFx_underbar_before = np.array([
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	])
RSQrq_hat_curr_p = np.array([
	[0.236276, 0.5112, 0, 0 ],
	[0.5112, 1.10667, 0, 0 ],
	[2, 1, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	])
FuFx_underbar_after = np.array([
	[-0.878592, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
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

print(np.linalg.inv(lmbd).T @ RSQrq_hat[:nu, nu:nu+nx])
FuFx_expected = FuFx_underbar_before - GuGx_hat[:, rank_k:] @ np.linalg.inv(lmbd).T @ RSQrq_hat[:nu, nu:nu+nx]
print(f"FuFx before:\n{FuFx_underbar_before}")
print(f"FuFx expected:\n{FuFx_expected}")
print(f"FuFx after:\n{FuFx_underbar_after}")
print(f"FuFx error: {np.linalg.norm(FuFx_underbar_after - FuFx_expected)}")