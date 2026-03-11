import numpy as np

GuGx_hat = np.array([
	[0.835423, 0.000000 ],
	])
Llt = np.array([
	[0.486082, 0.000000 ],
	[1.051673, 0.000000 ],
	[4.114528, 0.000000 ],
	[0.000000, 0.000000 ],
	[0.000000, 0.000000 ],
	])
nunxm1 = 1
nu = 1
rho_k = 0
lmbd = Llt[:nu-rho_k, :nu-rho_k]
ukxk = np.transpose(np.array([[-1.184461 ]]))
uk_before = np.transpose(np.array([[-6.945410 ]]))
uk_after = np.transpose(np.array([[-4.909694 ]]))

uk_expected = uk_before - np.linalg.inv(lmbd) @ GuGx_hat[:, rho_k:nu].T @ ukxk
print(f"Check uk update: {np.linalg.norm(uk_after - uk_expected)}")