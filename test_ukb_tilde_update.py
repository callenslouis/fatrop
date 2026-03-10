import numpy as np

GuGx_hat = np.array([
	[0.160541 ],
	[0.272966 ],
	])
nunxm1 = 2
nu = 1
rho_k = 0
ukxk = np.transpose(np.array([[0.968733, -0.680246 ]]))
uk_before = np.transpose(np.array([[-22.592018 ]]))
uk_after = np.transpose(np.array([[-22.561855 ]]))

uk_expected = uk_before - GuGx_hat[:, rho_k].T @ ukxk
print(f"Check uk update: {np.linalg.norm(uk_after - uk_expected)}")