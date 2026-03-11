import numpy as np

GuGx_hat = np.array([
	[0.802705 ],
	[1.364831 ],
	])
nunxm1 = 2
nu = 1
rho_k = 0
ukxk = np.transpose(np.array([[-0.709036, 0.497886 ]]))
uk_before = np.transpose(np.array([[-22.592018 ]]))
uk_after = np.transpose(np.array([[-22.702401 ]]))

uk_expected = uk_before - GuGx_hat[:, rho_k].T @ ukxk
print(f"Check uk update: {np.linalg.norm(uk_after - uk_expected)}")