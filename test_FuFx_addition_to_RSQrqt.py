import numpy as np

BAbt = np.array([
	[0.592845, 0.844266 ],
	[0, 1 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	[0, 0 ],
	])
FuFx = np.array([
	[-0.427068, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	[0, 0, 0, 0 ],
	])
FuFx_hessian = np.array([
	[0, 0.406084, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
	])
RSQrqt = np.array([
	[0.0383681 ],
	[-5.40985 ],
	])
nu = 0
nx = 1
nx_next = 1
RSQrqt_intermediate = np.array([
	[-0.214817 ],
	[-5.40985 ],
	])
RSQrqt_after = np.array([
	[-0.468002 ],
	[-5.40985 ],
	])



RSQrqt = RSQrqt[:nu+nx+1, :nu+nx]
FuFx = FuFx[:nx+nu, :nx_next]
BAbt = BAbt[:nx+nu+1, :nx_next]

A = np.block([FuFx, BAbt[:nx+nu,:]])
B = np.block([[BAbt.T],
              [FuFx.T, np.zeros((nx_next, 1))]])

RSQrqt_intermediate_expected = RSQrqt + BAbt @ FuFx.T
print(np.linalg.norm(RSQrqt_intermediate - RSQrqt_intermediate_expected))
RSQrqt_final_expected = RSQrqt_intermediate + np.block([[FuFx @ BAbt[:nu+nx,:].T], [np.zeros((1, nu+nx))]])
print(np.linalg.norm(RSQrqt_after - RSQrqt_final_expected))
print(np.linalg.norm(RSQrqt_after - (RSQrqt + (A @ B).T)))
