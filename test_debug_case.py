import numpy as np

K = 2
nu = [0, 0]
nx = [2, 1]
ng_ineq = [0, 0]
ng_eq = [0, 0]
r = [0]
modified_K = 2
modified_nu = [0, 1]
modified_nx = [2, 0]
modified_ng_ineq = [0, 0]
modified_ng_eq = [1, 0]
RSQrqt = [
np.array([
	[1.453906, 0.860652 ],
	[0.860652, 0.536581 ],
	[0.000000, 1.000000 ],
	])
,
np.array([
	[0.088527 ],
	[2.000000 ],
	])
]
RSQrqt_original = [
np.array([
	[1.453906, 0.860652 ],
	[0.860652, 0.536581 ],
	[0.000000, 1.000000 ],
	])
,
np.array([
	[0.088527 ],
	[2.000000 ],
	])
]
FuFx = [
np.array([
	[],
	[],
	])
]
FuFx_original = [
np.array([
	[0.238833 ],
	[0.406084 ],
	])
]
GuGx = [
np.array([
	[0.238833 ],
	[0.406084 ],
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
	[0.592845 ],
	[0.844266 ],
	[0.000000 ],
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
	[],
	])
]
BAbt_original = [
np.array([
	[0.592845 ],
	[0.844266 ],
	[0.000000 ],
	])
]
Jt = [
np.array([
	[0.000000 ],
	])
]
L = [
np.array([[1.000000],
])
]
U = [
np.array([[0.000000],
])
]
Pl = [
np.array([
	[1.000000 ],
	])
]
Pr = [
np.array([
	[1.000000 ],
	])
]



R1 = RSQrqt[1][:modified_nu[1], :modified_nu[1]]
G0x = GuGx[0]
Q0 = RSQrqt[0][modified_nu[0]:modified_nu[0]+modified_nx[0], modified_nu[0]:modified_nu[0]+modified_nx[0]]
H0u = Gg_eqt[0][:modified_nu[0]+modified_nx[0],:].T

r1 = RSQrqt[1][modified_nu[1]+modified_nx[1]:, :modified_nu[1]].T
q0 = RSQrqt[0][modified_nu[0]+modified_nx[0]:, modified_nu[0]:modified_nu[0]+modified_nx[0]].T
h0 = Gg_eqt[0][modified_nu[0]+modified_nx[0]:, :].T


### Original problem ###
KKT = np.block([
	[R1, G0x.T, np.zeros((modified_nu[1], modified_ng_eq[0]))],
	[G0x, Q0, H0u.T],
	[np.zeros((modified_ng_eq[0], modified_nu[1])), H0u, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r1], [q0], [h0]])

solution = np.linalg.solve(KKT, -rhs)
u1 = solution[:modified_nu[1]]
x0 = solution[modified_nu[1]:modified_nu[1]+modified_nx[0]]
lmbd0 = solution[modified_nu[1]+modified_nx[0]:]
print(f"original solution:\nu1 = \n{u1}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

### Initial stage ###
lmbd = np.linalg.cholesky(R1)
assert np.linalg.norm(R1 - lmbd @ lmbd.T) < 1e-8
G0x_tilde = G0x * np.linalg.inv(lmbd).T
Q0_tilde = Q0 - G0x_tilde @ G0x_tilde.T
r1_tilde = np.linalg.inv(lmbd) @ r1
q0_tilde = q0 - G0x_tilde @ r1_tilde

KKTI = np.block([
	[Q0_tilde, H0u.T],
	[H0u, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhsI = np.block([[q0_tilde], [h0]])
solutionI = np.linalg.solve(KKTI, -rhsI)
print(f"\nP:\n{Q0_tilde}")
print(f"p:\n{q0_tilde}")
print(f"H0u:\n{H0u}")
print(f"h0:\n{h0}")
x0 = solutionI[:modified_nx[0]]
lmbd0 = solutionI[modified_nx[0]:]

u1_tilde = -r1_tilde - G0x_tilde.T @ x0
print(f"\nG0x_tilde:\n{G0x_tilde}")
print(f"\nG0x * ukxk:\n{G0x_tilde.T @ x0}")
print(f"\nu1_tilde expected:\n{u1_tilde}")
print(f"\nu1_tilde without G:\n{-r1_tilde}")
print(f"\nu1_B_tilde from paper:\n{-np.linalg.inv(lmbd).T @ r1_tilde}")
print(f"multiplying with G_hat:\n{np.linalg.inv(lmbd).T @ G0x_tilde.T}")
print(f"\nu1_B_tilde with G:\n{-np.linalg.inv(lmbd).T @ r1_tilde - np.linalg.inv(lmbd).T @ G0x_tilde.T @ x0}")
u1 = np.linalg.inv(lmbd).T @ u1_tilde

print(f"=============================================================")
print(f"\n\nsolution:\nx0 = \n{x0}\nu1 = \n{u1}\nlmbd0 = \n{lmbd0}")
