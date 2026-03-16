import numpy as np
from test_debug_helper import *
rank_k_values = [1, 2]
Pl_r = [
np.array([
	[         1 ],
	])
,
np.array([
	[         1,          0 ],
	[         0,          1 ],
	])
]
Pr_r = [
np.array([
	[         1 ],
	])
,
np.array([
	[         0,          1,          0 ],
	[         1,          0,          0 ],
	[         0,          0,          1 ],
	])
]
L_r = [
np.array([[1],
])
,
np.array([[1, 0],
[0.590982, 1],
])
]
U_r = [
np.array([[0.731264],
])
,
np.array([[0.812169, 0.272656, 0.0952729],
[0, 0.31653, 0.0764855],
])
]

K = 2
nu = [1, 2]
nx = [1, 2]
ng_ineq = [0, 0]
ng_eq = [0, 2]
r = [1]
modified_K = 2
modified_nu = [1, 3]
modified_nx = [1, 1]
modified_ng_ineq = [0, 0]
modified_ng_eq = [1, 2]
RSQrqt = [
np.array([
	[   1.05175,   0.884491 ],
	[  0.884491,    0.77675 ],
	[         0,          1 ],
	])
,
np.array([
	[   1.59754,    1.60214,  -0.151728,    1.02673 ],
	[   1.60214,    1.72233,  -0.100807,    1.01113 ],
	[ -0.151728,  -0.100807,   0.184012,  -0.243659 ],
	[   1.02673,    1.01113,  -0.243659,   0.813282 ],
	[         2,          3,    2.53429,          4 ],
	])
]
RSQrqt_original = [
np.array([
	[   1.05175,   0.884491 ],
	[  0.884491,    0.77675 ],
	[         0,          1 ],
	])
,
np.array([
	[   1.59754,    1.60214,    1.02673,   0.481177 ],
	[   1.60214,    1.72233,    1.01113,   0.522483 ],
	[   1.02673,    1.01113,   0.813282,    0.25767 ],
	[  0.481177,   0.522483,    0.25767,   0.192648 ],
	[         2,          3,          4,          5 ],
	])
]
FuFx = [
np.array([
	[ 0.0613063 ],
	[0.00992804 ],
	])
]
FuFx_original = [
np.array([
	[ 0.0613063,  0.0902349 ],
	[0.00992804,  0.0969809 ],
	])
]
GuGx = [
np.array([
	[         0,          0,   0.052444 ],
	[         0,          0,   0.090861 ],
	])
]
GuGx_original = [
np.array([
	[         0,          0 ],
	[         0,          0 ],
	])
]
Gg_eqt = [
np.array([
	[  0.731264 ],
	[  0.683719 ],
	[   2.61878 ],
	])
,
np.array([
	[  0.272656,   0.477665 ],
	[  0.812169,   0.479977 ],
	[ 0.0952729,    0.13279 ],
	[  0.392785,   0.836079 ],
	[         0,          1 ],
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
	[  0.272656,   0.477665 ],
	[  0.812169,   0.479977 ],
	[  0.392785,   0.836079 ],
	[  0.337396,   0.648172 ],
	[         0,          1 ],
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
	[],
	[],
	])
]
BAbt = [
np.array([
	[  -3.19538 ],
	[  -4.62425 ],
	[  -10.7798 ],
	])
]
BAbt_original = [
np.array([
	[  0.592845,   0.844266 ],
	[  0.857946,   0.847252 ],
	[         2,          3 ],
	])
]
Jt = [
np.array([
	[  0.185532,  0.0353642 ],
	[  0.114367,  0.0217994 ],
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
	[         1,          0 ],
	[         0,          1 ],
	])
]
Pr = [
np.array([
	[         1,          0 ],
	[         0,          1 ],
	])
]


### STORE BLOCK MATRICES ###
R = []
S = []
Q = []
Gu = []
Gx = []
Fu = []
Fx = []
Hu = []
Hx = []
B = []
A = []
r = []
q = []
h = []
b = []
for k in range(K):
    R.append(RSQrqt[k][:modified_nu[k], :modified_nu[k]])
    S.append(RSQrqt[k][:modified_nu[k], modified_nu[k]:modified_nu[k]+modified_nx[k]].T)
    Q.append(RSQrqt[k][modified_nu[k]:modified_nu[k]+modified_nx[k], modified_nu[k]:modified_nu[k]+modified_nx[k]])
    if k < K-1:
        print(f"GuGx[{k}]:\n{GuGx[k]}\n")
        Gu.append(GuGx[k][:modified_nu[k], :])
        Gx.append(GuGx[k][modified_nu[k]:modified_nu[k]+modified_nx[k], :])
        print(f"Gu:\n{Gu[k]}\n")
        print(f"Gx:\n{Gx[k]}\n")
        Fu.append(FuFx[k][:modified_nu[k], :])
        Fx.append(FuFx[k][modified_nu[k]:modified_nu[k]+modified_nx[k], :])
        B.append(BAbt[k][:modified_nu[k], :modified_nx[k+1]].T)
        A.append(BAbt[k][modified_nu[k]:modified_nu[k]+modified_nx[k], :modified_nx[k+1]].T)
        b.append(BAbt[k][modified_nu[k]+modified_nx[k]:, :modified_nx[k+1]].T)
    Hu.append(Gg_eqt[k][:modified_nu[k], :].T)
    Hx.append(Gg_eqt[k][modified_nu[k]:modified_nu[k]+modified_nx[k], :].T)
    
    r.append(RSQrqt[k][modified_nu[k]+modified_nx[k]:, :modified_nu[k]].T)
    q.append(RSQrqt[k][modified_nu[k]+modified_nx[k]:, modified_nu[k]:modified_nu[k]+modified_nx[k]].T)
    h.append(Gg_eqt[k][modified_nu[k]+modified_nx[k]:, :modified_ng_eq[k]].T)

### Original problem ###
KKT, rhs = GetKKT(K, modified_nu, modified_nx, modified_ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b)
print_KKT(KKT, rhs)

solution = np.linalg.solve(KKT, -rhs)
extracted_solution = extract_solultion(K, modified_nu, modified_nx, modified_ng_eq, solution)
print_solution(extracted_solution)

solution2 = Solve(K, modified_nu, modified_nx, modified_ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b, Pl_r, Pr_r, L_r, U_r, [], rank_k_values)
print_solution(solution2)


exit()
print(H1u)
### Factorization of H1u ###
L = np.array([[1]])
U1 = H1u[0,0]
U2 = H1u[0,1]
U1, U2 = U2, U1
U = np.array([[U1, U2]])

Tl = L @ np.array([[-U1]])
Tr = np.block([[1, U2/U1], [0, 1]]) @ np.array([[0, 1], [1, 0]])
Tli = np.linalg.inv(Tl)
Tri = np.linalg.inv(Tr)
assert(np.linalg.norm(H1u - Tl @ np.array([[-1, 0]]) @ Tr) < 1e-8)

R1_tilde = Tri.T @ R1 @ Tri
r1_tilde = Tri.T @ r1
G0x_tilde = G0x @ Tri
G0u_tilde = G0u @ Tri
print(f"G0x:\n{G0x}\n")
print(f"G0u:\n{G0u}\n")
print(f"G0x_tilde:\n{G0x_tilde}\n")
print(f"G0u_tilde:\n{G0u_tilde}\n")
h1_tilde = Tli @ h1

KKT = np.block([
    [R1_tilde, np.array([[-1],[0]]), G0u_tilde.T, G0x_tilde.T, np.zeros((modified_nu[1], modified_ng_eq[0]))],
    [np.array([-1,0]), np.zeros((modified_ng_eq[1], modified_ng_eq[1] + modified_nu[0] + modified_nx[0] + modified_ng_eq[0]))],
    [G0u_tilde, np.zeros((modified_nu[0], modified_ng_eq[1])), R0, S0.T, H0u.T],
    [G0x_tilde, np.zeros((modified_nx[0], modified_ng_eq[1])), S0, Q0, H0x.T],
    [np.zeros((modified_ng_eq[0], modified_nu[1]+modified_ng_eq[1])), H0u, H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r1_tilde], [h1_tilde], [r0], [q0], [h0]])

solution = np.linalg.solve(KKT, -rhs)
ptr = 0
u1_tilde = solution[ptr:ptr+modified_nu[1]]; ptr += modified_nu[1]
lmbd1_tilde = solution[ptr:ptr+modified_ng_eq[1]]; ptr += modified_ng_eq[1]
u0 = solution[ptr:ptr+modified_nu[0]]; ptr += modified_nu[0]
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
lmbd0 = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]

u1 = Tri @ u1_tilde
lmbd1 = Tli.T @ lmbd1_tilde

print(f"solution after factorization of H1u:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")
### eliminating parts of u0 and lambda ###
R1_hat = R1_tilde[1:, 1:]
# print(f"\nR1_hat should be:\n{R1_hat}\n")
r1_hat = r1_tilde[1:, :]
# print(f"r1_hat should be:\n{r1_hat}\n")
G0x_hat = G0x_tilde[:, 1:]
G0u_hat = G0u_tilde[:, 1:]

KKT = np.block([
    [R1_hat, G0u_hat.T, G0x_hat.T, np.zeros((modified_nu[1]-1, modified_ng_eq[0]))],
    [G0u_hat, R0, S0.T, H0u.T],
    [G0x_hat, S0, Q0, H0x.T],
    [np.zeros((modified_ng_eq[0], modified_nu[1]-1)), H0u, H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r1_hat], [r0], [q0], [h0]])

solution = np.linalg.solve(KKT, -rhs)
ptr = 0
u1_hat = solution[ptr:ptr+modified_nu[1]-1]; ptr += modified_nu[1]-1
u0 = solution[ptr:ptr+modified_nu[0]]; ptr += modified_nu[0]
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
lmbd0 = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]
u1_tilde = np.block([[-h1_tilde], [u1_hat]])
u1 = Tri @ u1_tilde

lmbd1_tilde = (r1_tilde + R1_tilde @ u1_tilde + G0u_tilde.T @ u0 + G0x_tilde.T @ x0)[:1,:]
lmbd1 = Tli.T @ lmbd1_tilde

print(f"solution after elimination of parts of u0 and lambda:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")
### Factorization of R1 ###
L = np.array([[np.sqrt(R1_hat[0,0])]])
assert (np.linalg.norm(R1_hat - L @ L.T) < 1e-8)
Li = np.linalg.inv(L)

G0u_prime = G0u_hat @ Li.T
G0x_prime = G0x_hat @ Li.T
r1_prime = Li @ r1_hat

KKT = np.block([
    [np.eye(modified_nu[1]-1), G0u_prime.T, G0x_prime.T, np.zeros((modified_nu[1]-1, modified_ng_eq[0]))],
    [G0u_prime, R0, S0.T, H0u.T],
    [G0x_prime, S0, Q0, H0x.T],
    [np.zeros((modified_ng_eq[0], modified_nu[1]-1)), H0u, H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r1_prime], [r0], [q0], [h0]])
solution = np.linalg.solve(KKT, -rhs)
print("stage before tilde:")
print(f"KKT:\n{KKT}\nrhs:\n{rhs}\n")

ptr = 0
u1_prime = solution[ptr:ptr+modified_nu[1]-1]; ptr += modified_nu[1]-1
u0 = solution[ptr:ptr+modified_nu[0]]; ptr += modified_nu[0]
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
lmbd0 = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]
u1_hat = Li.T @ u1_prime
u1_tilde = np.block([[-h1_tilde], [u1_hat]])
u1 = Tri @ u1_tilde
lmbd1_tilde = (r1_tilde + R1_tilde @ u1_tilde + G0u_tilde.T @ u0 + G0x_tilde.T @ x0)[:1,:]
lmbd1 = Tli.T @ lmbd1_tilde
print(f"solution after factorization of R1:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")
### elimination of u1_prime ###

R0_tilde = R0 - G0u_prime.T @ G0u_prime
S0_tilde = S0 - G0x_prime @ G0u_prime.T
Q0_tilde = Q0 - G0x_prime.T @ G0x_prime
r0_tilde = r0 - G0u_prime.T @ r1_prime
q0_tilde = q0 - G0x_prime.T @ r1_prime

KKT = np.block([
    [R0_tilde, S0_tilde.T, H0u.T],
    [S0_tilde, Q0_tilde, H0x.T],
    [H0u, H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r0_tilde], [q0_tilde], [h0]])
print("stage at tilde:")
print(f"KKT:\n{KKT}\nrhs:\n{rhs}\n")
print(f"G0u_prime:\n{G0u_prime}")
print(f"G0x_prime:\n{G0x_prime}")
solution = np.linalg.solve(KKT, -rhs)
ptr = 0
u0 = solution[ptr:ptr+modified_nu[0]]; ptr += modified_nu[0]
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
lmbd0 = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]
u1_prime = -(r1_prime + G0u_prime.T @ u0 + G0x_prime.T @ x0)
u1_hat = Li.T @ u1_prime
u1_tilde = np.block([[-h1_tilde], [u1_hat]])
u1 = Tri @ u1_tilde
lmbd1_tilde = (r1_tilde + R1_tilde @ u1_tilde + G0u_tilde.T @ u0 + G0x_tilde.T @ x0)[:1,:]
lmbd1 = Tli.T @ lmbd1_tilde

print(f"solution after elimination of u1_prime:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

### decompose H0u ###
L = np.array([[1]])
U = np.array([[H0u[0,0]]])


print(f"solution after elimination of u1_prime:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

### decompose H0u ###
Tl0 = L @ np.array([[-U[0,0]]])
Tr0 = np.eye(1)
Tl0i = np.linalg.inv(Tl0)
Tr0i = np.linalg.inv(Tr0)

R0_hat = R0_tilde @ Tr0i
S0_hat = S0_tilde @ Tr0i
H0x_hat = Tl0i @ H0x
h0_hat = Tl0i @ h0

KKT = np.block([
    [R0_hat, S0_hat.T, -1],
    [S0_hat, Q0_tilde, H0x_hat.T],
    [-1, H0x_hat, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
])
rhs = np.block([[r0_tilde], [q0_tilde], [h0_hat]])

solution = np.linalg.solve(KKT, -rhs)
ptr = 0
u0_tilde = solution[ptr:ptr+modified_nu[0]]; ptr += modified_nu[0]
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
lmbd0_tilde = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]
u0 = Tr0i @ u0_tilde
u1_prime = -(r1_prime + G0u_prime.T @ u0 + G0x_prime.T @ x0)
u1_hat = Li.T @ u1_prime
u1_tilde = np.block([[-h1_tilde], [u1_hat]])
u1 = Tri @ u1_tilde
lmbd1_tilde = (r1_tilde + R1_tilde @ u1_tilde + G0u_tilde.T @ u0 + G0x_tilde.T @ x0)[:1,:]
lmbd1 = Tli.T @ lmbd1_tilde
print(f"solution after factorization of H0u:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

### eliminate u0 ###
P0 = Q0_tilde + H0x_hat.T @ S0_hat.T + S0_hat @ H0x_hat + H0x_hat.T @ R0_hat @ H0x_hat
p0 = q0_tilde + S0_hat.T @ h0_hat + H0x_hat.T @ R0_hat @ h0_hat + H0x_hat.T @ r0_tilde

KKT = np.block([
    [P0],
])
rhs = np.block([[p0]])
solution = np.linalg.solve(KKT, -rhs)
ptr = 0
x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
u0_tilde = H0x_hat @ x0 + h0_hat
lmbd0_hat = R0_hat * u0_tilde + S0_hat.T @ x0 + r0_tilde
lmbd0 = Tl0i.T @ np.array([[lmbd0_hat[0,0]]])
u0 = Tr0i @ u0_tilde

u1_prime = -(r1_prime + G0u_prime.T @ u0 + G0x_prime.T @ x0)
u1_hat = Li.T @ u1_prime
u1_tilde = np.block([[-h1_tilde], [u1_hat]])
u1 = Tri @ u1_tilde
lmbd1_tilde = (r1_tilde + R1_tilde @ u1_tilde + G0u_tilde.T @ u0 + G0x_tilde.T @ x0)[:1,:]
lmbd1 = Tli.T @ lmbd1_tilde
print(f"solution after factorization of H0u:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
print(f"=============================================================")

print(f"Initial stage:\nKKT = \n{KKT}\nrhs = \n{rhs}\n")



# print(f"P0 before:\n{Q0}\n")
# print(f"p0 before:\n{q0}\n")
# print(f"p0 difference:\n{- G0x_prime.T @ r1_prime}\n")
# print(f"G0x_prime:\n{G0x_prime}\n")
# print(f"r1_prime:\n{r1_prime}\n")
# # print(f"Gu_tilde:\n{G0u_tilde}\n")
# # print(f"Gx_tilde:\n{G0x_tilde}\n")
# # print(f"L:\n{L}\n")
# P0 = Q0 - G0x_prime.T @ G0x_prime
# p0 = q0 - G0x_prime.T @ r1_prime

# KKT = np.block([
# 	[P0, H0x.T],
# 	[H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
# ])
# rhs = np.block([[p0], [h0]])
# solution = np.linalg.solve(KKT, -rhs)
# ptr = 0
# x0 = solution[ptr:ptr+modified_nx[0]]; ptr += modified_nx[0]
# lmbd0 = solution[ptr:ptr+modified_ng_eq[0]]; ptr += modified_ng_eq[0]
# u1_prime = -(r1_prime + G0x_prime.T @ x0)
# u1_hat = Li.T @ u1_prime
# u1_tilde = np.block([[-h1_tilde], [u1_hat]])
# u1 = Tri @ u1_tilde
# lmbd1_tilde = (r1_tilde + R1_tilde @ u1_tilde + G0u_tilde.T @ u0 + G0x_tilde.T @ x0)[:1,:]
# lmbd1 = Tli.T @ lmbd1_tilde
# print(f"solution after elimination of u1_prime:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
# print(f"=============================================================")

# print(f"initial stage should be:\n{KKT}\n{rhs}")


# ### Factorization of H0u ###
# L1 = 1
# L2 = 0.691005028288494
# U1 = 0.857946
# L = np.array([[L1, 0], [L2, 1]])
# U = np.array([[U1], [0]])

# Tl0 = L @ np.array([[-U1, 0], [0, 1]])
# Tr0 = np.eye(1)
# Tl0i = np.linalg.inv(Tl0)
# Tr0i = np.linalg.inv(Tr0)
# assert(np.linalg.norm(H0u - Tl0 @ np.array([[-1], [0]]) @ Tr0) < 1e-8)

# R0_tilde = Tr0i.T @ R0 @ Tr0i
# r0_tilde = Tr0i.T @ r0_prime
# S0_tilde = S0 @ Tr0i
# H0x_tilde = Tl0i @ H0x
# h0_tilde = Tl0i @ h0

# KKT = np.block([
#     [R0_tilde, S0_tilde.T, np.array([[-1, 0]])],
#     [S0_tilde, Q0, H0x_tilde.T], 
#     [np.array([[-1, 0]]).T, H0x_tilde, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))]
# ])
# rhs = np.block([[r0_tilde], [q0_prime], [h0_tilde]])

# solution = np.linalg.solve(KKT, -rhs)
# u0_tilde = solution[:modified_nu[0]]
# x0 = solution[modified_nu[0]:modified_nu[0]+modified_nx[0]]
# lmbd0_tilde = solution[modified_nu[0]+modified_nx[0]:]
# print(f"lmbd0_tilde: {lmbd0_tilde}")

# u0 = Tr0i @ u0_tilde
# lmbd0 = Tl0i.T @ lmbd0_tilde
# u1_tilde = h1_tilde
# lmbd1_tilde = r1_tilde + R1_tilde @ u1_tilde + G0u_tilde @ u0 + G0x_tilde @ x0
# u1 = Tri @ u1_tilde
# lmbd1 = Tli.T @ lmbd1_tilde

# print(f"solution after factorization of H0u:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
# print(f"=============================================================")
# print(f"q0_prime should be:\n{q0_prime}\n")
# H0xa = H0x_tilde[:1,:]
# h0xa = h0_tilde[:1,:]
# P0 = Q0 + S0_tilde @ H0xa + H0xa.T @ S0_tilde.T + H0xa.T @ R0_tilde @ H0xa
# p0 = q0_prime + S0_tilde @ h0xa + H0xa.T @ R0_tilde @ h0xa + H0xa.T @ r0_tilde

# H0_hat = H0x_tilde[1:,:]
# h0_hat = h0_tilde[1:,:]

# KKT = np.block([
#     [P0, H0_hat.T],
#     [H0_hat, np.zeros((modified_ng_eq[0]-1, modified_ng_eq[0]-1))],
# ])
# rhs = np.block([[p0], [h0_hat]])
# solution = np.linalg.solve(KKT, -rhs)
# x0 = solution[0]
# v0 = solution[1:]
# u0_tilde = h0_tilde[:1,:] + H0x_tilde[:1,:] @ x0
# u0 = Tr0i @ u0_tilde
# lmbd0_tilde = R0_tilde * u0_tilde + S0_tilde.T @ x0 + r0_tilde
# lmbd0 = Tl0i.T @ np.array([[lmbd0_tilde[0,0]],[v0[0,0]]])

# u1_tilde = h1_tilde
# lmbd1_tilde = r1_tilde + R1_tilde @ u1_tilde + G0u_tilde @ u0 + G0x_tilde @ x0
# u1 = Tri @ u1_tilde
# lmbd1 = Tli.T @ lmbd1_tilde

# print(f"solution after factorization of H0u:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
# print(f"=============================================================")

# print(f"initial stage should be:\n{KKT}\n{rhs}")

# # ### Factorization of R0 ###
# # L0 = np.array([[np.sqrt(R0[0,0])]])
# # L0i = np.linalg.inv(L0)
# # assert(np.linalg.norm(R0 - L0 @ L0.T) < 1e-8)

# # r0_tilde = L0i @ r0_prime
# # S0_tilde = S0 @ L0i.T
# # H0u_tilde = H0u @ L0i.T

# # KKT = np.block([
# #     [np.eye(modified_nu[0]), S0_tilde.T, H0u_tilde.T],
# # 	[S0_tilde, Q0, H0x.T],
# # 	[H0u_tilde, H0x, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
# # ])
# # rhs = np.block([[r0_tilde], [q0_prime], [h0]])

# # solution = np.linalg.solve(KKT, -rhs)
# # u0_tilde = solution[:modified_nu[0]]
# # x0 = solution[modified_nu[0]:modified_nu[0]+modified_nx[0]]
# # lmbd0 = solution[modified_nu[0]+modified_nx[0]:]
# # u0 = L0i.T @ u0_tilde

# # u1_tilde = h1_tilde
# # lmbd1_tilde = r1_tilde + R1_tilde @ u1_tilde + G0u_tilde @ u0 + G0x_tilde @ x0
# # u1 = Tri @ u1_tilde
# # lmbd1 = Tli.T @ lmbd1_tilde

# # print(f"solution after factorization of R0:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
# # print(f"=============================================================")

# # ### elimination of u0 ###
# # P0 = Q0 - S0_tilde @ S0_tilde.T
# # p0 = q0_prime - S0_tilde @ r0_tilde
# # H0x_tilde = H0x - H0u_tilde @ S0_tilde.T
# # h0_tilde = h0 - H0u_tilde @ r0_tilde

# # KKT = np.block([
# #     [P0, H0x_tilde.T],
# # 	[H0x_tilde, np.zeros((modified_ng_eq[0], modified_ng_eq[0]))],
# # ])
# # print(KKT)
# # rhs = np.block([[p0], [h0_tilde]])

# # solution = np.linalg.solve(KKT, -rhs)
# # x0 = solution[:modified_nx[0]]
# # lmbd0 = solution[modified_nx[0]:]

# # u0_tilde = -(r0_tilde + S0_tilde.T@x0 + H0u_tilde.T @ lmbd0)
# # u0 = L0i.T @ u0_tilde

# # u1_tilde = h1_tilde
# # lmbd1_tilde = r1_tilde + R1_tilde @ u1_tilde + G0u_tilde @ u0 + G0x_tilde @ x0
# # u1 = Tri @ u1_tilde
# # lmbd1 = Tli.T @ lmbd1_tilde

# # print(f"solution after elimination of u0:\nu1 = \n{u1}\nlmbd1 = \n{lmbd1}\nu0 = \n{u0}\nx0 = \n{x0}\nlmbd0 = \n{lmbd0}")
# # print(f"=============================================================")