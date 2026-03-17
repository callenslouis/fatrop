import numpy as np
from test_debug_helper import *
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "build_docker"))

from factorization_info import *
from preprocess_info import *


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
        Gu.append(GuGx[k][:modified_nu[k], :])
        Gx.append(GuGx[k][modified_nu[k]:modified_nu[k]+modified_nx[k], :])
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
# print_KKT(KKT, rhs)

solution = np.linalg.solve(KKT, -rhs)
extracted_solution = extract_solultion(K, modified_nu, modified_nx, modified_ng_eq, solution)
print_solution(extracted_solution)

solution2 = Solve(K, modified_nu, modified_nx, modified_ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b, Pl_r, Pr_r, L_r, U_r, [], rank_k_values, Hut)
print_solution(solution2)