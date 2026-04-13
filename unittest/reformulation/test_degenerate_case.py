import numpy as np
import random

import numpy as np

def lu_no_permutation(M):
    m, n = M.shape
    k = min(m, n) # Number of pivots
    L = np.eye(m)
    U = M.astype(float).copy()
    
    for i in range(k):
        # Check for zero pivot to avoid division by zero
        if np.isclose(U[i, i], 0):
            # Without pivoting, if U[i,i] is 0, we skip elimination for this column
            continue 
            
        for j in range(i + 1, m):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]
            # Explicitly zero the lower entry to handle precision
            U[j, i] = 0

    return L, U

def lu_full_pivoting(M, ignore_pivoting=False, ignore_col_pivoting=False):
    m, n = M.shape
    k = min(m, n) # Number of pivots
    L = np.eye(m)
    U = M.astype(float).copy()
    P = np.eye(m)
    Q = np.eye(n)
    
    for i in range(k):
        # Find the index of the largest absolute value in the submatrix U[i:, i:]
        submatrix = U[i:, i:]
        if ignore_col_pivoting:
            submatrix = U[i:, i:i+1]
        max_idx = np.unravel_index(np.argmax(np.abs(submatrix)), submatrix.shape)
        pivot_row, pivot_col = max_idx[0] + i, max_idx[1] + i

        if ignore_pivoting:
            pivot_row, pivot_col = i, i

        if np.isclose(U[pivot_row, pivot_col], 0):
            # If the pivot is zero, we cannot eliminate this column
            break
        
        # Swap rows in U and L
        U[[i, pivot_row], :] = U[[pivot_row, i], :]
        L[[i, pivot_row], :i] = L[[pivot_row, i], :i]
        P[[i, pivot_row], :] = P[[pivot_row, i], :]
        
        # Swap columns in U and Q
        U[:, [i, pivot_col]] = U[:, [pivot_col, i]]
        Q[:, [i, pivot_col]] = Q[:, [pivot_col, i]]
        
        for j in range(i + 1, m):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]
            # Explicitly zero the lower entry to handle precision
            U[j, i] = 0

    assert np.allclose(P @ M @ Q, L @ U), "LU decomposition with pivoting failed to reconstruct the original matrix"
    return P, L, U, Q

def get_M():
    max_val = 10
    nx = random.randint(1, max_val)
    r = random.randint(1, nx)
    # r = nx-1
    nu = random.randint(1, max_val)
    ng = random.randint(1, max_val)

    # construct random nx x nx matrix with rank r
    A = np.random.rand(nx, nx)
    U, S, Vt = np.linalg.svd(A)
    S[r:] = 0
    A = U @ np.diag(S) @ Vt

    B = np.random.rand(ng, nx)
    C = np.random.rand(ng, nu)

    return A, B, C, nx, r, nu, ng


def test_degenerate_case():
    A, B, C, nx, r, nu, ng = get_M()

    L11, U11 = lu_no_permutation(A)
    assert np.allclose(L11 @ U11, A)
    assert L11.shape == (nx, nx)
    assert U11.shape == (nx, nx)
    L22, U22 = lu_no_permutation(C)
    assert np.allclose(L22 @ U22, C)
    assert L22.shape == (ng, ng)
    assert U22.shape == (ng, nu)

    L21 = np.zeros((ng, nx))
    L21[:ng, :r] = B[:, :r] @ np.linalg.inv(U11[:r, :r])

    L = np.block([[L11, np.zeros((nx, ng))], [L21, L22]])
    U = np.block([[U11, np.zeros((nx, nu))], [np.zeros((ng, nx)), U22]])
    M_reconstructed = L @ U
    M = np.block([[A, np.zeros((nx, nu))], [B, C]])
    # print(f"M:\n{M}\n\n")
    # print(f"M_reconstructed:\n{M_reconstructed}\n\n")

    assert M_reconstructed.shape == M.shape
    # for i in range(M.shape[0]):
    #     for j in range(M.shape[1]):
    #         if not np.isclose(M[i, j], M_reconstructed[i, j]):
    #             print(f"Mismatch at ({i}, {j}): {M[i, j]} vs {M_reconstructed[i, j]}")

    if r < nx:
        print(f"Checking rank deficient case")                
    assert np.allclose(M_reconstructed, M)

    print("OK" + (" (full rank)" if r == nx else " (rank deficient)"))

def test_degenerate_case_general():
    A, B, C, nx, r, nu, ng = get_M()

    ### Perform LU of top-left block
    L11, U11 = lu_no_permutation(A)
    K1 = L11[:r, :r]
    K2 = L11[r:, :r]
    V1 = U11[:r, :r]
    V2 = U11[:r, r:]
    assert np.allclose(L11 @ U11, A)
    assert L11.shape == (nx, nx)
    assert U11.shape == (nx, nx)
    
    ### Compute bottom left block of L
    K3 = B[:, :r] @ np.linalg.inv(V1)

    ### Compute K4
    K4 = B[:, r:] - K3 @ V2

    ### Perform second LU
    L22, U22 = lu_no_permutation(np.block([[K4, C]]))
    V3 = U22[:ng, :nx-r]
    V4 = U22[:ng, nx-r:]

    L = np.block([[K1, np.zeros((r, ng+nx-r))], 
                  [K3, L22[:ng, :ng], np.zeros((ng, nx-r))], 
                  [K2, np.zeros((nx-r, ng)), np.eye(nx-r)]])
    U = np.block([[V1, V2, np.zeros((r, nu))], 
                  [np.zeros((ng, r)), V3, V4],
                  [np.zeros((nx-r, nx+nu))]])
    P = np.block([[np.eye(r), np.zeros((r, nx-r+ng))],
                  [np.zeros((ng, nx)), np.eye(ng)],
                  [np.zeros((nx-r, r)), np.eye(nx-r), np.zeros((nx-r, ng))]])

    M_reconstructed = L @ U
    M = np.block([[A, np.zeros((nx, nu))], [B, C]])
    
    assert M_reconstructed.shape == M.shape             
    assert np.allclose(M_reconstructed, P @ M)

    print("OK" + (" (full rank)" if r == nx else " (rank deficient)"))

def test_degenerate_case_full_pivoting():
    A, B, C, nx, r, nu, ng = get_M()

    ### Perform LU of top-left block
    P1, L11, U11, Q1 = lu_full_pivoting(A)
    K1 = L11[:r, :r]
    K2 = L11[r:, :r]
    V1 = U11[:r, :r]
    V2 = U11[:r, r:]
    assert L11.shape == (nx, nx)
    assert U11.shape == (nx, nx)

    ### Compute bottom left block of L
    B_tilde = B @ Q1
    K3 = B_tilde[:,:r] @ np.linalg.inv(V1)

    ### Compute K4
    K4 = B_tilde[:, r:] - K3 @ V2

    ### Perform second LU
    P2, L22, U22, Q2 = lu_full_pivoting(np.block([[K4, C]]), ignore_pivoting=False, ignore_col_pivoting=True)
    V3 = U22[:ng, :nx-r]
    V4 = U22[:ng, nx-r:]

    L = np.block([[K1, np.zeros((r, ng+nx-r))],
                    [P2 @ K3, L22[:ng, :ng], np.zeros((ng, nx-r))], 
                    [K2, np.zeros((nx-r, ng)), np.eye(nx-r)]])
    U = np.block([[V1, V2, np.zeros((r, nu))],
                  [np.zeros((ng, r)), V3, V4],
                  [np.zeros((nx-r, nx+nu))]])
    P = np.block([[P1[:r,:nx], np.zeros((r, ng))],
                  [np.zeros((ng, nx)), P2[:ng,:ng]],
                  [P1[r:,:nx], np.zeros((nx-r, ng))]])
    Q = np.block([[Q1, np.zeros((nx, nu))],
                  [np.zeros((nu, nx)), np.eye(nu)]]) @ \
        np.block([[np.eye(r), np.zeros((r, nx-r+nu))],
                  [np.zeros((nx-r+nu, r)), Q2]])
        
    M_reconstructed = L @ U
    M = np.block([[A, np.zeros((nx, nu))], [B, C]])

    # print(f"P1:\n{P1}\n\n")
    # print(f"P1[:r,:r]:\n{P1[:r,:r]}\n\n")
    # print(f"Q1:\n{Q1}\n\n")
    # print(f"Q1[:r,:r]:\n{Q1[:r,:r]}\n\n")
    # print(f"P2:\n{P2}\n\n")
    # print(f"Q2:\n{Q2}\n\n")
    # print(f"P:\n{P}\n\n")
    # print(f"Q:\n{Q}\n\n")

    assert M_reconstructed.shape == M.shape

    # P_true, L_true, U_true, Q_true = lu_full_pivoting(M)
    # print(f"P_true:\n{P_true}\n\n")
    # print(f"L_true:\n{L_true}\n\n")
    # print(f"U_true:\n{U_true}\n\n")
    # print(f"Q_true:\n{Q_true}\n\n")

    # print(f"Pl M2^1 Pr\n{P2 @ B[:,:r]}")

    # print(f"M:\n{M}\n\n")
    # print(f"M_reconstructed:\n{M_reconstructed}\n\n")
    assert np.allclose(M_reconstructed, P @ M @ Q)

    print("OK" + (" (full rank)" if r == nx else " (rank deficient)"))

for _ in range(10000):
    # test_degenerate_case()
    # test_degenerate_case_general()
    test_degenerate_case_full_pivoting()

    