import numpy as np

def GetKKT(K, nu, nx, ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b):
    number_of_primal_vars = sum(nu) + sum(nx)
    number_of_eqs = sum(ng_eq) + sum(nx[1:])
    KKT = np.zeros((number_of_primal_vars + number_of_eqs, number_of_primal_vars + number_of_eqs))
    rhs = np.zeros((number_of_primal_vars + number_of_eqs, 1))
    ptr = 0
    for k in range(K-1, -1, -1):
        KKT[ptr:ptr+nu[k]+nx[k], ptr:ptr+nu[k]+nx[k]] = np.block([[R[k], S[k].T], [S[k], Q[k]]])
        rhs[ptr:ptr+nu[k]+nx[k]] = np.block([[r[k]], [q[k]]])
        
        Gg_eqt = np.block([[Hu[k], Hx[k]]]).T
        KKT[ptr:ptr+nu[k]+nx[k], ptr+nu[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]] = Gg_eqt[:nu[k]+nx[k], :]
        KKT[ptr+nu[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k], ptr:ptr+nu[k]+nx[k]] = Gg_eqt[:nu[k]+nx[k], :].T
        rhs[ptr+nu[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]] = h[k]
        
        if k > 0:
            # BAbt
            BAbt = np.block([[B[k-1], A[k-1]]]).T
            KKT[ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1], 
                ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = BAbt[:nu[k-1]+nx[k-1]]
            KKT[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k], 
                ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1]] = BAbt[:nu[k-1]+nx[k-1]].T
            rhs[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = b[k-1]

            # J
            KKT[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k],
                ptr+nu[k]:ptr+nu[k]+nx[k]] = -np.eye(nx[k])
            KKT[ptr+nu[k]:ptr+nu[k]+nx[k],
                ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]] = -np.eye(nx[k])
            
            # FuFx
            FuFx = np.block([[Fu[k-1]], [Fx[k-1]]])
            KKT[ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1], 
                ptr+nu[k]:ptr+nu[k]+nx[k]] = FuFx[:nx[k-1]+nu[k-1], :nx[k]]
            KKT[ptr+nu[k]:ptr+nu[k]+nx[k], 
                ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1]] = FuFx[:nx[k-1]+nu[k-1], :nx[k]].T
            
            # GuGx
            GuGx = np.block([[Gu[k-1]], [Gx[k-1]]])
            KKT[ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1], 
                ptr:ptr+nu[k]] = GuGx[:nx[k-1]+nu[k-1], :nu[k]]
            KKT[ptr:ptr+nu[k], 
                ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k]+nx[k]+nu[k-1]+nx[k-1]] = GuGx[:nx[k-1]+nu[k-1], :nu[k]].T
            

            ptr += nu[k] + nx[k] + ng_eq[k] + nx[k]

    return KKT, rhs

def print_KKT(KKT, rhs):
    print(f"KKT:")
    for row in KKT:
        for elem in row:
            if elem == 0:
                print(f"{elem:10.0f} ", end="")
            else:
                print(f"{elem:10.6f} ", end="")
        print()
    print(rhs.T)

def extract_solultion(K, nu, nx, ng_eq, sol):
    extracted_solution = {"u": [], "x": [], "lambda": [], "pi": []}

    ptr = 0
    for k in range(K-1, -1, -1):
        extracted_solution["u"].append(sol[ptr:ptr+nu[k], :])
        extracted_solution["x"].append(sol[ptr+nu[k]:ptr+nu[k]+nx[k], :])
        extracted_solution["lambda"].append(sol[ptr+nu[k]+nx[k]:ptr+nu[k]+nx[k]+ng_eq[k], :])
        extracted_solution["pi"].append(sol[ptr+nu[k]+nx[k]+ng_eq[k]:ptr+nu[k]+nx[k]+ng_eq[k]+(nx[k-1] if k > 0 else 0), :])
        ptr += nu[k] + nx[k] + ng_eq[k] + (nx[k-1] if k > 0 else 0)

    extracted_solution["u"] = extracted_solution["u"][::-1]
    extracted_solution["x"] = extracted_solution["x"][::-1]
    extracted_solution["lambda"] = extracted_solution["lambda"][::-1]
    extracted_solution["pi"] = extracted_solution["pi"][::-1]

    return extracted_solution

def print_solution(solution):
    print(f"u:")
    for k in range(len(solution["u"])):
        print(f"\t{solution['u'][k].T}")
    print(f"x:")
    for k in range(len(solution["x"])):
        print(f"\t{solution['x'][k].T}")
    print(f"lambda:")
    for k in range(len(solution["lambda"])):
        print(f"\t{solution['lambda'][k].T}")
    print(f"pi:")
    for k in range(len(solution["pi"])):
        print(f"\t{solution['pi'][k].T}")
    print()

def Solve(K, nu, nx, ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b,
          Pl, Pr, L, U, Lmbd, rank_k_values):
    # figure out what step to perform
    nu_c = nu.copy(); nx_c = nx.copy(); ng_eq_c = ng_eq.copy(); 
    R_c = R.copy(); S_c = S.copy(); Q_c = Q.copy(); 
    Gu_c = Gu.copy(); Gx_c = Gx.copy(); Fu_c = Fu.copy(); 
    Fx_c = Fx.copy(); Hu_c = Hu.copy(); Hx_c = Hx.copy(); 
    B_c = B.copy(); A_c = A.copy(); r_c = r.copy(); 
    q_c = q.copy(); h_c = h.copy(); b_c = b.copy()

    if nu[K-1] != 0:
        # decompose Hku and eliminate constraints partially
        rank = rank_k_values[K-1]; m = ng_eq[K-1]; n = nu[K-1]
        U1 = U[K-1][:rank, :rank]
        U2 = U[K-1][:rank, rank:]

        Tl = Pl[K-1] @ L[K-1] @ np.block([[-U1, np.zeros((rank, m-rank))], [np.zeros((m-rank, rank)), np.eye(m-rank)]])
        Tr = np.block([[np.eye(rank), np.linalg.inv(U1) @ U2], [np.zeros((n-rank, rank)), np.eye(n-rank)]]) @ Pr[K-1].T
        assert(np.linalg.norm(Hu[K-1] - Tl @ np.block([[-np.eye(rank), np.zeros((rank, n-rank))], [np.zeros((m-rank, n))]]) @ Tr) < 1e-6)

        Tli = np.linalg.inv(Tl)
        Tri = np.linalg.inv(Tr)

        # scale
        R_tilde = Tri.T @ R[K-1] @ Tri
        r_tilde = Tri.T @ r[K-1]
        S_tilde = S[K-1] @ Tri
        Hx_tilde = Tli @ Hx[K-1]
        h0_tilde = Tli @ h[K-1]

        Hu_c[K-1] = np.block([[-np.eye(rank), np.zeros((rank, n-rank))], [np.zeros((m-rank, n))]])
        R_c[K-1] = R_tilde
        S_c[K-1] = S_tilde
        Q_c[K-1] = Q[K-1]
        Hx_c[K-1] = Hx_tilde
        r_c[K-1] = r_tilde
        h_c[K-1] = h0_tilde
        if K > 1:
            Gu_c[K-2] = Gu[K-2] @ Tri
            Gx_c[K-2] = Gx[K-2] @ Tri

        ### intermediate check ###
        # Hu_c[K-1] = np.block([[-np.eye(rank), np.zeros((rank, n-rank))], [np.zeros((m-rank, n))]])
        # KKT, rhs = GetKKT(K, nu_c, nx_c, ng_eq_c, R_c, S_c, Q_c, Gu_c, Gx_c, Fu_c, Fx_c, Hu_c, Hx_c, B_c, A_c, r_c, q_c, h_c, b_c)
        # solution_vector = np.linalg.solve(KKT, -rhs)
        # solution = extract_solultion(K, nu_c, nx_c, ng_eq_c, solution_vector)
        # u_tilde = solution["u"][K-1]
        # u = Tri @ u_tilde
        # lmbd_tilde = solution["lambda"][K-1]
        # lmbd = Tli.T @ lmbd_tilde
        # solution["u"][K-1] = u
        # solution["lambda"][K-1] = lmbd
        # return solution
        ######

        # eliminate u1a and lambda
        Sa = S_tilde[:, :rank]; Sb = S_tilde[:, rank:]
        Ra = R_tilde[:rank, :rank]; Rb = R_tilde[rank:, rank:]
        ra = r_tilde[:rank,:]; rb = r_tilde[rank:,:]
        Ha = Hx_tilde[:rank, :]; Hb = Hx_tilde[rank:, :]
        ha = h0_tilde[:rank,:]; hb = h0_tilde[rank:,:]

        Q_tilde = Q[K-1] + Hx_tilde.T @ Sa.T + Sa @ Hx_tilde + Hx_tilde.T @ Ra @ Hx_tilde
        q_tilde = q[K-1] + Hx_tilde.T @ ra + Sa @ h0_tilde + Hx_tilde.T @ Ra @ h0_tilde

        if K > 1:
            Gua = Gu_c[K-2][:, :rank]; Gub = Gu_c[K-2][:, rank:]
            Gxa = Gx_c[K-2][:, :rank]; Gxb = Gx_c[K-2][:, rank:]
            Fu_c[K-2] = Fu[K-2] + Gua @ Hx_tilde
            Fx_c[K-2] = Fx[K-2] + Gxa @ Hx_tilde
            r_c[K-2] = r[K-2] + Gua @ ha
            q_c[K-2] = q[K-2] + Gxa @ ha

            Gu_c[K-2] = Gub
            Gx_c[K-2] = Gxb
        
        Sb_prime = Sb + Ha.T @ R_tilde[:rank, rank:]
        rb_prime = rb + R_tilde[rank:, :rank] @ h0_tilde
        
        R_c[K-1] = Rb
        S_c[K-1] = Sb_prime
        Q_c[K-1] = Q_tilde
        Hu_c[K-1] = np.zeros((m-rank, n))
        Hx_c[K-1] = Hb
        r_c[K-1] = rb_prime
        q_c[K-1] = q_tilde
        h_c[K-1] = hb  
        nu_c[K-1] = n - rank
        ng_eq_c[K-1] = m - rank

        ### intermediate check ###
        # KKT, rhs = GetKKT(K, nu_c, nx_c, ng_eq_c, R_c, S_c, Q_c, Gu_c, Gx_c, Fu_c, Fx_c, Hu_c, Hx_c, B_c, A_c, r_c, q_c, h_c, b_c)
        # solution_vector = np.linalg.solve(KKT, -rhs)
        # solution = extract_solultion(K, nu_c, nx_c, ng_eq_c, solution_vector)
        # # recover solution
        # ub_tilde = solution["u"][K-1]
        # ua_tilde = Ha @ solution["x"][K-1] + ha
        # u_tilde = np.block([[ua_tilde], [ub_tilde]])
        # u = Tri @ u_tilde
        # lmbd_tilde = (R_tilde @ u_tilde + S_tilde.T @ solution["x"][K-1] + r_tilde)[:rank]
        # if K > 1:
        #     lmbd_tilde += Gua.T @ solution["u"][K-2] + Gxa.T @ solution["x"][K-2]
        # lmbd = Tli.T @ lmbd_tilde
        # solution["u"][K-1] = u
        # solution["lambda"][K-1] = lmbd
        # return solution
        ######

        # # eliminate remaining controls
        Lmbd = np.linalg.cholesky(Rb)
        assert np.linalg.norm(Rb - Lmbd @ Lmbd.T) < 1e-8
        # TODO: check if this matches the Llt result

        Lmbdi = np.linalg.inv(Lmbd)
        
        S_hat = Sb_prime @ Lmbdi.T
        r_hat = Lmbdi @ rb_prime

        Q_hat = Q_tilde - S_hat @ S_hat.T
        q_hat = q_tilde - S_hat @ r_hat

        if K > 1:
            Gu_hat = Gub @ Lmbdi.T
            Gx_hat = Gxb @ Lmbdi.T
            Fu_c[K-2] = Fu_c[K-2] - Gu_hat @ S_hat.T
            Fx_c[K-2] = Fx_c[K-2] - Gx_hat @ S_hat.T
            R_c[K-2] = R_c[K-2] - Gu_hat @ Gu_hat.T
            S_c[K-2] = S_c[K-2] - Gx_hat @ Gu_hat.T
            Q_c[K-2] = Q_c[K-2] - Gx_hat @ Gx_hat.T
            r_c[K-2] = r_c[K-2] - Gu_hat @ r_hat
            q_c[K-2] = q_c[K-2] - Gx_hat @ r_hat
            Gx_c[K-2] = np.zeros((nx[K-2],0))
            Gu_c[K-2] = np.zeros((nu[K-2],0))

        nu_c[K-1] = 0
        ng_eq[K-1] = ng_eq_c[K-1] - rank

        R_c[K-1] = np.zeros((0,0))
        S_c[K-1] = np.zeros((nx[K-1],0))
        Q_c[K-1] = Q_hat
        r_c[K-1] = np.zeros((0,1))
        q_c[K-1] = q_hat

        # perform recursive call
        solution = Solve(K, nu_c, nx_c, ng_eq_c, R_c, S_c, Q_c, Gu_c, Gx_c, Fu_c, Fx_c,
              Hu_c, Hx_c, B_c, A_c, r_c, q_c, h_c, b_c, Pl, Pr, L, U,
              Lmbd, rank_k_values)
        
        # recover original solution
        u_b_hat = S_hat.T @ solution["x"][K-1] + r_hat
        if K > 1:
            u_b_hat += Gu_hat @ solution["u"][K-2] + Gx_hat @ solution["x"][K-2]
        u_b_tilde = - Lmbdi.T  @ u_b_hat
        u_tilde = np.block([[Ha @ solution["x"][K-1] + ha], [u_b_tilde]])
        u = Tri @ u_tilde
        lmbd_tilde = (R_tilde @ u_tilde + S_tilde.T @ solution["x"][K-1] + r_tilde)[:rank]
        if K > 1:
            lmbd_tilde += Gua.T @ solution["u"][K-2] + Gxa.T @ solution["x"][K-2]
        lmbd = Tli.T @ lmbd_tilde

        solution["u"][K-1] = u
        solution["lambda"][K-1] = lmbd

    elif K > 1:
        # eliminate states
        temp = np.block([[np.block([[B[K-2], A[K-2]]]).T, np.eye(nu[K-2]+nx[K-2])]]) @ \
            np.block([[Q[K-1], np.zeros((nx[K-1], nu[K-2] + nx[K-2])), q[K-1]],
                      [np.zeros((nu[K-2], nx[K-1])), R[K-2], S[K-2].T, r[K-2]],
                      [np.zeros((nx[K-2], nx[K-1])), S[K-2], Q[K-2], q[K-2]]]) @ \
            np.block([[B[K-2], A[K-2], b[K-2]],
                      [np.eye(nu[K-2] + nx[K-2] + 1)]])
        temp = temp + \
            np.block([[Fu[K-2], B[K-2].T],
                      [Fx[K-2], A[K-2].T]]) @ \
            np.block([[B[K-2], A[K-2], b[K-2]],
                      [Fu[K-2].T, Fx[K-2].T, np.zeros((nx[K-1], 1))]])
        R_bar = temp[:nu[K-2], :nu[K-2]]
        S_bar = temp[:nu[K-2], nu[K-2]:nu[K-2]+nx[K-2]]
        Q_bar = temp[nu[K-2]:nu[K-2]+nx[K-2], nu[K-2]:nu[K-2]+nx[K-2]]
        r_bar = temp[:nu[K-2], -1:]
        q_bar = temp[nu[K-2]:, -1:]

        temp = np.block([[Hu[K-2], Hx[K-2], h[K-2]],
                         [
                             np.block([[Hx[K-1], h[K-1]]]) @ \
                             np.block([[B[K-2], A[K-2], b[K-2]],
                                       [np.zeros((nx[K-1], nu[K-2] + nx[K-2])), 1]]),
                         ]])
        Hu_bar = temp[:, :nu[K-2]]
        Hx_bar = temp[:, nu[K-2]:nu[K-2]+nx[K-2]]
        h_bar = temp[:, -1:]

        R_c[K-2] = R_bar
        S_c[K-2] = S_bar
        Q_c[K-2] = Q_bar
        r_c[K-2] = r_bar
        q_c[K-2] = q_bar
        Hu_c[K-2] = Hu_bar
        Hx_c[K-2] = Hx_bar
        h_c[K-2] = h_bar
        nx_c[K-1] = 0
        nu_c[K-1] = 0
        ng_eq_c[K-1] = 0
        ng_eq_c[K-2] = h_bar.shape[0]
        
        # solution = Solve(K - 1, nu_c, nx_c, ng_eq_c, R_c, S_c, Q_c, Gu_c, Gx_c, Fu_c, Fx_c,
        #       Hu_c, Hx_c, B_c, A_c, r_c, q_c, h_c, b_c, Pl, Pr, L, U,
        #       Lmbd, rank_k_values)
        KKT, rhs = GetKKT(K-1, nu_c, nx_c, ng_eq_c, R_c, S_c, Q_c, Gu_c, Gx_c, Fu_c, Fx_c, Hu_c, Hx_c, B_c, A_c, r_c, q_c, h_c, b_c)
        solution_vector = np.linalg.solve(KKT, -rhs)
        solution = extract_solultion(K-1, nu_c, nx_c, ng_eq_c, solution_vector)
        
        # recover full solution
        x = B[K-2] @ solution["u"][K-2] + A[K-2] @ solution["x"][K-2] + b[K-2]
        solution["x"].append(x)
        lmbd = solution["lambda"][K-2]
        lmbd_true = lmbd[:ng_eq[K-2]]
        mu = lmbd[ng_eq[K-2]:]
        solution["lambda"][K-2] = lmbd_true
        solution["lambda"].append(mu)
        pi = Q[K-1] @ x + Hx[K-1].T @ solution["lambda"][K-1] + q[K-1] + Fu[K-2].T @ solution["u"][K-2] + Fx[K-2].T @ solution["x"][K-2]
        solution["pi"].append(pi)
        solution["u"].append(np.zeros((nu[K-1], 1)))

    else:
        KKT, rhs = GetKKT(K, nu, nx, ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b)
        solution_vector = np.linalg.solve(KKT, -rhs)
        solution = extract_solultion(K, nu, nx, ng_eq, solution_vector)

    return solution

        


        