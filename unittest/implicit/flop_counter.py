import numpy as np

class FlopCounter:
    def __init__(self):
        self.flops = 0
        self.implicit = False

        self.rank_mode = 'random'    

    def set_dimensions(self, K, nu, nx, r, ng, ng_ineq):
        self.K = K
        if isinstance(nu, int) or isinstance(nu, np.int64):
            nu = [nu for _ in range(K)]
        if isinstance(nx, int) or isinstance(nx, np.int64):
            nx = [nx for _ in range(K)]
        if isinstance(r, int) or isinstance(r, np.int64):
            r = [r for _ in range(K-1)]
        if isinstance(ng, int) or isinstance(ng, np.int64):
            ng = [ng for _ in range(K)]
        if isinstance(ng_ineq, int) or isinstance(ng_ineq, np.int64):
            ng_ineq = [ng_ineq for _ in range(K)]
        self.nu = nu
        self.nx = nx
        self.r = r
        self.ng = ng
        self.ng_ineq = ng_ineq

        self.rho = [0 for _ in range(K)]
        self.gamma = [0 for _ in range(K)]   

    def get_all_flops(self, data):
        impl = []
        expl = []
        reform = []
        for i in range(len(data['K'])):
            self.set_dimensions(data['K'][i], data['nu'][i], data['nx'][i], 
                                data['r'][i], data['ng'][i], data['ng_ineq'][i])
            self.get_flop_implicit(); impl.append(self.flops)
            self.get_flop_explicit(); expl.append(self.flops)
            self.get_flop_reformulated(); reform.append(self.flops)

        return np.array(impl), np.array(expl), np.array(reform)


    def get_flop_explicit(self):
        self.implicit = False
        self.flops = 0
        self.backward_recursion()
        self.first_stage()
        self.forward_substitution()

        return self.flops

    def get_flop_implicit(self):
        self.implicit = True
        self.flops = 0
        self.preprocess()
        self.backward_recursion()
        self.first_stage()
        self.forward_substitution()
        self.postprocess()

        return self.flops

    def get_flop_reformulated(self):
        for k in range(self.K-1):
            self.nu[k] += self.nx[k+1]
            self.ng[k] += self.nx[k+1]

        flops = self.get_flop_explicit()

        for k in range(self.K-1):
            self.nu[k] -= self.nx[k+1]
            self.ng[k] -= self.nx[k+1]

    def set_constraint_jacobian_rank_mode(self, mode):
        if mode != 'full' and mode != 'random':
            raise ValueError(f"Unknown constraint jacobian rank mode: {mode}")
        self.rank_mode = mode

    def backward_recursion(self):
        Hkp_rows = self.ng[self.K-1]
        gamma = Hkp_rows
        self.gamma_I = Hkp_rows
        for k in range(self.K-1, -1, -1):
            nu = self.nu[k]
            nx = self.nx[k]
            ng = self.ng[k]
            ng_ineq = self.ng_ineq[k]
            nunxm1 = self.nu[k-1] + self.nx[k-1] if k > 0 else 0
            nxp1 = self.nx[k+1] if k < self.K-1 else 0

            if k < self.K-1:
                self.gemm(nu + nx + 1, nxp1, nxp1)
                self.gead(1, nxp1)
                self.syrk(nu + nx + 1, nu + nx, nxp1)

                # second order dynamics contribution
                if self.implicit:
                    self.gemm(nu + nx + 1, nu + nx, nxp1)
                    self.gemm(nu + nx, nu + nx, nxp1)

                gamma = self.ng[k] + Hkp_rows
                if gamma > 0 and Hkp_rows > 0:
                    self.gemm(nu + nx + 1, Hkp_rows, nxp1)
                    self.gead(1, Hkp_rows)

            if not self.implicit:
                if self.ng_ineq[k] > 0:
                    for i in range(self.ng_ineq[k]):
                        self.colsc(nu + nx + 1)
                        self.flops += nu + nx + 1
                    self.syrk(nu + nx + 1, nu + nx, self.ng_ineq[k])

                self.flops += self.nu[k] + self.nx[k]

            # decomposition of constraint jacobian
            rank = min(gamma, nu + nx) if self.rank_mode == 'full' else 0 if min(gamma, nu + nx) == 0 else np.random.randint(0, min(gamma, nu + nx))
            self.lu_fact(gamma, nu + nx + 1, rank)
            if rank > 0:
                self.trsm(nu - rank + nx + 1, rank)
                self.gemm(nu - rank + nx + 1, nu + nx, rank)
                self.syrk(nu - rank + nx + 1, nu + nx - rank, rank)

            if self.implicit and k > 0 and self.r[k-1] < self.nx[k]:
                self.gemm(nunxm1, nu - rank, rank)
                self.gemm(nunxm1, nx, rank)
                self.gemm(1, nunxm1, rank)

            # shur complement step
            if nu - rank > 0:
                self.potrf(nu - rank, nu - rank)
                self.syrk(nx + 1, nx, nu - rank)
                # skip increased accuracy steps
                if self.implicit and k > 0 and self.r[k-1] < self.nx[k]:
                    self.trsm(nunxm1, nu - rank)
                    self.gemm(nunxm1, nunxm1, nu - rank)
                    self.trsm(1, nu - rank)
                    self.gemm(1, nunxm1, nu-rank)
                    self.gemm(nunxm1, nx, nu - rank)

            self.gamma[k] = gamma
            self.rho[k] = rank
            self.gamma_I = gamma - rank

    def first_stage(self):
        nx = self.nx[0]
        self.rank_I = min(self.gamma_I, nx + 1) if self.rank_mode == 'full' else 0 if min(self.gamma_I, nx+1) == 0 else np.random.randint(0, min(self.gamma_I, nx + 1))
        if self.gamma_I > 0:
            self.lu_fact(self.gamma_I, nx + 1, self.rank_I)

            self.trsm(nx - self.rank_I + 1, self.rank_I)
            self.gemm(nx - self.rank_I + 1, nx, self.rank_I)
            self.syrk(nx - self.rank_I + 1, nx - self.rank_I, self.rank_I)
            self.potrf(nx - self.rank_I + 1, nx - self.rank_I)
        else:
            self.potrf(nx + 1, nx)

        self.trsv(nx - self.rank_I)
        self.gemv(nx - self.rank_I, self.rank_I)
        self.gemv(nx, self.rank_I)
        self.trsv(self.rank_I)
        self.trsv(self.rank_I)

    def forward_substitution(self):
        for k in range(self.K):
            nx = self.nx[k]
            nu = self.nu[k]
            rho = self.rho[k]
            gammamrho = self.gamma[k] - rho
            gamma = self.gamma[k]
            numrho = nu - rho

            if numrho > 0:
                # skip increased accuracy
                self.gemv(nx, numrho)
                self.trsv(numrho)

                if self.implicit and k > 0 and self.r[k-1] < self.nx[k]:
                    nunxm1 = self.nu[k-1] + self.nx[k-1]
                    self.trsm(nunxm1, numrho)
                    self.gemv(nunxm1, numrho)

            if rho > 0:
                self.gemv(nx + numrho, rho)
                self.gemv(nu + nx, rho)
                if self.implicit and k > 0 and self.r[k-1] < self.nx[k]:
                    self.gemv(nunxm1, rho)
                self.trsv(rho)
                self.trsv(rho)

            # regularization steps
            if not self.implicit:
                if self.ng_ineq[k] > 0:
                    self.gemv(nu + nx, self.ng_ineq[k])
                    self.flops += self.ng_ineq[k]

            if k != self.K - 1:
                nxp1 = self.nx[k+1]
                nup1 = self.nu[k+1]
                gammamrhop1 = self.gamma[k+1] - self.rho[k+1]

                self.gemv(nu + nx, nxp1)
                self.gemv(nxp1, nxp1)
                self.gemv(gammamrho, nxp1)
                if self.implicit:
                    self.gemv(nu + nx, nxp1)

    def preprocess(self):
        if not self.implicit:
            self.flops += 0
            return
        
        # regularization
        for k in range(self.K):
            
            # D_s
            for i in range(self.ng_ineq[k]):
                self.colsc(self.nu[k] + self.nx[k] + 1)
            self.syrk(self.nu[k] + self.nx[k] + 1, self.nu[k] + self.nx[k], self.ng_ineq[k])
            self.gemv(self.nu[k] + self.nx[k], self.ng_ineq[k])

            # D_x
            self.flops += self.nu[k] * self.nx[k]

        # Preprocess loop
        for k in range(self.K-1):
            nx = self.nx[k]
            nu = self.nu[k]
            nx_next = self.nx[k+1]
            nu_next = self.nu[k+1]
            r = self.r[k]

            # decomposition
            self.lu_fact(nx_next, nx_next + nu + nx + 1, r)
            self.trsm(nx_next + nu + nx + 1, nx_next)
            self.trsm(nx_next + nu + nx + 1, r)

            # RSQrqt
            self.gemm(nx_next - r, nu_next + nx_next, r)
            self.gemm(nu_next + nx_next + 1, nx_next - r, r)
            
            # FuFx
            self.gemm(nu + nx, nx_next - r, r)

            if (k < self.K-2):
                nx_next_next = self.nx[k+2]
                self.gemm(nx_next - r, nx_next_next, r)
                self.gemm(nx_next - r, nx_next_next, r)

            # equality constraints
            self.gemm(nx_next - r, self.ng[k+1], r)

        # modify rhs
        #   only memory operations, no flops

    def postprocess(self):
        if not self.implicit:
            self.flops += 0
            return
        
        # postprocess loop
        for k in range(self.K):
            # copies

            if k > 0:
                self.gemv(self.nx[k] - self.r[k-1], self.nx[k])

                self.trsv(self.r[k-1])
                self.trsv(self.nx[k])
            
        # constraint regularization (inequalities)
        for k in range(self.K):
            self.gemv(self.nu[k] + self.nx[k], self.ng_ineq[k])
            self.flops += self.ng_ineq[k]

        
    def colsc(self, n):
        self.flops += n

    def gead(self, m, n):
        self.flops += 2 * m * n

    def gemm(self, m, n, k):
        self.flops += m * n * (2 * k + 2)
    
    def gemv(self, m, n):
        self.gemm(m, 1, n)

    def trsm(self, m, n):
        self.flops += m * n**2
    
    def trsv(self, m):
        self.trsm(m, 1)

    def potrf(self, m, n):
        self.flops += (m-n)*n**2 + n*(n-1)*(n+1)/6 + (n-1)*(n-2)*(n+1)/6 + n*(n-1)/2 + n*(n-1)/2 + n

    def syrk(self, m, n, k):
        self.flops += (2*k+2)*n*(2*m-n+1)/2

    def lu_fact(self, m, n, r):
        self.flops += 2*r*(m-1)*(n-1) + r*(m-1) + r*(r-1)*(2*r-1)/3 - r*(2*m+2*n-3)*(r-1)/2
