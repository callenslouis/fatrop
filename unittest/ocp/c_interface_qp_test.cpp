//
// Copyright (c) Lander Vanroye, KU Leuven
//
// Tests for the constant-data (QP) path of the v0-compatible C interface: with
// has_constant_hessian/has_constant_jacobian set, FatropOcpCMapping evaluates the callbacks once
// per solve and computes the Hessian, Jacobian, constraint violation, objective gradient and
// objective itself. These tests compare that path against the plain callback path.
//
// The C header is included through the internal header, which wraps it in namespace fatrop.
#include "fatrop/ocp/OCPCInterfaceInternal.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace fatrop;

namespace
{
    // ---------------------------------------------------------------------------
    // Random, feasible and convex OCP-QP. Per stage, with z = [u; x] (n = nu + nx):
    //   objective     0.5 z^T H z + q^T z
    //   dynamics      x_{k+1} = D z + b
    //   equalities    G z + g0 = 0
    //   inequalities  lo <= Gi z + gi0 <= up
    // All matrices are stored row-major.
    // ---------------------------------------------------------------------------
    struct Stage
    {
        int nu, nx, ng, ngi;
        std::vector<double> H, q, D, b, G, g0, Gi, gi0, lo, up;
        int n() const { return nu + nx; }
    };
    struct RandomQp
    {
        std::vector<Stage> stages;
    };

    double row_dot(const std::vector<double> &M, int row, int n, const double *u, int nu,
                   const double *x)
    {
        double res = 0.;
        for (int j = 0; j < nu; j++)
            res += M[row * n + j] * u[j];
        for (int j = nu; j < n; j++)
            res += M[row * n + j] * x[j - nu];
        return res;
    }

    RandomQp make_qp(unsigned seed, int K)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> unif(-1., 1.);
        std::uniform_real_distribution<double> margin(0.5, 2.);
        auto rand_vec = [&](int m) {
            std::vector<double> v(m);
            for (auto &e : v)
                e = unif(gen);
            return v;
        };
        RandomQp qp;
        qp.stages.resize(K);
        const int nx = 4;
        for (int k = 0; k < K; k++)
        {
            Stage &s = qp.stages[k];
            s.nx = nx;
            s.nu = (k == K - 1) ? 0 : 3;
            s.ng = (k == K - 1) ? 2 : 1;
            s.ngi = 3;
        }
        // feasible trajectory used to construct the constraints
        std::vector<double> x = rand_vec(nx);
        for (int k = 0; k < K; k++)
        {
            Stage &s = qp.stages[k];
            const int n = s.n();
            std::vector<double> u = rand_vec(s.nu);
            // H = M^T M + 0.1 I
            std::vector<double> M = rand_vec(n * n);
            s.H.assign(n * n, 0.);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    for (int l = 0; l < n; l++)
                        s.H[i * n + j] += M[l * n + i] * M[l * n + j];
                    if (i == j)
                        s.H[i * n + j] += 0.1;
                }
            s.q = rand_vec(n);
            s.G = rand_vec(s.ng * n);
            s.g0.resize(s.ng);
            for (int i = 0; i < s.ng; i++)
                s.g0[i] = -row_dot(s.G, i, n, u.data(), s.nu, x.data());
            s.Gi = rand_vec(s.ngi * n);
            s.gi0 = rand_vec(s.ngi);
            s.lo.resize(s.ngi);
            s.up.resize(s.ngi);
            for (int i = 0; i < s.ngi; i++)
            {
                const double val = row_dot(s.Gi, i, n, u.data(), s.nu, x.data()) + s.gi0[i];
                s.lo[i] = val - margin(gen);
                s.up[i] = val + margin(gen);
            }
            if (k != K - 1)
            {
                // scale the dynamics down to keep the trajectory well-scaled
                s.D = rand_vec(nx * n);
                for (auto &e : s.D)
                    e *= 0.5;
                s.b = rand_vec(nx);
                std::vector<double> x_next(nx);
                for (int i = 0; i < nx; i++)
                    x_next[i] = row_dot(s.D, i, n, u.data(), s.nu, x.data()) + s.b[i];
                x = x_next;
            }
        }
        return qp;
    }

    const Stage &stage(void *user_data, fatrop_int k)
    {
        return static_cast<RandomQp *>(user_data)->stages[k];
    }

    fatrop_int get_nx(fatrop_int k, void *ud) { return stage(ud, k).nx; }
    fatrop_int get_nu(fatrop_int k, void *ud) { return stage(ud, k).nu; }
    fatrop_int get_ng(fatrop_int k, void *ud) { return stage(ud, k).ng; }
    fatrop_int get_ng_ineq(fatrop_int k, void *ud) { return stage(ud, k).ngi; }
    fatrop_int get_horizon_length(void *ud)
    {
        return static_cast<RandomQp *>(ud)->stages.size();
    }
    fatrop_int no_params_k(fatrop_int, void *) { return 0; }
    fatrop_int no_params(void *) { return 0; }

    fatrop_int eval_BAbt(const double *states_kp1, const double *inputs_k, const double *states_k,
                         const double *, const double *, struct blasfeo_dmat *res, fatrop_int k,
                         void *ud)
    {
        const Stage &s = stage(ud, k);
        const int n = s.n();
        const int nx_next = stage(ud, k + 1).nx;
        for (int j = 0; j < nx_next; j++)
        {
            for (int i = 0; i < n; i++)
                blasfeo_dgein1(s.D[j * n + i], res, i, j);
            blasfeo_dgein1(row_dot(s.D, j, n, inputs_k, s.nu, states_k) + s.b[j] - states_kp1[j],
                           res, n, j);
        }
        return 0;
    }
    fatrop_int eval_b(const double *states_kp1, const double *inputs_k, const double *states_k,
                      const double *, const double *, double *res, fatrop_int k, void *ud)
    {
        const Stage &s = stage(ud, k);
        for (int j = 0; j < stage(ud, k + 1).nx; j++)
            res[j] = row_dot(s.D, j, s.n(), inputs_k, s.nu, states_k) + s.b[j] - states_kp1[j];
        return 0;
    }
    // Shared implementation of the (in)equality constraint callbacks.
    void eval_Ggt_impl(const std::vector<double> &G, const std::vector<double> &g0, int m,
                       const Stage &s, const double *inputs_k, const double *states_k,
                       struct blasfeo_dmat *res)
    {
        const int n = s.n();
        for (int j = 0; j < m; j++)
        {
            for (int i = 0; i < n; i++)
                blasfeo_dgein1(G[j * n + i], res, i, j);
            blasfeo_dgein1(row_dot(G, j, n, inputs_k, s.nu, states_k) + g0[j], res, n, j);
        }
    }
    fatrop_int eval_Ggt(const double *inputs_k, const double *states_k, const double *,
                        const double *, struct blasfeo_dmat *res, fatrop_int k, void *ud)
    {
        const Stage &s = stage(ud, k);
        eval_Ggt_impl(s.G, s.g0, s.ng, s, inputs_k, states_k, res);
        return 0;
    }
    fatrop_int eval_g(const double *inputs_k, const double *states_k, const double *,
                      const double *, double *res, fatrop_int k, void *ud)
    {
        const Stage &s = stage(ud, k);
        for (int j = 0; j < s.ng; j++)
            res[j] = row_dot(s.G, j, s.n(), inputs_k, s.nu, states_k) + s.g0[j];
        return 0;
    }
    fatrop_int eval_Ggt_ineq(const double *inputs_k, const double *states_k, const double *,
                             const double *, struct blasfeo_dmat *res, fatrop_int k, void *ud)
    {
        const Stage &s = stage(ud, k);
        eval_Ggt_impl(s.Gi, s.gi0, s.ngi, s, inputs_k, states_k, res);
        return 0;
    }
    fatrop_int eval_gineq(const double *inputs_k, const double *states_k, const double *,
                          const double *, double *res, fatrop_int k, void *ud)
    {
        const Stage &s = stage(ud, k);
        for (int j = 0; j < s.ngi; j++)
            res[j] = row_dot(s.Gi, j, s.n(), inputs_k, s.nu, states_k) + s.gi0[j];
        return 0;
    }
    // objective gradient H z + q (unscaled)
    std::vector<double> gradient(const Stage &s, const double *inputs_k, const double *states_k)
    {
        std::vector<double> grad(s.n());
        for (int i = 0; i < s.n(); i++)
            grad[i] = row_dot(s.H, i, s.n(), inputs_k, s.nu, states_k) + s.q[i];
        return grad;
    }
    fatrop_int eval_RSQrqt(const double *objective_scale, const double *inputs_k,
                           const double *states_k, const double *lam_dyn_k, const double *lam_eq_k,
                           const double *lam_ineq_k, const double *, const double *,
                           struct blasfeo_dmat *res, fatrop_int k, void *ud)
    {
        const Stage &s = stage(ud, k);
        const int n = s.n();
        std::vector<double> rhs = gradient(s, inputs_k, states_k);
        for (int i = 0; i < n; i++)
        {
            rhs[i] *= *objective_scale;
            for (int j = 0; j < n; j++)
                blasfeo_dgein1(*objective_scale * s.H[i * n + j], res, i, j);
        }
        // right-hand-side row: gradient of the Lagrangian
        if (lam_dyn_k)
            for (int l = 0; l < stage(ud, k + 1).nx; l++)
                for (int i = 0; i < n; i++)
                    rhs[i] += s.D[l * n + i] * lam_dyn_k[l];
        for (int l = 0; l < s.ng; l++)
            for (int i = 0; i < n; i++)
                rhs[i] += s.G[l * n + i] * lam_eq_k[l];
        for (int l = 0; l < s.ngi; l++)
            for (int i = 0; i < n; i++)
                rhs[i] += s.Gi[l * n + i] * lam_ineq_k[l];
        for (int i = 0; i < n; i++)
            blasfeo_dgein1(rhs[i], res, n, i);
        return 0;
    }
    fatrop_int eval_rq(const double *objective_scale, const double *inputs_k,
                       const double *states_k, const double *, const double *, double *res,
                       fatrop_int k, void *ud)
    {
        const Stage &s = stage(ud, k);
        std::vector<double> grad = gradient(s, inputs_k, states_k);
        for (int i = 0; i < s.n(); i++)
            res[i] = *objective_scale * grad[i];
        return 0;
    }
    fatrop_int eval_L(const double *objective_scale, const double *inputs_k,
                      const double *states_k, const double *, const double *, double *res,
                      fatrop_int k, void *ud)
    {
        const Stage &s = stage(ud, k);
        double obj = 0.;
        for (int i = 0; i < s.n(); i++)
        {
            const double zi = i < s.nu ? inputs_k[i] : states_k[i - s.nu];
            obj += zi * (0.5 * row_dot(s.H, i, s.n(), inputs_k, s.nu, states_k) + s.q[i]);
        }
        res[0] = *objective_scale * obj;
        return 0;
    }
    fatrop_int get_bounds(double *lower, double *upper, fatrop_int k, void *ud)
    {
        const Stage &s = stage(ud, k);
        for (int i = 0; i < s.ngi; i++)
        {
            lower[i] = s.lo[i];
            upper[i] = s.up[i];
        }
        return 0;
    }
    fatrop_int initial_uk(double *uk, fatrop_int k, void *ud)
    {
        for (int i = 0; i < stage(ud, k).nu; i++)
            uk[i] = 0.;
        return 0;
    }
    fatrop_int initial_xk(double *xk, fatrop_int k, void *ud)
    {
        for (int i = 0; i < stage(ud, k).nx; i++)
            xk[i] = 0.;
        return 0;
    }
    // "fall back to the stage-wise callbacks", like CasADi
    fatrop_int full_eval_lag_hess(double, const double *, const double *, const double *,
                                  const double *, struct blasfeo_dmat *,
                                  const FatropOcpCDims *, void *)
    {
        return 0;
    }
    fatrop_int full_eval_constr_jac(const double *, const double *, const double *,
                                    struct blasfeo_dmat *, struct blasfeo_dmat *,
                                    struct blasfeo_dmat *, const FatropOcpCDims *, void *)
    {
        return 0;
    }
    fatrop_int full_eval_contr_viol(const double *, const double *, const double *, double *,
                                    const FatropOcpCDims *, void *)
    {
        return 0;
    }
    fatrop_int full_eval_obj_grad(double, const double *, const double *, const double *,
                                  double *, const FatropOcpCDims *, void *)
    {
        return 0;
    }
    fatrop_int full_eval_obj(double, const double *, const double *, const double *, double *,
                             const FatropOcpCDims *, void *)
    {
        return 0;
    }

    FatropOcpCInterface make_interface(RandomQp *qp, bool constant)
    {
        FatropOcpCInterface ocp = {};
        ocp.get_nx = get_nx;
        ocp.get_nu = get_nu;
        ocp.get_ng = get_ng;
        ocp.get_ng_ineq = get_ng_ineq;
        ocp.get_horizon_length = get_horizon_length;
        ocp.get_n_stage_params = no_params_k;
        ocp.get_n_global_params = no_params;
        ocp.eval_BAbt = eval_BAbt;
        ocp.eval_b = eval_b;
        ocp.eval_Ggt = eval_Ggt;
        ocp.eval_g = eval_g;
        ocp.eval_Ggt_ineq = eval_Ggt_ineq;
        ocp.eval_gineq = eval_gineq;
        ocp.eval_RSQrqt = eval_RSQrqt;
        ocp.eval_rq = eval_rq;
        ocp.eval_L = eval_L;
        ocp.get_bounds = get_bounds;
        ocp.get_initial_uk = initial_uk;
        ocp.get_initial_xk = initial_xk;
        ocp.full_eval_lag_hess = full_eval_lag_hess;
        ocp.full_eval_constr_jac = full_eval_constr_jac;
        ocp.full_eval_contr_viol = full_eval_contr_viol;
        ocp.full_eval_obj_grad = full_eval_obj_grad;
        ocp.full_eval_obj = full_eval_obj;
        ocp.has_constant_hessian = constant;
        ocp.has_constant_jacobian = constant;
        ocp.user_data = qp;
        return ocp;
    }

    void fill_random(VecRealView &v, std::mt19937 &gen)
    {
        std::uniform_real_distribution<double> unif(-1., 1.);
        for (Index i = 0; i < v.m(); i++)
            v(i) = unif(gen);
    }
    void expect_vec_near(const VecRealView &a, const VecRealView &b, double tol)
    {
        ASSERT_EQ(a.m(), b.m());
        for (Index i = 0; i < a.m(); i++)
            EXPECT_NEAR(a(i), b(i), tol) << "at index " << i;
    }
    void expect_mat_near(Index m, Index n, const MatRealView &a, const MatRealView &b, double tol)
    {
        for (Index i = 0; i < m; i++)
            for (Index j = 0; j < n; j++)
                EXPECT_NEAR(a(i, j), b(i, j), tol) << "at (" << i << ", " << j << ")";
    }
    void expect_hess_near(const ProblemInfo<OcpType> &info, const Hessian<OcpType> &a,
                          const Hessian<OcpType> &b)
    {
        for (Index k = 0; k < info.dims.K; k++)
        {
            const Index n = info.dims.number_of_controls[k] + info.dims.number_of_states[k];
            expect_mat_near(n, n, a.RSQrqt[k], b.RSQrqt[k], 1e-12);
        }
    }
    void expect_jac_near(const ProblemInfo<OcpType> &info, const Jacobian<OcpType> &a,
                         const Jacobian<OcpType> &b)
    {
        for (Index k = 0; k < info.dims.K; k++)
        {
            const Index n = info.dims.number_of_controls[k] + info.dims.number_of_states[k];
            expect_mat_near(n, info.dims.number_of_eq_constraints[k], a.Gg_eqt[k], b.Gg_eqt[k],
                            1e-12);
            expect_mat_near(n, info.dims.number_of_ineq_constraints[k], a.Gg_ineqt[k],
                            b.Gg_ineqt[k], 1e-12);
            if (k != info.dims.K - 1)
                expect_mat_near(n, info.dims.number_of_states[k + 1], a.BAbt[k], b.BAbt[k],
                                1e-12);
        }
    }

    struct SolveResult
    {
        int ret;
        int iterations;
        std::vector<double> primal;
    };
    SolveResult solve(FatropOcpCSolver *solver)
    {
        SolveResult res;
        res.ret = fatrop_ocp_c_solve(solver);
        res.iterations = fatrop_ocp_c_get_stats(solver)->iterations_count;
        blasfeo_dvec *primal = const_cast<blasfeo_dvec *>(fatrop_ocp_c_get_primal(solver));
        for (int i = 0; i < primal->m; i++)
            res.primal.push_back(BLASFEO_DVECEL(primal, i));
        return res;
    }
    // Options as used by the benchmark. Without qp_reg_enabled, the QP algorithm can stop
    // with an INDEFINITE factorization close to convergence, with or without constant data.
    void set_options(FatropOcpCSolver *solver, bool qp_mode)
    {
        fatrop_ocp_c_set_option_int(solver, "print_level", 0);
        if (qp_mode)
        {
            fatrop_ocp_c_set_option_bool(solver, "qp", 1);
            fatrop_ocp_c_set_option_bool(solver, "qp_reg_enabled", 1);
        }
    }
    SolveResult solve_fresh(RandomQp *qp, bool constant, bool qp_mode)
    {
        FatropOcpCInterface ocp = make_interface(qp, constant);
        FatropOcpCSolver *solver = fatrop_ocp_c_create(&ocp, nullptr, nullptr);
        set_options(solver, qp_mode);
        SolveResult res = solve(solver);
        fatrop_ocp_c_destroy(solver);
        return res;
    }
    void expect_same_solution(const SolveResult &a, const SolveResult &b)
    {
        EXPECT_EQ(a.ret, 0);
        EXPECT_EQ(b.ret, 0);
        EXPECT_EQ(a.iterations, b.iterations);
        ASSERT_EQ(a.primal.size(), b.primal.size());
        for (size_t i = 0; i < a.primal.size(); i++)
            EXPECT_NEAR(a.primal[i], b.primal[i], 1e-8) << "at index " << i;
    }
} // namespace

// The evaluations computed from the constant data must match the callbacks.
TEST(CInterfaceQpTest, ConstantDataMatchesCallbacks)
{
    RandomQp qp = make_qp(42, 6);
    FatropOcpCInterface ocp_ref = make_interface(&qp, false);
    FatropOcpCInterface ocp_fast = make_interface(&qp, true);
    FatropOcpCMapping ref(&ocp_ref), fast(&ocp_fast);
    ProblemInfo<OcpType> info(ref.problem_dims());

    std::mt19937 gen(7);
    VecRealAllocated primal_x(info.number_of_primal_variables);
    VecRealAllocated primal_s(info.number_of_slack_variables);
    VecRealAllocated lam(info.number_of_eq_constraints);
    fill_random(primal_x, gen);
    fill_random(primal_s, gen);
    fill_random(lam, gen);

    // buffers are reused across the checks below, like the IP algorithm does
    Hessian<OcpType> hess_ref(info.dims), hess_fast(info.dims);
    Jacobian<OcpType> jac_ref(info.dims), jac_fast(info.dims);

    auto check_all = [&](Scalar objective_scale) {
        VecRealAllocated cv_ref(info.number_of_eq_constraints), cv_fast(info.number_of_eq_constraints);
        ref.eval_constraint_violation(info, primal_x, primal_s, cv_ref);
        fast.eval_constraint_violation(info, primal_x, primal_s, cv_fast);
        expect_vec_near(cv_ref, cv_fast, 1e-12);

        VecRealAllocated gx_ref(info.number_of_primal_variables),
            gx_fast(info.number_of_primal_variables);
        VecRealAllocated gs_ref(info.number_of_slack_variables),
            gs_fast(info.number_of_slack_variables);
        ref.eval_objective_gradient(info, objective_scale, primal_x, primal_s, gx_ref, gs_ref);
        fast.eval_objective_gradient(info, objective_scale, primal_x, primal_s, gx_fast, gs_fast);
        expect_vec_near(gx_ref, gx_fast, 1e-12);
        expect_vec_near(gs_ref, gs_fast, 1e-12);

        Scalar obj_ref = 0., obj_fast = 0.;
        ref.eval_objective(info, objective_scale, primal_x, primal_s, obj_ref);
        fast.eval_objective(info, objective_scale, primal_x, primal_s, obj_fast);
        EXPECT_NEAR(obj_ref, obj_fast, 1e-12);

        ref.eval_lag_hess(info, objective_scale, primal_x, primal_s, lam, hess_ref);
        fast.eval_lag_hess(info, objective_scale, primal_x, primal_s, lam, hess_fast);
        expect_hess_near(info, hess_ref, hess_fast);

        ref.eval_constr_jac(info, primal_x, primal_s, jac_ref);
        fast.eval_constr_jac(info, primal_x, primal_s, jac_fast);
        expect_jac_near(info, jac_ref, jac_fast);
    };

    check_all(1.);
    // a different objective scale on the same Hessian buffer must rescale the block
    check_all(0.3);
    check_all(0.);
    // after a data change and invalidation, the cached blocks must not be reused
    for (auto &s : qp.stages)
    {
        for (auto &e : s.H)
            e *= 2.;
        for (auto &e : s.q)
            e += 1.;
        for (auto &e : s.D)
            e *= -1.;
        for (auto &e : s.G)
            e *= 3.;
        for (auto &e : s.gi0)
            e -= 0.5;
    }
    fast.invalidate_constant_data();
    check_all(1.);
}

// Solving with the constant-data path must follow the same iterates as the callback path, for
// both the Mehrotra QP algorithm and the general IP algorithm.
TEST(CInterfaceQpTest, SolveMatchesCallbacks)
{
    for (bool qp_mode : {true, false})
    {
        SCOPED_TRACE(qp_mode ? "qp algorithm" : "ip algorithm");
        RandomQp qp = make_qp(3, 10);
        expect_same_solution(solve_fresh(&qp, false, qp_mode), solve_fresh(&qp, true, qp_mode));
    }
}

// Re-solving with the same solver object after the QP data changed must not reuse stale data.
TEST(CInterfaceQpTest, ResolveAfterDataChange)
{
    for (bool qp_mode : {true, false})
    {
        SCOPED_TRACE(qp_mode ? "qp algorithm" : "ip algorithm");
        RandomQp qp = make_qp(5, 8);
        FatropOcpCInterface ocp = make_interface(&qp, true);
        FatropOcpCSolver *solver = fatrop_ocp_c_create(&ocp, nullptr, nullptr);
        set_options(solver, qp_mode);
        EXPECT_EQ(solve(solver).ret, 0);

        // change the cost and the dynamics, keeping the problem feasible and convex
        for (auto &s : qp.stages)
        {
            for (auto &e : s.H)
                e *= 1.5;
            for (auto &e : s.q)
                e -= 0.7;
            for (auto &e : s.b)
                e += 0.1;
        }
        SolveResult resolved = solve(solver);
        fatrop_ocp_c_destroy(solver);
        expect_same_solution(solve_fresh(&qp, false, qp_mode), resolved);
    }
}
