// AugSystemSolver<ImplicitOcpType>::solve_rhs has to solve the same system as the last solve, only for another
// rhs (iterative refinement relies on this). These tests compare solve_rhs with solve and check the KKT residual
// for OCPs with the structure J[k] = [D C; 0 -I] (nz implicit states, ne explicit states, nv implicit
// equations), with inequalities, second order dynamics terms (FuFx) and rank deficient implicit blocks D (which
// turn states into controls of the next stage), with and without the fast solve_rhs.
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include "fatrop/ocp/type.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

using namespace fatrop;

namespace
{
    struct Case
    {
        Index ng_ineq;   // inequalities per stage
        double D_s;      // scale of the inequality regularization
        double fufx;     // scale of the second order dynamics terms FuFx
        double D_x;      // primal regularization
        Index rank_D;    // rank of the implicit block D (-1: full rank)
        bool fast_path;  // linsol_implicit_rhs_fast_path
    };

    double rnd(std::mt19937 &gen, double a = -1., double b = 1.)
    {
        return std::uniform_real_distribution<double>(a, b)(gen);
    }

    // max |constraint residual|, max |gradient residual| of the KKT system
    std::pair<double, double> kkt_residual(const ProblemInfo &info, const Jacobian<ImplicitOcpType> &jac,
                                           const Hessian<ImplicitOcpType> &hess, const VecRealView &D_x,
                                           const VecRealView &D_s, const VecRealView &f, const VecRealView &g,
                                           const VecRealView &x, const VecRealView &mult)
    {
        VecRealAllocated jx(info.number_of_eq_constraints), rg(info.number_of_eq_constraints);
        jac.apply_on_right(info, x, 0.0, jx, jx);
        rg = 0.;
        rg = rg + g + jx;
        rg.block(info.number_of_slack_variables, info.offset_g_eq_slack) =
            rg.block(info.number_of_slack_variables, info.offset_g_eq_slack) -
            D_s * mult.block(info.number_of_slack_variables, info.offset_g_eq_slack);
        VecRealAllocated grad(info.number_of_primal_variables), tmp(info.number_of_primal_variables);
        hess.apply_on_right(info, x, 0.0, tmp, tmp);
        grad = tmp + D_x * x;
        jac.transpose_apply_on_right(info, mult, 0.0, tmp, tmp);
        grad = grad + tmp + f;
        double rc = 0., rgr = 0.;
        for (Index i = 0; i < rg.m(); ++i) rc = std::max(rc, std::abs(rg(i)));
        for (Index i = 0; i < grad.m(); ++i) rgr = std::max(rgr, std::abs(grad(i)));
        return {rc, rgr};
    }

    double max_abs_diff(const VecRealView &a, const VecRealView &b)
    {
        double d = 0.;
        for (Index i = 0; i < a.m(); ++i) d = std::max(d, std::abs(a(i) - b(i)));
        return d;
    }

    void run_case(const Case &c)
    {
        std::mt19937 gen(3);
        const int K = 6;
        const Index nz = 4, ne = 2, nx = nz + ne, nu = 12, nv = 8, ng = 2;
        std::vector<Index> nxs(K, nx), nus(K, nu), ngs(K, ng), ngis(K, c.ng_ineq), ndyn(K - 1, nv + ne);
        nus[K - 1] = 0;
        ProblemDims dims(K, nus, nxs, ngs, ngis, ndyn);
        ProblemInfo info(dims);
        Jacobian<ImplicitOcpType> jac(dims);
        Hessian<ImplicitOcpType> hess(dims);
        for (int k = 0; k < K; ++k)
        {
            const Index n = nus[k] + nx;
            if (k < K - 1)
            {
                for (Index i = 0; i < n; ++i)
                    for (Index j = 0; j < nv + ne; ++j) jac.BAbt[k](i, j) = rnd(gen);
                jac.Jt[k] = 0.;
                for (Index i = 0; i < nx; ++i)
                    for (Index j = 0; j < nv; ++j) jac.Jt[k](i, j) = rnd(gen); // [D C]^T
                if (c.rank_D >= 0)
                {
                    // D = P Q with P (nv x rank_D), Q (rank_D x nz)
                    std::vector<double> P(nv * c.rank_D), Q(c.rank_D * nz);
                    for (auto &v : P) v = rnd(gen);
                    for (auto &v : Q) v = rnd(gen);
                    for (Index i = 0; i < nz; ++i)
                        for (Index j = 0; j < nv; ++j)
                        {
                            double d = 0.;
                            for (Index l = 0; l < c.rank_D; ++l) d += P[j * c.rank_D + l] * Q[l * nz + i];
                            jac.Jt[k](i, j) = d;
                        }
                }
                for (Index j = 0; j < ne; ++j) jac.Jt[k](nz + j, nv + j) = -1.0; // [0 -I]^T
                for (Index i = 0; i < n; ++i)
                    for (Index j = 0; j < nx; ++j) hess.FuFx[k](i, j) = c.fufx * rnd(gen, 0., 1.);
            }
            for (Index i = 0; i < n; ++i)
                for (Index j = 0; j < ng; ++j) jac.Gg_eqt[k](i, j) = rnd(gen);
            for (Index i = 0; i < n; ++i)
                for (Index j = 0; j < c.ng_ineq; ++j) jac.Gg_ineqt[k](i, j) = rnd(gen);
            MatRealAllocated M(n, n), H(n, n);
            for (Index i = 0; i < n; ++i)
                for (Index j = 0; j < n; ++j) M(i, j) = rnd(gen);
            syrk_ln(n, n, 1.0, M, 0, 0, M, 0, 0, 0.0, H, 0, 0, H, 0, 0);
            trtr_l(n, H, 0, 0, H, 0, 0);
            hess.RSQrqt[k].block(n, n, 0, 0) = H;
        }
        VecRealAllocated D_x(info.number_of_primal_variables), D_s(info.number_of_slack_variables);
        VecRealAllocated f(info.number_of_primal_variables), g(info.number_of_eq_constraints);
        VecRealAllocated f2(info.number_of_primal_variables), g2(info.number_of_eq_constraints);
        for (Index i = 0; i < f.m(); ++i) { f(i) = rnd(gen); f2(i) = rnd(gen); D_x(i) = c.D_x; }
        for (Index i = 0; i < g.m(); ++i) { g(i) = rnd(gen); g2(i) = rnd(gen); }
        for (Index i = 0; i < D_s.m(); ++i) D_s(i) = c.D_s * rnd(gen, 0.5, 1.5);

        AugSystemSolver<ImplicitOcpType> solver(info);
        solver.set_nb_of_implicit_states(nz);
        solver.set_rhs_fast_path(c.fast_path);
        const Index n_x = info.number_of_primal_variables, n_m = info.number_of_eq_constraints;
        VecRealAllocated x1(n_x), m1(n_m), x2(n_x), m2(n_m), x3(n_x), m3(n_m), x4(n_x), m4(n_m);
        const double tol = 1e-9;

        // solve_rhs with the rhs of the last solve gives the same solution
        ASSERT_EQ(solver.solve(info, jac, hess, D_x, D_s, f, g, x1, m1), LinsolReturnFlag::SUCCESS);
        solver.solve_rhs(info, jac, hess, D_s, f, g, x2, m2);
        EXPECT_NEAR(max_abs_diff(x1, x2), 0., tol);
        EXPECT_NEAR(max_abs_diff(m1, m2), 0., tol);
        auto r1 = kkt_residual(info, jac, hess, D_x, D_s, f, g, x1, m1);
        auto r2 = kkt_residual(info, jac, hess, D_x, D_s, f, g, x2, m2);
        EXPECT_NEAR(r1.first, 0., tol);
        EXPECT_NEAR(r1.second, 0., tol);
        EXPECT_NEAR(r2.first, 0., tol);
        EXPECT_NEAR(r2.second, 0., tol);

        // solve_rhs for another rhs (twice: it must not change the stored factorization)
        solver.solve_rhs(info, jac, hess, D_s, f2, g2, x3, m3);
        solver.solve_rhs(info, jac, hess, D_s, f2, g2, x3, m3);
        auto r3 = kkt_residual(info, jac, hess, D_x, D_s, f2, g2, x3, m3);
        EXPECT_NEAR(r3.first, 0., tol);
        EXPECT_NEAR(r3.second, 0., tol);
        ASSERT_EQ(solver.solve(info, jac, hess, D_x, D_s, f2, g2, x4, m4), LinsolReturnFlag::SUCCESS);
        EXPECT_NEAR(max_abs_diff(x3, x4), 0., tol);
        EXPECT_NEAR(max_abs_diff(m3, m4), 0., tol);
    }
} // namespace

class ImplicitSolveRhsTest : public ::testing::TestWithParam<Case>
{
};

TEST_P(ImplicitSolveRhsTest, SolveRhsMatchesSolve) { run_case(GetParam()); }

INSTANTIATE_TEST_SUITE_P(
    Cases, ImplicitSolveRhsTest,
    ::testing::Values(
        // ng_ineq, D_s, fufx, D_x, rank_D, fast_path
        Case{0, 1.0, 0.0, 1.0, -1, true}, Case{0, 1.0, 0.5, 1.0, -1, true}, Case{3, 2.5, 0.0, 1.0, -1, true},
        Case{3, 2.5, 0.5, 1.0, -1, true}, Case{3, 2.5, 0.5, 0.0, -1, true}, Case{3, 2.5, 0.5, 1.0, -1, false},
        Case{0, 1.0, 0.5, 1.0, 3, true}, Case{3, 2.5, 0.0, 1.0, 3, true}, Case{3, 2.5, 0.5, 1.0, 3, true},
        Case{3, 2.5, 0.5, 0.0, 2, true}, Case{3, 2.5, 0.5, 1.0, 2, false}));
