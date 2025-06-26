//
// Copyright (c) Lander Vanroye, KU Leuven
//
#include "../random_matrix.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/dims.hpp" // inherit
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp" //inherit
#include "fatrop/ocp/type.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <casadi/casadi.hpp>

using namespace fatrop;
using namespace casadi;

MatRealAllocated get_inverse(const MatRealView &A)
{
    fatrop_dbg_assert(A.m() == A.n() && "Matrix must be square for inversion");
    MatRealAllocated A_inv(A.m(), A.m());
    MatRealAllocated LU(A.m(), A.m());
    blasfeo_dgetrf_np(A.m(), A.m(), const_cast<MAT *>(&A.mat()), 0, 0, &LU.mat(), 0, 0);

    // Solve the system LU * X = I, where I is the identity matrix
    MatRealAllocated I = ::test::identity_matrix(A.m());

    // (1) solve L Y = I
    blasfeo_dtrsm_llnu(A.m(), A.m(), 1.0, &LU.mat(), 0, 0, &I.mat(), 0, 0, &A_inv.mat(), 0, 0);
    // (2) solve U X = Y
    blasfeo_dtrsm_lunn(A.m(), A.m(), 1.0, &LU.mat(), 0, 0, &A_inv.mat(), 0, 0, &A_inv.mat(), 0, 0);

    // std::cout << "Inverse of A: \n" << A << "\nis given by \n"
    //           << A_inv << std::endl;

    // check if A_inv contains any NaN or Inf values
    for (Index i = 0; i < A_inv.m(); ++i)
    {
        for (Index j = 0; j < A_inv.n(); ++j)
        {
            if (std::isnan(A_inv(i, j)) || std::isinf(A_inv(i, j)))
            {
                throw std::runtime_error("Inverse contains NaN or Inf values");
            }
        }
    }

    // check result
    MatRealAllocated I_check = ::test::identity_matrix(A.m());
    blasfeo_dgemm_nn(A.m(), A.m(), A.m(), 1.0, 
                     const_cast<MAT *>(&A.mat()), 0, 0, 
                     const_cast<MAT *>(&A_inv.mat()), 0, 0, 0.0, 
                     const_cast<MAT *>(&I_check.mat()), 0, 0,
                     const_cast<MAT *>(&I_check.mat()), 0, 0);
    // std::cout << "identity check: \n"
    //           << I_check << std::endl;

    return A_inv;
}

class AugSystemSolverTest : public ::testing::Test
{
protected:
    // Create OcpDims object
    int K = 10;                                                   // Number of stages
    std::vector<Index> nx = {20, 10, 10, 10, 10, 2, 0, 1, 10, 5}; // State dimensions for each stage
    std::vector<Index> nu = {1, 4, 2, 10, 1, 30, 4, 1, 10, 0};    // Input dimensions for each stage
    std::vector<Index> ng = {9, 3, 4, 3, 4, 0, 1, 0, 1, 5}; // Equality constraints for each stage
    std::vector<Index> ng_ineq = {0, 5, 10, 0,   0,
                                  0, 0, 0,  10, 0}; // Inequality constraints for each stage

    ProblemDims dims{K, nu, nx, ng, ng_ineq};

    ProblemInfo info{dims};
    // Create Jacobian object
    Jacobian<OcpType> jacobian{dims};
    MatRealAllocated full_matrix_jacobian =
        MatRealAllocated(info.number_of_eq_constraints, info.number_of_primal_variables);
    Hessian<OcpType> hessian{dims};
    MatRealAllocated full_matrix_hessian =
        MatRealAllocated(info.number_of_primal_variables, info.number_of_primal_variables);
    VecRealAllocated x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated mult = VecRealAllocated(info.number_of_eq_constraints);
    VecRealAllocated rhs_x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated rhs_g = VecRealAllocated(info.number_of_eq_constraints);
    VecRealAllocated D_x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated D_s = VecRealAllocated(info.number_of_slack_variables);
    VecRealAllocated D_eq = VecRealAllocated(info.number_of_g_eq_path);
    MatRealAllocated full_kkt_matrix =
        MatRealAllocated(info.number_of_primal_variables + info.number_of_eq_constraints,
                         info.number_of_primal_variables + info.number_of_eq_constraints);
    AugSystemSolver<OcpType> solver = AugSystemSolver<OcpType>(info);
    void SetUp()
    {
        x = 0;
        full_matrix_jacobian = 0.;
        // fill the jacobian with random values
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            if (k < info.dims.K - 1)
            {
                Index nx_next = info.dims.number_of_states[k + 1];
                jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0) =
                    ::test::random_matrix(nu + nx, nx_next);
            }
            jacobian.Gg_eqt[k].block(nu + nx, info.dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.dims.number_of_eq_constraints[k]);
            jacobian.Gg_ineqt[k].block(nu + nx, info.dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.dims.number_of_ineq_constraints[k]);
        }
        // fill the Hessian with random values
        for (Index k = 0; k < dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
        }
        // add dynamics constraints
        for (Index k = 0; k < info.dims.K - 1; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index offs_ux = info.offsets_primal_u[k];
            Index offs_x_next = info.offsets_primal_x[k + 1];
            Index nx_next = info.dims.number_of_states[k + 1];
            Index offs_eq_dyn = info.offsets_g_eq_dyn[k];
            full_matrix_jacobian.block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0));
            full_matrix_jacobian.block(nx_next, nx_next, offs_eq_dyn, offs_x_next).diagonal() =
                -1.0;
        }
        // equality path equality constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng = info.dims.number_of_eq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_eq = info.offsets_g_eq_path[k];
            full_matrix_jacobian.block(ng, nu + nx, offset_g_eq, offset_ux) =
                transpose(jacobian.Gg_eqt[k].block(nu + nx, ng, 0, 0));
        }
        // inequality path constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng_ineq = info.dims.number_of_ineq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_ineq = info.offsets_g_eq_slack[k];
            full_matrix_jacobian.block(ng_ineq, nu + nx, offset_g_ineq, offset_ux) =
                transpose(jacobian.Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));
        }
        full_matrix_hessian = 0.;
        // populate the full matrixTEST_F(ImplicitAugSystemSolverVsReformulatioTest, TestSolve)
        for (Index k = 0; k < dims.K; k++)
        {
            Index nu = dims.number_of_controls[k];
            Index nx = dims.number_of_states[k];
            Index offs_ux = info.offsets_primal_u[k];
            full_matrix_hessian.block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        // set up the full KKT matrix
        full_kkt_matrix.block(info.number_of_primal_variables, info.number_of_primal_variables, 0,
                              0) = full_matrix_hessian;
        full_kkt_matrix.block(info.number_of_primal_variables, info.number_of_eq_constraints, 0,
                              info.number_of_primal_variables) = transpose(full_matrix_jacobian);
        full_kkt_matrix.block(info.number_of_eq_constraints, info.number_of_primal_variables,
                              info.number_of_primal_variables, 0) = full_matrix_jacobian;
        // fill the x vector with random values
        for (Index i = 0; i < info.number_of_primal_variables; ++i)
        {
            rhs_x(i) = 1.0 * i;
            D_x = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info.number_of_eq_constraints; ++i)
        {
            rhs_g(i) = 1.0 * i;
        }

        for (Index i = 0; i < info.number_of_g_eq_path; ++i)
        {
            D_eq(i) = 1.0 * (i + 1);
        }
        for (Index i = 0; i < info.number_of_slack_variables; ++i)
        {
            D_s(i) = 1.0 * (i + 0.1);
        }
    };
};

class ImplicitAugSystemSolverTest : public ::testing::Test
{
// protected:
public:
    // Create OcpDims object
    int K = 10;                                                   // Number of stages
    std::vector<Index> nx = {20, 10, 10, 10, 10, 2, 0, 1, 10, 5}; // State dimensions for each stage
    std::vector<Index> nu = {1, 4, 2, 10, 1, 30, 4, 1, 10, 0};    // Input dimensions for each stage
    std::vector<Index> ng = {9, 3, 4, 3, 4, 0, 1, 0, 1, 5}; // Equality constraints for each stage
    std::vector<Index> ng_ineq = {0, 5, 10, 0,   0,
                                  0, 0, 0,  10, 0}; // Inequality constraints for each stage

    ProblemDims dims{K, nu, nx, ng, ng_ineq};

    ProblemInfo info{dims};
    // Create Jacobian object
    Jacobian<ImplicitOcpType> jacobian{dims};
    MatRealAllocated full_matrix_jacobian =
        MatRealAllocated(info.number_of_eq_constraints, info.number_of_primal_variables);
    Hessian<ImplicitOcpType> hessian{dims};
    MatRealAllocated full_matrix_hessian =
        MatRealAllocated(info.number_of_primal_variables, info.number_of_primal_variables);
    VecRealAllocated x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated mult = VecRealAllocated(info.number_of_eq_constraints);
    VecRealAllocated rhs_x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated rhs_g = VecRealAllocated(info.number_of_eq_constraints);
    VecRealAllocated D_x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated D_s = VecRealAllocated(info.number_of_slack_variables);
    VecRealAllocated D_eq = VecRealAllocated(info.number_of_g_eq_path);
    MatRealAllocated full_kkt_matrix =
        MatRealAllocated(info.number_of_primal_variables + info.number_of_eq_constraints,
                         info.number_of_primal_variables + info.number_of_eq_constraints);
    AugSystemSolver<ImplicitOcpType> solver = AugSystemSolver<ImplicitOcpType>(info);
    void SetUp()
    {
        // std::cout << "creating ImplicitAugSystemSolverTest" << std::endl;
        x = 0;
        full_matrix_jacobian = 0.;

        bool CREATE_EXPLICIT_EQUIVALENT = false;

        // fill the jacobian with random values
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            if (k < info.dims.K - 1)
            {
                Index nx_next = info.dims.number_of_states[k + 1];
                jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0) =
                    ::test::random_matrix(nu + nx, nx_next);
                if (jacobian.ASSUME_INVERSE_GIVEN){
                    if (CREATE_EXPLICIT_EQUIVALENT){
                        jacobian.Jt_inv[k].block(nx_next, nx_next, 0, 0) =
                            ::test::identity_matrix(nx_next, -1.0);
                    } else {
                        jacobian.Jt_inv[k].block(nx_next, nx_next, 0, 0) =
                            ::test::random_matrix(nx_next, nx_next);
                    }
                    jacobian.Jt[k].block(nx_next, nx_next, 0, 0) =
                        get_inverse(jacobian.Jt_inv[k].block(nx_next, nx_next, 0, 0));
                } else {
                    if (CREATE_EXPLICIT_EQUIVALENT){
                        jacobian.Jt[k].block(nx_next, nx_next, 0, 0) =
                            ::test::identity_matrix(nx_next, -1.0);
                    } else {
                        jacobian.Jt[k].block(nx_next, nx_next, 0, 0) =
                            ::test::random_matrix(nx_next, nx_next);
                    }
                }
            }
            jacobian.Gg_eqt[k].block(nu + nx, info.dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.dims.number_of_eq_constraints[k]);
            jacobian.Gg_ineqt[k].block(nu + nx, info.dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.dims.number_of_ineq_constraints[k]);
        }
        // fill the Hessian with random values
        for (Index k = 0; k < dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
        }
        // add dynamics constraints
        for (Index k = 0; k < info.dims.K - 1; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index offs_ux = info.offsets_primal_u[k];
            Index offs_x_next = info.offsets_primal_x[k + 1];
            Index nx_next = info.dims.number_of_states[k + 1];
            Index offs_eq_dyn = info.offsets_g_eq_dyn[k];
            full_matrix_jacobian.block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0));
            full_matrix_jacobian.block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                transpose(jacobian.Jt[k]);
            hessian.FuFxt[k].block(nx + nu, nx_next, 0, 0) =
                ::test::random_matrix(nx + nu, nx_next);
            if (CREATE_EXPLICIT_EQUIVALENT){
                hessian.FuFxt[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::empty_matrix(nx + nu, nx_next);
            } else {
                hessian.FuFxt[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::random_matrix(nx + nu, nx_next);
            }
            full_matrix_hessian.block(nx_next, nu + nx, offs_x_next, offs_ux) = 
                transpose(hessian.FuFxt[k]);
            full_matrix_hessian.block(nu + nx, nx_next, offs_ux, offs_x_next) =
                hessian.FuFxt[k];
        }
        // equality path equality constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng = info.dims.number_of_eq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_eq = info.offsets_g_eq_path[k];
            full_matrix_jacobian.block(ng, nu + nx, offset_g_eq, offset_ux) =
                transpose(jacobian.Gg_eqt[k].block(nu + nx, ng, 0, 0));
        }
        // inequality path constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng_ineq = info.dims.number_of_ineq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_ineq = info.offsets_g_eq_slack[k];
            full_matrix_jacobian.block(ng_ineq, nu + nx, offset_g_ineq, offset_ux) =
                transpose(jacobian.Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));
        }
        full_matrix_hessian = 0.;
        // populate the full matrix
        for (Index k = 0; k < dims.K; k++)
        {
            Index nu = dims.number_of_controls[k];
            Index nx = dims.number_of_states[k];
            Index offs_ux = info.offsets_primal_u[k];
            full_matrix_hessian.block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        // set up the full KKT matrix
        full_kkt_matrix.block(info.number_of_primal_variables, info.number_of_primal_variables, 0,
                              0) = full_matrix_hessian;
        full_kkt_matrix.block(info.number_of_primal_variables, info.number_of_eq_constraints, 0,
                              info.number_of_primal_variables) = transpose(full_matrix_jacobian);
        full_kkt_matrix.block(info.number_of_eq_constraints, info.number_of_primal_variables,
                              info.number_of_primal_variables, 0) = full_matrix_jacobian;

        // fill the x vector with random values
        for (Index i = 0; i < info.number_of_primal_variables; ++i)
        {
            rhs_x(i) = 1.0 * i;
            D_x = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info.number_of_eq_constraints; ++i)
        {
            rhs_g(i) = 1.0 * i;
        }

        for (Index i = 0; i < info.number_of_g_eq_path; ++i)
        {
            D_eq(i) = 1.0 * (i + 1);
        }
        for (Index i = 0; i < info.number_of_slack_variables; ++i)
        {
            D_s(i) = 1.0 * (i + 0.1);
        }
    // std::cout << "Created ImplicitAugSystemSolverTest" << std::endl;
    };
};

class BasicImplicitAugSystemSolverTest : public ::testing::Test
{
// protected:
public:
    // Create OcpDims object
    // int K = 2;                                                   // Number of stages
    // std::vector<Index> nx = {2, 7}; // State dimensions for each stage
    // std::vector<Index> nu = {1, 0};    // Input dimensions for each stage
    // std::vector<Index> ng = {0, 0}; // Equality constraints for each stage
    // std::vector<Index> ng_ineq = {0, 0}; // Inequality constraints for each stage
    int K = 8;                                                   // Number of stages
    std::vector<Index> nx = {1, 4, 25, 3, 4, 2, 1, 1}; // State dimensions for each stage
    std::vector<Index> nu = {1, 1, 1, 1, 10, 0, 1, 0};    // Input dimensions for each stage
    std::vector<Index> ng = {0, 0, 0, 0, 0, 0, 0, 0}; // Equality constraints for each stage
    std::vector<Index> ng_ineq = {0, 1, 3, 0, 1, 1, 3, 2}; // Inequality constraints for each stage

    // 00 does not work
    // 10 works
    // 01 does not
    // its a jacobian problem
    bool USE_IDENTITY_J = 0;
    bool USE_ZERO_F = 0;

    ProblemDims dims{K, nu, nx, ng, ng_ineq};
    ProblemInfo info{dims};
    // Create Jacobian object
    Jacobian<ImplicitOcpType> jacobian{dims};
    MatRealAllocated full_matrix_jacobian =
        MatRealAllocated(info.number_of_eq_constraints, info.number_of_primal_variables);

    Hessian<ImplicitOcpType> hessian{dims};
    MatRealAllocated full_matrix_hessian =
        MatRealAllocated(info.number_of_primal_variables, info.number_of_primal_variables);
    
    VecRealAllocated x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated mult = VecRealAllocated(info.number_of_eq_constraints);
    VecRealAllocated rhs_x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated rhs_g = VecRealAllocated(info.number_of_eq_constraints);
    VecRealAllocated D_x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated D_s = VecRealAllocated(info.number_of_slack_variables);
    VecRealAllocated D_eq = VecRealAllocated(info.number_of_g_eq_path);
    MatRealAllocated full_kkt_matrix =
        MatRealAllocated(info.number_of_primal_variables + info.number_of_eq_constraints,
                         info.number_of_primal_variables + info.number_of_eq_constraints);
    VecRealAllocated full_kkt_rhs = 
        VecRealAllocated(info.number_of_primal_variables + info.number_of_eq_constraints);
    AugSystemSolver<ImplicitOcpType> solver = AugSystemSolver<ImplicitOcpType>(info);
    void SetUp()
    {
        // std::cout << "creating ImplicitAugSystemSolverTest" << std::endl;
        x = 0;
        full_matrix_jacobian = 0.;
        full_matrix_hessian = 0.;

        // fill the jacobian with random values
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            if (k < info.dims.K - 1)
            {
                Index nx_next = info.dims.number_of_states[k + 1];
                jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0) =
                    ::test::random_matrix(nu + nx, nx_next);

                if (USE_IDENTITY_J){
                    jacobian.Jt[k].block(nx_next, nx_next, 0, 0) =
                        ::test::identity_matrix(nx_next, -1.0);
                } else {
                    jacobian.Jt[k].block(nx_next, nx_next, 0, 0) =
                        ::test::random_matrix(nx_next, nx_next);
                }

                jacobian.Jt_inv[k].block(nx_next, nx_next, 0, 0) =
                    get_inverse(jacobian.Jt[k].block(nx_next, nx_next, 0, 0));
            }
            jacobian.Gg_eqt[k].block(nu + nx, info.dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.dims.number_of_eq_constraints[k]);
            jacobian.Gg_ineqt[k].block(nu + nx, info.dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.dims.number_of_ineq_constraints[k]);
        }
        // fill the Hessian with random values
        for (Index k = 0; k < dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
        }
        // add dynamics constraints
        for (Index k = 0; k < info.dims.K - 1; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index offs_ux = info.offsets_primal_u[k];
            Index offs_x_next = info.offsets_primal_x[k + 1];
            Index nx_next = info.dims.number_of_states[k + 1];
            Index offs_eq_dyn = info.offsets_g_eq_dyn[k];
            full_matrix_jacobian.block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0));
            // full_matrix_jacobian.block(nx_next, nx_next, offs_eq_dyn, offs_x_next).diagonal() =
            //     -1.0;
            full_matrix_jacobian.block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                transpose(jacobian.Jt[k]);

            if (USE_ZERO_F){
                hessian.FuFxt[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::empty_matrix(nx + nu, nx_next);    
            } else {
                hessian.FuFxt[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::random_matrix(nx + nu, nx_next);
            }

            full_matrix_hessian.block(nx_next, nu + nx, offs_x_next, offs_ux) = 
                transpose(hessian.FuFxt[k]);
            full_matrix_hessian.block(nu + nx, nx_next, offs_ux, offs_x_next) =
                hessian.FuFxt[k];
        }
        // equality path equality constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng = info.dims.number_of_eq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_eq = info.offsets_g_eq_path[k];
            full_matrix_jacobian.block(ng, nu + nx, offset_g_eq, offset_ux) =
                transpose(jacobian.Gg_eqt[k].block(nu + nx, ng, 0, 0));
        }
        // inequality path constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng_ineq = info.dims.number_of_ineq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_ineq = info.offsets_g_eq_slack[k];
            full_matrix_jacobian.block(ng_ineq, nu + nx, offset_g_ineq, offset_ux) =
                transpose(jacobian.Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));
        }
        // populate the full matrix
        for (Index k = 0; k < dims.K; k++)
        {
            Index nu = dims.number_of_controls[k];
            Index nx = dims.number_of_states[k];
            Index offs_ux = info.offsets_primal_u[k];
            full_matrix_hessian.block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        // set up the full KKT matrix
        full_kkt_matrix.block(info.number_of_primal_variables, info.number_of_primal_variables, 0,
                              0) = full_matrix_hessian;
        full_kkt_matrix.block(info.number_of_primal_variables, info.number_of_eq_constraints, 0,
                              info.number_of_primal_variables) = transpose(full_matrix_jacobian);
        full_kkt_matrix.block(info.number_of_eq_constraints, info.number_of_primal_variables,
                              info.number_of_primal_variables, 0) = full_matrix_jacobian;

        // std::cout << "Full KKT matrix: \n"
        //           << full_kkt_matrix << std::endl;
        // fill the x vector with random values
        for (Index i = 0; i < info.number_of_primal_variables; ++i)
        {
            rhs_x(i) = 1.0 * i;
            D_x = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info.number_of_eq_constraints; ++i)
        {
            rhs_g(i) = 1.0 * i;
        }

        for (Index i = 0; i < info.number_of_g_eq_path; ++i)
        {
            D_eq(i) = 1.0 * (i + 1);
        }
        for (Index i = 0; i < info.number_of_slack_variables; ++i)
        {
            D_s(i) = 1.0 * (i + 0.1);
        }

        // fill the full KKT rhs vector
        full_kkt_rhs = 0.0;
        for (Index i = 0; i < info.number_of_primal_variables; ++i)
        {
            full_kkt_rhs(i) = rhs_x(i);
        }
        for (Index i = 0; i < info.number_of_eq_constraints; ++i)
        {
            full_kkt_rhs(info.number_of_primal_variables + i) = rhs_g(i);
        }
        // std::cout << "Full KKT rhs vector: \n"
        //           << full_kkt_rhs << std::endl;
    // std::cout << "Created ImplicitAugSystemSolverTest" << std::endl;
    };
};

class ImplicitVsReformulationTester
{
    public:
        ImplicitVsReformulationTester(){};

        void UpdateRandomly(bool random_dimensions=false){
            std::vector<Index> nx;
            std::vector<Index> nu;
            std::vector<Index> ng;
            std::vector<Index> ng_ineq;
            if (random_dimensions){
                K = ::test::random_int(5, 50);
                nx = std::vector<Index>(K, 0);
                nu = std::vector<Index>(K, 0);
                ng = std::vector<Index>(K, 0);
                ng_ineq = std::vector<Index>(K, 0);
                for (int k = 0; k < K; k++){
                    nx[k] = ::test::random_int(5, 50);
                    nu[k] = ::test::random_int(0, 20);
                    ng[k] = ::test::random_int(0, nx[k] + nu[k]-1);
                    ng_ineq[k] = ::test::random_int(0, 10);
                }
                nu[K - 1] = 0; // last stage has no control
                ng[K - 1] = std::min(ng[K - 1], nx[K - 1] - 1);

            } else {
                K = 10;
                nx = {20, 10, 10, 10, 10, 20, 30, 1, 10, 5};
                nu = {1, 4, 2, 1, 1, 3, 4, 1, 1, 0};
                ng = {9, 3, 4, 3, 4, 0, 1, 0, 1, 5};
                ng_ineq = {0, 5, 10, 0,   0, 0, 0, 0,  10, 0};
            }

            try{
                Prepare(K, nx, nu, ng, ng_ineq);
            } catch (const std::exception& e){
                return UpdateRandomly(random_dimensions);
            }
        };

        void UpdateRandomly(int nx_, int nu_, int K){
            if (K < 0){K = ::test::random_int(5, 50);}
            std::vector<Index> nx = std::vector<Index>(K, nx_);
            std::vector<Index> nu = std::vector<Index>(K, nu_);
            std::vector<Index> ng = std::vector<Index>(K, 0);
            std::vector<Index> ng_ineq = std::vector<Index>(K, 0);
            for (int k = 0; k < K; k++){
                if (nx_ < 0) { nx[k] = ::test::random_int(5, 20); }
                if (nu_ < 0) { nu[k] = ::test::random_int(0, 20); }
                ng[k] = 0*::test::random_int(0, nx[k] + nu[k]-1);
                ng_ineq[k] = 0*::test::random_int(0, 10);
            }
            nu[K - 1] = 0; // last stage has no control
            ng[K - 1] = std::min(ng[K - 1], nx[K - 1] - 1);

            try{
                Prepare(K, nx, nu, ng, ng_ineq);
            } catch (const std::exception& e){
                UpdateRandomly(nx_, nu_, K);
            }
        };
        
        void Prepare(int K, std::vector<Index> nx, std::vector<Index> nu, std::vector<Index> ng, std::vector<Index> ng_ineq){
            // implicit
            try{
            dims_i.emplace(ProblemDims{K, nu, nx, ng, ng_ineq});
            info_i.emplace(ProblemInfo(dims_i.value()));
            jacobian_i.emplace(Jacobian<ImplicitOcpType>(dims_i.value()));
            hessian_i.emplace(Hessian<ImplicitOcpType>(dims_i.value()));
            x_i.emplace(VecRealAllocated(info_i.value().number_of_primal_variables));
            mult_i.emplace(VecRealAllocated(info_i.value().number_of_eq_constraints));
            rhs_x_i.emplace(VecRealAllocated(info_i.value().number_of_primal_variables));
            rhs_g_i.emplace(VecRealAllocated(info_i.value().number_of_eq_constraints));
            D_x_i.emplace(VecRealAllocated(info_i.value().number_of_primal_variables));
            D_s_i.emplace(VecRealAllocated(info_i.value().number_of_slack_variables));
            D_eq_i.emplace(VecRealAllocated(info_i.value().number_of_g_eq_path));
            solver_i.emplace(AugSystemSolver<ImplicitOcpType>(info_i.value()));

            // reformulation
            std::vector<Index> nu_r = std::vector<Index>(info_i.value().dims.K, 0);
            std::vector<Index> ng_r = std::vector<Index>(info_i.value().dims.K, 0);
            for (int i = 0; i < info_i.value().dims.K - 1; i++){
                nu_r[i] = nu[i] + info_i.value().dims.number_of_states[i+1];
                ng_r[i] = ng[i] + info_i.value().dims.number_of_states[i+1];
                if (ng_r[i] >= nx[i] + nu_r[i]){
                    throw std::runtime_error("Something is wrong");
                }
            }
            dims_r.emplace(ProblemDims(K, nu_r, nx, ng_r, ng_ineq));
            info_r.emplace(ProblemInfo(dims_r.value()));
            jacobian_r.emplace(Jacobian<OcpType>(dims_r.value()));
            hessian_r.emplace(Hessian<OcpType>(dims_r.value()));
            x_r.emplace(VecRealAllocated(info_r.value().number_of_primal_variables));
            mult_r.emplace(VecRealAllocated(info_r.value().number_of_eq_constraints));
            rhs_x_r.emplace(VecRealAllocated(info_r.value().number_of_primal_variables));
            rhs_g_r.emplace(VecRealAllocated(info_r.value().number_of_eq_constraints));
            D_x_r.emplace(VecRealAllocated(info_r.value().number_of_primal_variables));
            D_s_r.emplace(VecRealAllocated(info_r.value().number_of_slack_variables));
            D_eq_r.emplace(VecRealAllocated(info_r.value().number_of_g_eq_path));
            solver_r.emplace(AugSystemSolver<OcpType>(info_r.value()));
            } catch (const std::exception& e){
                std::cout << "Error in ImplicitVsReformulationTester::UpdateRandomly: " << e.what() << std::endl;
                for (int i = 0; i < K; i++){
                    std::cout << nx[i] << " " << nu[i] << " " << ng[i] << " " << ng_ineq[i] << std::endl;
                    if (ng[i] >= nx[i] + nu[i]){
                        std::cout << "Error in ImplicitVsReformulationTester::UpdateRandomly: ng[" << i << "] >= nx[" << i << "] + nu[" << i << "]" << std::endl;
                        std::cout << "\tK = " << K << std::endl;
                    }
                }
                // return UpdateRandomly(random_dimensions);
                throw std::runtime_error("Error in ImplicitVsReformulationTester::UpdateRandomly");
            }

             // fill the jacobian with random values
            for (Index k = 0; k < info_i.value().dims.K; ++k)
            {
                Index nu = info_i.value().dims.number_of_controls[k];
                Index nu_r = info_r.value().dims.number_of_controls[k];
                Index nx = info_i.value().dims.number_of_states[k];
                if (k < info_i.value().dims.K - 1)
                {
                    Index nx_next = info_i.value().dims.number_of_states[k + 1];
                    jacobian_i.value().BAbt[k].block(nu + nx, nx_next, 0, 0) =
                        ::test::random_matrix(nu + nx, nx_next);
                    jacobian_i.value().Jt_inv[k].block(nx_next, nx_next, 0, 0) =
                        ::test::random_matrix(nx_next, nx_next);
                    jacobian_i.value().Jt[k].block(nx_next, nx_next, 0, 0) =
                        get_inverse(jacobian_i.value().Jt_inv[k].block(nx_next, nx_next, 0, 0));

                    jacobian_r.value().BAbt[k].block(nu_r + nx, nx_next, 0, 0) =
                        ::test::random_matrix(nu_r + nx, nx_next);
                }
                jacobian_i.value().Gg_eqt[k].block(nu + nx, info_i.value().dims.number_of_eq_constraints[k], 0, 0) =
                    ::test::random_matrix(nu + nx, info_i.value().dims.number_of_eq_constraints[k]);
                jacobian_i.value().Gg_ineqt[k].block(nu + nx, info_i.value().dims.number_of_ineq_constraints[k], 0, 0) =
                    ::test::random_matrix(nu + nx, info_i.value().dims.number_of_ineq_constraints[k]);

                jacobian_r.value().Gg_eqt[k].block(nu_r + nx, info_r.value().dims.number_of_eq_constraints[k], 0, 0) =
                    ::test::random_matrix(nu_r + nx, info_r.value().dims.number_of_eq_constraints[k]);
                jacobian_r.value().Gg_ineqt[k].block(nu_r + nx, info_r.value().dims.number_of_ineq_constraints[k], 0, 0) =
                    ::test::random_matrix(nu_r + nx, info_r.value().dims.number_of_ineq_constraints[k]);
            }
            // fill the Hessian with random values
            for (Index k = 0; k < dims_i.value().K; ++k)
            {
                Index nu = info_i.value().dims.number_of_controls[k];
                Index nu_r = info_r.value().dims.number_of_controls[k];
                Index nx = info_i.value().dims.number_of_states[k];
                hessian_i.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);

                hessian_r.value().RSQrqt[k].block(nu_r + nx, nu_r + nx, 0, 0) =
                    ::test::random_spd_matrix(nu_r + nx);
            }
            // add dynamics constraints
            for (Index k = 0; k < info_i.value().dims.K - 1; ++k)
            {
                Index nu = info_i.value().dims.number_of_controls[k];
                Index nu_r = info_r.value().dims.number_of_controls[k];
                Index nx = info_i.value().dims.number_of_states[k];
                Index offs_ux = info_i.value().offsets_primal_u[k];
                Index offs_x_next = info_i.value().offsets_primal_x[k + 1];
                Index nx_next = info_i.value().dims.number_of_states[k + 1];
                Index offs_eq_dyn = info_i.value().offsets_g_eq_dyn[k];
                
                hessian_i.value().FuFxt[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::random_matrix(nx + nu, nx_next);
            }

            // Implicit OCP
            for (Index i = 0; i < info_i.value().number_of_primal_variables; ++i){
                rhs_x_i.value()(i) = 1.0 * i;
                D_x_i.value() = 1.0 * (i + 0.1);
            }
            for (Index i = 0; i < info_i.value().number_of_eq_constraints; ++i){
                rhs_g_i.value()(i) = 1.0 * i;
            }

            for (Index i = 0; i < info_i.value().number_of_g_eq_path; ++i){
                D_eq_i.value()(i) = 1.0 * (i + 1);
            }
            for (Index i = 0; i < info_i.value().number_of_slack_variables; ++i){
                D_s_i.value()(i) = 1.0 * (i + 0.1);
            }

            // Reformulation
            for (Index i = 0; i < info_r.value().number_of_primal_variables; ++i){
                rhs_x_r.value()(i) = 1.0 * i;
                D_x_r.value() = 1.0 * (i + 0.1);
            }
            for (Index i = 0; i < info_r.value().number_of_eq_constraints; ++i){
                rhs_g_r.value()(i) = 1.0 * i;
            }

            for (Index i = 0; i < info_r.value().number_of_g_eq_path; ++i){
                D_eq_r.value()(i) = 1.0 * (i + 1);
            }
            for (Index i = 0; i < info_r.value().number_of_slack_variables; ++i){
                D_s_r.value()(i) = 1.0 * (i + 0.1);
            }
        }

        int K;
        std::optional<ProblemDims> dims_i;
        std::optional<ProblemInfo> info_i;
        std::optional<Jacobian<ImplicitOcpType>> jacobian_i;
        std::optional<Hessian<ImplicitOcpType>> hessian_i;
        std::optional<VecRealAllocated> x_i;
        std::optional<VecRealAllocated> mult_i;
        std::optional<VecRealAllocated> rhs_x_i;
        std::optional<VecRealAllocated> rhs_g_i;
        std::optional<VecRealAllocated> D_x_i;
        std::optional<VecRealAllocated> D_s_i;
        std::optional<VecRealAllocated> D_eq_i;
        std::optional<AugSystemSolver<ImplicitOcpType>> solver_i;

        std::optional<ProblemDims> dims_r;
        std::optional<ProblemInfo> info_r;
        std::optional<Jacobian<OcpType>> jacobian_r;
        std::optional<Hessian<OcpType>> hessian_r;
        std::optional<VecRealAllocated> x_r;
        std::optional<VecRealAllocated> mult_r;
        std::optional<VecRealAllocated> rhs_x_r;
        std::optional<VecRealAllocated> rhs_g_r;
        std::optional<VecRealAllocated> D_x_r;
        std::optional<VecRealAllocated> D_s_r;
        std::optional<VecRealAllocated> D_eq_r;
        std::optional<AugSystemSolver<OcpType>> solver_r;

};

class ImplicitAugSystemSolverVsReformulationTest : public ::testing::Test
{
// protected:
public:
   ImplicitVsReformulationTester tester;

    void SetUp()
    {
        tester.UpdateRandomly();
    };
};

void PrintSolutionOfOcpTypeSolver(ImplicitAugSystemSolverTest &implicit_solver, 
                                  VecRealAllocated &original_x,
                                  VecRealAllocated &original_mult){
//    std::cout << "mult: " << original_mult << std::endl;
}


TEST_F(AugSystemSolverTest, TestSolve)
{
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_s, rhs_x, rhs_g, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    VecRealAllocated jac_x(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + rhs_g + jac_x;
    rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) =
        rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) -
        D_s * mult.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    VecRealAllocated grad(info.number_of_primal_variables);
    VecRealAllocated tmp(info.number_of_primal_variables);
    grad = 0;
    hessian.apply_on_right(info, x, 0.0, tmp, tmp);
    grad = grad + tmp + D_x * x;
    jacobian.transpose_apply_on_right(info, mult, 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + rhs_x;
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }
}

TEST_F(AugSystemSolverTest, TestSolveRhs)
{
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_s, rhs_x, rhs_g, x, mult);
    EXPECT_TRUE(ret == LinsolReturnFlag::SUCCESS);
    solver.solve_rhs(info, jacobian, hessian, D_s, rhs_x, rhs_g, x, mult);
    VecRealAllocated jac_x(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + rhs_g + jac_x;
    rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) =
        rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) -
        D_s * mult.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    VecRealAllocated grad(info.number_of_primal_variables);
    VecRealAllocated tmp(info.number_of_primal_variables);
    grad = 0;
    hessian.apply_on_right(info, x, 0.0, tmp, tmp);
    grad = grad + tmp + D_x * x;
    jacobian.transpose_apply_on_right(info, mult, 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + rhs_x;
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }
}

TEST_F(AugSystemSolverTest, TestSolveDegen)
{
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_eq, D_s, rhs_x, rhs_g, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    VecRealAllocated jac_x(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + rhs_g + jac_x;
    rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) =
        rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) -
        D_s * mult.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    rhs_gg.block(info.number_of_g_eq_path, info.offset_g_eq_path) =
        rhs_gg.block(info.number_of_g_eq_path, info.offset_g_eq_path) -
        D_eq * mult.block(info.number_of_g_eq_path, info.offset_g_eq_path);
    VecRealAllocated grad(info.number_of_primal_variables);
    VecRealAllocated tmp(info.number_of_primal_variables);
    grad = 0;
    hessian.apply_on_right(info, x, 0.0, tmp, tmp);
    grad = grad + tmp + D_x * x;
    jacobian.transpose_apply_on_right(info, mult, 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + rhs_x;
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }
}

TEST_F(AugSystemSolverTest, TestSolveDegenRhs)
{
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_eq, D_s, rhs_x, rhs_g, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    ret = solver.solve_rhs(info, jacobian, hessian, D_eq, D_s, rhs_x, rhs_g, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    VecRealAllocated jac_x(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + rhs_g + jac_x;
    rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) =
        rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) -
        D_s * mult.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    rhs_gg.block(info.number_of_g_eq_path, info.offset_g_eq_path) =
        rhs_gg.block(info.number_of_g_eq_path, info.offset_g_eq_path) -
        D_eq * mult.block(info.number_of_g_eq_path, info.offset_g_eq_path);
    VecRealAllocated grad(info.number_of_primal_variables);
    VecRealAllocated tmp(info.number_of_primal_variables);
    grad = 0;
    hessian.apply_on_right(info, x, 0.0, tmp, tmp);
    grad = grad + tmp + D_x * x;
    jacobian.transpose_apply_on_right(info, mult, 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + rhs_x;
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }
}


TEST_F(ImplicitAugSystemSolverTest, TestSolve)
{
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_s, rhs_x, rhs_g, x, mult);
    PrintSolutionOfOcpTypeSolver(*this, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    VecRealAllocated jac_x(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + rhs_g + jac_x;
    rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) =
        rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) -
        D_s * mult.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    VecRealAllocated grad(info.number_of_primal_variables);
    VecRealAllocated tmp(info.number_of_primal_variables);
    grad = 0;
    hessian.apply_on_right(info, x, 0.0, tmp, tmp);
    grad = grad + tmp + D_x * x;
    jacobian.transpose_apply_on_right(info, mult, 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + rhs_x;
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }
}

TEST_F(ImplicitAugSystemSolverTest, TestSolveRhs)
{
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_s, rhs_x, rhs_g, x, mult);
    PrintSolutionOfOcpTypeSolver(*this, x, mult);
    EXPECT_TRUE(ret == LinsolReturnFlag::SUCCESS);
    solver.solve_rhs(info, jacobian, hessian, D_s, rhs_x, rhs_g, x, mult);
    VecRealAllocated jac_x(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + rhs_g + jac_x;
    rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) =
        rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) -
        D_s * mult.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    VecRealAllocated grad(info.number_of_primal_variables);
    VecRealAllocated tmp(info.number_of_primal_variables);
    grad = 0;
    hessian.apply_on_right(info, x, 0.0, tmp, tmp);
    grad = grad + tmp + D_x * x;
    jacobian.transpose_apply_on_right(info, mult, 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + rhs_x;
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }
}

TEST_F(ImplicitAugSystemSolverTest, TestSolveDegen)
{
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_eq, D_s, rhs_x, rhs_g, x, mult);
    PrintSolutionOfOcpTypeSolver(*this, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    VecRealAllocated jac_x(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + rhs_g + jac_x;
    rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) =
        rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) -
        D_s * mult.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    rhs_gg.block(info.number_of_g_eq_path, info.offset_g_eq_path) =
        rhs_gg.block(info.number_of_g_eq_path, info.offset_g_eq_path) -
        D_eq * mult.block(info.number_of_g_eq_path, info.offset_g_eq_path);
    VecRealAllocated grad(info.number_of_primal_variables);
    VecRealAllocated tmp(info.number_of_primal_variables);
    grad = 0;
    hessian.apply_on_right(info, x, 0.0, tmp, tmp);
    grad = grad + tmp + D_x * x;
    jacobian.transpose_apply_on_right(info, mult, 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + rhs_x;
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }
}

TEST_F(ImplicitAugSystemSolverTest, TestSolveDegenRhs)
{
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_eq, D_s, rhs_x, rhs_g, x, mult);
    PrintSolutionOfOcpTypeSolver(*this, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    ret = solver.solve_rhs(info, jacobian, hessian, D_eq, D_s, rhs_x, rhs_g, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    VecRealAllocated jac_x(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + rhs_g + jac_x;
    rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) =
        rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) -
        D_s * mult.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    rhs_gg.block(info.number_of_g_eq_path, info.offset_g_eq_path) =
        rhs_gg.block(info.number_of_g_eq_path, info.offset_g_eq_path) -
        D_eq * mult.block(info.number_of_g_eq_path, info.offset_g_eq_path);
    VecRealAllocated grad(info.number_of_primal_variables);
    VecRealAllocated tmp(info.number_of_primal_variables);
    grad = 0;
    hessian.apply_on_right(info, x, 0.0, tmp, tmp);
    grad = grad + tmp + D_x * x;
    jacobian.transpose_apply_on_right(info, mult, 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + rhs_x;
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }
}


TEST_F(BasicImplicitAugSystemSolverTest, TestSolve)
{
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_s, rhs_x, rhs_g, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    VecRealAllocated jac_x(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + rhs_g + jac_x;
    rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) =
        rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) -
        D_s * mult.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    VecRealAllocated grad(info.number_of_primal_variables);
    VecRealAllocated tmp(info.number_of_primal_variables);
    grad = 0;
    hessian.apply_on_right(info, x, 0.0, tmp, tmp);
    grad = grad + tmp + D_x * x;
    jacobian.transpose_apply_on_right(info, mult, 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + rhs_x;

    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        // std::cout << "i: " << i << std::endl;
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        // std::cout << "i: " << i << std::endl;
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }
}

TEST_F(ImplicitAugSystemSolverVsReformulationTest, TestSolve)
{
    // Implicit OCP // ;
    auto start_i = std::chrono::high_resolution_clock::now();
    Index ret = tester.solver_i.value().solve(tester.info_i.value(), tester.jacobian_i.value(), tester.hessian_i.value(), 
        tester.D_x_i.value(), tester.D_eq_i.value(), tester.D_s_i.value(), tester.rhs_x_i.value(), tester.rhs_g_i.value(), tester.x_i.value(), tester.mult_i.value());
    auto stop_i = std::chrono::high_resolution_clock::now();
    auto duration_i = std::chrono::duration_cast<std::chrono::microseconds>(stop_i - start_i);
    std::cout << "Implicit OCP solve duration: " << duration_i.count() << " microseconds" << std::endl;
    std::cout << "\tpreprocessing (jac):  " << tester.solver_i.value().duration_preprocess_jac.count() << " microseconds" << std::endl;
    std::cout << "\tpreprocessing (hess): " << tester.solver_i.value().duration_preprocess_hess.count() << " microseconds" << std::endl;
    std::cout << "\t solve:               " << tester.solver_i.value().duration_solve.count() << " microseconds" << std::endl;
    std::cout << "\tpostprocessing:       " << tester.solver_i.value().duration_postprocess.count() << " microseconds" << std::endl;

    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    ret = tester.solver_i.value().solve_rhs(tester.info_i.value(), tester.jacobian_i.value(), tester.hessian_i.value(), 
        tester.D_eq_i.value(), tester.D_s_i.value(), tester.rhs_x_i.value(), tester.rhs_g_i.value(), tester.x_i.value(), tester.mult_i.value());
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    VecRealAllocated jac_x(tester.info_i.value().number_of_eq_constraints);
    tester.jacobian_i.value().apply_on_right(tester.info_i.value(), tester.x_i.value(), 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(tester.info_i.value().number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + tester.rhs_g_i.value() + jac_x;
    rhs_gg.block(tester.info_i.value().number_of_slack_variables, tester.info_i.value().offset_g_eq_slack) =
        rhs_gg.block(tester.info_i.value().number_of_slack_variables, tester.info_i.value().offset_g_eq_slack) -
        tester.D_s_i.value() * tester.mult_i.value().block(tester.info_i.value().number_of_slack_variables, tester.info_i.value().offset_g_eq_slack);
    rhs_gg.block(tester.info_i.value().number_of_g_eq_path, tester.info_i.value().offset_g_eq_path) =
        rhs_gg.block(tester.info_i.value().number_of_g_eq_path, tester.info_i.value().offset_g_eq_path) -
        tester.D_eq_i.value() * tester.mult_i.value().block(tester.info_i.value().number_of_g_eq_path, tester.info_i.value().offset_g_eq_path);
    VecRealAllocated grad(tester.info_i.value().number_of_primal_variables);
    VecRealAllocated tmp(tester.info_i.value().number_of_primal_variables);
    grad = 0;
    tester.hessian_i.value().apply_on_right(tester.info_i.value(), tester.x_i.value(), 0.0, tmp, tmp);
    grad = grad + tmp + tester.D_x_i.value() * tester.x_i.value();
    tester.jacobian_i.value().transpose_apply_on_right(tester.info_i.value(), tester.mult_i.value(), 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + tester.rhs_x_i.value();
    for (Index i = 0; i < tester.info_i.value().number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < tester.info_i.value().number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }



    // Reformulation //
    auto start_r = std::chrono::high_resolution_clock::now();
    Index ret_r = tester.solver_r.value().solve(tester.info_r.value(), tester.jacobian_r.value(), tester.hessian_r.value(), 
        tester.D_x_r.value(), tester.D_eq_r.value(), tester.D_s_r.value(), tester.rhs_x_r.value(), tester.rhs_g_r.value(), tester.x_r.value(), tester.mult_r.value());
    auto stop_r = std::chrono::high_resolution_clock::now();
    auto duration_r = std::chrono::duration_cast<std::chrono::microseconds>(stop_r - start_r);
    std::cout << "Reformulation OCP solve duration: " << duration_r.count() << " microseconds" << std::endl;

    EXPECT_EQ(ret_r, LinsolReturnFlag::SUCCESS);
    ret_r = tester.solver_r.value().solve_rhs(tester.info_r.value(), tester.jacobian_r.value(), tester.hessian_r.value(), 
        tester.D_eq_r.value(), tester.D_s_r.value(), tester.rhs_x_r.value(), tester.rhs_g_r.value(), tester.x_r.value(), tester.mult_r.value());
    EXPECT_EQ(ret_r, LinsolReturnFlag::SUCCESS);
    VecRealAllocated jac_x_r(tester.info_r.value().number_of_eq_constraints);
    tester.jacobian_r.value().apply_on_right(tester.info_r.value(), tester.x_r.value(), 0.0, jac_x_r, jac_x_r);
    VecRealAllocated rhs_gg_r(tester.info_r.value().number_of_eq_constraints);
    rhs_gg_r = 0.;
    rhs_gg_r = rhs_gg_r + tester.rhs_g_r.value() + jac_x_r;
    rhs_gg_r.block(tester.info_r.value().number_of_slack_variables, tester.info_r.value().offset_g_eq_slack) =
        rhs_gg_r.block(tester.info_r.value().number_of_slack_variables, tester.info_r.value().offset_g_eq_slack) -
        tester.D_s_r.value() * tester.mult_r.value().block(tester.info_r.value().number_of_slack_variables, tester.info_r.value().offset_g_eq_slack);
    rhs_gg_r.block(tester.info_r.value().number_of_g_eq_path, tester.info_r.value().offset_g_eq_path) =
        rhs_gg_r.block(tester.info_r.value().number_of_g_eq_path, tester.info_r.value().offset_g_eq_path) -
        tester.D_eq_r.value() * tester.mult_r.value().block(tester.info_r.value().number_of_g_eq_path, tester.info_r.value().offset_g_eq_path);
    VecRealAllocated grad_r(tester.info_r.value().number_of_primal_variables);
    VecRealAllocated tmp_r(tester.info_r.value().number_of_primal_variables);
    grad_r = 0;
    tester.hessian_r.value().apply_on_right(tester.info_r.value(), tester.x_r.value(), 0.0, tmp_r, tmp_r);
    grad_r = grad_r + tmp_r + tester.D_x_r.value() * tester.x_r.value();
    tester.jacobian_r.value().transpose_apply_on_right(tester.info_r.value(), tester.mult_r.value(), 0.0, tmp_r, tmp_r);
    grad_r = grad_r + tmp_r;
    grad_r = grad_r + tester.rhs_x_r.value();
    for (Index i = 0; i < tester.info_r.value().number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg_r(i), 0, 1e-5);
    }
    for (Index i = 0; i < tester.info_r.value().number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad_r(i), 0, 1e-5);
    }
}

TEST_F(ImplicitAugSystemSolverVsReformulationTest, TestTimings)
{
    int nb_runs = 1;
    double time_implicit_us = 0.0;
    double time_implicit_copying_rhs = 0.0;
    double time_implicit_preprocess_jac_us = 0.0;
    double time_implicit_preprocess_hess_us = 0.0;
    double time_implicit_preprocess_hess_copy_us = 0.0;
    double time_implicit_preprocess_hess_scaling_us = 0.0;
    double time_implicit_only_solve_us = 0.0;
    double time_implicit_postprocess_us = 0.0;
    double time_reformulation_us = 0.0;

    for (int counter = 0; counter < nb_runs; counter++){
        tester.UpdateRandomly(true);

        // Reformulation //
        auto start_r = std::chrono::high_resolution_clock::now();
        Index ret_r = tester.solver_r.value().solve(tester.info_r.value(), tester.jacobian_r.value(), tester.hessian_r.value(), 
            tester.D_x_r.value(), tester.D_eq_r.value(), tester.D_s_r.value(), tester.rhs_x_r.value(), tester.rhs_g_r.value(), tester.x_r.value(), tester.mult_r.value());
        auto stop_r = std::chrono::high_resolution_clock::now();
        auto duration_r = std::chrono::duration_cast<std::chrono::microseconds>(stop_r - start_r);
        time_reformulation_us += duration_r.count();

        EXPECT_EQ(ret_r, LinsolReturnFlag::SUCCESS);
        ret_r = tester.solver_r.value().solve_rhs(tester.info_r.value(), tester.jacobian_r.value(), tester.hessian_r.value(), 
            tester.D_eq_r.value(), tester.D_s_r.value(), tester.rhs_x_r.value(), tester.rhs_g_r.value(), tester.x_r.value(), tester.mult_r.value());
        EXPECT_EQ(ret_r, LinsolReturnFlag::SUCCESS);
        /*
        VecRealAllocated jac_x_r(tester.info_r.value().number_of_eq_constraints);
        tester.jacobian_r.value().apply_on_right(tester.info_r.value(), tester.x_r.value(), 0.0, jac_x_r, jac_x_r);
        VecRealAllocated rhs_gg_r(tester.info_r.value().number_of_eq_constraints);
        rhs_gg_r = 0.;
        rhs_gg_r = rhs_gg_r + tester.rhs_g_r.value() + jac_x_r;
        rhs_gg_r.block(tester.info_r.value().number_of_slack_variables, tester.info_r.value().offset_g_eq_slack) =
            rhs_gg_r.block(tester.info_r.value().number_of_slack_variables, tester.info_r.value().offset_g_eq_slack) -
            tester.D_s_r.value() * tester.mult_r.value().block(tester.info_r.value().number_of_slack_variables, tester.info_r.value().offset_g_eq_slack);
        rhs_gg_r.block(tester.info_r.value().number_of_g_eq_path, tester.info_r.value().offset_g_eq_path) =
            rhs_gg_r.block(tester.info_r.value().number_of_g_eq_path, tester.info_r.value().offset_g_eq_path) -
            tester.D_eq_r.value() * tester.mult_r.value().block(tester.info_r.value().number_of_g_eq_path, tester.info_r.value().offset_g_eq_path);
        VecRealAllocated grad_r(tester.info_r.value().number_of_primal_variables);
        VecRealAllocated tmp_r(tester.info_r.value().number_of_primal_variables);
        grad_r = 0;
        tester.hessian_r.value().apply_on_right(tester.info_r.value(), tester.x_r.value(), 0.0, tmp_r, tmp_r);
        grad_r = grad_r + tmp_r + tester.D_x_r.value() * tester.x_r.value();
        tester.jacobian_r.value().transpose_apply_on_right(tester.info_r.value(), tester.mult_r.value(), 0.0, tmp_r, tmp_r);
        grad_r = grad_r + tmp_r;
        grad_r = grad_r + tester.rhs_x_r.value();
        for (Index i = 0; i < tester.info_r.value().number_of_eq_constraints; ++i)
        {
            EXPECT_NEAR(rhs_gg_r(i), 0, 1e-5);
        }
        for (Index i = 0; i < tester.info_r.value().number_of_primal_variables; ++i)
        {
            EXPECT_NEAR(grad_r(i), 0, 1e-5);
        }
        */


        // Implicit OCP //
        auto start_i = std::chrono::high_resolution_clock::now();
        Index ret = tester.solver_i.value().solve(tester.info_i.value(), tester.jacobian_i.value(), tester.hessian_i.value(), 
            tester.D_x_i.value(), tester.D_eq_i.value(), tester.D_s_i.value(), tester.rhs_x_i.value(), tester.rhs_g_i.value(), tester.x_i.value(), tester.mult_i.value());
        auto stop_i = std::chrono::high_resolution_clock::now();
        auto duration_i = std::chrono::duration_cast<std::chrono::microseconds>(stop_i - start_i);
        time_implicit_us += duration_i.count();
        time_implicit_copying_rhs += tester.solver_i.value().duration_copying_rhs.count();
        time_implicit_preprocess_jac_us += tester.solver_i.value().duration_preprocess_jac.count();
        time_implicit_preprocess_hess_us += tester.solver_i.value().duration_preprocess_hess.count();
        time_implicit_preprocess_hess_copy_us += tester.hessian_i.value().duration_copy_RSQrqt.count();
        time_implicit_preprocess_hess_scaling_us += tester.hessian_i.value().duration_modifying_RSQrqt.count();
        time_implicit_only_solve_us += tester.solver_i.value().duration_solve.count();
        time_implicit_postprocess_us += tester.solver_i.value().duration_postprocess.count();
        
        EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
        ret = tester.solver_i.value().solve_rhs(tester.info_i.value(), tester.jacobian_i.value(), tester.hessian_i.value(), 
            tester.D_eq_i.value(), tester.D_s_i.value(), tester.rhs_x_i.value(), tester.rhs_g_i.value(), tester.x_i.value(), tester.mult_i.value());
        EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
        /*
        VecRealAllocated jac_x(tester.info_i.value().number_of_eq_constraints);
        tester.jacobian_i.value().apply_on_right(tester.info_i.value(), tester.x_i.value(), 0.0, jac_x, jac_x);
        VecRealAllocated rhs_gg(tester.info_i.value().number_of_eq_constraints);
        rhs_gg = 0.;
        rhs_gg = rhs_gg + tester.rhs_g_i.value() + jac_x;
        rhs_gg.block(tester.info_i.value().number_of_slack_variables, tester.info_i.value().offset_g_eq_slack) =
            rhs_gg.block(tester.info_i.value().number_of_slack_variables, tester.info_i.value().offset_g_eq_slack) -
            tester.D_s_i.value() * tester.mult_i.value().block(tester.info_i.value().number_of_slack_variables, tester.info_i.value().offset_g_eq_slack);
        rhs_gg.block(tester.info_i.value().number_of_g_eq_path, tester.info_i.value().offset_g_eq_path) =
            rhs_gg.block(tester.info_i.value().number_of_g_eq_path, tester.info_i.value().offset_g_eq_path) -
            tester.D_eq_i.value() * tester.mult_i.value().block(tester.info_i.value().number_of_g_eq_path, tester.info_i.value().offset_g_eq_path);
        VecRealAllocated grad(tester.info_i.value().number_of_primal_variables);
        VecRealAllocated tmp(tester.info_i.value().number_of_primal_variables);
        grad = 0;
        tester.hessian_i.value().apply_on_right(tester.info_i.value(), tester.x_i.value(), 0.0, tmp, tmp);
        grad = grad + tmp + tester.D_x_i.value() * tester.x_i.value();
        tester.jacobian_i.value().transpose_apply_on_right(tester.info_i.value(), tester.mult_i.value(), 0.0, tmp, tmp);
        grad = grad + tmp;
        grad = grad + tester.rhs_x_i.value();
        for (Index i = 0; i < tester.info_i.value().number_of_eq_constraints; ++i)
        {
            EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
        }
        for (Index i = 0; i < tester.info_i.value().number_of_primal_variables; ++i)
        {
            EXPECT_NEAR(grad(i), 0, 1e-5);
        }
        */
    }

    std::cout << std::endl;
    std::cout << "Average implicit OCP solve duration:              " << time_implicit_us / nb_runs << " microseconds" << std::endl;
    std::cout << "Average implicit OCP solve duration (only solve): " << time_implicit_only_solve_us / nb_runs << " microseconds" << std::endl;
    std::cout << "Average reformulation OCP solve duration:         " << time_reformulation_us / nb_runs << " microseconds" << std::endl;
    std::cout << std::endl;

    std::cout << "Implicit OCP timings breakdown:" << std::endl;
    std::cout << "\tAverage copying rhs:                " << time_implicit_copying_rhs / nb_runs << " microseconds" << std::endl;
    std::cout << "\tAverage preprocessing (jacobian):   " << time_implicit_preprocess_jac_us / nb_runs << " microseconds" << std::endl;
    std::cout << "\tAverage preprocessing (hessian):    " << time_implicit_preprocess_hess_us / nb_runs << " microseconds" << std::endl;
    std::cout << "\t\tcopying RSQrqt:               " << time_implicit_preprocess_hess_copy_us / nb_runs << " microseconds" << std::endl;
    std::cout << "\t\tmodifying RSQrqt:             " << time_implicit_preprocess_hess_scaling_us / nb_runs << " microseconds" << std::endl;
    std::cout << "\tAverage solve:                      " << time_implicit_only_solve_us / nb_runs << " microseconds" << std::endl;
    std::cout << "\tAverage postprocessing:             " << time_implicit_postprocess_us / nb_runs << " microseconds" << std::endl;

    // print out percentages of [preprocess jac, preprocess hess, solving, postprocess]
    std::cout << "preprocess_jac_rel = " << 
        (time_implicit_preprocess_jac_us / time_implicit_us) << std::endl;
    std::cout << "preprocess_hess_rel = " <<
        (time_implicit_preprocess_hess_us / time_implicit_us) << std::endl;
    std::cout << "solve_rel = " <<
        (time_implicit_only_solve_us / time_implicit_us) << std::endl;
    std::cout << "postprocess_rel = " <<
        (time_implicit_postprocess_us / time_implicit_us) << std::endl;
    std::cout << "other_rel = " << 
        (1.0 - (time_implicit_preprocess_jac_us + time_implicit_preprocess_hess_us + time_implicit_only_solve_us + time_implicit_postprocess_us) / time_implicit_us) << std::endl;

    std::cout << "\ntotal_time_implicit = " << time_implicit_us  << std::endl;
    std::cout << "total_time_reformulation = " << time_reformulation_us << std::endl;
}

void PrintAverages(std::vector<std::vector<double>>& times, const std::string& name)
{
    std::cout << "avg_" << name << " = [";
    for (size_t i = 0; i < times.size(); ++i)
    {
        double avg = std::accumulate(times[i].begin(), times[i].end(), 0.0) / times[i].size();
        std::cout << avg;
        if (i < times.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

TEST_F(ImplicitAugSystemSolverVsReformulationTest, TestTimings7dof)
{
    int nb_runs = 500;
    std::vector<int> nx_values = {7, 14, 21, 28, 35};
    std::vector<int> nu_values = {7, 7, 7, 7, 7};
    std::vector<std::vector<double>> times_total_implicit(nx_values.size(), std::vector<double>(nb_runs));
    std::vector<std::vector<double>> times_total_reformulation(nx_values.size(), std::vector<double>(nb_runs));
    std::vector<std::vector<double>> times_preprocessing_jac(nx_values.size(), std::vector<double>(nb_runs));
    std::vector<std::vector<double>> times_preprocessing_hess(nx_values.size(), std::vector<double>(nb_runs));
    std::vector<std::vector<double>> times_solve(nx_values.size(), std::vector<double>(nb_runs));
    std::vector<std::vector<double>> times_postprocess(nx_values.size(), std::vector<double>(nb_runs));
    std::vector<std::vector<int>> Ks(nx_values.size(), std::vector<int>(nb_runs));
    std::vector<std::vector<int>> ng(nx_values.size(), std::vector<int>(nb_runs));

    double temp_time_1 = 0.0;
    double temp_time_2 = 0.0;

    for (int i = 0; i < nx_values.size(); i++){
        for (int counter = 0; counter < nb_runs; counter++){
            tester.UpdateRandomly(nx_values[i], nu_values[i], -1);
            // tester.UpdateRandomly(21, 7, 5+10*i);

            // Reformulation //
            auto start_r = std::chrono::high_resolution_clock::now();
            Index ret_r = tester.solver_r.value().solve(tester.info_r.value(), tester.jacobian_r.value(), tester.hessian_r.value(), 
                tester.D_x_r.value(), tester.D_eq_r.value(), tester.D_s_r.value(), tester.rhs_x_r.value(), tester.rhs_g_r.value(), tester.x_r.value(), tester.mult_r.value());
            auto stop_r = std::chrono::high_resolution_clock::now();
            auto duration_r = std::chrono::duration_cast<std::chrono::microseconds>(stop_r - start_r);
            times_total_reformulation[i][counter] = duration_r.count();

            // Implicit OCP //
            auto start_i = std::chrono::high_resolution_clock::now();
            Index ret = tester.solver_i.value().solve(tester.info_i.value(), tester.jacobian_i.value(), tester.hessian_i.value(), 
                tester.D_x_i.value(), tester.D_eq_i.value(), tester.D_s_i.value(), tester.rhs_x_i.value(), tester.rhs_g_i.value(), tester.x_i.value(), tester.mult_i.value());
            auto stop_i = std::chrono::high_resolution_clock::now();
            auto duration_i = std::chrono::duration_cast<std::chrono::microseconds>(stop_i - start_i);
            times_total_implicit[i][counter] = duration_i.count();
            times_preprocessing_jac[i][counter] = tester.solver_i.value().duration_preprocess_jac.count();
            // times_preprocessing_jac[i][counter] = tester.jacobian_i.value().dgemm_time;
            temp_time_1 += tester.solver_i.value().duration_preprocess_jac.count();
            temp_time_2 += tester.jacobian_i.value().dgemm_time;
            times_preprocessing_hess[i][counter] = tester.solver_i.value().duration_preprocess_hess.count();
            times_solve[i][counter] = tester.solver_i.value().duration_solve.count();
            times_postprocess[i][counter] = tester.solver_i.value().duration_postprocess.count();
            Ks[i][counter] = tester.info_i.value().dims.K;
            ng[i][counter] = tester.info_i.value().dims.number_of_eq_constraints[0];
        }
    }

    std::cout << "\n\nTime 1: " << temp_time_1 << " microseconds" << std::endl;
    std::cout << "Time 2: " << temp_time_2 << " microseconds\n\n" << std::endl;

    PrintAverages(times_total_implicit, "total_implicit");
    PrintAverages(times_total_reformulation, "total_reformulation");
    PrintAverages(times_preprocessing_jac, "preprocessing_jac");
    PrintAverages(times_preprocessing_hess, "preprocessing_hess");
    PrintAverages(times_solve, "solve");
    PrintAverages(times_postprocess, "postprocess");
}

TEST_F(ImplicitAugSystemSolverVsReformulationTest, TestTimingsPreProcessHess)
{
    int nb_runs = 1;
    std::map<std::string, std::vector<double>> results;
    std::map<std::string, double> local_result;

    for (int counter = 0; counter < nb_runs; counter++){
        tester.UpdateRandomly(true);

        // Implicit OCP //
        local_result = tester.hessian_i.value().TestPreProcessImplementation(
            tester.info_i.value(), tester.jacobian_i.value(), 
            tester.rhs_x_i.value(), tester.rhs_g_i.value());
        
        for (const auto& [key, value] : local_result)
        {
            results[key].push_back(value);
        }
    }

    // print out all averages
    for (const auto& [key, values] : results)
    {
        std::cout << "avg_" << key << " = ";
        double avg = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        std::cout << avg << std::endl;;
    }
}


class TestFunctionEvaluation {
    public:
        casadi::Function f;
        casadi::Function eval_Jk;
        casadi::Function eval_Jk_inv;
        casadi::Function eval_Bk;
        casadi::Function eval_Jk_inv_Bk;

        std::vector<double> Jk_inv_nz;
        std::vector<double> Bk_nz;
        std::vector<double> Jk_inv_Bk_nz;

        Sparsity sp_Jk_inv;
        Sparsity sp_Bk;
        Sparsity sp_Jk_inv_Bk;

        TestFunctionEvaluation(){};

        void SetUp(){};

        void Randomize(int nx, int nu, int nx_next){
            casadi::MX xk = casadi::MX::sym("xk", nx);
            casadi::MX uk = casadi::MX::sym("uk", nu);
            casadi::MX xk_next = casadi::MX::sym("xk_next", nx_next);

            f = casadi::Function("f", {xk, uk, xk_next},
                {xk_next*sumsqr(uk) + (xk_next * sumsqr(sin(xk)))});

            eval_Jk = casadi::Function("eval_J", {xk, uk, xk_next},
                        {jacobian(f({xk, uk, xk_next})[0], xk_next)});
            eval_Jk_inv = casadi::Function("eval_Jk_inv", {xk, uk, xk_next},
                        {inv(eval_Jk({xk, uk, xk_next})[0])});
            eval_Bk = casadi::Function("eval_Bk", {xk, uk, xk_next},
                        {jacobian(f({xk, uk, xk_next})[0], uk)});
            eval_Jk_inv_Bk = casadi::Function("eval_Jk_inv_Bk", {xk, uk, xk_next},
                        {mtimes(eval_Jk_inv({xk, uk, xk_next})[0], eval_Bk({xk, uk, xk_next})[0])});

            sp_Jk_inv = eval_Jk_inv.sparsity_out(0);
            sp_Bk = eval_Bk.sparsity_out(0);
            sp_Jk_inv_Bk = eval_Jk_inv_Bk.sparsity_out(0);

            Jk_inv_nz.resize(sp_Jk_inv.nnz());
            Bk_nz.resize(sp_Bk.nnz());
            Jk_inv_Bk_nz.resize(sp_Jk_inv_Bk.nnz());
        }

        void Test(int nb_runs){
            double option_1_time_eval_Jk_inv = 0.0;
            double option_1_time_eval_Bk = 0.0;
            double option_1_time_store_Jk_inv_Bk = 0.0;
            double option_1_time_dgemm = 0.0;

            double option_2_time_eval_Jk_inv_Bk = 0.0;
            double option_2_time_store_Jk_inv_Bk = 0.0;

            for (int i = 0; i < nb_runs; ++i) {
                // Randomize inputs
                int nx = ::test::random_int(1, 20);
                int nu = ::test::random_int(1, 20);
                int nx_next = ::test::random_int(1, 20);
                Randomize(nx, nu, nx_next);

                VecRealAllocated xk = ::test::random_vector(nx);
                VecRealAllocated uk = ::test::random_vector(nu);
                VecRealAllocated xk_next = ::test::random_vector(nx_next);

                MatRealAllocated Jk_inv_numeric = MatRealAllocated(nx_next, nx_next);
                MatRealAllocated Bk_numeric = MatRealAllocated(nx_next, nu);
                MatRealAllocated Jk_inv_Bk_numeric_1 = MatRealAllocated(nx_next, nu);
                MatRealAllocated Jk_inv_Bk_numeric_2 = MatRealAllocated(nx_next, nu);

                std::vector<const double*> arg_in(3);
                arg_in[0] = &xk(0); arg_in[1] = &uk(0); arg_in[2] = &xk_next(0);
                std::vector<double*> arg_out_1(1);
                std::vector<double*> arg_out_2(1);
                std::vector<double*> arg_out_3(1);

                arg_out_1[0] = &Jk_inv_nz[0];
                arg_out_2[0] = &Bk_nz[0];
                arg_out_3[0] = &Jk_inv_Bk_nz[0];

                // (1) evaluate Jk_inv and Bk and multiply numerically
                auto start = std::chrono::high_resolution_clock::now();
                eval_Jk_inv(arg_in, arg_out_1);
                auto stop = std::chrono::high_resolution_clock::now();
                option_1_time_eval_Jk_inv += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
                
                start = std::chrono::high_resolution_clock::now();
                eval_Bk(arg_in, arg_out_2);
                stop = std::chrono::high_resolution_clock::now();
                option_1_time_eval_Bk += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

                start = std::chrono::high_resolution_clock::now();
                int nz_ptr = 0;
                for (int i = 0; i < nx_next; ++i) {
                    for (int j = 0; j < nx_next; ++j) {
                        if (sp_Jk_inv.has_nz(i, j)) {
                            Jk_inv_numeric(i, j) = Jk_inv_nz[nz_ptr];
                            nz_ptr++;
                        }
                    }
                }
                nz_ptr = 0;
                for (int i = 0; i < nx_next; ++i) {
                    for (int j = 0; j < nu; ++j) {
                        if (sp_Bk.has_nz(i, j)) {
                            Bk_numeric(i, j) = Bk_nz[nz_ptr];
                            nz_ptr++;
                        }
                    }
                }
                stop = std::chrono::high_resolution_clock::now();
                option_1_time_store_Jk_inv_Bk += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

                start = std::chrono::high_resolution_clock::now();
                blasfeo_dgemm_nn(nx_next, nu, nx_next, 1.0, &Jk_inv_numeric.mat(), 0, 0, 
                    &Bk_numeric.mat(), 0, 0, 0.0, &Jk_inv_Bk_numeric_1.mat(), 0, 0, 
                    &Jk_inv_Bk_numeric_1.mat(), 0, 0);
                stop = std::chrono::high_resolution_clock::now();
                option_1_time_dgemm += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

                // (2) evaluate Jk_inv_Bk
                start = std::chrono::high_resolution_clock::now();
                eval_Jk_inv_Bk(arg_in, arg_out_3);
                stop = std::chrono::high_resolution_clock::now();
                option_2_time_eval_Jk_inv_Bk += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

                start = std::chrono::high_resolution_clock::now();
                nz_ptr = 0;
                for (int i = 0; i < nx_next; ++i) {
                    for (int j = 0; j < nu; ++j) {
                        if (sp_Jk_inv_Bk.has_nz(i, j)) {
                            Jk_inv_Bk_numeric_2(i, j) = Jk_inv_Bk_nz[nz_ptr];
                            nz_ptr++;
                        }
                    }
                }
                stop = std::chrono::high_resolution_clock::now();
                option_2_time_store_Jk_inv_Bk += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

                // (3) compare results
                for (Index i = 0; i < nx_next; ++i) {
                    for (Index j = 0; j < nu; ++j) {
                        EXPECT_NEAR(Jk_inv_Bk_numeric_1(i, j), Jk_inv_Bk_numeric_2(i, j), 1e-5);
                    }
                }

            }

            std::cout << "=====================================================" << std::endl;
            std::cout << "option 1 timings:" << std::endl;
            std::cout << "\teval Jk inv:            " << option_1_time_eval_Jk_inv / nb_runs << " microseconds" << std::endl;
            std::cout << "\teval Bk:                " << option_1_time_eval_Bk / nb_runs << " microseconds" << std::endl;
            std::cout << "\tstore Jk inv and Bk:    " << option_1_time_store_Jk_inv_Bk / nb_runs << " microseconds" << std::endl;
            std::cout << "\tdgemm:                  " << option_1_time_dgemm / nb_runs << " microseconds" << std::endl;
            std::cout << "total (without storing):  " 
                      << (option_1_time_eval_Jk_inv + option_1_time_eval_Bk + option_1_time_dgemm) / nb_runs 
                      << " microseconds" << std::endl;
            std::cout << "total:                    " 
                      << (option_1_time_eval_Jk_inv + option_1_time_eval_Bk + option_1_time_store_Jk_inv_Bk + option_1_time_dgemm) / nb_runs 
                      << " microseconds" << std::endl;
            std::cout << "=====================================================" << std::endl;
            std::cout << "option 2 timings:" << std::endl;
            std::cout << "\teval Jk inv Bk:         " << option_2_time_eval_Jk_inv_Bk / nb_runs << " microseconds" << std::endl;
            std::cout << "\tstore Jk inv Bk:        " << option_2_time_store_Jk_inv_Bk / nb_runs << " microseconds" << std::endl;
            std::cout << "total (without storing):  " 
                      << (option_2_time_eval_Jk_inv_Bk) / nb_runs 
                      << " microseconds" << std::endl;
            std::cout << "total:                    " 
                      << (option_2_time_eval_Jk_inv_Bk + option_2_time_store_Jk_inv_Bk) / nb_runs 
                      << " microseconds" << std::endl;
            std::cout << "=====================================================" << std::endl;
        }
};


/*
class TestFunctionEvaluation {
    public:

    void Setup(){};

    void Test(){
        int nx = 20;
        MX x = MX::sym("x", nx);
        MX f = sumsqr(sumsqr(x)  + sumsqr(sin(x)) - cos(x) + x * sin(x));
        MX g = sumsqr(-x);
        // MX f = sumsqr(x);
        // MX g = sumsqr(x);

        Function A = Function("A", {x}, {hessian(f, x)});
        Function B = Function("B", {x}, {hessian(g, x)});
        Function C = Function("C", {x}, {mtimes(A(x)[0], B(x)[0])});
        Sparsity sp_A = A.sparsity_out(0);
        Sparsity sp_B = B.sparsity_out(0);
        Sparsity sp_C = C.sparsity_out(0);

        VecRealAllocated x_numeric = ::test::random_vector(nx);
        MatRealAllocated A_numeric(nx, nx);
        MatRealAllocated B_numeric(nx, nx);
        MatRealAllocated C_numeric(nx, nx);
        MatRealAllocated D_numeric(nx, nx);
        std::vector<double> A_nz(nx*nx);
        std::vector<double> B_nz(nx*nx);
        std::vector<double> C_nz(nx*nx);

        // define input and output vectors
        std::vector<const double*> arg_in(1);
        std::vector<double*> arg_out_1(1);
        std::vector<double*> arg_out_2(1);
        std::vector<double*> arg_out_3(1);
        arg_in[0] = &x_numeric(0);
        arg_out_1[0] = &A_nz[0];
        arg_out_2[0] = &B_nz[0];
        arg_out_3[0] = &C_nz[0];

        // (1): evaluate A, B and compute A @ B numerically
        auto start = std::chrono::high_resolution_clock::now();
        A(arg_in, arg_out_1);
        B(arg_in, arg_out_2);
        int nz_ptr = 0;
        for (int i = 0; i < nx; ++i){
            for (int j = 0; j < nx; ++j){
                // std::cout << "i: " << i << ", j: " << j << ", nz_ptr: " << nz_ptr << std::endl;
                if (sp_A.has_nz(i, j)){ A_numeric(i, j) = A_nz[nz_ptr]; nz_ptr++;}
            }
        }
        nz_ptr = 0;
        for (int i = 0; i < nx; ++i){
            for (int j = 0; j < nx; ++j){
                // std::cout << "i: " << i << ", j: " << j << ", nz_ptr: " << nz_ptr << std::endl;
                if (sp_B.has_nz(i, j)){ B_numeric(i, j) = B_nz[nz_ptr]; nz_ptr++;}
            }
        }

        blasfeo_dgemm_nn(nx, nx, nx, 1.0, &A_numeric.mat(), 0, 0, 
            &B_numeric.mat(), 0, 0, 0.0, &D_numeric.mat(), 0, 0, 
            &D_numeric.mat(), 0, 0);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time for A @ B: " << duration.count() << " microseconds" << std::endl;

        // (2) evaluate C numerically
        start = std::chrono::high_resolution_clock::now();
        C(arg_in, arg_out_3);
        nz_ptr = 0;
        for (int i = 0; i < nx; ++i){
            for (int j = 0; j < nx; ++j){
                // std::cout << "i: " << i << ", j: " << j << ", nz_ptr: " << nz_ptr << std::endl;
                if (sp_C.has_nz(i, j)){ C_numeric(j, i) = C_nz[nz_ptr]; nz_ptr++;}
            }
        }
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time for C: " << duration.count() << " microseconds" << std::endl;

        // compare results
        for (Index i = 0; i < nx; ++i)
        {
            for (Index j = 0; j < nx; ++j)
            {
                EXPECT_NEAR(D_numeric(i, j), C_numeric(i, j), 1e-5);
            }
        }
        // std::cout << "A_numeric:\n" << A_numeric << std::endl;
        // std::cout << "B_numeric:\n" << B_numeric << std::endl;
        // std::cout << "D_numeric:\n" << D_numeric << std::endl;
        // std::cout << "C_numeric:\n" << C_numeric << std::endl;
    }
};
*/

TEST_F(ImplicitAugSystemSolverVsReformulationTest, TestFunctionEvaluation)
{
    // consider a general nonlinear function f(xk, uk, xk+1) and it's
    // linearization Jk @ xk+1 + Ak @ xk + Bk @ uk + bk

    // we want to compare two cases:
    // (1) evaluate Jk, Jk_inv, Ak, Bk and bk and compute Jk_inv @ Ak,
    //     Jk_inv @ Bk, Jk_inv @ bk
    // (2) evaluate symbolically constructed Jk_inv @ Ak, Jk_inv @ Bk and
    //     Jk_inv @ bk symbolically 

    TestFunctionEvaluation tester;
    tester.Test(5000);

}