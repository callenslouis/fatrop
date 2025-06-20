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

using namespace fatrop;

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
    // MatRealAllocated I_check = ::test::identity_matrix(A.m());
    // blasfeo_dgemm_nn(A.m(), A.m(), A.m(), 1.0, 
    //                  const_cast<MAT *>(&A.mat()), 0, 0, 
    //                  const_cast<MAT *>(&A_inv.mat()), 0, 0, 0.0, 
    //                  const_cast<MAT *>(&I_check.mat()), 0, 0,
    //                  const_cast<MAT *>(&I_check.mat()), 0, 0);
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
                        // jacobian.Jt_inv[k].block(nx_next, nx_next, 0, 0) =
                        //     ::test::random_matrix(nx_next, nx_next);
                        jacobian.Jt_inv[k].block(nx_next, nx_next, 0, 0) =
                            ::test::identity_matrix(nx_next, 1.0);
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
            // full_matrix_jacobian.block(nx_next, nx_next, offs_eq_dyn, offs_x_next).diagonal() =
            //     -1.0;
            full_matrix_jacobian.block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = jacobian.Jt[k];
            hessian.FuFxt[k].block(nx + nu, nx_next, 0, 0) =
                ::test::random_matrix(nx + nu, nx_next);
            if (CREATE_EXPLICIT_EQUIVALENT){
                hessian.FuFxt[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::empty_matrix(nx + nu, nx_next);
            } else {
                // hessian.FuFxt[k].block(nx + nu, nx_next, 0, 0) =
                //     ::test::random_matrix(nx + nu, nx_next);
                hessian.FuFxt[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::empty_matrix(nx + nu, nx_next);
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
    // std::vector<Index> nx = {1, 1}; // State dimensions for each stage
    // std::vector<Index> nu = {2, 0};    // Input dimensions for each stage
    // std::vector<Index> ng = {0, 0}; // Equality constraints for each stage
    // std::vector<Index> ng_ineq = {0, 0}; // Inequality constraints for each stage
    int K = 4;                                                   // Number of stages
    std::vector<Index> nx = {1, 2, 1, 1}; // State dimensions for each stage
    std::vector<Index> nu = {1, 1, 1, 0};    // Input dimensions for each stage
    std::vector<Index> ng = {0, 0, 0, 0}; // Equality constraints for each stage
    std::vector<Index> ng_ineq = {0, 0, 0, 0}; // Inequality constraints for each stage

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

                // jacobian.Jt[k].block(nx_next, nx_next, 0, 0) =
                //     ::test::random_matrix(nx_next, nx_next);
                jacobian.Jt[k].block(nx_next, nx_next, 0, 0) =
                    ::test::identity_matrix(nx_next, -1.0);

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
            full_matrix_jacobian.block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = jacobian.Jt[k];

            hessian.FuFxt[k].block(nx + nu, nx_next, 0, 0) =
                ::test::random_matrix(nx + nu, nx_next);

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

        std::cout << "Full KKT matrix: \n"
                  << full_kkt_matrix << std::endl;
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
        std::cout << "Full KKT rhs vector: \n"
                  << full_kkt_rhs << std::endl;
    // std::cout << "Created ImplicitAugSystemSolverTest" << std::endl;
    };
};

void PrintSolutionOfOcpTypeSolver(ImplicitAugSystemSolverTest &implicit_solver, 
                                  VecRealAllocated &original_x,
                                  VecRealAllocated &original_mult){
   std::cout << "mult: " << original_mult << std::endl;
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

    // compute KKT-matrix @ x
    VecRealAllocated expected_rhs(info.number_of_primal_variables + info.number_of_eq_constraints);
    VecRealAllocated solution_vector(info.number_of_primal_variables + info.number_of_eq_constraints);
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        solution_vector(i) = x(i);
    }
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        solution_vector(info.number_of_primal_variables + i) = mult(i);
    }

    std::cout << "Full solution vector: \n"
              << solution_vector << std::endl;
    // gemv_n(full_kkt_matrix.m(), full_kkt_matrix.n(), 1.0, full_kkt_matrix, 0, 0, 
    //        solution_vector, 0, 0.0, expected_rhs, 0, expected_rhs, 0);

    // check if the rhs is equal to the expected rhs
    std::cout << "Comparing KKT @ sol with [rhs_x; rhs_g]" << std::endl;
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        std::cout << "[" << i << "]: \t" << -expected_rhs(i) << "\t-\t" << rhs_x(i) 
                  << "\t\t(" << std::abs(-expected_rhs(i) - rhs_x(i)) << ")" << std::endl;
    }
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        std::cout << "[" << i << "]: \t" << -expected_rhs(info.number_of_primal_variables + i) 
                  << "\t-\t" << rhs_g(i) 
                  << "\t\t(" << std::abs(-expected_rhs(info.number_of_primal_variables + i) - rhs_g(i)) << ")" 
                  << std::endl;
    }
    std::cout << "---------------------------" << std::endl;
    std::cout << "Comparing KKT @ sol with [grad; rhs_gg]" << std::endl;
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        std::cout << "[" << i << "]: \t" << expected_rhs(i) << "\t-\t" << grad(i) 
                  << "\t\t(" << std::abs(expected_rhs(i) - grad(i)) << ")" << std::endl;
    }
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        std::cout << "[" << i << "]: \t" << expected_rhs(info.number_of_primal_variables + i) 
                  << "\t-\t" << rhs_gg(i) 
                  << "\t\t(" << std::abs(expected_rhs(info.number_of_primal_variables + i) - rhs_gg(i)) << ")" 
                  << std::endl;
    }

    // test hessian.apply_on_right
    VecRealAllocated v(info.number_of_primal_variables);
    VecRealAllocated h(info.number_of_primal_variables);
    h = 0.0;

    for (int i = 0; i < info.number_of_primal_variables; ++i)
    {
        v = 0; v(i) = 1.0;
        hessian.apply_on_right(info, v, 0.0, h, h);
        std::cout << "column " << i << " of hessian should be: " << h << std::endl;
    }
    
    // test jacobian.apply_on_right
    VecRealAllocated j(info.number_of_eq_constraints);
    for (int i = 0; i < info.number_of_primal_variables; ++i)
    {
        v = 0; v(i) = 1.0;
        jacobian.apply_on_right(info, v, 0.0, j, j);
        std::cout << "column " << i << " of jacobian should be: " << j << std::endl;
    }
    
    // test jacobian.transpose_apply_on_right
    v = 1;
    jacobian.transpose_apply_on_right(info, v, 0.0, h, h);
    for (int i = 0; i < info.number_of_eq_constraints; ++i)
    {
        std::cout << "column " << i << " of jacobian transpose should be: " << h(i) << std::endl;
    }

    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }
}