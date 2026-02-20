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

using namespace fatrop;

class GeneralImplicitAugSystemSolverTest : public ::testing::Test
{
// protected:
public:
    bool J_matrix_is_idendity = true;
    bool no_second_order_effects = true;

    // Create OcpDims object
    int K = 10;                                                   // Number of stages
    std::vector<Index> nx = {20, 10, 10, 10, 10, 2, 0, 1, 10, 5}; // State dimensions for each stage
    std::vector<Index> r =  {20,  10, 10, 10, 10, 2, 0, 1, 10, 5};
    std::vector<Index> nu = {1, 4, 2, 10, 1, 30, 4, 1, 10, 0};    // Input dimensions for each stage
    std::vector<Index> ng = {9, 3, 4, 3, 4, 0, 1, 0, 1, 5}; // Equality constraints for each stage
    std::vector<Index> ng_ineq = {0, 5, 10, 0,   0,
                                  0, 0, 0,  10, 0}; // Inequality constraints for each stage
    // int K = 3;                                                   // Number of stages
    // std::vector<Index> nx = {2, 2, 2}; // State dimensions for each stage
    // std::vector<Index> r =  {2, 1, 2};
    // std::vector<Index> nu = {1, 3, 1};    // Input dimensions for each stage
    // std::vector<Index> ng = {2, 1, 1}; // Equality constraints for each stage
    // std::vector<Index> ng_ineq = {0, 0, 0}; // Inequa

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

    /*
    // // Normal case (manual treatment of dynamics as equality constraints)
    std::vector<Index> nu2 = {0, 1, 0};    // Input dimensions for each stage
    std::vector<Index> ng2 = {1, 0, 0}; // Equality constraints for each stage

    ProblemDims dims2{K, nu2, r, ng2, ng_ineq};
    ProblemInfo info2{dims2};
    // Create Jacobian object
    Jacobian<OcpType> jacobian2{dims2};
    MatRealAllocated full_matrix_jacobian2 =
        MatRealAllocated(info2.number_of_eq_constraints, info2.number_of_primal_variables);
    Hessian<OcpType> hessian2{dims2};
    MatRealAllocated full_matrix_hessian2 =
        MatRealAllocated(info2.number_of_primal_variables, info2.number_of_primal_variables);
    VecRealAllocated x2 = VecRealAllocated(info2.number_of_primal_variables);
    VecRealAllocated mult2 = VecRealAllocated(info2.number_of_eq_constraints);
    VecRealAllocated rhs_x2 = VecRealAllocated(info2.number_of_primal_variables);
    VecRealAllocated rhs_g2 = VecRealAllocated(info2.number_of_eq_constraints);
    VecRealAllocated D_x2 = VecRealAllocated(info2.number_of_primal_variables);
    VecRealAllocated D_s2 = VecRealAllocated(info2.number_of_slack_variables);
    VecRealAllocated D_eq2 = VecRealAllocated(info2.number_of_g_eq_path);
    MatRealAllocated full_kkt_matrix2 =
        MatRealAllocated(info2.number_of_primal_variables + info2.number_of_eq_constraints,
                         info2.number_of_primal_variables + info2.number_of_eq_constraints);
    AugSystemSolver<OcpType> solver2 = AugSystemSolver<OcpType>(info2);
    */


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

                if (J_matrix_is_idendity){
                    jacobian.Jt[k].block(r[k+1], r[k+1], 0, 0) =
                        ::test::identity_matrix(r[k+1], -1);
                } else {
                    jacobian.Jt[k].block(nx_next, nx_next, 0, 0) =
                        ::test::random_matrix(nx_next, nx_next);
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
            if (no_second_order_effects){
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













        /*
        x2 = 0;
        full_matrix_jacobian2 = 0.;

        // fill the jacobian with random values
        for (Index k = 0; k < info2.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index nu2 = info2.dims.number_of_controls[k];
            Index nx2 = info2.dims.number_of_states[k];
            if (k < info2.dims.K - 1)
            {
                Index nx_next = info.dims.number_of_states[k + 1];
                Index nx_next2 = info2.dims.number_of_states[k + 1];

                // jacobian2.BAbt[k].block(nu2 + nx2, nx_next2, 0, 0) =
                //     jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0);
                jacobian2.BAbt[k].block(nu2, nx_next2, 0, 0) =
                    jacobian.BAbt[k].block(nu2, nx_next2, r[k], 0);
                jacobian2.BAbt[k].block(nx2, nx_next2, nu2, 0) =
                    jacobian.BAbt[k].block(nx2, nx_next2, 0, 0);                
                jacobian2.Gg_eqt[k].block(nu + nx, info2.dims.number_of_eq_constraints[k], 0, 0) =
                    jacobian.BAbt[k].block(nu + nx, info2.dims.number_of_eq_constraints[k], 0, r[k+1]);
            } else {
                // jacobian2.Gg_eqt[k] should be empty
            }
            jacobian2.Gg_ineqt[k].block(nu + nx, info2.dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info2.dims.number_of_ineq_constraints[k]);
        }
        // fill the Hessian with random values
        for (Index k = 0; k < dims2.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index nu2 = info2.dims.number_of_controls[k];
            Index nx2 = info2.dims.number_of_states[k];
            // we know nu is zero for the implicit version
            hessian2.RSQrqt[k].block(nu2, nu2, 0, 0) = hessian.RSQrqt[k].block(nu2, nu2, nx-r[k], nx-r[k]);
            hessian2.RSQrqt[k].block(nx2, nx2, nu2, nu2) = hessian.RSQrqt[k].block(r[k], r[k], 0, 0);
            hessian2.RSQrqt[k].block(nx2, nu2, nu2, 0) = hessian.RSQrqt[k].block(nx2, nu2, 0, nx-r[k]);
            hessian2.RSQrqt[k].block(nu2, nx2, 0, nu2) = hessian.RSQrqt[k].block(nu2, nx2, nx-r[k], 0);
        }
        // add dynamics constraints
        for (Index k = 0; k < info2.dims.K - 1; ++k)
        {
            Index nu2 = info2.dims.number_of_controls[k];
            Index nx2 = info2.dims.number_of_states[k];
            Index offs_ux2 = info2.offsets_primal_u[k];
            Index offs_x_next2 = info2.offsets_primal_x[k + 1];
            Index nx_next2 = info2.dims.number_of_states[k + 1];
            Index offs_eq_dyn2 = info2.offsets_g_eq_dyn[k];
            full_matrix_jacobian2.block(nx_next2, nu2 + nx2, offs_eq_dyn2, offs_ux2) =
                transpose(jacobian2.BAbt[k].block(nu2 + nx2, nx_next2, 0, 0));

            full_matrix_jacobian2.block(nx_next2, nx_next2, offs_eq_dyn2, offs_x_next2) = 
                ::test::identity_matrix(nx_next2, -1);
        }
        // equality path equality constraints
        for (Index k = 0; k < info2.dims.K; ++k)
        {
            Index nu2 = info2.dims.number_of_controls[k];
            Index nx2 = info2.dims.number_of_states[k];
            Index ng2 = info2.dims.number_of_eq_constraints[k];
            Index offset_ux2 = info2.offsets_primal_u[k];
            Index offset_g_eq2 = info2.offsets_g_eq_path[k];
            full_matrix_jacobian2.block(ng2, nu2 + nx2, offset_g_eq2, offset_ux2) =
                transpose(jacobian2.Gg_eqt[k].block(nu2 + nx2, ng2, 0, 0));
        }
        // inequality path constraints
        for (Index k = 0; k < info2.dims.K; ++k)
        {
            Index nu2 = info2.dims.number_of_controls[k];
            Index nx2 = info2.dims.number_of_states[k];
            Index ng_ineq2 = info2.dims.number_of_ineq_constraints[k];
            Index offset_ux2 = info2.offsets_primal_u[k];
            Index offset_g_ineq2 = info2.offsets_g_eq_slack[k];
            full_matrix_jacobian2.block(ng_ineq2, nu2 + nx2, offset_g_ineq2, offset_ux2) =
                transpose(jacobian2.Gg_ineqt[k].block(nu2 + nx2, ng_ineq2, 0, 0));
        }
        full_matrix_hessian2 = 0.;
        // populate the full matrix
        for (Index k = 0; k < dims2.K; k++)
        {
            Index nu2 = dims2.number_of_controls[k];
            Index nx2 = dims2.number_of_states[k];
            Index offs_ux2 = info2.offsets_primal_u[k];
            full_matrix_hessian2.block(nu2 + nx2, nu2 + nx2, offs_ux2, offs_ux2) =
                hessian2.RSQrqt[k].block(nu2 + nx2, nu2 + nx2, 0, 0);
        }
        // set up the full KKT matrix
        full_kkt_matrix2.block(info2.number_of_primal_variables, info2.number_of_primal_variables, 0,
                              0) = full_matrix_hessian2;
        full_kkt_matrix2.block(info2.number_of_primal_variables, info2.number_of_eq_constraints, 0,
                              info2.number_of_primal_variables) = transpose(full_matrix_jacobian2);
        full_kkt_matrix2.block(info2.number_of_eq_constraints, info2.number_of_primal_variables,
                              info2.number_of_primal_variables, 0) = full_matrix_jacobian2;

        // fill the x vector with random values
        for (Index k = 0; k < info2.dims.K; ++k)
        {
            Index nu2 = info2.dims.number_of_controls[k];
            Index nx2 = info2.dims.number_of_states[k];
            rhs_x2.block(nu2, info2.offsets_primal_u[k]) =
                rhs_x.block(nu2, info.offsets_primal_x[k] + r[k]);
            rhs_x2.block(nx2, info2.offsets_primal_x[k]) =
                rhs_x.block(nx2, info.offsets_primal_x[k]);
        }
        for (Index i = 0; i < info2.number_of_primal_variables; ++i)
        {
            // rhs_x2(i) = 0 * 1.0 * i;
            D_x2 = 0 * 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info2.number_of_eq_constraints; ++i)
        {
            // rhs_g2(i) = 0 * 1.0 * i;
        }
        for (Index k = 0; k < info2.dims.K; ++k)
        {
            if (k < info2.dims.K-1){
                Index nx_next = info2.dims.number_of_states[k + 1];
                rhs_g2.block(nx_next, info2.offsets_g_eq_dyn[k]) =
                    rhs_g.block(nx_next, info.offsets_g_eq_dyn[k]);
                Index ng2 = info2.dims.number_of_eq_constraints[k];
                rhs_g2.block(ng2, info2.offsets_g_eq_path[k]) =
                    rhs_g.block(ng2, info.offsets_g_eq_dyn[k] + r[k+1]);
            }
        }

        for (Index i = 0; i < info2.number_of_g_eq_path; ++i)
        {
            D_eq2(i) = 0 * 1.0 * (i + 1);
        }
        for (Index i = 0; i < info2.number_of_slack_variables; ++i)
        {
            D_s2(i) = 0 * 1.0 * (i + 0.1);
        }
        */
    };
};

TEST_F(GeneralImplicitAugSystemSolverTest, TestSolve)
{
    // IMPLICIT OCP VERSION //
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_s, rhs_x, rhs_g, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);

    /*
    // print the full KKT matrix and rhs
    std::cout << "KKT = np.array([\n";
    for (Index i = 0; i < full_kkt_matrix.m(); i++){
        std::cout << "\t[";
        for (Index j = 0; j < full_kkt_matrix.n(); j++){
            std::cout << full_kkt_matrix(i,j);
            if (j < full_kkt_matrix.n() - 1){
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < full_kkt_matrix.m() - 1){
            std::cout << ",\n";
        }
    }
    std::cout << "\n])" << std::endl;

    VecRealAllocated full_rhs = VecRealAllocated(info.number_of_primal_variables + info.number_of_eq_constraints);
    for (Index i = 0; i < info.number_of_primal_variables; ++i){full_rhs(i) = rhs_x(i);}
    for (Index i = 0; i < info.number_of_eq_constraints; ++i){full_rhs(info.number_of_primal_variables + i) = rhs_g(i);}

    std::cout << "rhs = np.array([\n";
    for (Index i = 0; i < full_rhs.m(); i++){
        std::cout << "\t" << full_rhs(i);
        if (i < full_rhs.m() - 1){
            std::cout << ",";
        }
    }
    std::cout << "\n])" << std::endl;

    // print obtained solution //
    std::cout << "Obtained solution x:" << std::endl << x << std::endl;
    std::cout << "Obtained solution mult:" << std::endl << mult << std::endl;
    */

    /*
    // NORMAL OCP VERSION //
    Index ret2 = solver2.solve(info2, jacobian2, hessian2, D_x2, D_s2, rhs_x2, rhs_g2, x2, mult2);
    EXPECT_EQ(ret2, LinsolReturnFlag::SUCCESS);

    // print the full KKT matrix and rhs
    std::cout << "KKT2 = np.array([\n";
    for (Index i = 0; i < full_kkt_matrix2.m(); i++){
        std::cout << "\t[";
        for (Index j = 0; j < full_kkt_matrix2.n(); j++){
            std::cout << full_kkt_matrix2(i,j);
            if (j < full_kkt_matrix2.n() - 1){
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < full_kkt_matrix2.m() - 1){
            std::cout << ",\n";
        }
    }
    std::cout << "\n])" << std::endl;

    VecRealAllocated full_rhs2 = VecRealAllocated(info2.number_of_primal_variables + info2.number_of_eq_constraints);
    for (Index i = 0; i < info2.number_of_primal_variables; ++i){full_rhs2(i) = rhs_x2(i);}
    for (Index i = 0; i < info2.number_of_eq_constraints; ++i){full_rhs2(info2.number_of_primal_variables + i) = rhs_g2(i);}

    std::cout << "rhs2 = np.array([\n";
    for (Index i = 0; i < full_rhs2.m(); i++){
        std::cout << "\t" << full_rhs2(i);
        if (i < full_rhs2.m() - 1){
            std::cout << ",";
        }
    }
    std::cout << "\n])" << std::endl;

    // print obtained solution //
    std::cout << "Obtained solution x2:" << std::endl << x2 << std::endl;
    std::cout << "Obtained solution mult2:" << std::endl << mult2 << std::endl;
    */




    // Solution checking
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