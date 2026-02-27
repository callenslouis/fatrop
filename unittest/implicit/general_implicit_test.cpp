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
#include <random>

using namespace fatrop;

class GeneralImplicitAugSystemSolverTest : public ::testing::Test
{
// protected:
public:
    bool J_matrix_is_idendity = false;
    bool no_second_order_effects = true;

    // Create OcpDims object
    int K = 10;                                                   // Number of stages
    std::vector<Index> nx = {20, 10, 10, 10, 10, 2, 0, 1, 10, 5}; // State dimensions for each stage
    std::vector<Index> r =  {20, 5, 2, 10, 9, 1, 0, 1, 6, 0};
    std::vector<Index> nu = {1, 4, 2, 10, 1, 30, 4, 5, 10, 2};    // Input dimensions for each stage
    std::vector<Index> ng = {9, 3, 4, 3, 4, 2, 1, 0, 1, 5}; // Equality constraints for each stage
    std::vector<Index> ng_ineq = {0, 5, 10, 4, 0, 0, 0, 0, 10, 0}; // Inequality constraints for each stage
    // int K = 3;
    // std::vector<Index> nx = {1, 2, 5};
    // std::vector<Index> r =  {1, 1, 2};
    // std::vector<Index> nu = {1, 4, 0};
    // std::vector<Index> ng = {0, 0, 2};
    // std::vector<Index> ng_ineq = {0, 3, 4};
    // int K = 2;
    // std::vector<Index> nx = {1, 3};
    // std::vector<Index> r =  {1, 0};
    // std::vector<Index> nu = {5, 2};
    // std::vector<Index> ng = {2, 1};
    // std::vector<Index> ng_ineq = {4, 2};
    // int K = 3;
    // std::vector<Index> nx = {2, 2, 2};
    // std::vector<Index> r =  {2, 0, 0};
    // std::vector<Index> nu = {3, 4, 2};
    // std::vector<Index> ng = {2, 1, 1};
    // std::vector<Index> ng_ineq = {1, 2, 2};
    // int K = 2;
    // std::vector<Index> nx = {2, 2};
    // std::vector<Index> r =  {2, 1};
    // std::vector<Index> nu = {0, 0};
    // std::vector<Index> ng = {0, 0};
    // std::vector<Index> ng_ineq = {0, 0};

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
                    if (k == 0 && false){
                        for (Index i = 0; i < r[k+1]; i++){
                            jacobian.Jt[k](r[k+1] - 1 - i, i) = -1.0;
                        }
                    } else {
                        jacobian.Jt[k].block(r[k+1], r[k+1], 0, 0) =
                            ::test::identity_matrix(r[k+1], -1);
                    }
                } else {
                    jacobian.Jt[k].block(nx_next, nx_next, 0, 0) =
                        // ::test::random_matrix(r[k+1], r[k+1]);
                        ::test::random_degenerate_matrix(nx_next, r[k+1]);
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
            hessian.FuFxt[k].block(nx_next, nx + nu, 0, 0) =
                ::test::random_matrix(nx_next, nx + nu);
            if (no_second_order_effects){
                hessian.FuFxt[k].block(nx_next, nx + nu, 0, 0) =
                    ::test::empty_matrix(nx_next, nx + nu);
            } else {
                hessian.FuFxt[k].block(nx_next, nx + nu, 0, 0) =
                    ::test::random_matrix(nx_next, nx + nu);
            }
            full_matrix_hessian.block(nx_next, nu + nx, offs_x_next, offs_ux) = 
                hessian.FuFxt[k];
            full_matrix_hessian.block(nu + nx, nx_next, offs_ux, offs_x_next) =
                transpose(hessian.FuFxt[k]);
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
            D_x(i) = 1.0 * (i + 0.1);
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
            D_s(i) =  1.0 * (i + 0.1);
        }
    };
};


class GeneralImplicitAugSystemSolverTest2 : public ::testing::Test
{
// protected:
public:
    bool no_second_order_effects = true;

    // Create OcpDims object
    int K;
    std::vector<Index> nx;
    std::vector<Index> r;
    std::vector<Index> nu;
    std::vector<Index> ng;
    std::vector<Index> ng_ineq;
    std::optional<ProblemDims> dims;
    std::optional<ProblemInfo> info;
    std::optional<Jacobian<ImplicitOcpType>> jacobian;
    std::optional<Hessian<ImplicitOcpType>> hessian;
    std::optional<MatRealAllocated> full_matrix_jacobian;
    std::optional<MatRealAllocated> full_matrix_hessian;
    std::optional<VecRealAllocated> x;
    std::optional<VecRealAllocated> mult;
    std::optional<VecRealAllocated> rhs_x;
    std::optional<VecRealAllocated> rhs_g;
    std::optional<VecRealAllocated> D_x;
    std::optional<VecRealAllocated> D_s;
    std::optional<VecRealAllocated> D_eq;
    std::optional<MatRealAllocated> full_kkt_matrix;
    std::optional<AugSystemSolver<ImplicitOcpType>> solver;

    std::vector<int> RandomVector(int size, int min_val, int max_val)
    {
        std::vector<int> vec(size);
        for (int i = 0; i < size; ++i){
            vec[i] = rand() % (max_val - min_val + 1) + min_val;
        }
        return vec;
    }

    void GetRandomDimensions()
    {
        srand(time(0));
        K = rand() % 20 + 2; // Random K between 2 and 21
        nx = RandomVector(K, 0, 20);
        r = std::vector<Index>(K, 100);
        for (int k = 0; k < K; ++k){ 
            while (r[k] > nx[k]){ r[k] = rand() % (nx[k]+1);}
        }
        nu = RandomVector(K, 0, 20);
        ng = RandomVector(K, 0, 20);
        for (int k = 0; k < K; ++k){
            bool okay = false;
            while (!okay){
                int max_allowed_ng = nx[k] + nu[k];
                if (k < K-1){ max_allowed_ng -= nx[k+1] - r[k+1];}
                if (ng[k] <= max_allowed_ng){
                    okay = true;
                } else {
                    // randomize both the nb of constraints and the nb of controls
                    ng[k] = rand() % (20 + 1);
                    nu[k] = rand() % (20 + 1);
                }
            }
        }
        ng_ineq = RandomVector(K, 0, 20);

        dims.emplace(ProblemDims{K, nu, nx, ng, ng_ineq});
        info.emplace(ProblemInfo(dims.value()));
        jacobian.emplace(Jacobian<ImplicitOcpType>(dims.value()));
        full_matrix_jacobian =
            MatRealAllocated(info->number_of_eq_constraints, info->number_of_primal_variables);
        hessian.emplace(Hessian<ImplicitOcpType>(dims.value()));
        full_matrix_hessian =
            MatRealAllocated(info->number_of_primal_variables, info->number_of_primal_variables);
        x = VecRealAllocated(info->number_of_primal_variables);
        mult = VecRealAllocated(info->number_of_eq_constraints);
        rhs_x = VecRealAllocated(info->number_of_primal_variables);
        rhs_g = VecRealAllocated(info->number_of_eq_constraints);
        D_x = VecRealAllocated(info->number_of_primal_variables);
        D_s = VecRealAllocated(info->number_of_slack_variables);
        D_eq = VecRealAllocated(info->number_of_g_eq_path);
        full_kkt_matrix =
            MatRealAllocated(info->number_of_primal_variables + info->number_of_eq_constraints,
                             info->number_of_primal_variables + info->number_of_eq_constraints);
        solver.emplace(AugSystemSolver<ImplicitOcpType>(info.value()));
    }

    void Randomize(){
        GetRandomDimensions();
        x = 0;
        full_matrix_jacobian.value() = 0.;

        // fill the jacobian with random values
        for (Index k = 0; k < info.value().dims.K; ++k)
        {
            Index nu = info.value().dims.number_of_controls[k];
            Index nx = info.value().dims.number_of_states[k];
            if (k < info.value().dims.K - 1)
            {
                Index nx_next = info.value().dims.number_of_states[k + 1];
                jacobian.value().BAbt[k].block(nu + nx, nx_next, 0, 0) =
                    ::test::random_matrix(nu + nx, nx_next);

                jacobian.value().Jt[k].block(nx_next, nx_next, 0, 0) =
                    // ::test::random_matrix(r[k+1], r[k+1]);
                    ::test::random_degenerate_matrix(nx_next, r[k+1]);
            }
            jacobian.value().Gg_eqt[k].block(nu + nx, info.value().dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.value().dims.number_of_eq_constraints[k]);
            jacobian.value().Gg_ineqt[k].block(nu + nx, info.value().dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.value().dims.number_of_ineq_constraints[k]);
        }
        // fill the Hessian with random values
        for (Index k = 0; k < dims.value().K; ++k)
        {
            Index nu = info.value().dims.number_of_controls[k];
            Index nx = info.value().dims.number_of_states[k];
            hessian.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
        }
        // add dynamics constraints
        for (Index k = 0; k < info.value().dims.K - 1; ++k)
        {
            Index nu = info.value().dims.number_of_controls[k];
            Index nx = info.value().dims.number_of_states[k];
            Index offs_ux = info.value().offsets_primal_u[k];
            Index offs_x_next = info.value().offsets_primal_x[k + 1];
            Index nx_next = info.value().dims.number_of_states[k + 1];
            Index offs_eq_dyn = info.value().offsets_g_eq_dyn[k];
            full_matrix_jacobian.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian.value().BAbt[k].block(nu + nx, nx_next, 0, 0));
            full_matrix_jacobian.value().block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                transpose(jacobian.value().Jt[k]);
            hessian.value().FuFxt[k].block(nx_next, nx + nu, 0, 0) =
                ::test::random_matrix(nx_next, nx + nu);
            if (no_second_order_effects){
                hessian.value().FuFxt[k].block(nx_next, nx + nu, 0, 0) =
                    ::test::empty_matrix(nx_next, nx + nu);
            } else {
                hessian.value().FuFxt[k].block(nx_next, nx + nu, 0, 0) =
                    ::test::random_matrix(nx_next, nx + nu);
            }
            full_matrix_hessian.value().block(nx_next, nu + nx, offs_x_next, offs_ux) = 
                hessian.value().FuFxt[k];
            full_matrix_hessian.value().block(nu + nx, nx_next, offs_ux, offs_x_next) =
                transpose(hessian.value().FuFxt[k]);
        }
        // equality path equality constraints
        for (Index k = 0; k < info.value().dims.K; ++k)
        {
            Index nu = info.value().dims.number_of_controls[k];
            Index nx = info.value().dims.number_of_states[k];
            Index ng = info.value().dims.number_of_eq_constraints[k];
            Index offset_ux = info.value().offsets_primal_u[k];
            Index offset_g_eq = info.value().offsets_g_eq_path[k];
            full_matrix_jacobian.value().block(ng, nu + nx, offset_g_eq, offset_ux) =
                transpose(jacobian.value().Gg_eqt[k].block(nu + nx, ng, 0, 0));
        }
        // inequality path constraints
        for (Index k = 0; k < info.value().dims.K; ++k)
        {
            Index nu = info.value().dims.number_of_controls[k];
            Index nx = info.value().dims.number_of_states[k];
            Index ng_ineq = info.value().dims.number_of_ineq_constraints[k];
            Index offset_ux = info.value().offsets_primal_u[k];
            Index offset_g_ineq = info.value().offsets_g_eq_slack[k];
            full_matrix_jacobian.value().block(ng_ineq, nu + nx, offset_g_ineq, offset_ux) =
                transpose(jacobian.value().Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));
        }
        full_matrix_hessian.value() = 0.;
        // populate the full matrix
        for (Index k = 0; k < dims.value().K; k++)
        {
            Index nu = dims.value().number_of_controls[k];
            Index nx = dims.value().number_of_states[k];
            Index offs_ux = info.value().offsets_primal_u[k];
            full_matrix_hessian.value().block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        // set up the full KKT matrix
        full_kkt_matrix.value().block(info.value().number_of_primal_variables, info.value().number_of_primal_variables, 0,
                              0) = full_matrix_hessian.value();
        full_kkt_matrix.value().block(info.value().number_of_primal_variables, info.value().number_of_eq_constraints, 0,
                              info.value().number_of_primal_variables) = transpose(full_matrix_jacobian.value());
        full_kkt_matrix.value().block(info.value().number_of_eq_constraints, info.value().number_of_primal_variables,
                              info.value().number_of_primal_variables, 0) = full_matrix_jacobian.value();

        // fill the x vector with random values
        for (Index i = 0; i < info.value().number_of_primal_variables; ++i)
        {
            rhs_x.value()(i) = rand() % 10;
            D_x.value()(i) = rand() % 10;
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info.value().number_of_eq_constraints; ++i)
        {
            rhs_g.value()(i) = rand() % 10;
        }

        for (Index i = 0; i < info.value().number_of_g_eq_path; ++i)
        {
            D_eq.value()(i) = rand() % 10;
        }
        for (Index i = 0; i < info.value().number_of_slack_variables; ++i)
        {
            D_s.value()(i) =  rand() % 10;
        }
    }

    void SetUp()
    {
        Randomize();
    };
};

TEST_F(GeneralImplicitAugSystemSolverTest, TestSolve)
{
    // IMPLICIT OCP VERSION //
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_s, rhs_x, rhs_g, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);

    // print the full KKT matrix and rhs
    bool print_full_kkt = true;
    if (print_full_kkt){
    std::cout << "KKT = np.array([\n";
    for (Index i = 0; i < full_kkt_matrix.m(); i++){
        std::cout << "\t[";
        for (Index j = 0; j < full_kkt_matrix.n(); j++){
            std::cout << std::setw(9) << std::setprecision(6) << full_kkt_matrix(i,j);
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
    for (Index i = 0; i < info.number_of_primal_variables; ++i){full_rhs(i) = rhs_x(i) + D_x(i)*x(i);}
    for (Index i = 0; i < info.number_of_eq_constraints; ++i){full_rhs(info.number_of_primal_variables + i) = rhs_g(i);}
    for (Index i = 0; i < info.number_of_slack_variables; ++i){
        full_rhs(info.number_of_primal_variables + info.offset_g_eq_slack + i) -= D_s(i) * mult(info.offset_g_eq_slack + i);
    }

    VecRealAllocated true_rhs(info.number_of_primal_variables + info.number_of_eq_constraints);
    true_rhs.block(info.number_of_primal_variables, 0) = rhs_x + D_x * x;
    true_rhs.block(info.number_of_eq_constraints, info.number_of_primal_variables) = rhs_g;
    true_rhs.block(info.number_of_slack_variables, info.offset_g_eq_slack + info.number_of_primal_variables) =
        true_rhs.block(info.number_of_slack_variables, info.offset_g_eq_slack + info.number_of_primal_variables) - D_s * mult.block(info.offset_g_eq_slack, info.offset_g_eq_slack);
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
    }
    
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
    std::cout << "Halfway through the tests" << std::endl;
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    }
    std::cout << "rhs_gg: " << rhs_gg << std::endl;
    std::cout << "grad:   " << grad << std::endl;
}

TEST_F(GeneralImplicitAugSystemSolverTest2, TestRandomSolve)
{
    /*
    Index ret = solver.value().solve(info.value(), jacobian.value(), hessian.value(), D_x.value(), D_s.value(), rhs_x.value(), rhs_g.value(), x.value(), mult.value());
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
   
    // Solution checking
    VecRealAllocated jac_x(info.value().number_of_eq_constraints);
    jacobian.value().apply_on_right(info.value(), x.value(), 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.value().number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + rhs_g.value() + jac_x;
    rhs_gg.block(info.value().number_of_slack_variables, info.value().offset_g_eq_slack) =
        rhs_gg.block(info.value().number_of_slack_variables, info.value().offset_g_eq_slack) -
        D_s.value() * mult.value().block(info.value().number_of_slack_variables, info.value().offset_g_eq_slack);
    VecRealAllocated grad(info.value().number_of_primal_variables);
    VecRealAllocated tmp(info.value().number_of_primal_variables);
    grad = 0;
    hessian.value().apply_on_right(info.value(), x.value(), 0.0, tmp, tmp);
    grad = grad + tmp + D_x.value() * x.value();
    jacobian.value().transpose_apply_on_right(info.value(), mult.value(), 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + rhs_x.value();
    for (Index i = 0; i < info.value().number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(rhs_gg(i), 0, 1e-5);
    }
    for (Index i = 0; i < info.value().number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(grad(i), 0, 1e-5);
    } 
    */   
}