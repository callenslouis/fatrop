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
#include <fstream>
#include <filesystem>
#include <chrono>
#include <random>
#include <fstream>

using namespace fatrop;

class RandomBenchmarkTest : public ::testing::Test
{
// protected:
public:
    bool full_rank = true;
    bool constant_dimensions = true;

    int K;
    std::vector<Index> nx;
    std::vector<Index> r;
    std::vector<Index> nu;
    std::vector<Index> ng;
    std::vector<Index> ng_ineq;

    /// Explicit case ///
    std::optional<ProblemDims> dims_expl;
    std::optional<ProblemInfo> info_expl;
    std::optional<Jacobian<OcpType>> jacobian_expl;
    std::optional<Hessian<OcpType>> hessian_expl;
    std::optional<MatRealAllocated> full_matrix_jacobian_expl;
    std::optional<MatRealAllocated> full_matrix_hessian_expl;
    std::optional<VecRealAllocated> x_expl;
    std::optional<VecRealAllocated> mult_expl;
    std::optional<VecRealAllocated> rhs_x_expl;
    std::optional<VecRealAllocated> rhs_g_expl;
    std::optional<VecRealAllocated> D_x_expl;
    std::optional<VecRealAllocated> D_s_expl;
    std::optional<VecRealAllocated> D_eq_expl;
    std::optional<MatRealAllocated> full_kkt_matrix_expl;
    std::optional<AugSystemSolver<OcpType>> solver_expl;

    /// Implicit case ///
    std::optional<ProblemDims> dims_impl;
    std::optional<ProblemInfo> info_impl;
    std::optional<Jacobian<ImplicitOcpType>> jacobian_impl;
    std::optional<Hessian<ImplicitOcpType>> hessian_impl;
    std::optional<MatRealAllocated> full_matrix_jacobian_impl;
    std::optional<MatRealAllocated> full_matrix_hessian_impl;
    std::optional<VecRealAllocated> x_impl;
    std::optional<VecRealAllocated> mult_impl;
    std::optional<VecRealAllocated> rhs_x_impl;
    std::optional<VecRealAllocated> rhs_g_impl;
    std::optional<VecRealAllocated> D_x_impl;
    std::optional<VecRealAllocated> D_s_impl;
    std::optional<VecRealAllocated> D_eq_impl;
    std::optional<MatRealAllocated> full_kkt_matrix_impl;
    std::optional<AugSystemSolver<ImplicitOcpType>> solver_impl;

    /// Reformulated case ///
    std::optional<ProblemDims> dims_reform;
    std::optional<ProblemInfo> info_reform;
    std::optional<Jacobian<OcpType>> jacobian_reform;
    std::optional<Hessian<OcpType>> hessian_reform;
    std::optional<MatRealAllocated> full_matrix_jacobian_reform;
    std::optional<MatRealAllocated> full_matrix_hessian_reform;
    std::optional<VecRealAllocated> x_reform;
    std::optional<VecRealAllocated> mult_reform;
    std::optional<VecRealAllocated> rhs_x_reform;
    std::optional<VecRealAllocated> rhs_g_reform;
    std::optional<VecRealAllocated> D_x_reform;
    std::optional<VecRealAllocated> D_s_reform;
    std::optional<VecRealAllocated> D_eq_reform;
    std::optional<MatRealAllocated> full_kkt_matrix_reform;
    std::optional<AugSystemSolver<OcpType>> solver_reform;

    std::vector<int> RandomVector(int size, int min_val, int max_val)
    {
        std::vector<int> vec(size);
        if (size == 0){ return vec;}

        vec[0] = rand() % (max_val - min_val + 1) + min_val;
        for (int i = 0; i < size; ++i){
            if (constant_dimensions){
                vec[i] = vec[0];
            } else {
                vec[i] = rand() % (max_val - min_val + 1) + min_val;
            }
        }
        return vec;
    }

    void ClearOptionals(){
        solver_expl.reset();
        hessian_expl.reset();
        jacobian_expl.reset();
        info_expl.reset();
        dims_expl.reset();
        full_matrix_jacobian_expl.reset();
        full_matrix_hessian_expl.reset();
        x_expl.reset();
        mult_expl.reset();
        rhs_x_expl.reset();
        rhs_g_expl.reset();
        D_x_expl.reset();
        D_s_expl.reset();
        D_eq_expl.reset();
        full_kkt_matrix_expl.reset();

        solver_impl.reset();
        hessian_impl.reset();
        jacobian_impl.reset();
        info_impl.reset();
        dims_impl.reset();
        full_matrix_jacobian_impl.reset();
        full_matrix_hessian_impl.reset();
        x_impl.reset();
        mult_impl.reset();
        rhs_x_impl.reset();
        rhs_g_impl.reset();
        D_x_impl.reset();
        D_s_impl.reset();
        D_eq_impl.reset();
        full_kkt_matrix_impl.reset();

        solver_reform.reset();
        hessian_reform.reset();
        jacobian_reform.reset();
        info_reform.reset();
        dims_reform.reset();
        full_matrix_jacobian_reform.reset();
        full_matrix_hessian_reform.reset();
        x_reform.reset();
        mult_reform.reset();
        rhs_x_reform.reset();
        rhs_g_reform.reset();
        D_x_reform.reset();
        D_s_reform.reset();
        D_eq_reform.reset();
        full_kkt_matrix_reform.reset();
    }

    void GetRandomDimensions()
    {
        ClearOptionals();
        int max_val = 50;
        K = rand() % 4 + 2; // Random K between 2 and 21
        nx = RandomVector(K, 0, 1+0*max_val);
        if (full_rank){
            r = nx;
        } else {
            r = std::vector<Index>(K, 100);
            for (int k = 0; k < K; ++k){ 
                while (r[k] > nx[k]){ r[k] = rand() % (nx[k]+1);}
            }
        }
        nu = RandomVector(K, 0, max_val);
        ng = RandomVector(K, 0, 1+0*max_val);
        for (int k = 0; k < K; ++k){
            bool okay = false;
            while (!okay){
                int max_allowed_ng = nx[k] + nu[k];
                if (k < K-1){ max_allowed_ng -= (nx[k+1] - r[k+1]);}
                if (ng[k] <= max_allowed_ng){
                    okay = true;
                } else {
                    // randomize both the nb of constraints and the nb of controls
                    if (constant_dimensions){
                        ng = RandomVector(K, 0, max_val);
                        nu = RandomVector(K, 0, max_val);
                    } else {
                        ng[k] = rand() % (max_val + 1);
                        nu[k] = rand() % (max_val + 1);
                    }
                }
            }
        }
        ng_ineq = RandomVector(K, 0, 0*max_val);

        // std::cout << "K: " << K << std::endl;
        // std::cout << "nx:      "; for (auto val : nx){ std::cout << std::setw(4) << val << " ";} std::cout << std::endl;
        // std::cout << "r:       "; for (auto val : r){ std::cout << std::setw(4) << val << " ";} std::cout << std::endl;
        // std::cout << "nu:      "; for (auto val : nu){ std::cout << std::setw(4) << val << " ";} std::cout << std::endl;
        // std::cout << "ng:      "; for (auto val : ng){ std::cout << std::setw(4) << val << " ";} std::cout << std::endl;
        // std::cout << "ng_ineq: "; for (auto val : ng_ineq){ std::cout << std::setw(4) << val << " ";} std::cout << std::endl;

    }

    void AllocateSolvers(){
        AllocateExplicitSolver();
        AllocateImplicitSolver();
        AllocateReformulatedSolver();
        // std::cout << "KKT size expl:   " << full_kkt_matrix_expl.value().m() << " x " << full_kkt_matrix_expl.value().n() << std::endl;
        // std::cout << "KKT size impl:   " << full_kkt_matrix_impl.value().m() << " x " << full_kkt_matrix_impl.value().n() << std::endl;
        // std::cout << "KKT size reform: " << full_kkt_matrix_reform.value().m() << " x " << full_kkt_matrix_reform.value().n() << std::endl;
    }

    void AllocateExplicitSolver(){
        dims_expl.emplace(ProblemDims{K, nu, nx, ng, ng_ineq});
        info_expl.emplace(ProblemInfo(dims_expl.value()));
        jacobian_expl.emplace(Jacobian<OcpType>(dims_expl.value()));
        full_matrix_jacobian_expl =
            MatRealAllocated(info_expl->number_of_eq_constraints, info_expl->number_of_primal_variables);
        hessian_expl.emplace(Hessian<OcpType>(dims_expl.value()));
        full_matrix_hessian_expl =
            MatRealAllocated(info_expl->number_of_primal_variables, info_expl->number_of_primal_variables);
        x_expl = VecRealAllocated(info_expl->number_of_primal_variables);
        mult_expl = VecRealAllocated(info_expl->number_of_eq_constraints);
        rhs_x_expl = VecRealAllocated(info_expl->number_of_primal_variables);
        rhs_g_expl = VecRealAllocated(info_expl->number_of_eq_constraints);
        D_x_expl = VecRealAllocated(info_expl->number_of_primal_variables);
        D_s_expl = VecRealAllocated(info_expl->number_of_slack_variables);
        D_eq_expl = VecRealAllocated(info_expl->number_of_g_eq_path);
        full_kkt_matrix_expl =
            MatRealAllocated(info_expl->number_of_primal_variables + info_expl->number_of_eq_constraints,
                             info_expl->number_of_primal_variables + info_expl->number_of_eq_constraints);
        solver_expl.emplace(AugSystemSolver<OcpType>(info_expl.value()));
    }

    void AllocateImplicitSolver(){
        dims_impl.emplace(ProblemDims{K, nu, nx, ng, ng_ineq});
        info_impl.emplace(ProblemInfo(dims_impl.value()));
        jacobian_impl.emplace(Jacobian<ImplicitOcpType>(dims_impl.value()));
        full_matrix_jacobian_impl =
            MatRealAllocated(info_impl->number_of_eq_constraints, info_impl->number_of_primal_variables);
        hessian_impl.emplace(Hessian<ImplicitOcpType>(dims_impl.value()));
        full_matrix_hessian_impl =
            MatRealAllocated(info_impl->number_of_primal_variables, info_impl->number_of_primal_variables);
        x_impl = VecRealAllocated(info_impl->number_of_primal_variables);
        mult_impl = VecRealAllocated(info_impl->number_of_eq_constraints);
        rhs_x_impl = VecRealAllocated(info_impl->number_of_primal_variables);
        rhs_g_impl = VecRealAllocated(info_impl->number_of_eq_constraints);
        D_x_impl = VecRealAllocated(info_impl->number_of_primal_variables);
        D_s_impl = VecRealAllocated(info_impl->number_of_slack_variables);
        D_eq_impl = VecRealAllocated(info_impl->number_of_g_eq_path);
        full_kkt_matrix_impl =
            MatRealAllocated(info_impl->number_of_primal_variables + info_impl->number_of_eq_constraints,
                             info_impl->number_of_primal_variables + info_impl->number_of_eq_constraints);
        solver_impl.emplace(AugSystemSolver<ImplicitOcpType>(info_impl.value()));
    }

    void AllocateReformulatedSolver(){
        std::vector<Index> nu_reform = nu;
        std::vector<Index> ng_reform = ng;
        for (int k = 0; k < K-1; ++k){
            nu_reform[k] += nx[k+1];
            ng_reform[k] += nx[k+1];
        }

        dims_reform.emplace(ProblemDims{K, nu_reform, nx, ng_reform, ng_ineq});
        info_reform.emplace(ProblemInfo(dims_reform.value()));
        jacobian_reform.emplace(Jacobian<OcpType>(dims_reform.value()));
        full_matrix_jacobian_reform =
            MatRealAllocated(info_reform->number_of_eq_constraints, info_reform->number_of_primal_variables);
        hessian_reform.emplace(Hessian<OcpType>(dims_reform.value()));
        full_matrix_hessian_reform =
            MatRealAllocated(info_reform->number_of_primal_variables, info_reform->number_of_primal_variables);
        x_reform = VecRealAllocated(info_reform->number_of_primal_variables);
        mult_reform = VecRealAllocated(info_reform->number_of_eq_constraints);
        rhs_x_reform = VecRealAllocated(info_reform->number_of_primal_variables);
        rhs_g_reform = VecRealAllocated(info_reform->number_of_eq_constraints);
        D_x_reform = VecRealAllocated(info_reform->number_of_primal_variables);
        D_s_reform = VecRealAllocated(info_reform->number_of_slack_variables);
        D_eq_reform = VecRealAllocated(info_reform->number_of_g_eq_path);
        full_kkt_matrix_reform =
            MatRealAllocated(info_reform->number_of_primal_variables + info_reform->number_of_eq_constraints,
                             info_reform->number_of_primal_variables + info_reform->number_of_eq_constraints);
        solver_reform.emplace(AugSystemSolver<OcpType>(info_reform.value()));
    }

    void FillExplicitSolver(){
        x_expl = 0;
        full_matrix_jacobian_expl.value() = 0.;
        full_matrix_hessian_expl.value() = 0.;

        for (Index k = 0; k < info_expl.value().dims.K; ++k)
        {
            Index nu = info_expl.value().dims.number_of_controls[k];
            Index nx = info_expl.value().dims.number_of_states[k];
            Index offs_eq_dyn = info_expl.value().offsets_g_eq_dyn[k];
            Index offs_ux = info_expl.value().offsets_primal_u[k];
            Index offset_g_eq = info_expl.value().offsets_g_eq_path[k];
            Index offset_g_ineq = info_expl.value().offsets_g_eq_slack[k];
            Index ng = info_expl.value().dims.number_of_eq_constraints[k];
            Index ng_ineq = info_expl.value().dims.number_of_ineq_constraints[k];
            if (k < info_expl.value().dims.K - 1)
            {
                Index nx_next = info_expl.value().dims.number_of_states[k + 1];
                Index offs_x_next = info_expl.value().offsets_primal_x[k + 1];
                jacobian_expl.value().BAbt[k].block(nu + nx, nx_next, 0, 0) =
                    ::test::random_matrix(nu + nx, nx_next);
                full_matrix_jacobian_expl.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian_expl.value().BAbt[k].block(nu + nx, nx_next, 0, 0));
            }
            jacobian_expl.value().Gg_eqt[k].block(nu + nx, info_expl.value().dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_expl.value().dims.number_of_eq_constraints[k]);
            full_matrix_jacobian_expl.value().block(ng, nu + nx, offset_g_eq, offs_ux) =
                transpose(jacobian_expl.value().Gg_eqt[k].block(nu + nx, ng, 0, 0));

            jacobian_expl.value().Gg_ineqt[k].block(nu + nx, info_expl.value().dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_expl.value().dims.number_of_ineq_constraints[k]);
            full_matrix_jacobian_expl.value().block(ng_ineq, nu + nx, offset_g_ineq, offs_ux) =
                transpose(jacobian_expl.value().Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));

            hessian_expl.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
            full_matrix_hessian_expl.value().block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian_expl.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        
        // set up the full KKT matrix
        full_kkt_matrix_expl.value().block(info_expl.value().number_of_primal_variables, info_expl.value().number_of_primal_variables, 0,
                              0) = full_matrix_hessian_expl.value();
        full_kkt_matrix_expl.value().block(info_expl.value().number_of_primal_variables, info_expl.value().number_of_eq_constraints, 0,
                              info_expl.value().number_of_primal_variables) = transpose(full_matrix_jacobian_expl.value());
        full_kkt_matrix_expl.value().block(info_expl.value().number_of_eq_constraints, info_expl.value().number_of_primal_variables,
                              info_expl.value().number_of_primal_variables, 0) = full_matrix_jacobian_expl.value();

        // fill the x vector with random values
        for (Index i = 0; i < info_expl.value().number_of_primal_variables; ++i){
            rhs_x_expl.value()(i) = 1.0 * i; D_x_expl.value()(i) = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info_expl.value().number_of_eq_constraints; ++i){
            rhs_g_expl.value()(i) = 1.0 * i;
        }
        for (Index i = 0; i < info_expl.value().number_of_g_eq_path; ++i){
            D_eq_expl.value()(i) = 10.0 * (i + 1);
        }
        for (Index i = 0; i < info_expl.value().number_of_slack_variables; ++i){
            D_s_expl.value()(i) =  1.0 + 0*10.0 * (i + 0.1);
        }
    }

    void FillImplicitSolver(){
        x_impl = 0;
        full_matrix_jacobian_impl.value() = 0.;
        full_matrix_hessian_impl.value() = 0.;

        for (Index k = 0; k < info_impl.value().dims.K; ++k)
        {
            Index nu = info_impl.value().dims.number_of_controls[k];
            Index nx = info_impl.value().dims.number_of_states[k];
            Index offs_eq_dyn = info_impl.value().offsets_g_eq_dyn[k];
            Index offs_ux = info_impl.value().offsets_primal_u[k];
            Index offset_g_eq = info_impl.value().offsets_g_eq_path[k];
            Index offset_g_ineq = info_impl.value().offsets_g_eq_slack[k];
            Index ng = info_impl.value().dims.number_of_eq_constraints[k];
            Index ng_ineq = info_impl.value().dims.number_of_ineq_constraints[k];
            if (k < info_impl.value().dims.K - 1)
            {
                Index nx_next = info_impl.value().dims.number_of_states[k + 1];
                Index offs_x_next = info_impl.value().offsets_primal_x[k + 1];
                jacobian_impl.value().BAbt[k].block(nu + nx, nx_next, 0, 0) =
                    ::test::random_matrix(nu + nx, nx_next);
                full_matrix_jacobian_impl.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian_impl.value().BAbt[k].block(nu + nx, nx_next, 0, 0));

                jacobian_impl.value().Jt[k].block(nx_next, nx_next, 0, 0) =
                ::test::random_degenerate_matrix(nx_next, r[k+1]);
                full_matrix_jacobian_impl.value().block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                    transpose(jacobian_impl.value().Jt[k]);

                hessian_impl.value().FuFx[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::random_matrix(nx + nu, nx_next);
                full_matrix_hessian_impl.value().block(nx_next, nu + nx, offs_x_next, offs_ux) = 
                    transpose(hessian_impl.value().FuFx[k]);
                full_matrix_hessian_impl.value().block(nu + nx, nx_next, offs_ux, offs_x_next) =
                    hessian_impl.value().FuFx[k];
            }
            jacobian_impl.value().Gg_eqt[k].block(nu + nx, info_impl.value().dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_impl.value().dims.number_of_eq_constraints[k]);
            full_matrix_jacobian_impl.value().block(ng, nu + nx, offset_g_eq, offs_ux) =
                transpose(jacobian_impl.value().Gg_eqt[k].block(nu + nx, ng, 0, 0));

            jacobian_impl.value().Gg_ineqt[k].block(nu + nx, info_impl.value().dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_impl.value().dims.number_of_ineq_constraints[k]);
            full_matrix_jacobian_impl.value().block(ng_ineq, nu + nx, offset_g_ineq, offs_ux) =
                transpose(jacobian_impl.value().Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));

            hessian_impl.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
            full_matrix_hessian_impl.value().block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian_impl.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        
        // set up the full KKT matrix
        full_kkt_matrix_impl.value().block(info_impl.value().number_of_primal_variables, info_impl.value().number_of_primal_variables, 0,
                              0) = full_matrix_hessian_impl.value();
        full_kkt_matrix_impl.value().block(info_impl.value().number_of_primal_variables, info_impl.value().number_of_eq_constraints, 0,
                              info_impl.value().number_of_primal_variables) = transpose(full_matrix_jacobian_impl.value());
        full_kkt_matrix_impl.value().block(info_impl.value().number_of_eq_constraints, info_impl.value().number_of_primal_variables,
                              info_impl.value().number_of_primal_variables, 0) = full_matrix_jacobian_impl.value();

        // fill the x vector with random values
        for (Index i = 0; i < info_impl.value().number_of_primal_variables; ++i){
            rhs_x_impl.value()(i) = 1.0 * i; D_x_impl.value()(i) = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info_impl.value().number_of_eq_constraints; ++i){
            rhs_g_impl.value()(i) = 1.0 * i;
        }
        for (Index i = 0; i < info_impl.value().number_of_g_eq_path; ++i){
            D_eq_impl.value()(i) = 10.0 * (i + 1);
        }
        for (Index i = 0; i < info_impl.value().number_of_slack_variables; ++i){
            D_s_impl.value()(i) =  1.0 + 0*10.0 * (i + 0.1);
        }
    }

    void FillReformulatedSolver(){
        x_reform = 0;
        full_matrix_jacobian_reform.value() = 0.;
        full_matrix_hessian_reform.value() = 0.;

        for (Index k = 0; k < info_reform.value().dims.K; ++k)
        {
            Index nu = info_reform.value().dims.number_of_controls[k];
            Index nx = info_reform.value().dims.number_of_states[k];
            Index offs_eq_dyn = info_reform.value().offsets_g_eq_dyn[k];
            Index offs_ux = info_reform.value().offsets_primal_u[k];
            Index offset_g_eq = info_reform.value().offsets_g_eq_path[k];
            Index offset_g_ineq = info_reform.value().offsets_g_eq_slack[k];
            Index ng = info_reform.value().dims.number_of_eq_constraints[k];
            Index ng_ineq = info_reform.value().dims.number_of_ineq_constraints[k];
            Index nu_true = (k < info_reform.value().dims.K - 1) ? nu - info_reform.value().dims.number_of_states[k+1] : nu;
            Index ng_true = (k < info_reform.value().dims.K - 1) ? ng - info_reform.value().dims.number_of_states[k+1] : ng;
            if (k < info_reform.value().dims.K - 1)
            {
                Index nx_next = info_reform.value().dims.number_of_states[k + 1];
                Index offs_x_next = info_reform.value().offsets_primal_x[k + 1];
                jacobian_reform.value().BAbt[k].block(nx_next, nx_next, nu_true, 0) =
                    ::test::identity_matrix(nx_next, -1);
                full_matrix_jacobian_reform.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                    transpose(jacobian_reform.value().BAbt[k].block(nu + nx, nx_next, 0, 0));
            }
            jacobian_reform.value().Gg_eqt[k].block(nu + nx, info_reform.value().dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_reform.value().dims.number_of_eq_constraints[k]);
            jacobian_reform.value().Gg_eqt[k].block(nu - nu_true, ng_true, nu_true, 0) =
                ::test::empty_matrix(nu - nu_true, ng_true);
            full_matrix_jacobian_reform.value().block(ng, nu + nx, offset_g_eq, offs_ux) =
                transpose(jacobian_reform.value().Gg_eqt[k].block(nu + nx, ng, 0, 0));

            jacobian_reform.value().Gg_ineqt[k].block(nu + nx, info_reform.value().dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_reform.value().dims.number_of_ineq_constraints[k]);
            jacobian_reform.value().Gg_ineqt[k].block(nu - nu_true, ng_ineq, nu_true, 0) =
                ::test::empty_matrix(nu - nu_true, ng_ineq);
            full_matrix_jacobian_reform.value().block(ng_ineq, nu + nx, offset_g_ineq, offs_ux) =
                transpose(jacobian_reform.value().Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));

            hessian_reform.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
            // hessian_reform.value().RSQrqt[k].block(nu - nu_true, nu + nx, nu_true, 0) = ::test::empty_matrix(nu_true, nu + nx);
            // hessian_reform.value().RSQrqt[k].block(nu + nx, nu - nu_true, 0, nu_true) = ::test::empty_matrix(nu + nx, nu_true);
            full_matrix_hessian_reform.value().block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian_reform.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        
        // set up the full KKT matrix
        full_kkt_matrix_reform.value().block(info_reform.value().number_of_primal_variables, info_reform.value().number_of_primal_variables, 0,
                              0) = full_matrix_hessian_reform.value();
        full_kkt_matrix_reform.value().block(info_reform.value().number_of_primal_variables, info_reform.value().number_of_eq_constraints, 0,
                              info_reform.value().number_of_primal_variables) = transpose(full_matrix_jacobian_reform.value());
        full_kkt_matrix_reform.value().block(info_reform.value().number_of_eq_constraints, info_reform.value().number_of_primal_variables,
                              info_reform.value().number_of_primal_variables, 0) = full_matrix_jacobian_reform.value();

        // fill the x vector with random values
        for (Index i = 0; i < info_reform.value().number_of_primal_variables; ++i){
            rhs_x_reform.value()(i) = 1.0 * i; D_x_reform.value()(i) = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info_reform.value().number_of_eq_constraints; ++i){
            rhs_g_reform.value()(i) = 1.0 * i;
        }
        for (Index i = 0; i < info_reform.value().number_of_g_eq_path; ++i){
            D_eq_reform.value()(i) = 10.0 * (i + 1);
        }
        for (Index i = 0; i < info_reform.value().number_of_slack_variables; ++i){
            D_s_reform.value()(i) =  1.0 + 0*10.0 * (i + 0.1);
        }
    }

    void Randomize(){
        GetRandomDimensions();
        AllocateSolvers();
        FillExplicitSolver();
        FillImplicitSolver();
        FillReformulatedSolver();

        solver_impl.value().set_performance_mode(true);
    }

    void SetUp()
    {
        Randomize();
    };
};


TEST_F(RandomBenchmarkTest, Test)
{
    int nb_runs = 10000;
    int nb_runs_completed = 0;

    long int total_ns_expl = 0; long int total_ns_impl = 0; long int total_ns_reform = 0;
    long int total_ns_impl_solve = 0; long int total_ns_impl_preprocess = 0; long int total_ns_impl_postprocess = 0;
    long int ns_expl, ns_impl, ns_impl_solve, ns_impl_preprocess, ns_impl_postprocess, ns_reform;

    long int total_pre_jac = 0;
    long int total_pre_hess = 0;
    long int total_pre_reg = 0;
    long int total_pre_decomp = 0;
    long int total_pre_info = 0;
    long int total_pre_rhs = 0;
    long int ns_pre_jac, ns_pre_hess, ns_pre_reg, ns_pre_decomp, ns_pre_info, ns_pre_rhs;

    long int total_decomp_copies = 0;
    long int total_decomp_decomp = 0;
    long int total_decomp_scale1 = 0;
    long int total_decomp_scale2 = 0;
    long int total_decomp_permutation = 0;
    long int total_decomp_store = 0;
    long int ns_decomp_copies, ns_decomp_decomp, ns_decomp_scale1, ns_decomp_scale2, ns_decomp_permutation, ns_decomp_store;

    long int total_lu_reform = 0;
    long int ns_lu_reform = 0;
    long int total_lu_impl = 0;
    long int ns_lu_impl = 0;

    long int total_ns_impl_solve_inner = 0;

    std::ofstream f("random_benchmark_results.csv");
    f << "K,nu,nx,r,ng,ng_ineq,t_expl,t_impl,t_impl_pre,t_impl_solve,t_impl_post,t_reform,lu_impl,lu_reform,impl_decomp\n";
    
    int nb_consecutive_failures = 0;
    while (nb_runs_completed < nb_runs){
        // std::cout << "Run " << nb_runs_completed + 1 << "/" << nb_runs << std::endl;
        // overwrite current status
        std::cout << "progress: " << std::setw(3) << std::setprecision(2) << 100.0 * nb_runs_completed / nb_runs << " %";
        if (nb_consecutive_failures > 0){ std::cout << " (+" << nb_consecutive_failures << " fails)"; }
        std::cout << "\r" << std::flush;
        Randomize();

        // explicit //
        auto start = std::chrono::steady_clock::now();
        Index ret_expl = solver_expl.value().solve(info_expl.value(), jacobian_expl.value(), hessian_expl.value(), D_x_expl.value(), D_s_expl.value(), rhs_x_expl.value(), rhs_g_expl.value(), x_expl.value(), mult_expl.value());
        auto stop = std::chrono::steady_clock::now();
        ns_expl = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

        // reformulated //
        start = std::chrono::steady_clock::now();
        Index ret_reform = solver_reform.value().solve(info_reform.value(), jacobian_reform.value(), hessian_reform.value(), D_x_reform.value(), D_s_reform.value(), rhs_x_reform.value(), rhs_g_reform.value(), x_reform.value(), mult_reform.value());
        stop = std::chrono::steady_clock::now();
        ns_reform = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

        // implicit //
        start = std::chrono::steady_clock::now();
        Index ret_impl = 0;
        try{
        Index ret_impl = solver_impl.value().solve(info_impl.value(), jacobian_impl.value(), hessian_impl.value(), D_x_impl.value(), D_s_impl.value(), rhs_x_impl.value(), rhs_g_impl.value(), x_impl.value(), mult_impl.value());
        } catch (std::exception &e){
            std::cout << "Exception caught during solve!" << std::endl;
            nb_consecutive_failures++;
            continue;
        }
        stop = std::chrono::steady_clock::now();
        ns_impl = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        ns_impl_preprocess = solver_impl.value().duration_preprocess.count();
        ns_impl_solve = solver_impl.value().duration_solve.count();
        ns_impl_postprocess = solver_impl.value().duration_postprocess.count();
        ns_pre_jac = solver_impl.value().duration_preprocess_jac.count();
        ns_pre_hess = solver_impl.value().duration_preprocess_hess.count();
        ns_pre_reg = solver_impl.value().duration_preprocess_regularization.count();
        ns_pre_decomp = solver_impl.value().duration_preprocess_decomposition.count();
        ns_pre_info = solver_impl.value().duration_preprocess_info.count();
        ns_pre_rhs = solver_impl.value().duration_preprocess_modify_rhs.count();

        ns_decomp_copies = solver_impl.value().duration_decomp_copies.count();
        ns_decomp_decomp = solver_impl.value().duration_decomp_decomp.count();
        ns_decomp_scale1 = solver_impl.value().duration_decomp_scale1.count();
        ns_decomp_scale2 = solver_impl.value().duration_decomp_scale2.count();
        ns_decomp_permutation = solver_impl.value().duration_decomp_permutation.count();
        ns_decomp_store = solver_impl.value().duration_decomp_store.count();

        if (ret_expl != LinsolReturnFlag::SUCCESS || 
                ret_reform != LinsolReturnFlag::SUCCESS || 
                ret_impl != LinsolReturnFlag::SUCCESS){
            nb_consecutive_failures++;
            continue;
        }
        total_ns_expl += ns_expl;
        total_ns_reform += ns_reform;
        total_ns_impl += ns_impl;
        total_ns_impl_preprocess += ns_impl_preprocess;
        total_ns_impl_solve += ns_impl_solve;
        total_ns_impl_postprocess += ns_impl_postprocess;
        total_ns_impl_solve_inner += solver_impl.value().duration_inner_solve.count();

        total_pre_jac += ns_pre_jac;
        total_pre_hess += ns_pre_hess;
        total_pre_reg += ns_pre_reg;
        total_pre_decomp += ns_pre_decomp;
        total_pre_info += ns_pre_info;
        total_pre_rhs += ns_pre_rhs;

        total_decomp_copies += ns_decomp_copies;
        total_decomp_decomp += ns_decomp_decomp;
        total_decomp_scale1 += ns_decomp_scale1;
        total_decomp_scale2 += ns_decomp_scale2;
        total_decomp_permutation += ns_decomp_permutation;
        total_decomp_store += ns_decomp_store;

        ns_lu_impl = solver_impl.value().duration_lu_factorization.count() + ns_decomp_decomp;
        ns_lu_reform = solver_reform.value().duration_lu_factorization.count();
        total_lu_impl += ns_lu_impl;
        total_lu_reform += ns_lu_reform;

        nb_runs_completed++;
        nb_consecutive_failures = 0;

        f << K << "," << nu[0] << "," << nx[0] << "," << r[0] << "," << ng[0] << "," << ng_ineq[0];
        f << "," << ns_expl << "," << ns_impl << "," << ns_impl_preprocess << "," << ns_impl_solve << "," << ns_impl_postprocess << "," << ns_reform << "," << ns_lu_impl << "," << ns_lu_reform << "," << ns_decomp_decomp << "\n";
        
    }

    std::cout << "Average time explicit:     " << total_ns_expl / nb_runs_completed << " ns" << std::endl;
    std::cout << "Average time reformulated: " << total_ns_reform / nb_runs_completed << " ns" << std::endl;
    std::cout << "Average time implicit:     " << total_ns_impl / nb_runs_completed << " ns (";
    std::cout << total_ns_impl_preprocess / nb_runs_completed << " - ";
    std::cout << total_ns_impl_solve / nb_runs_completed << " - ";
    std::cout << total_ns_impl_postprocess / nb_runs_completed << ")" << std::endl;
    std::cout << "Time spent in LU factorization: " << std::endl;
    std::cout << "  - Implicit:     " << std::setw(3) << std::setprecision(3) << 100.0 * total_lu_impl / total_ns_impl << " % (" << total_lu_impl << ")" << std::endl;
    std::cout << "  - Reformulated: " << std::setw(3) << std::setprecision(3) << 100.0 * total_lu_reform / total_ns_reform << " % (" << total_lu_reform << ")" << std::endl;
    std::cout << "Average time implicit preprocess breakdown: " << std::endl;
    std::cout << "  - Jacobian:       " << std::setw(3) << std::setprecision(3) << 100.0 * total_pre_jac / total_ns_impl_preprocess << " %" << std::endl;
    std::cout << "  - Hessian:        " << std::setw(3) << std::setprecision(3) << 100.0 * total_pre_hess / total_ns_impl_preprocess << " %" << std::endl;
    std::cout << "  - Regularization: " << std::setw(3) << std::setprecision(3) << 100.0 * total_pre_reg / total_ns_impl_preprocess << " %" << std::endl;
    std::cout << "  - Decomposition:  " << std::setw(3) << std::setprecision(3) << 100.0 * total_pre_decomp / total_ns_impl_preprocess << " %" << std::endl;
    std::cout << "  - Info:           " << std::setw(3) << std::setprecision(3) << 100.0 * total_pre_info / total_ns_impl_preprocess << " %" << std::endl;
    std::cout << "  - RHS:            " << std::setw(3) << std::setprecision(3) << 100.0 * total_pre_rhs / total_ns_impl_preprocess << " %" << std::endl;

    std::cout << "decomposition breakdown: " << std::endl;
    std::cout << "  - Copies:         " << std::setw(3) << std::setprecision(3) << 100.0 * total_decomp_copies / total_pre_decomp << " %" << std::endl;
    std::cout << "  - Decomposition:  " << std::setw(3) << std::setprecision(3) << 100.0 * total_decomp_decomp / total_pre_decomp << " %" << std::endl;
    std::cout << "  - Scaling1:       " << std::setw(3) << std::setprecision(3) << 100.0 * total_decomp_scale1 / total_pre_decomp << " %" << std::endl;
    std::cout << "  - Scaling2:       " << std::setw(3) << std::setprecision(3) << 100.0 * total_decomp_scale2 / total_pre_decomp << " %" << std::endl;
    std::cout << "  - Permutation:    " << std::setw(3) << std::setprecision(3) << 100.0 * total_decomp_permutation / total_pre_decomp << " %" << std::endl;
    std::cout << "  - Store:          " << std::setw(3) << std::setprecision(3) << 100.0 * total_decomp_store / total_pre_decomp << " %" << std::endl;


    std::cout << "implicit solve modifications breakdown: (" << total_ns_impl_solve_inner / nb_runs << ")" << std::endl;
    // duration_lu_factorization, duration_RSQrqt_copy, duration_FuFx_addition, duration_GuGx_addition, duration_GuGx_hat_addition, duration_ukb_tilde_addition, duration_lambdatilde_addition, duration_FuFx_addition_forward
    std::cout << "  - LU factorization:      " << std::setw(3) << std::setprecision(3) << 100.0 * solver_impl.value().duration_lu_factorization.count() / ns_impl_solve << " % (" << solver_impl.value().duration_lu_factorization.count() << ")" << std::endl;
    std::cout << "  - RSQrqt copy:           " << std::setw(3) << std::setprecision(3) << 100.0 * solver_impl.value().duration_RSQrqt_copy.count() / ns_impl_solve << " % (" << solver_impl.value().duration_RSQrqt_copy.count() << ")" << std::endl;
    std::cout << "  - FuFx addition:         " << std::setw(3) << std::setprecision(3) << 100.0 * solver_impl.value().duration_FuFx_addition.count() / ns_impl_solve << " % (" << solver_impl.value().duration_FuFx_addition.count() << ")" << std::endl;
    std::cout << "  - GuGx addition:         " << std::setw(3) << std::setprecision(3) << 100.0 * solver_impl.value().duration_GuGx_addition.count() / ns_impl_solve << " % (" << solver_impl.value().duration_GuGx_addition.count() << ")" << std::endl;
    std::cout << "  - GuGx hat addition:     " << std::setw(3) << std::setprecision(3) << 100.0 * solver_impl.value().duration_GuGx_hat_addition.count() / ns_impl_solve << " % (" << solver_impl.value().duration_GuGx_hat_addition.count() << ")" << std::endl;
    std::cout << "  - ukb_tilde addition:    " << std::setw(3) << std::setprecision(3) << 100.0 * solver_impl.value().duration_ukb_tilde_addition.count() / ns_impl_solve << " % (" << solver_impl.value().duration_ukb_tilde_addition.count() << ")" << std::endl;
    std::cout << "  - lambdatilde addition:  " << std::setw(3) << std::setprecision(3) << 100.0 * solver_impl.value().duration_lambdatilde_addition.count() / ns_impl_solve << " % (" << solver_impl.value().duration_lambdatilde_addition.count() << ")" << std::endl;
    std::cout << "  - FuFx addition forward: " << std::setw(3) << std::setprecision(3) << 100.0 * solver_impl.value().duration_FuFx_addition_forward.count() / ns_impl_solve << " % (" << solver_impl.value().duration_FuFx_addition_forward.count() << ")" << std::endl;

}