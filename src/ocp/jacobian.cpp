//
// Copyright (c) Lander Vanroye, KU Leuven
//
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/common/exception.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/problem_info.hpp"
using namespace fatrop;

Jacobian<OcpType>::Jacobian(const ProblemDims &dims)
{
    // reserve memory for the Jacobian matrices
    BAbt.reserve(dims.K - 1);
    Gg_eqt.reserve(dims.K);
    Gg_ineqt.reserve(dims.K);
    // allocate memory for the Jacobian matrices
    for (int k = 0; k < dims.K - 1; ++k)
        BAbt.emplace_back(dims.number_of_states[k] + dims.number_of_controls[k] + 1,
                          dims.number_of_states[k + 1]);
    for (int k = 0; k < dims.K; ++k)
    {
        Gg_eqt.emplace_back(dims.number_of_states[k] + dims.number_of_controls[k] + 1,
                            dims.number_of_eq_constraints[k]);
    }
    for (int k = 0; k < dims.K; ++k)
        Gg_ineqt.emplace_back(dims.number_of_states[k] + dims.number_of_controls[k] + 1,
                              dims.number_of_ineq_constraints[k]);
};
void Jacobian<OcpType>::apply_on_right(const OcpInfo &info, const VecRealView &x, Scalar alpha,
                                       const VecRealView &y, VecRealView &out) const
{
    out = alpha * y;
    // dynamics constraints BA*ux - x_next
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_x_next = info.offsets_primal_x[k + 1];
        Index offset_dyn_eq = info.offsets_g_eq_dyn[k];
        // apply out[offs:offs+nx] =  BAbt.T @ x[offs:offs+nu+nx] - x_next[offs:offs+nx]
        gemv_t(nu + nx, nx_next, 1.0, BAbt[k], 0, 0, x, offset_ux, 1.0, out, offset_dyn_eq, out,
               offset_dyn_eq);
        axpy(nx_next, -1.0, x, offset_x_next, out, offset_dyn_eq, out, offset_dyn_eq);
    }
    // equality path constraints
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_eq = info.offsets_g_eq_path[k];
        // apply out[offs:offs+ng] =  Gg_eqt.T @ x[offs:offs+nu+nx]
        gemv_t(nu + nx, ng, 1.0, Gg_eqt[k], 0, 0, x, offset_ux, 1.0, out, offset_g_eq, out,
               offset_g_eq);
    }
    // slack equality path constraints Gg_ineqt @ x
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_ineq = info.offsets_g_eq_slack[k];
        // apply out[offs:offs+ng_ineq] =  Gg_ineqt.T @ x[offs:offs+nu+nx]
        gemv_t(nu + nx, ng_ineq, 1.0, Gg_ineqt[k], 0, 0, x, offset_ux, 1.0, out, offset_g_ineq, out,
               offset_g_ineq);
    }
};
void Jacobian<OcpType>::transpose_apply_on_right(const OcpInfo &info, const VecRealView &mult_eq,
                                                 Scalar alpha, const VecRealView &y,
                                                 VecRealView &out) const
{
    // set the output to zero, we will add the contributions
    // dynamics constraints'contributions [BA.T ; 0; -I] @ mult_eq
    out = alpha * y;
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index offs_ux = info.offsets_primal_u[k];
        Index offs_x_next = info.offsets_primal_x[k + 1];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_g_dyn = info.offsets_g_eq_dyn[k];
        // apply out[offs_ux:offs_ux + nu + nx] +=  BAbt @ mult_eq[offs_g_dyn:offs_g_dyn + nx_next]
        gemv_n(nu + nx, nx_next, 1.0, BAbt[k], 0, 0, mult_eq, offset_g_dyn, 1.0, out, offs_ux, out,
               offs_ux);
        // apply out[offs_ux:offs_ux + nu + nx] -= mult_eq[offs_f_dyn:offs_g_dyn + nx_next]
        axpy(nx_next, -1.0, mult_eq, offset_g_dyn, out, offs_x_next, out, offs_x_next);
    };
    // equality path constraints' contributions Gg_eqt @ mult_eq
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_eq = info.offsets_g_eq_path[k];
        // apply out[offs:offs+nu+nx] +=  Gg_eqt @ mult_eq[offs:offs+ng]
        gemv_n(nu + nx, ng, 1.0, Gg_eqt[k], 0, 0, mult_eq, offset_g_eq, 1.0, out, offset_ux, out,
               offset_ux);
    }
    // inequality path constraints' contributions Gg_ineqt @ mult_eq
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_ineq = info.offsets_g_eq_slack[k];
        // apply out[offs:offs+nu+nx] +=  Gg_ineqt @ mult_eq[offs:offs+ng_ineq]
        gemv_n(nu + nx, ng_ineq, 1.0, Gg_ineqt[k], 0, 0, mult_eq, offset_g_ineq, 1.0, out,
               offset_ux, out, offset_ux);
    }
}
void Jacobian<OcpType>::get_rhs(const OcpInfo &info, VecRealView &rhs) const
{
    // dynamics constraints' right-hand side
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
        // the rhs is the last row of the BAbt[k] matrix
        rowex(nx_next, 1.0, BAbt[k], nu + nx, 0, rhs, offset_eq_dyn);
    }
    // equality path constraints' right-hand side
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_eq_path = info.offsets_g_eq_path[k];
        // the rhs is the last row of the Gg_eqt[k] matrix
        rowex(ng, 1.0, Gg_eqt[k], nu + nx, 0, rhs, offset_eq_path);
    }
    // inequality path constraints' right-hand side
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_eq_ineq = info.offsets_g_eq_slack[k];
        // the rhs is the last row of the Gg_ineqt[k] matrix
        rowex(ng_ineq, 1.0, Gg_ineqt[k], nu + nx, 0, rhs, offset_eq_ineq);
    }
};
void Jacobian<OcpType>::set_rhs(const OcpInfo &info, const VecRealView &rhs)
{
    // dynamics constraints' right-hand side
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
        // the rhs is the last row of the BAbt[k] matrix
        rowin(nx_next, 1.0, rhs, offset_eq_dyn, BAbt[k], nu + nx, 0);
    }
    // equality path constraints' right-hand side
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_eq_path = info.offsets_g_eq_path[k];
        // the rhs is the last row of the Gg_eqt[k] matrix
        rowin(ng, 1.0, rhs, offset_eq_path, Gg_eqt[k], nu + nx, 0);
    }
    // inequality path constraints' right-hand side
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_eq_ineq = info.offsets_g_eq_slack[k];
        // the rhs is the last row of the Gg_ineqt[k] matrix
        rowin(ng_ineq, 1.0, rhs, offset_eq_ineq, Gg_ineqt[k], nu + nx, 0);
    }
};

// make printable
namespace fatrop
{
    std::ostream &operator<<(std::ostream &os, const Jacobian<OcpType> &jac)
    {
        os << "Jacobian<OcpType> object with horizon length " << jac.Gg_eqt.size();
        for (int k = 0; k < jac.Gg_eqt.size(); ++k)
        {
            os << "\n ----- Stage " << k << ": -----\n";
            os << "Gg_eq:\n" << transpose(jac.Gg_eqt[k]) << "\n";
            os << "Gg_ineq:\n" << transpose(jac.Gg_ineqt[k]) << "\n";
            if (k < jac.BAbt.size())
                os << "BAb:\n" << transpose(jac.BAbt[k]) << "\n";
        }
        return os;
    }
}


















//////////////////////////////
// ImplicitOcpType-specific //
//////////////////////////////

void Jacobian<ImplicitOcpType>::PreProcess(const ProblemInfo &info,
                                           VecRealView &f,
                                           VecRealView &g){
    // Make sure to store the current BAbt into BAbt_original before modifying BAbt matrices
    for (int k = 0; k < info.dims.K - 1; ++k){
        BAbt_original[k] = BAbt[k];

        // AugSystemSolver<OcpType> overwrites the entries corresponding to
        // the vector b. We do the same here
        // this is necessary to make sure we have the correct BAbt matrix
        // when calling apply_on_right
        rowin(info.dims.number_of_states[k + 1], 1.0, 
              g, info.offsets_g_eq_dyn[k], 
              BAbt_original[k], info.dims.number_of_states[k] + 
                info.dims.number_of_controls[k], 0);
    }   

    // Compute BAbt = BAbt_original * Jt^-1
    for (int k = 0; k < info.dims.K - 1; ++k){
        Index nx_next = info.dims.number_of_states[k + 1];
        Index nx = info.dims.number_of_states[k];
        Index nu = info.dims.number_of_controls[k];

        if (ASSUME_INVERSE_GIVEN){
            blasfeo_dgemm_nn(nx + nu + 1, nx_next, nx_next, -1.0, 
                             &BAbt[k].mat(), 0, 0, &Jt_inv[k].mat(), 0, 0, 0.0,
                             &BAbt[k].mat(), 0, 0, &BAbt[k].mat(), 0, 0);

            // apply transformation to rhs also, since AugSystemSolver<OcpType> 
            // overwrites the entries corresponding to the vector b in BAbt
            // by considering g (b <-- -J^-1 @ b)
            blasfeo_dgemv_t(nx_next, nx_next, -1.0,
                            &Jt_inv[k].mat(), 0, 0, 
                            &g.vec(), info.offsets_g_eq_dyn[k], 0.0,
                            &g.vec(), info.offsets_g_eq_dyn[k],
                            &g.vec(), info.offsets_g_eq_dyn[k]);

        } else {
            // (1) compute X1 = BAbt_original * Pr^T
            Pr[k].apply_on_cols(nx_next, &BAbt[k].mat());
            // (2) compute X2 * U^T = X1
            trsm_rlnn(nx+nu+1, nx_next, 1.0, Jt_LU[k], 0, 0, 
                    BAbt[k], 0, 0, BAbt[k], 0, 0);
            // (3) compute X3 * L^T = X2
            trsm_runu(nx+nu+1, nx_next, 1.0, Jt_LU[k], 0, 0, 
                    BAbt[k], 0, 0, BAbt[k], 0, 0);
            // (4) compute BAbt = X3 * Pl^T
            Pl[k].apply_on_cols(nx_next, &BAbt[k].mat());            
        }
    }    
}

void Jacobian<ImplicitOcpType>::ResetPreProcess(const ProblemInfo &info){
    for (int k = 0; k < info.dims.K - 1; ++k){
        BAbt[k] = BAbt_original[k];
    }   
}

void Jacobian<ImplicitOcpType>::PrepareInverseOfJ(const ProblemInfo &info){
    if (ASSUME_INVERSE_GIVEN)
    {
        return;
    }

    // compute LU-factorizations of matrices J such that we can later easily compute J^-1 * BAbt
    for (int k = 0; k < info.dims.K - 1; ++k)
    {
        int m = info.dims.number_of_states[k + 1];
        int n = info.dims.number_of_states[k + 1];
        int n_max = info.dims.number_of_states[k + 1];
        MatRealAllocated J(m, m);
        getr(info.dims.number_of_states[k + 1], info.dims.number_of_states[k + 1],
             Jt[k], 0, 0, J, 0, 0);

        // Check if J^T = Jt
        for (Index i = 0; i < m; ++i)
        {
            for (Index j = 0; j < n; ++j)
            {
                if (std::abs(J(i, j) - Jt[k](j, i)) > 1e-10)
                {
                    std::cout << "Jacobian<ImplicitOcpType>::PrepareInverseOfJ: "
                                            "Jt does not match J^T" << std::endl;
                }
            }
        }
        
        std::cout << "J: " << std::endl;
        std::cout << J << std::endl;
        std::cout << "Jt[" << k << "] = \n" << Jt[k] << std::endl;

        gecp(info.dims.number_of_states[k + 1], info.dims.number_of_states[k + 1],
             Jt[k], 0, 0, Jt_LU[k], 0, 0);
        lu_fact_transposed(m, n, n_max, rho[k], Jt_LU[k], Pl[k], Pr[k]);

        // sanity check: test Jt = (Pl^T * L * U * Pr)^T = Pr * U^T * L^T * Pl
        MatRealAllocated Lt(m, m);
        MatRealAllocated Ut(m, n);

        for (Index i = 0; i < m; ++i)
        {
            for (Index j = 0; j < n; ++j)
            {
                if (i < j)
                    Lt(i, j) = Jt_LU[k](i, j);
                else if (i == j)
                    Lt(i, j) = 1.0;
                else
                    Lt(i, j) = 0.0;

                if (i >= j)
                    Ut(i, j) = Jt_LU[k](i, j);
                else
                    Ut(i, j) = 0.0;
            }
        }

        MatRealAllocated L(m, m);
        MatRealAllocated U(m, n);
        for (Index i = 0; i < m; ++i)
        {
            for (Index j = 0; j < n; ++j)
            {
                if (i > j)
                    L(i, j) = Jt_LU[k](j, i);
                else if (i == j)
                    L(i, j) = 1.0;
                else
                    L(i, j) = 0.0;

                if (i <= j)
                    U(i, j) = Jt_LU[k](j, i);
                else
                    U(i, j) = 0.0;
            }
        }

        // Check if Lt  = L^T and Ut = U^T
        for (Index i = 0; i < m; ++i)
        {
            for (Index j = 0; j < n; ++j)
            {
                if (std::abs(Lt(i, j) - L(j, i)) > 1e-10 ||
                    std::abs(Ut(i, j) - U(j, i)) > 1e-10)
                {
                    std::cout << "Jacobian<ImplicitOcpType>::PrepareInverseOfJ: "
                                            "Lt or Ut does not match L^T or U^T" << std::endl;
                }
            }
        }

        MatRealAllocated temp1(m, m);
        // Compute U^T * L^T
        for (Index i = 0; i < m; ++i)
        {
            for (Index j = 0; j < n; ++j)
            {
                Scalar sum = 0;
                for (Index k = 0; k < m; ++k)
                {
                    sum += Ut(i, k) * Lt(k, j);
                }
                temp1(i, j) = sum;
            }
        }

        MatRealAllocated temp2(m, m);
        // Compute L * U
        for (Index i = 0; i < m; ++i)
        {
            for (Index j = 0; j < n; ++j)
            {
                Scalar sum = 0;
                for (Index k = 0; k < m; ++k)
                {
                    sum += L(i, k) * U(k, j);
                }
                temp2(i, j) = sum;
            }
        }

        // Check if Ut * Lt = (L * U)^T
        for (Index i = 0; i < m; ++i)
        {
            for (Index j = 0; j < n; ++j)
            {
                if (std::abs(temp1(i, j) - temp2(j, i)) > 1e-10)
                {
                    std::cout << "Jacobian<ImplicitOcpType>::PrepareInverseOfJ: "
                                            "Ut * Lt does not match L * U" << std::endl;
                }
            }
        }
        Pr[k].apply_on_rows(m, &temp2.mat());
        Pl[k].apply_on_cols(m, &temp2.mat());

        // Pl[k].apply_on_rows(m, &Jt[k].mat());
        // Pr[k].apply_on_cols(m, &Jt[k].mat());

        std::cout << "J: " << std::endl;
        std::cout << J << std::endl;
        // std::cout << "temp2: " << std::endl;
        // std::cout << temp2 << std::endl;

        // std::cout << "Jt[" << k << "] = \n" << Jt[k] << std::endl;
        // std::cout << "temp1 = Pl * L * U * Pr^T: \n" << temp1 << std::endl;

        std::cout << "Jt[" << k << "] = \n" << Jt[k] << std::endl;
        std::cout << "temp2 = Pr * L * U * Pl^T: \n" << temp2 << std::endl;

        // Compare Jt with Pl * L * U * Pr^T
        for (Index i = 0; i < m; ++i)
        {
            for (Index j = 0; j < m; ++j)
            {
                // if (std::abs(Jt[k](i, j) - temp1(i, j)) > 1e-4)
                // {
                //     std::cout << "Jacobian<ImplicitOcpType>::PrepareInverseOfJ: "
                //                             "Jt does not match Pl * L * U * Pr^T" << std::endl;
                // }

                if (std::abs(J(i, j) - temp2(i, j)) > 1e-4)
                {
                    std::cout << "Jacobian<ImplicitOcpType>::PrepareInverseOfJ: "
                                            "J does not match Pr * L * U * Pl^T" << std::endl;
                }

                // if (std::abs(Jt[k](i, j) - temp2(j, i)) > 1e-4)
                // {
                //     std::cout << "Jacobian<ImplicitOcpType>::PrepareInverseOfJ: "
                //                             "Jt or J does not match Pl * L * U * Pr^T or Pr * L * U * Pl^T" << std::endl;
                // }
            }
        }
    }
}

void Jacobian<ImplicitOcpType>::apply_on_right(const OcpInfo& info, const VecRealView& x, Scalar alpha, const VecRealView& y, VecRealView& out) const{
    if (print_debug){ std::cout << "Jacobian<ImplicitOcpType>::apply_on_right" << std::endl;}
    out = alpha * y;
    // dynamics constraints BA*ux - x_next
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_x_next = info.offsets_primal_x[k + 1];
        Index offset_dyn_eq = info.offsets_g_eq_dyn[k];
        // apply out[offs:offs+nx] =  BAbt.T @ x[offs:offs+nu+nx] + Jt @ x_next[offs:offs+nx]
        gemv_t(nu + nx, nx_next, 1.0, BAbt[k], 0, 0, x, offset_ux, 1.0, out, offset_dyn_eq, out,
               offset_dyn_eq);
        // axpy(nx_next, -1.0, x, offset_x_next, out, offset_dyn_eq, out, offset_dyn_eq);
        gemv_t(nx_next, nx_next, 1.0, Jt[k], 0, 0, x, offset_x_next, 1.0, out,
               offset_dyn_eq, out, offset_dyn_eq);
    }
    // equality path constraints
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_eq = info.offsets_g_eq_path[k];
        // apply out[offs:offs+ng] =  Gg_eqt.T @ x[offs:offs+nu+nx]
        gemv_t(nu + nx, ng, 1.0, Gg_eqt[k], 0, 0, x, offset_ux, 1.0, out, offset_g_eq, out,
               offset_g_eq);
    }
    // slack equality path constraints Gg_ineqt @ x
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_ineq = info.offsets_g_eq_slack[k];
        // apply out[offs:offs+ng_ineq] =  Gg_ineqt.T @ x[offs:offs+nu+nx]
        gemv_t(nu + nx, ng_ineq, 1.0, Gg_ineqt[k], 0, 0, x, offset_ux, 1.0, out, offset_g_ineq, out,
               offset_g_ineq);
    }
    if (print_debug){ std::cout << "Jacobian<ImplicitOcpType>::apply_on_right done" << std::endl;}
};

void Jacobian<ImplicitOcpType>::transpose_apply_on_right(const OcpInfo &info, const VecRealView &mult_eq,
                                                 Scalar alpha, const VecRealView &y,
                                                 VecRealView &out) const
{
    if (print_debug){ std::cout << "Jacobian<ImplicitOcpType>::transpose_apply_on_right" << std::endl;}
    // set the output to zero, we will add the contributions
    // dynamics constraints'contributions [BA.T ; 0; -I] @ mult_eq
    out = alpha * y;
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index offs_ux = info.offsets_primal_u[k];
        Index offs_x_next = info.offsets_primal_x[k + 1];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_g_dyn = info.offsets_g_eq_dyn[k];
        // apply out[offs_ux:offs_ux + nu + nx] +=  BAbt @ mult_eq[offs_g_dyn:offs_g_dyn + nx_next]
        gemv_n(nu + nx, nx_next, 1.0, BAbt[k], 0, 0, mult_eq, offset_g_dyn, 1.0, out, offs_ux, out,
               offs_ux);
        // axpy(nx_next, -1.0, mult_eq, offset_g_dyn, out, offs_x_next, out, offs_x_next);
        gemv_n(nx_next, nx_next, 1.0, Jt[k], 0, 0, mult_eq, offset_g_dyn, 1.0, out,
               offs_x_next, out, offs_x_next);
    };
    // equality path constraints' contributions Gg_eqt @ mult_eq
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_eq = info.offsets_g_eq_path[k];
        // apply out[offs:offs+nu+nx] +=  Gg_eqt @ mult_eq[offs:offs+ng]
        gemv_n(nu + nx, ng, 1.0, Gg_eqt[k], 0, 0, mult_eq, offset_g_eq, 1.0, out, offset_ux, out,
               offset_ux);
    }
    // inequality path constraints' contributions Gg_ineqt @ mult_eq
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_ineq = info.offsets_g_eq_slack[k];
        // apply out[offs:offs+nu+nx] +=  Gg_ineqt @ mult_eq[offs:offs+ng_ineq]
        gemv_n(nu + nx, ng_ineq, 1.0, Gg_ineqt[k], 0, 0, mult_eq, offset_g_ineq, 1.0, out,
               offset_ux, out, offset_ux);
    }
    if (print_debug){ std::cout << "Jacobian<ImplicitOcpType>::transpose_apply_on_right done" << std::endl;}
}