//
// Copyright (c) Lander Vanroye, KU Leuven
//
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/common/exception.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/problem_info.hpp"

#include <chrono>
#include <map>
#include <utility>

using namespace fatrop;

Hessian<OcpType>::Hessian(const ProblemDims &dims)
{
    // reserve memory for the Jacobian matrices
    RSQrqt.reserve(dims.K);
    // allocate memory for the Jacobian matrices
    for (Index k = 0; k < dims.K; ++k)
        RSQrqt.emplace_back(dims.number_of_states[k] + dims.number_of_controls[k] + 1,
                            dims.number_of_states[k] + dims.number_of_controls[k]);
};
void Hessian<OcpType>::apply_on_right(const OcpInfo &info, const VecRealView &x, Scalar alpha,
                                      const VecRealView &y, VecRealView &out) const
{
    for (Index k = 0; k < info.dims.K; ++k)
    {
        // get the dimensions of the Hessian matrix
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        // get the offsets of the current variables in x, and out
        Index offset_ux = info.offsets_primal_u[k];
        // apply out[offs:offs+nu+nx] =  RSQ @ x[offs:offs+nu+nx]
        gemv_t(nu + nx, nu + nx, 1.0, RSQrqt[k], 0, 0, x, offset_ux, alpha, y, offset_ux, out,
               offset_ux);
    }
};
void Hessian<OcpType>::get_rhs(const OcpInfo &info, VecRealView &out) const
{
    for (Index k = 0; k < info.dims.K; ++k)
    {
        // get the dimensions of the Hessian matrix
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        // get the offsets of the current variables in x, and out
        Index offset_ux = info.offsets_primal_u[k];
        // the rhs is the last row of the RSQrqt[k] matrix
        rowex(nu + nx, 1.0, RSQrqt[k], nu + nx, 0, out, offset_ux);
    }
};
void Hessian<OcpType>::set_rhs(const OcpInfo &info, const VecRealView &in)
{
    for (Index k = 0; k < info.dims.K; ++k)
    {
        // get the dimensions of the Hessian matrix
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        // get the offsets of the current variables in x, and out
        Index offset_ux = info.offsets_primal_u[k];
        // the rhs is the last row of the RSQrqt[k] matrix
        rowin(nu + nx, 1.0, in, offset_ux, RSQrqt[k], nu + nx, 0);
    }
};
void Hessian<OcpType>::set_zero()
{
    for (auto &RSQ : RSQrqt)
        gese(RSQ.m(), RSQ.n(), 0.0, RSQ, 0, 0);
}
// make printable
namespace fatrop
{

    std::ostream &operator<<(std::ostream &os, const Hessian<OcpType> &hess)
    {
        os << "Hessian<OcpType> object with horizon length " << hess.RSQrqt.size();
        for (int k = 0; k < hess.RSQrqt.size(); ++k)
        {
            os << "\n ----- Stage " << k << ": -----\n";
            os << "RSQrq:\n" << transpose(hess.RSQrqt[k]) << "\n";
        }
        return os;
    }
}






//////////////////////////////
// ImplicitOcpType-specific //
//////////////////////////////

void Hessian<ImplicitOcpType>::PreProcess(const ProblemInfo &info, 
                                          Jacobian<ImplicitOcpType> &jacobian,
                                          VecRealView &f,
                                          VecRealView &g)
{
    // GENERAL CASE
    for (int k = 0; k < info.dims.K; ++k){
        // consider the right-hand-side for the vector [r; q] (this is also 
        // what AugSystemSolver<OcpType> does in its preprocess step)
        rowin(info.dims.number_of_states[k] + info.dims.number_of_controls[k],
              1.0, f, info.offsets_primal_u[k], RSQrqt[k],
              info.dims.number_of_states[k] + info.dims.number_of_controls[k], 0); 

        // RSQrqt_original[k] = RSQrqt[k];
        gecp(info.dims.number_of_states[k] + info.dims.number_of_controls[k] + 1,
             info.dims.number_of_states[k] + info.dims.number_of_controls[k], RSQrqt[k], 0, 0,
             RSQrqt_original[k], 0, 0);
    }
    // FuFx and GuGx are only modified when J is rank deficient, AugSystemSolver<ImplicitOcpType>
    // stores (and restores) them only then
}

void Hessian<ImplicitOcpType>::ResetPreProcess(const ProblemInfo &info, 
                                               const Jacobian<ImplicitOcpType> &jacobian)
{
    // PreProcess copied the original RSQrqt into RSQrqt_original: swap the buffers instead of copying them back
    // (RSQrqt_original then holds the preprocessed matrices, it is overwritten by the next PreProcess)
    (void)info; (void)jacobian;
    std::swap(RSQrqt, RSQrqt_original);
    // FuFx and GuGx are restored by AugSystemSolver<ImplicitOcpType> if they were modified
}

void Hessian<ImplicitOcpType>::apply_on_right(const OcpInfo& info, 
                                              const VecRealView& x, 
                                              Scalar alpha, 
                                              const VecRealView& y, 
                                              VecRealView& out) const {
    if (print_debug){ std::cout << "Hessian<ImplicitOcpType>::apply_on_right" << std::endl;}
    for (Index k = 0; k < info.dims.K; ++k)
    {
        // get the dimensions of the Hessian matrix
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        // get the offsets of the current variables in x, and out
        Index offset_ux = info.offsets_primal_u[k];
        // apply out[offs:offs+nu+nx] =  RSQ @ x[offs:offs+nu+nx]
        gemv_t(nu + nx, nu + nx, 1.0, RSQrqt[k], 0, 0, x, offset_ux, alpha, y, offset_ux, out,
               offset_ux);
    }

    // add additional terms
    for (Index k = 0; k < info.dims.K-1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index nu_next = info.dims.number_of_controls[k + 1];
        gemv_t(nu + nx, nx_next, 1.0, FuFx[k], 0, 0, 
               x, info.offsets_primal_u[k], 1.0, out, info.offsets_primal_x[k + 1], 
               out, info.offsets_primal_x[k + 1]);
        gemv_n(nu + nx, nx_next, 1.0, FuFx[k], 0, 0, 
               x, info.offsets_primal_x[k + 1], 1.0, out, info.offsets_primal_u[k], 
               out, info.offsets_primal_u[k]);
        if (GuGx_is_zero[k]) continue; // GuGx is only nonzero during the preprocessing of a rank deficient J
        gemv_t(nu + nx, nu_next, 1.0, GuGx[k], 0, 0, 
               x, info.offsets_primal_u[k], 1.0, out, info.offsets_primal_u[k + 1], 
               out, info.offsets_primal_u[k + 1]);
        gemv_n(nu + nx, nu_next, 1.0, GuGx[k], 0, 0, 
               x, info.offsets_primal_u[k + 1], 1.0, out, info.offsets_primal_u[k], 
               out, info.offsets_primal_u[k]);

    }
    if (print_debug){ std::cout << "Hessian<ImplicitOcpType>::apply_on_right done" << std::endl;}
}
