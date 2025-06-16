//
// Copyright (c) Lander Vanroye, KU Leuven
//
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/common/exception.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/problem_info.hpp"
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
                                          const Jacobian<ImplicitOcpType> &jacobian)
{
    std::cout << "preprocessing hessian" << std::endl;
    for (int k = 0; k < info.dims.K - 1; ++k){
        RSQrqt_original[k] = RSQrqt[k];
    }   

    for (int k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        
        gemm_nt(nx+nu+1, nx_next, nx+nu, 1.0, jacobian.BAbt[k], 0, 0, 
                FuFxt[k], 0, 0, 1.0, RSQrqt_original[k], 0, 0, RSQrqt[k], 0, 0);
        gemm_nt(nu+nx, nx_next, nu+nx, 1.0, FuFxt[k], 0, 0,
                jacobian.BAbt[k], 0, 0, 1.0, RSQrqt[k], 0, 0, RSQrqt[k], 0, 0);
    }
    std::cout << "\tPreprocessing done" << std::endl;
}