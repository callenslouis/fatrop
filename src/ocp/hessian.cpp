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

        RSQrqt_original[k] = RSQrqt[k];
    }
    for (int k = 0; k < info.dims.K-1; ++k){
        FuFxt_original[k] = FuFxt[k];
    }
    return;

    auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < info.dims.K; ++k){
        RSQrqt_original[k] = RSQrqt[k];

        // AugSystemSolver<OcpType> overwrites the entries corresponding to
        // the vectors r and q. We do the same here
        // this is necessary to make sure we have the correct RSQrqt matrix
        // when calling apply_on_right
        rowin(info.dims.number_of_states[k] + info.dims.number_of_controls[k],
              1.0, f, info.offsets_primal_u[k], RSQrqt_original[k],
              info.dims.number_of_states[k] + info.dims.number_of_controls[k], 0); 
    }   
    auto end = std::chrono::high_resolution_clock::now();
    duration_copy_RSQrqt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];

        // gemm_nt(nx+nu+1, nx+nu, nx_next, 1.0, jacobian.BAbt[k], 0, 0, 
        //         FuFxt[k], 0, 0, 1.0, RSQrqt[k], 0, 0, RSQrqt[k], 0, 0);
        // gemm_nt(nu+nx, nx+nu, nx_next, 1.0, FuFxt[k], 0, 0,
        //         jacobian.BAbt[k], 0, 0, 1.0, RSQrqt[k], 0, 0, RSQrqt[k], 0, 0);
        syrk_ln_mn(nx+nu+1, nx+nu, nx_next, 1.0, jacobian.BAbt[k], 0, 0, 
                   FuFxt[k], 0, 0, 1.0, RSQrqt[k], 0, 0, RSQrqt[k], 0, 0);
        syrk_ln_mn(nu+nx, nx+nu, nx_next, 1.0, FuFxt[k], 0, 0,
                   jacobian.BAbt[k], 0, 0, 1.0, RSQrqt[k], 0, 0, RSQrqt[k], 0, 0);
        // blasfeo_dsyr2k_ln(nx+nu+1, nx_next, 1.0, &jacobian.BAbt[k].mat(), 0, 0, 
        //                   &FuFxt[k].mat(), 0, 0, 1.0, &RSQrqt[k].mat(), 0, 0, &RSQrqt[k].mat(), 0, 0);

        // apply transformation to rhs also, since AugSystemSolver<OcpType> 
        // overwrites the entries corresponding to the vectors r and q in 
        // RSQrqt by considering f
        // [r^T q^T] <-- [r^T q^T] + b^T @ [Fu Fx]
        // [r; q] <-- [r; q] + [Fu Fx]^T @ b (--> do this in the rhs column vector)
        gemv_n(nu + nx, nx_next, 1.0, FuFxt[k], 0, 0,
               g, info.offsets_g_eq_dyn[k], 1.0, 
               f, info.offsets_primal_u[k], f, info.offsets_primal_u[k]);
    }
    end = std::chrono::high_resolution_clock::now();
    duration_modifying_RSQrqt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

std::map<std::string, double> Hessian<ImplicitOcpType>::TestPreProcessImplementation(
                                          const ProblemInfo &info, 
                                          Jacobian<ImplicitOcpType> &jacobian,
                                          VecRealView &f,
                                          VecRealView &g){
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();


    ///////////////////
    /// copy RSQrqt ///
    ///////////////////
    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < info.dims.K; ++k){
        RSQrqt_original[k] = RSQrqt[k];
        rowin(info.dims.number_of_states[k] + info.dims.number_of_controls[k],
              1.0, f, info.offsets_primal_u[k], RSQrqt_original[k],
              info.dims.number_of_states[k] + info.dims.number_of_controls[k], 0); 
    }   
    end = std::chrono::high_resolution_clock::now();
    auto duration_copy_RSQrqt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);




    ///////////////
    /// gemm_nt ///
    ///////////////
    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];

        gemm_nt(nx+nu+1, nx+nu, nx_next, 1.0, jacobian.BAbt[k], 0, 0, 
                FuFxt[k], 0, 0, 1.0, RSQrqt[k], 0, 0, RSQrqt[k], 0, 0);
        gemm_nt(nu+nx, nx+nu, nx_next, 1.0, FuFxt[k], 0, 0,
                jacobian.BAbt[k], 0, 0, 1.0, RSQrqt[k], 0, 0, RSQrqt[k], 0, 0);
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration_gemm_nt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);





    ///////////////
    /// gemm_nn ///
    ///////////////
    std::vector<MatRealAllocated> FuFx;
    FuFx.reserve(info.dims.K - 1);
    for (int k = 0; k < FuFxt.size(); ++k)
    {   
        FuFx.emplace_back(FuFxt[k].n(), FuFxt[k].m());
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        blasfeo_dgetr(nu + nx, nx_next, &FuFxt[k].mat(), 0, 0,
                      &FuFx[k].mat(), 0, 0);
    }
    std::vector<MatRealAllocated> BAb;
    BAb.reserve(jacobian.BAbt.size());
    for (int k = 0; k < jacobian.BAbt.size(); ++k)
    {
        BAb.emplace_back(jacobian.BAbt[k].n(), jacobian.BAbt[k].m());
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        blasfeo_dgetr(nu + nx + 1, nx_next, &jacobian.BAbt[k].mat(), 0, 0,
                      &BAb[k].mat(), 0, 0);
    }
    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];

        blasfeo_dgemm_nn(nx+nu+1, nx+nu, nx_next, 1.0, &jacobian.BAbt[k].mat(), 0, 0, 
                &FuFxt[k].mat(), 0, 0, 1.0, &RSQrqt[k].mat(), 0, 0, &RSQrqt[k].mat(), 0, 0);
        blasfeo_dgemm_nn(nu+nx, nx+nu, nx_next, 1.0, &FuFxt[k].mat(), 0, 0,
                &BAb[k].mat(), 0, 0, 1.0, &RSQrqt[k].mat(), 0, 0, &RSQrqt[k].mat(), 0, 0);
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration_gemm_nn = std::chrono::duration_cast<std::chrono::microseconds>(end - start);




    //////////////////
    /// syrk_ln_mn ///      ---> seems to be fastest
    //////////////////
    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];

        syrk_ln_mn(nx+nu+1, nx+nu, nx_next, 1.0, jacobian.BAbt[k], 0, 0, 
                   FuFxt[k], 0, 0, 1.0, RSQrqt[k], 0, 0, RSQrqt[k], 0, 0);
        syrk_ln_mn(nu+nx, nx+nu, nx_next, 1.0, FuFxt[k], 0, 0,
                   jacobian.BAbt[k], 0, 0, 1.0, RSQrqt[k], 0, 0, RSQrqt[k], 0, 0);
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration_syrk_ln_mn = std::chrono::duration_cast<std::chrono::microseconds>(end - start);




    ////////////////
    /// syr2k_ln ///
    ////////////////
    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < info.dims.K - 1; ++k){
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];

        blasfeo_dsyr2k_ln(nx+nu+1, nx_next, 1.0, &jacobian.BAbt[k].mat(), 0, 0, 
                          &FuFxt[k].mat(), 0, 0, 1.0, &RSQrqt[k].mat(), 0, 0, &RSQrqt[k].mat(), 0, 0);
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration_blasfeo_syr2k = std::chrono::duration_cast<std::chrono::microseconds>(end - start);




    //////////////
    /// gemv_n ///
    //////////////
    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < info.dims.K - 1; ++k) {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        gemv_n(nu + nx, nx_next, 1.0, FuFxt[k], 0, 0,
               g, info.offsets_g_eq_dyn[k], 1.0, 
               f, info.offsets_primal_u[k], f, info.offsets_primal_u[k]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration_gemv_n = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return {
        {"duration_copy_RSQrqt", duration_copy_RSQrqt.count()},
        {"duration_gemm_nt", duration_gemm_nt.count()},
        {"duration_syrk_ln_mn", duration_syrk_ln_mn.count()},
        {"duration_blasfeo_syr2k", duration_blasfeo_syr2k.count()},
        {"duration_gemv_n", duration_gemv_n.count()},
        {"duration_gemm_nn", duration_gemm_nn.count()},
    };
}

void Hessian<ImplicitOcpType>::ResetPreProcess(const ProblemInfo &info, 
                                               const Jacobian<ImplicitOcpType> &jacobian)
{
    for (int k = 0; k < info.dims.K; ++k){
        // std::cout << "RSQrqt[" << k << "]:" << std::endl << RSQrqt[k] << std::endl;
        // std::cout << "RSQrqt_original[" << k << "]:" << std::endl << RSQrqt_original[k] << std::endl;
        // if (k < info.dims.K-1){ std::cout << "FuFxt[" << k << "]:" << std::endl << FuFxt[k] << std::endl;}
        RSQrqt[k] = RSQrqt_original[k];
    }
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
        gemv_t(nu + nx, nx_next, 1.0, FuFxt[k], 0, 0, 
                x, info.offsets_primal_u[k], 1.0, out, info.offsets_primal_x[k + 1], 
                out, info.offsets_primal_x[k + 1]);
        gemv_n(nu + nx, nx_next, 1.0, FuFxt[k], 0, 0, x, info.offsets_primal_x[k + 1],
                1.0, out, info.offsets_primal_u[k], out, info.offsets_primal_u[k]);
    }
    if (print_debug){ std::cout << "Hessian<ImplicitOcpType>::apply_on_right done" << std::endl;}
}
