//
// Copyright (C) 2024 Lander Vanroye, KU Leuven
//

#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
using namespace fatrop;

bool check_reg(const Index m, MAT *sA, const Index ai, const Index aj)
{
    for (Index i = 0; i < m; i++)
    {
        if (blasfeo_matel_wrap(sA, ai + i, aj + i) < 1e-8)
            return false;
    }
    return true;
}

AugSystemSolver<OcpType>::AugSystemSolver(const ProblemInfo &info)
{
    Index max_number_of_controls =
        *std::max_element(info.dims.number_of_controls.begin(), info.dims.number_of_controls.end());
    Index max_number_of_states =
        *std::max_element(info.dims.number_of_states.begin(), info.dims.number_of_states.end());
    Index max_number_of_variables = *std::max_element(info.number_of_stage_variables.begin(),
                                                      info.number_of_stage_variables.end());
    Index max_number_of_ineq_constraints = *std::max_element(
        info.dims.number_of_ineq_constraints.begin(), info.dims.number_of_ineq_constraints.end());
    Index max_number_of_eq_consttraints = *std::max_element(
        info.dims.number_of_eq_constraints.begin(), info.dims.number_of_eq_constraints.end());

    AL.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    Ggt_stripe.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    GgLt.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    RSQrqt_hat.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    Llt_shift.emplace_back(max_number_of_variables + 1, max_number_of_controls);
    GgIt_tilde.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    GgLIt.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    HhIt.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    PpIt_hat.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    LlIt.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    Ggt_ineq_temp.emplace_back(max_number_of_variables + 1, max_number_of_ineq_constraints);

    Ppt.reserve(info.dims.K);
    Hh.reserve(info.dims.K);
    RSQrqt_tilde.reserve(info.dims.K);
    Ggt_tilde.reserve(info.dims.K);
    Llt.reserve(info.dims.K);
    for (Index k = 0; k < info.dims.K; k++)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index ng_eq = info.dims.number_of_eq_constraints[k];
        Ppt.emplace_back(nx + 1, nx);
        Hh.emplace_back(nx, nx + 1);
        RSQrqt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Ggt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Llt.emplace_back(nu + nx + 1, nu);
    }

    v_AL.emplace_back(max_number_of_variables);
    v_Ggt_stripe.emplace_back(max_number_of_variables);
    v_GgLt.emplace_back(max_number_of_variables);
    v_RSQrqt_hat.emplace_back(max_number_of_variables);
    v_Llt_shift.emplace_back(max_number_of_controls);
    v_GgIt_tilde.emplace_back(info.dims.number_of_states[0]);
    v_GgLIt.emplace_back(info.dims.number_of_states[0]);
    v_HhIt.emplace_back(info.dims.number_of_states[0]);
    v_PpIt_hat.emplace_back(info.dims.number_of_states[0]);
    v_LlIt.emplace_back(info.dims.number_of_states[0]);
    v_Ggt_ineq_temp.emplace_back(max_number_of_ineq_constraints);
    v_tmp.emplace_back(max_number_of_variables);

    v_Ppt.reserve(info.dims.K);
    v_Hh.reserve(info.dims.K);
    v_RSQrqt_tilde.reserve(info.dims.K);
    v_Ggt_tilde.reserve(info.dims.K);
    v_Llt.reserve(info.dims.K);

    for (Index k = 0; k < info.dims.K; k++)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        v_Ppt.emplace_back(nx);
        v_Hh.emplace_back(nx);
        v_RSQrqt_tilde.emplace_back(nu + nx);
        v_Ggt_tilde.emplace_back(nu + nx);
        v_Llt.emplace_back(nu + nx);
    }

    PlI.emplace_back(info.dims.number_of_states[0]);
    PrI.emplace_back(info.dims.number_of_states[0]);

    Pl.reserve(info.dims.K);
    Pr.reserve(info.dims.K);

    for (Index k = 0; k < info.dims.K; k++)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Pl.emplace_back(max_number_of_controls);
        Pr.emplace_back(max_number_of_controls);
    }

    gamma.resize(info.dims.K);
    rho.resize(info.dims.K);
};

LinsolReturnFlag AugSystemSolver<OcpType>::solve(const ProblemInfo &info,
                                           Jacobian<OcpType> &jacobian, Hessian<OcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_s,
                                           const VecRealView &f, const VecRealView &g,
                                           VecRealView &x, VecRealView &eq_mult)
{
    MatRealView *RSQrq_hat_curr_p;
    Index rank_k;
    /////////////// recursion ///////////////
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offset_ineq_k = info.offsets_slack[k];
        const Index offset_u = info.offsets_primal_u[k];
        const Index offset_eq_path = info.offsets_g_eq_path[k];
        const Index offset_eq_slack = info.offsets_g_eq_slack[k];
        //////// SUBSDYN
        Index gamma_k;
        if (k == info.dims.K - 1)
        {
            gamma_k = ng;
            gamma[k] = gamma_k;
            rowin(ng, 1.0, g, offset_eq_path, jacobian.Gg_eqt[k], nu + nx, 0);
            gecp(nx + nu + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            gecp(nx + nu + 1, nu + nx, hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        else
        {
            const Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index Hp1_size = gamma[k + 1] - rho[k + 1];
            if (Hp1_size + ng > nu + nx)
                return LinsolReturnFlag::NOFULL_RANK;
            gamma_k = Hp1_size + ng;
            // AL <- [BAb]^T_k P_kp1
            rowin(nxp1, 1.0, g, offset_eq_dyn, jacobian.BAbt[k], nu + nx, 0);
            gemm_nt(nu + nx + 1, nxp1, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Ppt[k + 1], 0, 0, 0.0,
                    AL[0], 0, 0, AL[0], 0, 0);
            // AL[-1,:] <- AL[-1,:] + p_kp1^T
            gead(1, nxp1, 1.0, Ppt[k + 1], nxp1, 0, AL[0], nx + nu, 0);
            // RSQrqt_stripe <- AL[BA] + RSQrqt
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            syrk_ln_mn(nu + nx + 1, nu + nx, nxp1, 1.0, AL[0], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                       hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
            //// inequalities
            gamma[k] = gamma_k;
            // if ng[k]>0
            if (gamma_k > 0)
            {
                // if Gk nonempty
                if (ng > 0)
                {
                    // Ggt_stripe  <- Ggt_k
                    rowin(ng, 1.0, g, offset_eq_path, jacobian.Gg_eqt[k], nu + nx, 0);
                    gecp(nu + nx + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
                }
                // if Hkp1 nonempty
                if (Hp1_size > 0)
                {
                    // Ggt_stripe <- [Ggt_k [BAb_k^T]H_kp1]
                    gemm_nt(nu + nx + 1, Hp1_size, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Hh[k + 1], 0,
                            0, 0.0, Ggt_stripe[0], 0, ng, Ggt_stripe[0], 0, ng);
                    // Ggt_stripe[-1,ng:] <- Ggt_stripe[-1,ng:] + h_kp1^T
                    gead_transposed(1, Hp1_size, 1.0, Hh[k + 1], 0, nxp1, Ggt_stripe[0], nu + nx,
                                    ng);
                }
            }
            else
            {
                rho[k] = 0;
                rank_k = 0;
                RSQrq_hat_curr_p = &RSQrqt_tilde[k];
            }
        }
        // inequalities + inertia correction
        {
            if (ng_ineq > 0)
            {
                rowin(ng_ineq, 1.0, g, offset_eq_slack, jacobian.Gg_ineqt[k], nu + nx, 0);
                gecp(nu + nx + 1, ng_ineq, jacobian.Gg_ineqt[k], 0, 0, Ggt_ineq_temp[0], 0, 0);
                for (Index i = 0; i < ng_ineq; i++)
                {
                    Scalar scaling_factor = 1.0 / D_s(offset_ineq_k + i);
                    colsc(nu + nx + 1, scaling_factor, Ggt_ineq_temp[0], 0, i);
                }
                // add the penalty
                syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, Ggt_ineq_temp[0], 0, 0,
                           jacobian.Gg_ineqt[k], 0, 0, 1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k],
                           0, 0);
            }
            // inertia correction
            diaad(nu + nx, 1.0, D_x, offset_u, RSQrqt_tilde[k], 0, 0);
        }
        //////// TRANSFORM_AND_SUBSEQ
        {
            // symmetric transformation, done a little different than in paper, in order to fuse LA
            // operations LU_FACT_TRANSPOSE(Ggtstripe[:gamma_k, nu+nx+1], nu max) if(k==K-2)
            // blasfeo_print_dmat(1, gamma_k, Ggt_stripe[0], nu+nx, 0);
            lu_fact_transposed(gamma_k, nu + nx + 1, nu, rank_k, Ggt_stripe[0], Pl[k], Pr[k],
                               lu_fact_tol);

            rho[k] = rank_k;
            if (gamma_k - rank_k > 0)
            {
                // transfer eq's to next stage
                if (gamma_k - rank_k > nx)
                    return LinsolReturnFlag::NOFULL_RANK;
                getr(nx + 1, gamma_k - rank_k, Ggt_stripe[0], nu, rank_k, Hh[k], 0, 0);
            }
            if (rank_k > 0)
            {
                // Ggt_tilde_k <- Ggt_stripe[rho_k:nu+nx+1, :rho] L-T (note that this is slightly
                // different from the implementation)
                trsm_rlnn(nu - rank_k + nx + 1, rank_k, -1.0, Ggt_stripe[0], 0, 0, Ggt_stripe[0],
                          rank_k, 0, Ggt_tilde[k], 0, 0);
                // the following command copies the top block matrix (LU) to the bottom because it
                // it needed later
                gecp(rank_k, gamma_k, Ggt_stripe[0], 0, 0, Ggt_tilde[k], nu - rank_k + nx + 1, 0);
                // permutations
                trtr_l(nu + nx, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0,
                       0); // copy lower part of RSQ to upper part
                Pr[k].apply_on_rows(rank_k, &RSQrqt_tilde[k].mat()); // TODO make use of symmetry
                Pr[k].apply_on_cols(rank_k, &RSQrqt_tilde[k].mat());
                // GL <- Ggt_tilde_k @ RSQ[:rho,:nu+nx] + RSQrqt[rho:nu+nx+1, rho:] (with
                // RSQ[:rho,:nu+nx] = RSQrqt[:nu+nx,:rho]^T) GEMM_NT(nu - rank_k + nx + 1, nu + nx,
                // rank_k, 1.0, Ggt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0, 1.0, RSQrqt_tilde_p
                // + k, rank_k, 0, GgLt[0], 0, 0); split up because valgrind was giving invalid read
                // errors when C matrix has nonzero row offset GgLt[0].print();
                gecp(nu - rank_k + nx + 1, nu + nx, RSQrqt_tilde[k], rank_k, 0, GgLt[0], 0, 0);
                gemm_nt(nu - rank_k + nx + 1, nu + nx, rank_k, 1.0, Ggt_tilde[k], 0, 0,
                        RSQrqt_tilde[k], 0, 0, 1.0, GgLt[0], 0, 0, GgLt[0], 0, 0);
                // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] +
                // GgLt[rank_k:, :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
                syrk_ln_mn(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt[0], 0, 0,
                           Ggt_tilde[k], 0, 0, 1.0, GgLt[0], 0, rank_k, RSQrqt_hat[0], 0, 0);
                // GEMM_NT(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt[0], 0, 0,
                // Ggt_tilde[k], 0, 0, 1.0, GgLt[0], 0, rank_k, RSQrqt_hat[0], 0, 0);
                RSQrq_hat_curr_p = &RSQrqt_hat[0];
            }
            else
            {
                RSQrq_hat_curr_p = &RSQrqt_tilde[k];
            }
        }
        //////// SCHUR
        {
            if (nu - rank_k > 0)
            {
                // DLlt_k = [chol(R_hatk); Llk@chol(R_hatk)^-T]
                potrf_l_mn(nu - rank_k + nx + 1, nu - rank_k, RSQrq_hat_curr_p[0], 0, 0, Llt[k], 0,
                           0);
                if (!check_reg(nu - rank_k, &Llt[k].mat(), 0, 0))
                    return LinsolReturnFlag::INDEFINITE;
                // Pp_k = Qq_hatk - L_k^T @ Ll_k
                // SYRK_LN_MN(nx+1, nx, nu-rank_k, -1.0,Llt_p+k, nu-rank_k,0, Llt_p+k,
                // nu-rank_k,0, 1.0, RSQrq_hat_curr[0], nu-rank_k, nu-rank_k,Pp+k,0,0); // feature
                // not implmented yet
                gecp(nx + 1, nu - rank_k, Llt[k], nu - rank_k, 0, Llt_shift[0], 0,
                     0); // needless operation because feature not implemented yet
                // SYRK_LN_MN(nx + 1, nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0,
                // 0, 1.0, RSQrq_hat_curr[0], nu - rank_k, nu - rank_k, Ppt[k], 0, 0);
                gecp(nx + 1, nx, RSQrq_hat_curr_p[0], nu - rank_k, nu - rank_k, Ppt[k], 0, 0);
                syrk_ln_mn(nx + 1, nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0, 0,
                           1.0, Ppt[k], 0, 0, Ppt[k], 0, 0);
                // next steps are for better accuracy
                if (increased_accuracy)
                {
                    // copy eta
                    getr(nu - rank_k, gamma_k - rank_k, Ggt_stripe[0], rank_k, rank_k,
                         Ggt_stripe[0], 0, 0);
                    // blasfeo_print_dmat(gamma_k-rank_k, nu-rank_k, Ggt_stripe[0], 0,0);
                    // eta L^-T
                    trsm_rltn(gamma_k - rank_k, nu - rank_k, 1.0, Llt[k], 0, 0, Ggt_stripe[0], 0, 0,
                              Ggt_stripe[0], 0, 0);
                    // ([S^T \\ r^T] L^-T) @ (L^-1 eta^T)
                    // (eta L^-T) @ ([S^T \\ r^T] L^-T)^T
                    gemm_nt(gamma_k - rank_k, nx + 1, nu - rank_k, -1.0, Ggt_stripe[0], 0, 0,
                            Llt[k], nu - rank_k, 0, 1.0, Hh[k], 0, 0, Hh[k], 0, 0);
                    // keep (L^-1 eta^T) for forward recursion
                    getr(gamma_k - rank_k, nu - rank_k, Ggt_stripe[0], 0, 0, Ggt_tilde[k], 0,
                         rank_k);
                }
            }
            else
            {
                gecp(nx + 1, nx, RSQrq_hat_curr_p[0], 0, 0, Ppt[k], 0, 0);
            }
            trtr_l(nx, Ppt[k], 0, 0, Ppt[k], 0, 0);
        }
    }
    rankI = 0;
    //////// FIRST_STAGE
    {
        const Index nx = info.dims.number_of_states[0];
        Index gamma_I = gamma[0] - rho[0];
        if (gamma_I > nx)
        {
            return LinsolReturnFlag::NOFULL_RANK;
        }
        if (gamma_I > 0)
        {
            getr(gamma_I, nx + 1, Hh[0], 0, 0, HhIt[0], 0, 0); // transposition may be avoided
            // HhIt[0].print();
            lu_fact_transposed(gamma_I, nx + 1, nx, rankI, HhIt[0], PlI[0], PrI[0], lu_fact_tol);
            if (rankI < gamma_I)
                return LinsolReturnFlag::NOFULL_RANK;
            // PpIt_tilde <- Ggt[rankI:nx+1, :rankI] L-T (note that this is slightly different from
            // the implementation)
            trsm_rlnn(nx - rankI + 1, rankI, -1.0, HhIt[0], 0, 0, HhIt[0], rankI, 0, GgIt_tilde[0],
                      0, 0);
            // permutations
            PrI[0].apply_on_rows(rankI, &Ppt[0].mat()); // TODO make use of symmetry
            PrI[0].apply_on_cols(rankI, &Ppt[0].mat());
            // // GL <- GgIt_tilde @ Pp[:rankI,:nx] + Ppt[rankI:nx+1, rankI:] (with Pp[:rankI,:nx] =
            // Ppt[:nx,:rankI]^T) GEMM_NT(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde[0], 0, 0,
            // Ppt[0], 0, 0, 1.0, Ppt[0], rankI, 0, GgLIt[0], 0, 0); split up because valgrind was
            // giving invalid read errors when C matrix has nonzero row offset
            gecp(nx - rankI + 1, nx, Ppt[0], rankI, 0, GgLIt[0], 0, 0);
            gemm_nt(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde[0], 0, 0, Ppt[0], 0, 0, 1.0,
                    GgLIt[0], 0, 0, GgLIt[0], 0, 0);
            // // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] + GgLt[rank_k:,
            // :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
            syrk_ln_mn(nx - rankI + 1, nx - rankI, rankI, 1.0, GgLIt[0], 0, 0, GgIt_tilde[0], 0, 0,
                       1.0, GgLIt[0], 0, rankI, PpIt_hat[0], 0, 0);
            // TODO skipped if nx-rankI = 0
            potrf_l_mn(nx - rankI + 1, nx - rankI, PpIt_hat[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx - rankI, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
        else
        {
            rankI = 0;
            potrf_l_mn(nx + 1, nx, Ppt[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
    }
    ////// FORWARD_SUBSTITUTION:
    // first stage
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_u = info.offsets_primal_u[0];
        const Index offs_x = info.offsets_primal_x[0];
        const Index offs_g = info.offsets_g_eq_path[0];
        // calculate xIb
        rowex(nx - rankI, -1.0, LlIt[0], nx - rankI, 0, x, offs_x + rankI);
        // assume TRSV_LTN allows aliasing, this is the case in normal BLAS
        trsv_ltn(nx - rankI, LlIt[0], 0, 0, x, offs_x + rankI, x, offs_x + rankI);
        // calculate xIa
        rowex(rankI, 1.0, GgIt_tilde[0], nx - rankI, 0, x, offs_x);
        // assume aliasing is possible for last two elements
        gemv_t(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, x, offs_x + rankI, 1.0, x, offs_x, x,
               offs_x);
        //// lag
        rowex(rankI, -1.0, Ppt[0], nx, 0, eq_mult, offs_g);
        // assume aliasing is possible for last two elements
        gemv_t(nx, rankI, -1.0, Ppt[0], 0, 0, x, offs_x, 1.0, eq_mult, offs_g, eq_mult, offs_g);

        // U^-T
        trsv_lnn(rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        // L^-T
        trsv_unu(rankI, rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        PlI[0].apply_inverse(rankI, &eq_mult.vec(), offs_g);
        PrI[0].apply_inverse(rankI, &x.vec(), offs_x);
    }
    // other stages
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index rho_k = rho[k];
        const Index numrho_k = nu - rho_k;
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index gammamrho_k = gamma[k] - rho[k];
        const Index gamma_k = gamma[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (numrho_k > 0)
        {
            /// calculate ukb_tilde
            // -Lkxk - lk
            rowex(numrho_k, -1.0, Llt[k], numrho_k + nx, 0, x, offs + rho_k);
            if (increased_accuracy)
            {
                gemv_n(nu - rho_k, gamma_k - rho_k, -1.0, Ggt_tilde[k], 0, rho_k, eq_mult, offs_g_k,
                       1.0, x, offs + rho_k, x, offs + rho_k);
            }
            // assume aliasing of last two eliments is allowed
            gemv_t(nx, numrho_k, -1.0, Llt[k], numrho_k, 0, x, offs_x, 1.0, x, offs + rho_k, x,
                   offs + rho_k);
            trsv_ltn(numrho_k, Llt[k], 0, 0, x, offs + rho_k, x, offs + rho_k);
        }
        /// calcualate uka_tilde
        if (rho_k > 0)
        {
            rowex(rho_k, 1.0, Ggt_tilde[k], numrho_k + nx, 0, x, offs);
            gemv_t(nx + numrho_k, rho_k, 1.0, Ggt_tilde[k], 0, 0, x, offs + rho_k, 1.0, x, offs, x,
                   offs);
            // calculate lamda_tilde_k
            // copy vk to right location
            veccp(gammamrho_k, eq_mult, offs_g_k, v_tmp[0], 0);
            veccp(gammamrho_k, v_tmp[0], 0, eq_mult, offs_g_k + rho_k);
            rowex(rho_k, -1.0, RSQrqt_tilde[k], nu + nx, 0, eq_mult, offs_g_k);
            // assume aliasing of last two eliments is allowed
            gemv_t(nu + nx, rho_k, -1.0, RSQrqt_tilde[k], 0, 0, x, offs, 1.0, eq_mult, offs_g_k,
                   eq_mult, offs_g_k);
            // nu-rank_k+nx,0
            // needless copy because feature not implemented yet in trsv_lnn
            gecp(rho_k, gamma_k, Ggt_tilde[k], nu - rho_k + nx + 1, 0, AL[0], 0, 0);
            // U^-T
            trsv_lnn(rho_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            // L^-T
            trsv_unu(rho_k, gamma_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            Pl[k].apply_inverse(rho_k, &eq_mult.vec(), offs_g_k);
            Pr[k].apply_inverse(rho_k, &x.vec(), offs);
        }
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        }
        if (k != info.dims.K - 1)
        {
            const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offsxp1 = info.offsets_primal_x[k + 1];
            const Index offs_g_kp1 = info.offsets_g_eq_path[k + 1];
            const Index gammamrho_kp1 = gamma[k + 1] - rho[k + 1];
            // calculate xkp1
            rowex(nxp1, 1.0, jacobian.BAbt[k], nu + nx, 0, x, offsxp1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offsxp1, x,
                   offsxp1);
            // calculate lam_dyn xp1
            rowex(nxp1, 1.0, Ppt[k + 1], nxp1, 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offsxp1, 1.0, eq_mult, offs_dyn_eq_k,
                   eq_mult, offs_dyn_eq_k);
            gemv_t(gammamrho_kp1, nxp1, 1.0, Hh[k + 1], 0, 0, eq_mult, offs_g_kp1, 1.0, eq_mult,
                   offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}
LinsolReturnFlag AugSystemSolver<OcpType>::solve(const ProblemInfo &info,
                                           Jacobian<OcpType> &jacobian, Hessian<OcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_eq,
                                           const VecRealView &D_s, const VecRealView &f,
                                           const VecRealView &g, VecRealView &x,
                                           VecRealView &eq_mult)
{
    MatRealView *RSQrq_hat_curr_p;
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_ineq_k = info.offsets_slack[k];
        const Index offset_u = info.offsets_primal_u[k];
        const Index offset_eq_k = info.offsets_eq[k];
        const Index offset_g_eq_k = info.offsets_g_eq_path[k];
        const Index offset_g_ineq_k = info.offsets_g_eq_slack[k];
        // const fatrop_int offs_g_ineq_k = offs_g_ineq_p[k];
        //////// SUBSDYN
        if (k == info.dims.K - 1)
        {
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            gecp(nx + nu + 1, nu + nx, hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        else
        {
            const Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            // AL <- [BAb]^T_k P_kp1
            rowin(nxp1, 1.0, g, offset_eq_dyn, jacobian.BAbt[k], nu + nx, 0);
            gemm_nt(nu + nx + 1, nxp1, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Ppt[k + 1], 0, 0, 0.0,
                    AL[0], 0, 0, AL[0], 0, 0);
            // AL[-1,:] <- AL[-1,:] + p_kp1^T
            gead(1, nxp1, 1.0, Ppt[k + 1], nxp1, 0, AL[0], nx + nu, 0);
            // RSQrqt_stripe <- AL[BA] + RSQrqt
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            syrk_ln_mn(nu + nx + 1, nu + nx, nxp1, 1.0, AL[0], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                       hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        // equality penalty
        {
            rowin(ng, 1.0, g, offset_g_eq_k, jacobian.Gg_eqt[k], nu + nx, 0);
            gecp(nu + nx + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
            for (Index i = 0; i < ng; i++)
            {
                Scalar scaling_factor = 1.0 / D_eq(offset_eq_k + i);
                colsc(nu + nx + 1, scaling_factor, Ggt_stripe[0], 0, i);
            }
            // add the penalty
            syrk_ln_mn(nu + nx + 1, nu + nx, ng, 1.0, Ggt_stripe[0], 0, 0, jacobian.Gg_eqt[k], 0, 0,
                       1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        // inequalities + inertia correction
        {
            if (ng_ineq > 0)
            {
                rowin(ng_ineq, 1.0, g, offset_g_ineq_k, jacobian.Gg_ineqt[k], nu + nx, 0);
                gecp(nu + nx + 1, ng_ineq, jacobian.Gg_ineqt[k], 0, 0, Ggt_ineq_temp[0], 0, 0);
                for (Index i = 0; i < ng_ineq; i++)
                {
                    Scalar scaling_factor = 1.0 / D_s(offs_ineq_k + i);
                    colsc(nu + nx + 1, scaling_factor, Ggt_ineq_temp[0], 0, i);
                }
                // add the penalty
                syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, Ggt_ineq_temp[0], 0, 0,
                           jacobian.Gg_ineqt[k], 0, 0, 1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k],
                           0, 0);
            }
            // inertia correction
            diaad(nu + nx, 1.0, D_x, offset_u, RSQrqt_tilde[k], 0, 0);
        }

        //////// TRANSFORM_AND_SUBSEQ
        {
            RSQrq_hat_curr_p = &RSQrqt_tilde[k];
        }
        //////// SCHUR
        {
            // DLlt_k = [chol(R_hatk); Llk@chol(R_hatk)^-T]
            potrf_l_mn(nu + nx + 1, nu, *RSQrq_hat_curr_p, 0, 0, Llt[k], 0, 0);
            if (!check_reg(nu, &Llt[k].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
            // Pp_k = Qq_hatk - L_k^T @ Ll_k
            // SYRK_LN_MN(nx+1, nx, nu-rank_k, -1.0,Llt_p+k, nu-rank_k,0, Llt_p+k, nu-rank_k,0, 1.0,
            // RSQrq_hat_curr_p, nu-rank_k, nu-rank_k,Pp+k,0,0); // feature not implmented yet
            gecp(nx + 1, nu, Llt[k], nu, 0, Llt_shift[0], 0,
                 0); // needless operation because feature not implemented yet
            syrk_ln_mn(nx + 1, nx, nu, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0, 0, 1.0,
                       *RSQrq_hat_curr_p, nu, nu, Ppt[k], 0, 0);
        }
        trtr_l(nx, Ppt[k], 0, 0, Ppt[k], 0, 0);
    }
    //////// FIRST_STAGE
    {
        const Index nx = info.dims.number_of_states[0];
        {
            potrf_l_mn(nx + 1, nx, Ppt[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
    }
    ////// FORWARD_SUBSTITUTION:
    // first stage
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_x = info.offsets_primal_x[0];
        // calculate xIb
        rowex(nx, -1.0, LlIt[0], nx, 0, x, offs_x);
        // assume TRSV_LTN allows aliasing, this is the case in normal BLAS
        trsv_ltn(nx, LlIt[0], 0, 0, x, offs_x, x, offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        rowex(nu, -1.0, Llt[k], nu + nx, 0, x, offs);
        gemv_t(nx, nu, -1.0, Llt[k], nu, 0, x, offs_x, 1.0, x, offs, x, offs);
        trsv_ltn(nu, Llt[k], 0, 0, x, offs, x, offs);
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offs_x_p1 = info.offsets_primal_x[k + 1];
            const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
            // calculate xkp1
            rowex(nxp1, 1.0, jacobian.BAbt[k], nu + nx, 0, x, offs_x_p1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offs_x_p1, x,
                   offs_x_p1);
            // calculate lam_dyn xp1
            rowex(nxp1, 1.0, Ppt[k + 1], nxp1, 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offs_x_p1, 1.0, eq_mult, offs_dyn_eq_k,
                   eq_mult, offs_dyn_eq_k);
        }
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index offs_g_eq_k = info.offsets_g_eq_path[k];
        const Index offs_eq_k = info.offsets_eq[k];
        if (ng > 0)
        {
            gemv_t(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, x, offs, 1.0, g, offs_g_eq_k,
                   eq_mult, offs_g_eq_k);
            eq_mult.block(ng, offs_g_eq_k) =
                eq_mult.block(ng, offs_g_eq_k) / D_eq.block(ng, offs_eq_k);
        }
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_slack = info.offsets_slack[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}

LinsolReturnFlag AugSystemSolver<OcpType>::solve_rhs(const ProblemInfo &info,
                                               const Jacobian<OcpType> &jacobian,
                                               const Hessian<OcpType> &hessian,
                                               const VecRealView &D_s, const VecRealView &f,
                                               const VecRealView &g, VecRealView &x,
                                               VecRealView &eq_mult)
{
    VecRealView *v_RSQrq_hat_curr_p;
    Index rank_k;
    /////////////// recursion ///////////////

    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offset_ineq_k = info.offsets_slack[k];
        const Index offs_g_ineq_k = info.offsets_g_eq_slack[k];
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index offs = info.offsets_primal_u[k];
        //         //////// SUBSDYN
        Index gamma_k;
        if (k == info.dims.K - 1)
        {
            gamma_k = ng;
            gamma[k] = gamma_k;
            veccp(ng, g, offs_g_k, v_Ggt_stripe[0], 0);
            veccp(nu + nx, f, offs, v_RSQrqt_tilde[k], 0);
        }
        else
        {
            const Index offs_dyn_k = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index Hp1_size = gamma[k + 1] - rho[k + 1];
            gamma_k = Hp1_size + ng;
            gemv_n(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, g, offs_dyn_k, 0.0, v_AL[0], 0, v_AL[0], 0);
            axpy(nxp1, 1.0, v_Ppt[k + 1], 0, v_AL[0], 0, v_AL[0], 0);
            gemv_n(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, v_AL[0], 0, 1.0, f, offs,
                   v_RSQrqt_tilde[k], 0);
            if (gamma_k > 0)
            {
                if (ng > 0)
                {
                    veccp(ng, g, offs_g_k, v_Ggt_stripe[0], 0);
                }
                if (Hp1_size > 0)
                {
                    gemv_n(Hp1_size, nxp1, 1.0, Hh[k + 1], 0, 0, g, offs_dyn_k, 0.0,
                           v_Ggt_stripe[0], ng, v_Ggt_stripe[0], ng);
                    axpy(Hp1_size, 1.0, v_Hh[k + 1], 0, v_Ggt_stripe[0], ng, v_Ggt_stripe[0], ng);
                }
            }
            else
            {
                rank_k = 0;
                v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
            }
        }
        if (ng_ineq > 0)
        {
            for (Index i = 0; i < ng_ineq; i++)
            {
                Scalar scaling_factor = D_s(offset_ineq_k + i);
                Scalar grad_barrier = g(offs_g_ineq_k + i);
                v_Ggt_ineq_temp[0](i) = grad_barrier / scaling_factor;
            }
            gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, v_Ggt_ineq_temp[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        {
            rank_k = rho[k];
            gecp(rank_k, gamma_k, Ggt_tilde[k], nu - rank_k + nx + 1, 0, Ggt_stripe[0], 0, 0);
            Pl[k].apply(rank_k, &v_Ggt_stripe[0].vec(), 0);
            trsv_utu(rank_k, Ggt_stripe[0], 0, 0, v_Ggt_stripe[0], 0, v_Ggt_stripe[0], 0);
            gemv_t(rank_k, gamma_k - rank_k, -1.0, Ggt_stripe[0], 0, rank_k, v_Ggt_stripe[0], 0,
                   1.0, v_Ggt_stripe[0], rank_k, v_Ggt_stripe[0], rank_k);

            if (gamma_k - rank_k > 0)
            {
                veccp(gamma_k - rank_k, v_Ggt_stripe[0], rank_k, v_Hh[k], 0);
            }
            if (rank_k > 0)
            {
                veccpsc(rank_k, -1.0, v_Ggt_stripe[0], 0, v_Ggt_tilde[k], 0);
                trsv_ltn(rank_k, Ggt_stripe[0], 0, 0, v_Ggt_tilde[k], 0, v_Ggt_tilde[k], 0);
                Pr[k].apply(rank_k, &v_RSQrqt_tilde[k].vec(), 0);
                veccp(nu + nx, v_RSQrqt_tilde[k], 0, v_GgLt[0], 0);
                gemv_n(nu + nx, rank_k, 1.0, RSQrqt_tilde[k], 0, 0, v_Ggt_tilde[k], 0, 1.0,
                       v_GgLt[0], 0, v_GgLt[0], 0);
                gemv_n(nu + nx - rank_k, rank_k, 1.0, Ggt_tilde[k], 0, 0, v_GgLt[0], 0, 1.0,
                       v_GgLt[0], rank_k, v_RSQrqt_hat[0], 0);
                v_RSQrq_hat_curr_p = &v_RSQrqt_hat[0];
            }
            else
            {
                v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
            }
        }
        //         //////// SCHUR
        {
            if (nu - rank_k > 0)
            {
                trsv_lnn(nu - rank_k, Llt[k], 0, 0, *v_RSQrq_hat_curr_p, 0, v_Llt[k], 0);
                gecp(nx + 1, nu - rank_k, Llt[k], nu - rank_k, 0, Llt_shift[0], 0, 0);
                veccp(nu - rank_k, v_Llt[k], 0, v_Llt_shift[0], 0);
                veccp(nx, *v_RSQrq_hat_curr_p, nu - rank_k, v_Ppt[k], 0);
                gemv_n(nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, v_Llt_shift[0], 0, 1.0, v_Ppt[k],
                       0, v_Ppt[k], 0);
                if (increased_accuracy)
                {
                    gemv_t(nu - rank_k, gamma_k - rank_k, -1.0, Ggt_tilde[k], 0, rank_k, v_Llt[k],
                           0, 1.0, v_Hh[k], 0, v_Hh[k], 0);
                }
            }
            else
            {
                veccp(nx, *v_RSQrq_hat_curr_p, 0, v_Ppt[k], 0);
            }
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        Index gamma_I = gamma[0] - rho[0];
        if (gamma_I > 0)
        {
            veccp(gamma_I, v_Hh[0], 0, v_HhIt[0], 0);
            PlI[0].apply(rankI, &v_HhIt[0].vec(), 0);
            trsv_utu(rankI, HhIt[0], 0, 0, v_HhIt[0], 0, v_HhIt[0], 0);
            gemv_t(rankI, gamma_I - rankI, -1.0, HhIt[0], 0, rankI, v_HhIt[0], 0, 1.0, v_HhIt[0],
                   rankI, v_HhIt[0], rankI);
            veccpsc(rankI, -1.0, v_HhIt[0], 0, v_GgIt_tilde[0], 0);
            trsv_ltn(rankI, HhIt[0], 0, 0, v_GgIt_tilde[0], 0, v_GgIt_tilde[0], 0);
            PrI[0].apply(rankI, &v_Ppt[0].vec(), 0);
            veccp(nx, v_Ppt[0], 0, v_GgLIt[0], 0);
            gemv_n(nx, rankI, 1.0, Ppt[0], 0, 0, v_GgIt_tilde[0], 0, 1.0, v_GgLIt[0], 0, v_GgLIt[0],
                   0);
            gemv_n(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, v_GgLIt[0], 0, 1.0, v_GgLIt[0],
                   rankI, v_PpIt_hat[0], 0);
            trsv_lnn(nx - rankI, LlIt[0], 0, 0, v_PpIt_hat[0], 0, v_LlIt[0], 0);
        }
        else
        {
            trsv_lnn(nx, LlIt[0], 0, 0, v_Ppt[0], 0, v_LlIt[0], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_u = info.offsets_primal_u[0];
        const Index offs_x = info.offsets_primal_x[0];
        const Index offs_g = info.offsets_g_eq_path[0];
        veccpsc(nx - rankI, -1.0, v_LlIt[0], 0, x, offs_x + rankI);
        trsv_ltn(nx - rankI, LlIt[0], 0, 0, x, offs_x + rankI, x, offs_x + rankI);
        veccp(rankI, v_GgIt_tilde[0], 0, x, offs_x);
        gemv_t(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, x, offs_x + rankI, 1.0, x, offs_x, x,
               offs_x);
        veccpsc(rankI, -1.0, v_Ppt[0], 0, eq_mult, offs_g);
        gemv_t(nx, rankI, -1.0, Ppt[0], 0, 0, x, nu, 1.0, eq_mult, offs_g, eq_mult, offs_g);
        trsv_lnn(rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        trsv_unu(rankI, rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        PlI[0].apply_inverse(rankI, &eq_mult.vec(), offs_g);
        PrI[0].apply_inverse(rankI, &x.vec(), offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {

        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index rho_k = rho[k];
        const Index numrho_k = nu - rho_k;
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index gammamrho_k = gamma[k] - rho[k];
        const Index gamma_k = gamma[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (numrho_k > 0)
        {
            veccpsc(numrho_k, -1.0, v_Llt[k], 0, x, offs + rho_k);
            if (increased_accuracy)
            {
                gemv_n(nu - rho_k, gamma_k - rho_k, -1.0, Ggt_tilde[k], 0, rho_k, eq_mult, offs_g_k,
                       1.0, x, offs + rho_k, x, offs + rho_k);
            }
            gemv_t(nx, numrho_k, -1.0, Llt[k], numrho_k, 0, x, offs_x, 1.0, x, offs + rho_k, x,
                   offs + rho_k);
            trsv_ltn(numrho_k, Llt[k], 0, 0, x, offs + rho_k, x, offs + rho_k);
        }
        //         /// calcualate uka_tilde
        if (rho_k > 0)
        {
            // ROWEX(rho_k, 1.0, Ggt_tilde[k], numrho_k + nx, 0, ux[0], offs);
            veccp(rho_k, v_Ggt_tilde[k], 0, x, offs);
            gemv_t(nx + numrho_k, rho_k, 1.0, Ggt_tilde[k], 0, 0, x, offs + rho_k, 1.0, x, offs, x,
                   offs);
            veccp(gammamrho_k, eq_mult, offs_g_k, v_tmp[0], 0);
            veccp(gammamrho_k, v_tmp[0], 0, eq_mult, offs_g_k + rho_k);
            veccpsc(rho_k, -1.0, v_RSQrqt_tilde[k], 0, eq_mult, offs_g_k);
            gemv_t(nu + nx, rho_k, -1.0, RSQrqt_tilde[k], 0, 0, x, offs, 1.0, eq_mult, offs_g_k,
                   eq_mult, offs_g_k);
            gecp(rho_k, gamma_k, Ggt_tilde[k], nu - rho_k + nx + 1, 0, AL[0], 0, 0);
            trsv_lnn(rho_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            trsv_unu(rho_k, gamma_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            Pl[k].apply_inverse(rho_k, &eq_mult.vec(), offs_g_k);
            Pr[k].apply_inverse(rho_k, &x.vec(), offs);
        }
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        }
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offsxp1 = info.offsets_primal_x[k + 1];
            const Index offs_g_kp1 = info.offsets_g_eq_path[k + 1];
            const Index offs_dyn_k = info.offsets_g_eq_dyn[k];
            const Index gammamrho_kp1 = gamma[k + 1] - rho[k + 1];
            veccp(nxp1, g, offs_dyn_k, x, offsxp1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offsxp1, x,
                   offsxp1);
            veccp(nxp1, v_Ppt[k + 1], 0, eq_mult, offs_dyn_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offsxp1, 1.0, eq_mult, offs_dyn_k, eq_mult,
                   offs_dyn_k);
            gemv_t(gammamrho_kp1, nxp1, 1.0, Hh[k + 1], 0, 0, eq_mult, offs_g_kp1, 1.0, eq_mult,
                   offs_dyn_k, eq_mult, offs_dyn_k);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}
LinsolReturnFlag AugSystemSolver<OcpType>::solve_rhs(const ProblemInfo &info,
                                               const Jacobian<OcpType> &jacobian,
                                               const Hessian<OcpType> &hessian,
                                               const VecRealView &D_eq, const VecRealView &D_s,
                                               const VecRealView &f, const VecRealView &g,
                                               VecRealView &x, VecRealView &eq_mult)
{
    VecRealView *v_RSQrq_hat_curr_p;
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index offs_ux_k = info.offsets_primal_u[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_g_dyn = info.offsets_g_eq_dyn[k];
        const Index offs_g_eq = info.offsets_g_eq_path[k];
        const Index offs_ge_eq_ineq = info.offsets_g_eq_slack[k];
        //     //////// SUBSDYN
        if (k == info.dims.K - 1)
        {
            veccp(nu + nx, f, offs_ux_k, v_RSQrqt_tilde[k], 0);
        }
        else
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            gemv_n(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, g, offs_g_dyn, 0.0, v_AL[0], 0, v_AL[0], 0);
            axpy(nxp1, 1.0, v_Ppt[k + 1], 0, v_AL[0], 0, v_AL[0], 0);
            gemv_n(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, v_AL[0], 0, 1.0, f, offs_ux_k,
                   v_RSQrqt_tilde[k], 0);
        }
        if (ng > 0)
        {
            const Index offs_eq_k = info.offsets_eq[k];
            for (Index i = 0; i < ng; i++)
            {
                Scalar scaling_factor = D_eq(offs_eq_k + i);
                v_Ggt_stripe[0](i) = g(offs_g_eq + i) / scaling_factor;
            }
            gemv_n(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, v_Ggt_stripe[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        if (ng_ineq > 0)
        {
            const Index offs_ineq_k = info.offsets_slack[k];
            for (Index i = 0; i < ng_ineq; i++)
            {
                Scalar scaling_factor = D_s(offs_ineq_k + i);
                v_Ggt_ineq_temp[0](i) = g(offs_ge_eq_ineq + i) / scaling_factor;
            }
            gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, v_Ggt_ineq_temp[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        {
            v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
        }
        {
            trsv_lnn(nu, Llt[k], 0, 0, *v_RSQrq_hat_curr_p, 0, v_Llt[k], 0);
            veccp(nu, v_Llt[k], 0, v_Llt_shift[0], 0);
            gemv_n(nx, nu, -1.0, Llt[k], nu, 0, v_Llt_shift[0], 0, 1.0, v_RSQrqt_tilde[k], nu,
                   v_Ppt[k], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        {
            trsv_lnn(nx, LlIt[0], 0, 0, v_Ppt[0], 0, v_LlIt[0], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_x = info.offsets_primal_x[0];
        veccpsc(nx, -1.0, v_LlIt[0], 0, x, offs_x);
        trsv_ltn(nx, LlIt[0], 0, 0, x, offs_x, x, offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
        veccpsc(nu, -1.0, v_Llt[k], 0, x, offs);
        gemv_t(nx, nu, -1.0, Llt[k], nu, 0, x, offs_x, 1.0, x, offs, x, offs);
        trsv_ltn(nu, Llt[k], 0, 0, x, offs, x, offs);
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offs_x_p1 = info.offsets_primal_x[k + 1];
            veccp(nxp1, g, offs_dyn_eq_k, x, offs_x_p1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offs_x_p1, x,
                   offs_x_p1);
            veccp(nxp1, v_Ppt[k + 1], 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offs_x_p1, 1.0, eq_mult,
                   offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
        }
    }
    // // calculate lam_eq xk
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index offs_eq = info.offsets_eq[k];
        if (ng > 0)
        {
            gemv_t(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, x, offs, 1.0, g, offs_g_k,
                   eq_mult, offs_g_k);
            eq_mult.block(ng, offs_g_k) =
                eq_mult.block(ng, offs_g_k) / D_eq.block(ng, offs_eq);
        }
    }

    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_gineq_k = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_gineq_k,
                   eq_mult, offs_gineq_k);
            eq_mult.block(ng_ineq, offs_gineq_k) =
                eq_mult.block(ng_ineq, offs_gineq_k) / D_s.block(ng_ineq, offs_slack);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}













































void PrintNpArray(MatRealAllocated const &A, std::string name, int m=-1, int n=-1, bool with_name=true, std::ostream& o = std::cout){
    if (m < 0){m = A.m();}
    if (n < 0){n = A.n();}

    if (with_name){
        o << name << " = np.array([\n\t";
    } else {
        o << "np.array([\n\t";
    }

    if (m == 0){ o << "[]";}
    else{
        for (int i = 0; i < m; i++){
            o << "[";
            for (int j = 0; j < n; j++){
                o << std::setw(10) << std::setprecision(10) << A(i,j);
                if (j < n - 1){ o << ",";}
                o << " ";
            }
            o << "],\n\t";        
        }
    }
    o << "])" << std::endl;
}

void PrintNpArray(VecRealAllocated const &v, std::string name){
    std::cout << name << " = np.transpose(np.array([[";
    for (int i = 0; i < v.m(); i++){
        std::cout << v(i);
        if (i < v.m() - 1){ std::cout << ",";}
        std::cout << " ";
    }
    std::cout << "]]))" << std::endl;
}

void PrintNpArray(VecRealAllocated const &v, int offset, int length, std::string name){
    std::cout << name << " = np.transpose(np.array([[";
    for (int i = 0; i < length; i++){
        std::cout << v(offset + i);
        if (i < length - 1){ std::cout << ",";}
        std::cout << " ";
    }
    std::cout << "]]))" << std::endl;
}

MatRealAllocated PermutationVectorToMatrix(PermutationMatrix &P){
    MatRealAllocated result(P.size(), P.size());
    for (int i = 0; i < P.size(); i++){
        result(i, i) = 1.0;
    }
    P.apply_on_cols(P.size(), &result.mat());
    return result;
}

void PrintPreProcessNpInfo(const ProblemInfo &info, 
                           const ProblemInfo & modified_info,
                           const Hessian<ImplicitOcpType> &hess, 
                           const Jacobian<ImplicitOcpType> &jacobian,
                           const VecRealView &x, const VecRealView &eq_mult,
                           const VecRealView &D_x){
    // Print dimensions
    // std::ostream& o = std::cout;
    std::string filename = "preprocess_info.py";
    std::ofstream o(filename);
    o << "import numpy as np\n";

    // o << "==============================================================\n";
    o << "K = " << info.dims.K << "\n";
    o << "nu = [";
    std::vector<int> nu = {};
    for (int k = 0; k < info.dims.K; k++){
        o << info.dims.number_of_controls[k];
        if (k < info.dims.K - 1){ o << ", ";}
        nu.push_back(info.dims.number_of_controls[k]);
    }
    o << "]\n";
    o << "nx = [";
    std::vector<int> nx = {};
    for (int k = 0; k < info.dims.K; k++){
        o << info.dims.number_of_states[k];
        if (k < info.dims.K - 1){ o << ", ";}
        nx.push_back(info.dims.number_of_states[k]);
    }
    o << "]\n";
    o << "ng_ineq = [";
    std::vector<int> ng_ineq = {};
    for (int k = 0; k < info.dims.K; k++){
        o << info.dims.number_of_ineq_constraints[k];
        if (k < info.dims.K - 1){ o << ", ";}
        ng_ineq.push_back(info.dims.number_of_ineq_constraints[k]);
    }
    o << "]\n";
    o << "ng_eq = [";
    std::vector<int> ng_eq = {};
    for (int k = 0; k < info.dims.K; k++){
        o << info.dims.number_of_eq_constraints[k];
        if (k < info.dims.K - 1){ o << ", ";}
        ng_eq.push_back(info.dims.number_of_eq_constraints[k]);
    }
    o << "]\n";
    o << "r = [";
    std::vector<int> r = {};
    for (int k = 0; k < info.dims.K - 1; k++){
        o << jacobian.J_ranks[k];
        if (k < info.dims.K - 2){ o << ", ";}
        r.push_back(jacobian.J_ranks[k]);
    }
    o << "]\n";
    
    o << "D_x = [";
    for (int k = 0; k < info.dims.K; k++){
        o << "[";
        int n = info.dims.number_of_controls[k] + info.dims.number_of_states[k];
        for (int i = 0; i < n; i++){
            o << D_x(info.offsets_primal_u[k] + i);
            if (i < n - 1){ o << ", ";}
        }
        o << "]";
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";

    // modified info
    o << "modified_K = " << modified_info.dims.K << "\n";
    o << "modified_nu = [";
    std::vector<int> modified_nu = {};
    for (int k = 0; k < modified_info.dims.K; k++){
        o << modified_info.dims.number_of_controls[k];
        if (k < modified_info.dims.K - 1){ o << ", ";}
        modified_nu.push_back(modified_info.dims.number_of_controls[k]);
    }
    o << "]\n";
    o << "modified_nx = [";
    std::vector<int> modified_nx = {};
    for (int k = 0; k < modified_info.dims.K; k++){
        o << modified_info.dims.number_of_states[k];
        if (k < modified_info.dims.K - 1){ o << ", ";}
        modified_nx.push_back(modified_info.dims.number_of_states[k]);
    }
    o << "]\n";
    o << "modified_ng_ineq = [";
    std::vector<int> modified_ng_ineq = {};
    for (int k = 0; k < modified_info.dims.K; k++){
        o << modified_info.dims.number_of_ineq_constraints[k];
        if (k < modified_info.dims.K - 1){ o << ", ";}
        modified_ng_ineq.push_back(modified_info.dims.number_of_ineq_constraints[k]);
    }
    o << "]\n";
    o << "modified_ng_eq = [";
    std::vector<int> modified_ng_eq = {};
    for (int k = 0; k < modified_info.dims.K; k++){
        o << modified_info.dims.number_of_eq_constraints[k];
        if (k < modified_info.dims.K - 1){ o << ", ";}
        modified_ng_eq.push_back(modified_info.dims.number_of_eq_constraints[k]);
    }
    o << "]\n";

    // hessian attributes
    o << "RSQrqt = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(hess.RSQrqt[k], "", modified_nu[k] + modified_nx[k] + 1, modified_nu[k] + modified_nx[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "RSQrqt_original = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(hess.RSQrqt_original[k], "", nu[k] + nx[k] + 1, nu[k] + nx[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "FuFx" << " = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(hess.FuFx[k], "", modified_nu[k] + modified_nx[k], modified_nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";
    o << "FuFx_original" << " = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(hess.FuFx_original[k], "", nu[k] + nx[k], nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";
    o << "GuGx" << " = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(hess.GuGx[k], "", modified_nu[k] + modified_nx[k], modified_nu[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";
    o << "GuGx_original" << " = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(hess.GuGx_original[k], "", nu[k] + nx[k], nu[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";

    // jacobian attributes
    o << "Gg_eqt = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(jacobian.Gg_eqt[k], "",  modified_nu[k] + modified_nx[k] + 1, modified_ng_eq[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "Gg_eqt_original = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(jacobian.Gg_eqt_original[k], "", nu[k] + nx[k] + 1, ng_eq[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "Gg_ineqt = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(jacobian.Gg_ineqt[k], "", modified_nu[k] + modified_nx[k] + 1, modified_ng_ineq[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "Gg_ineqt_original = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(jacobian.Gg_ineqt_original[k], "", nu[k] + nx[k] + 1, ng_ineq[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "BAbt = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(jacobian.BAbt[k], "", modified_nu[k] + modified_nx[k] + 1, modified_nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";
    o << "BAbt_original = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(jacobian.BAbt_original[k], "", nu[k] + nx[k] + 1, nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";

    std::vector<MatRealAllocated> Jt_LU;
    Jt_LU.reserve(info.dims.K - 1);
    std::vector<PermutationMatrix> Pl;
    Pl.reserve(info.dims.K - 1);
    std::vector<PermutationMatrix> Pr;
    Pr.reserve(info.dims.K - 1);
    
    o << "Jt = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(jacobian.Jt[k], "", nx[k+1], nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
        Jt_LU.push_back(jacobian.Jt[k]);
        Pl.push_back(PermutationMatrix(nx[k+1]));
        Pr.push_back(PermutationMatrix(nx[k+1]));
        int rho;
        lu_fact_transposed(nx[k+1], nx[k+1], nx[k+1], rho, Jt_LU[k], Pl[k], Pr[k]);
        if (rho != jacobian.J_ranks[k]){
            std::cerr << "Error: LU factorization rank does not match J_ranks." << std::endl;
        }
    }
    o << "]\n";

    // LU info
    o << "L = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        o << "np.array([";
        int nx_next = info.dims.number_of_states[k+1];
        if (nx_next == 0){ o << "[]";}
        for (int i = 0; i < nx_next; i++){
            o << "[";
            for (int j = 0; j < nx_next; j++){
                if (i > j){
                    // o << jacobian.Jt_LU[k](j,i);
                    o << std::setw(10) << std::setprecision(10) << Jt_LU[k](j,i);
                } else if (i == j){
                    o << 1.0;
                } else {
                    o << 0.0;
                }
                if (j < nx_next - 1){
                    o << ", ";
                }
            }
            o << "]";
            if (i < nx_next){
                o << ",\n";
            }
        }
        o << "])\n";
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";

    o << "U = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        o << "np.array([";
        int nx_next = info.dims.number_of_states[k+1];
        if (nx_next == 0){ o << "[]";}
        for (int i = 0; i < nx_next; i++){
            o << "[";
            for (int j = 0; j < nx_next; j++){
                if (i <= j){
                    // o << jacobian.Jt_LU[k](j,i);
                    o << std::setw(10) << std::setprecision(10) << Jt_LU[k](j,i);
                } else {
                    o << 0.0;
                }
                if (j < nx_next - 1){
                    o << ", ";
                }
            }
            o << "]";
            if (i < nx_next){
                o << ",\n";
            }
        }
        o << "])\n";
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";

    o << "Pl = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PermutationMatrix Pl_copy = jacobian.Pl_pre[k];
        PrintNpArray(PermutationVectorToMatrix(Pl_copy), "", nx[k+1], nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";
    o << "Pr = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PermutationMatrix Pr_copy = jacobian.Pr_pre[k];
        PrintNpArray(PermutationVectorToMatrix(Pr_copy), "", nx[k+1], nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";

    o << "x = np.array([";
    for (int i = 0; i < info.number_of_primal_variables; i++){
        o << std::setw(10) << std::setprecision(10) << x(i);
        if (i < info.number_of_primal_variables - 1){
            o << ", ";
        }
    }
    o << "])\n";
    o << "eq_mult = np.array([";
    for (int i = 0; i < info.number_of_eq_constraints; i++){
        o << std::setw(10) << std::setprecision(10) << eq_mult(i);
        if (i < info.number_of_eq_constraints - 1){
            o << ", ";
        }
    }
    o << "])\n";
    // o << "==============================================================\n";
    o.close();
}

void PrintFactorizationInfo(const ProblemInfo &info,
                            const std::vector<Index>& rank_k_values,
                            std::vector<PermutationMatrix>& Pl,
                            std::vector<PermutationMatrix>& Pr,
                            std::vector<MatRealAllocated>& LU,
                            std::vector<Index>& gamma_k_values,
                            std::vector<MatRealAllocated>& Gg_eqt,
                            std::vector<MatRealAllocated>& Llt,
                            std::vector<MatRealAllocated>& R_shur){
    std::string filename = "factorization_info.py";

    // std::ostream& o = std::cout;
    std::ofstream o(filename);
    o << "import numpy as np\n";

    // o << "==============================================================\n";

    o << "rank_k_values = [";
    for (size_t i = 0; i < rank_k_values.size(); i++){
        o << rank_k_values[i];
        if (i < rank_k_values.size() - 1){
            o << ", ";
        }
    }
    o << "]\n";

    o << "Pl_r = [\n";
    for (size_t i = 0; i < Pl.size(); i++){
        PermutationMatrix Pl_copy = Pl[i];
        PrintNpArray(PermutationVectorToMatrix(Pl_copy), "", gamma_k_values[i], 
            gamma_k_values[i], false, o);
        if (i < Pl.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    o << "Pr_r = [\n";
    for (size_t i = 0; i < Pr.size(); i++){
        PermutationMatrix Pr_copy = Pr[i];
        PrintNpArray(PermutationVectorToMatrix(Pr_copy), "", info.dims.number_of_controls[i], info.dims.number_of_controls[i], false, o);
        if (i < Pr.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    // LU factorization
    o << "L_r = [\n";
    for (int k = 0; k < LU.size(); k++){
        int m = gamma_k_values[k];
        o << "np.array([";
        int n = info.dims.number_of_controls[k];
        if (m == 0){ o << "[]";}
        for (int i = 0; i < m; i++){
            o << "[";
            for (int j = 0; j < m; j++){
                if (i > j){
                    o << std::setw(10) << std::setprecision(10) << LU[k](j,i);
                } else if (i == j){
                    o << 1.0;
                } else {
                    o << 0.0;
                }
                if (j < m - 1){
                    o << ", ";
                }
            }
            o << "]";
            if (i < m){
                o << ",\n";
            }
        }
        o << "])\n";
        if (k < LU.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    o << "U_r = [\n";
    for (int k = 0; k < LU.size(); k++){
        int m = gamma_k_values[k];
        o << "np.array([";
        int n = info.dims.number_of_controls[k];
        if (m == 0){ o << "[]";}
        for (int i = 0; i < m; i++){
            o << "[";
            for (int j = 0; j < n; j++){
                if (i <= j){
                    o << std::setw(10) << std::setprecision(10) << LU[k](j,i);
                } else {
                    o << 0.0;
                }
                if (j < n - 1){
                    o << ", ";
                }
            }
            o << "]";
            if (i < m){
                o << ",\n";
            }
        }
        o << "])\n";
        if (k < LU.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    o << "Hut = [\n";
    for (size_t i = 0; i < Gg_eqt.size(); i++){
        PrintNpArray(Gg_eqt[i], "", info.dims.number_of_controls[i], gamma_k_values[i], false, o);
        if (i < Gg_eqt.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    // shur step
    o << "Lmbd = [\n";
    for (size_t i = 0; i < Llt.size(); i++){
        int n = info.dims.number_of_controls[i] - rank_k_values[i];
        PrintNpArray(Llt[i], "", n, n, false, o);
        if (i < Llt.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    o << "R_shur = [\n";
    for (size_t i = 0; i < R_shur.size(); i++){
        int m = info.dims.number_of_controls[i] - rank_k_values[i];
        int n = info.dims.number_of_controls[i] - rank_k_values[i];
        PrintNpArray(R_shur[i], "", m, n, false, o);
        if (i < R_shur.size() - 1){ o << ",\n";}
    }
    o << "]\n";
    // o << "==============================================================\n";
    o.close();
}


ModifiedAugSystemSolver::ModifiedAugSystemSolver(const ProblemInfo &info)
{
    std::vector<Index> number_of_controls = info.dims.number_of_controls;
    std::vector<Index> number_of_states = info.dims.number_of_states;
    std::vector<Index> number_of_ineq_constraints = info.dims.number_of_ineq_constraints;
    std::vector<Index> number_of_eq_constraints = info.dims.number_of_eq_constraints;

    for (Index i = 1; i < number_of_controls.size(); i++)
    {
        number_of_controls[i] += info.dims.number_of_states[i];
    }
    for (Index i = 0; i < number_of_eq_constraints.size()-1; i++)
    {
        // There can be no more constraints than number of controls and states.
        // Otherwise, the constraints can not be satisfied (assuming the 
        // jacobian is not degenerate)
        // The preprocessing should check the rank of the J-matrix to make
        // sure we don't end up in this case.
        number_of_eq_constraints[i] = std::min(
            number_of_eq_constraints[i] + info.dims.number_of_states[i+1],
            number_of_controls[i] + number_of_states[i]);
    }
    ProblemInfo new_info = ProblemInfo(ProblemDims(info.dims.K, number_of_controls, number_of_states, number_of_eq_constraints,
                       number_of_ineq_constraints));
    

    Index max_number_of_controls =
        *std::max_element(new_info.dims.number_of_controls.begin(), new_info.dims.number_of_controls.end());
    Index max_number_of_states =
        *std::max_element(new_info.dims.number_of_states.begin(), new_info.dims.number_of_states.end());
    Index max_number_of_variables = *std::max_element(new_info.number_of_stage_variables.begin(),
                                                      new_info.number_of_stage_variables.end());
    Index max_number_of_ineq_constraints = *std::max_element(
        new_info.dims.number_of_ineq_constraints.begin(), new_info.dims.number_of_ineq_constraints.end());
    Index max_number_of_eq_consttraints = *std::max_element(
        new_info.dims.number_of_eq_constraints.begin(), new_info.dims.number_of_eq_constraints.end());

    AL.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    Ggt_stripe.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    GgLt.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    RSQrqt_hat.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    Llt_shift.emplace_back(max_number_of_variables + 1, max_number_of_controls);
    GgIt_tilde.emplace_back(new_info.dims.number_of_states[0] + 1, new_info.dims.number_of_states[0]);
    GgLIt.emplace_back(new_info.dims.number_of_states[0] + 1, new_info.dims.number_of_states[0]);
    HhIt.emplace_back(new_info.dims.number_of_states[0] + 1, new_info.dims.number_of_states[0]);
    PpIt_hat.emplace_back(new_info.dims.number_of_states[0] + 1, new_info.dims.number_of_states[0]);
    LlIt.emplace_back(new_info.dims.number_of_states[0] + 1, new_info.dims.number_of_states[0]);
    Ggt_ineq_temp.emplace_back(max_number_of_variables + 1, max_number_of_ineq_constraints);

    FuFx_underbar.reserve(new_info.dims.K-1);
    GuGx_tilde.reserve(new_info.dims.K-1);
    GuGx_hat.reserve(new_info.dims.K-1);
    RSQrqt_underbar.reserve(new_info.dims.K);
    Ppt.reserve(new_info.dims.K);
    Hh.reserve(new_info.dims.K);
    RSQrqt_tilde.reserve(new_info.dims.K);
    Ggt_tilde.reserve(new_info.dims.K);
    Llt.reserve(new_info.dims.K);
    for (Index k = 0; k < new_info.dims.K; k++)
    {
        Index nu = new_info.dims.number_of_controls[k];
        Index nx = new_info.dims.number_of_states[k];
        Index ng_ineq = new_info.dims.number_of_ineq_constraints[k];
        Index ng_eq = new_info.dims.number_of_eq_constraints[k];
        Ppt.emplace_back(nx + 1, nx);
        Hh.emplace_back(nx, nx + 1);
        RSQrqt_underbar.emplace_back(nu + nx + 1, nx + nu);
        RSQrqt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Ggt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Llt.emplace_back(nu + nx + 1, nu);
        if (k < new_info.dims.K - 1){
            Index nu_next = new_info.dims.number_of_controls[k+1];
            Index nx_next = new_info.dims.number_of_states[k+1];
            FuFx_underbar.emplace_back(nu + nx, nx_next);
            GuGx_tilde.emplace_back(nu + nx, nu_next);
            GuGx_hat.emplace_back(nu + nx, nu_next);
        }
    }

    v_r_tilde.emplace_back(1, max_number_of_variables);
    v_AL.emplace_back(max_number_of_variables);
    v_Ggt_stripe.emplace_back(max_number_of_variables);
    v_GgLt.emplace_back(max_number_of_variables);
    v_RSQrqt_hat.emplace_back(max_number_of_variables);
    v_Llt_shift.emplace_back(max_number_of_controls);
    v_GgIt_tilde.emplace_back(new_info.dims.number_of_states[0]);
    v_GgLIt.emplace_back(new_info.dims.number_of_states[0]);
    v_HhIt.emplace_back(new_info.dims.number_of_states[0]);
    v_PpIt_hat.emplace_back(new_info.dims.number_of_states[0]);
    v_LlIt.emplace_back(new_info.dims.number_of_states[0]);
    v_Ggt_ineq_temp.emplace_back(max_number_of_ineq_constraints);
    v_tmp.emplace_back(max_number_of_variables);

    v_Ppt.reserve(new_info.dims.K);
    v_Hh.reserve(new_info.dims.K);
    v_RSQrqt_tilde.reserve(new_info.dims.K);
    v_Ggt_tilde.reserve(new_info.dims.K);
    v_Llt.reserve(new_info.dims.K);

    for (Index k = 0; k < new_info.dims.K; k++)
    {
        Index nu = new_info.dims.number_of_controls[k];
        Index nx = new_info.dims.number_of_states[k];
        v_Ppt.emplace_back(nx);
        v_Hh.emplace_back(nx);
        v_RSQrqt_tilde.emplace_back(nu + nx);
        v_Ggt_tilde.emplace_back(nu + nx);
        v_Llt.emplace_back(nu + nx);
    }

    PlI.emplace_back(new_info.dims.number_of_states[0]);
    PrI.emplace_back(new_info.dims.number_of_states[0]);

    Pl.reserve(new_info.dims.K);
    Pr.reserve(new_info.dims.K);

    for (Index k = 0; k < new_info.dims.K; k++)
    {
        Index nu = new_info.dims.number_of_controls[k];
        Index nx = new_info.dims.number_of_states[k];
        Pl.emplace_back(max_number_of_controls);
        Pr.emplace_back(max_number_of_controls);
    }

    gamma.resize(new_info.dims.K);
    rho.resize(new_info.dims.K);

    // for debugging
    if (write_factorization_file){
    rank_k_values = std::vector<Index>(info.dims.K);
    LU.reserve(info.dims.K);
    for (Index k = 0; k < info.dims.K; k++){
        LU.emplace_back(max_number_of_controls, max_number_of_eq_consttraints);
    }
    gamma_k_values = std::vector<Index>(info.dims.K);
    Ggt_eq.reserve(info.dims.K);
    for (Index k = 0; k < info.dims.K; k++){ 
        Ggt_eq.emplace_back(max_number_of_controls + max_number_of_states + 1, max_number_of_eq_consttraints);
    }
    R_shur.reserve(info.dims.K);
    for (Index k = 0; k < info.dims.K; k++){ 
        R_shur.emplace_back(max_number_of_states, max_number_of_states);
    }
    }
};

LinsolReturnFlag ModifiedAugSystemSolver::solve(const ProblemInfo &info,
                                           Jacobian<ImplicitOcpType> &jacobian, Hessian<ImplicitOcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_s,
                                           const VecRealView &f, const VecRealView &g,
                                           VecRealView &x, VecRealView &eq_mult)
{
    MatRealView *RSQrq_hat_curr_p;
    Index rank_k;

    /////////////// recursion ///////////////
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offset_ineq_k = info.offsets_slack[k];
        const Index offset_u = info.offsets_primal_u[k];
        const Index offset_eq_path = info.offsets_g_eq_path[k];
        const Index offset_eq_slack = info.offsets_g_eq_slack[k];
        const Index nunxm1 = (k > 0) ? info.dims.number_of_controls[k-1] + info.dims.number_of_states[k-1] : 0;

        if (k == info.dims.K - 1){
            if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            gecp(nx + nu + 1, nu + nx, hessian.RSQrqt[k], 0, 0, RSQrqt_underbar[k], 0, 0);
            if (k > 0){
                gecp(nunxm1, nx, hessian.FuFx[k-1], 0, 0, FuFx_underbar[k-1], 0, 0);
            }
        }
        if (k > 0){
            if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
            gecp(nunxm1 + 1, nunxm1, hessian.RSQrqt[k-1], 0, 0, RSQrqt_underbar[k-1], 0, 0);
        }
        // std::cout << "\n\nk = " << k << std::endl;
        // PrintNpArray(RSQrqt_underbar[0], "RSQrqt_underbar[0]");
        // PrintNpArray(RSQrqt_underbar[1], "RSQrqt_underbar[1]");
        // std::cout << " === SUBSDYN === " << std::endl; 
        //////// SUBSDYN
        Index gamma_k;
        if (k == info.dims.K - 1)
        {
            gamma_k = ng;
            gamma[k] = gamma_k;
            rowin(ng, 1.0, g, offset_eq_path, jacobian.Gg_eqt[k], nu + nx, 0);
            gecp(nx + nu + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
            if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
            gecp(nx + nu + 1, nu + nx, RSQrqt_underbar[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        else
        {
            const Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index Hp1_size = gamma[k + 1] - rho[k + 1];
            if (Hp1_size + ng > nu + nx){
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                return LinsolReturnFlag::NOFULL_RANK;
            }
            gamma_k = Hp1_size + ng;
            // AL <- [BAb]^T_k P_kp1
            rowin(nxp1, 1.0, g, offset_eq_dyn, jacobian.BAbt[k], nu + nx, 0);
            gemm_nt(nu + nx + 1, nxp1, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Ppt[k + 1], 0, 0, 0.0,
                    AL[0], 0, 0, AL[0], 0, 0);
            // AL[-1,:] <- AL[-1,:] + p_kp1^T
            gead(1, nxp1, 1.0, Ppt[k + 1], nxp1, 0, AL[0], nx + nu, 0);
            // RSQrqt_stripe <- AL[BA] + RSQrqt
            syrk_ln_mn(nu + nx + 1, nu + nx, nxp1, 1.0, AL[0], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                       RSQrqt_underbar[k], 0, 0, RSQrqt_tilde[k], 0, 0);

            // Add second order dynamics contribution
            // std::cout << "--------------------------------------------" << std::endl;
            // std::cout << "test FuFx addition to RSQrqt" << std::endl;
            // PrintNpArray(jacobian.BAbt[k], "BAbt");
            // PrintNpArray(FuFx_underbar[k], "FuFx");
            // PrintNpArray(hessian.FuFx[k], "FuFx_hessian");
            // PrintNpArray(RSQrqt_tilde[k], "RSQrqt_tilde");
            // std::cout << "nu = " << nu << std::endl;
            // std::cout << "nx = " << nx << std::endl;
            // std::cout << "nx_next = " << nxp1 << std::endl;
            gemm_nt(nu + nx + 1, nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, FuFx_underbar[k], 0, 0, 1.0,
                    RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
            // PrintNpArray(RSQrqt_tilde[k], "RSQrqt_intermediate");
            gemm_nt(nu + nx, nu + nx, nxp1, 1.0, FuFx_underbar[k], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                    RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
            // PrintNpArray(RSQrqt_tilde[k], "RSQrqt_tilde_after");
            // std::cout << "--------------------------------------------" << std::endl;

            //// inequalities
            gamma[k] = gamma_k;
            // if ng[k]>0
            if (gamma_k > 0)
            {
                // if Gk nonempty
                if (ng > 0)
                {
                    // Ggt_stripe  <- Ggt_k
                    rowin(ng, 1.0, g, offset_eq_path, jacobian.Gg_eqt[k], nu + nx, 0);
                    gecp(nu + nx + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
                }
                // if Hkp1 nonempty
                if (Hp1_size > 0)
                {
                    // Ggt_stripe <- [Ggt_k [BAb_k^T]H_kp1]
                    gemm_nt(nu + nx + 1, Hp1_size, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Hh[k + 1], 0,
                            0, 0.0, Ggt_stripe[0], 0, ng, Ggt_stripe[0], 0, ng);
                    // Ggt_stripe[-1,ng:] <- Ggt_stripe[-1,ng:] + h_kp1^T
                    gead_transposed(1, Hp1_size, 1.0, Hh[k + 1], 0, nxp1, Ggt_stripe[0], nu + nx,
                                    ng);
                }
            }
            else
            {
                rho[k] = 0;
                rank_k = 0;
                RSQrq_hat_curr_p = &RSQrqt_tilde[k];
            }
        }
        // inequalities + inertia correction
        {
            // We've already covered this in pre-processing
            // if (ng_ineq > 0)
            // {
            //     rowin(ng_ineq, 1.0, g, offset_eq_slack, jacobian.Gg_ineqt[k], nu + nx, 0);
            //     gecp(nu + nx + 1, ng_ineq, jacobian.Gg_ineqt[k], 0, 0, Ggt_ineq_temp[0], 0, 0);
            //     for (Index i = 0; i < ng_ineq; i++)
            //     {
            //         Scalar scaling_factor = 1.0 / D_s(offset_ineq_k + i);
            //         colsc(nu + nx + 1, scaling_factor, Ggt_ineq_temp[0], 0, i);
            //     }
            //     // add the penalty
            //     syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, Ggt_ineq_temp[0], 0, 0,
            //                jacobian.Gg_ineqt[k], 0, 0, 1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k],
            //                0, 0);
            // }
            // // inertia correction
            // diaad(nu + nx, 1.0, D_x, offset_u, RSQrqt_tilde[k], 0, 0);
        }
        // PrintNpArray(RSQrqt_underbar[0], "RSQrqt_underbar[0]");
        // PrintNpArray(RSQrqt_underbar[1], "RSQrqt_underbar[1]");
        // std::cout << " === TRANSFORM AND SUBSEQ === " << std::endl;
        //////// TRANSFORM_AND_SUBSEQ
        {
            // symmetric transformation, done a little different than in paper, in order to fuse LA
            // operations LU_FACT_TRANSPOSE(Ggtstripe[:gamma_k, nu+nx+1], nu max) if(k==K-2)
            // blasfeo_print_dmat(1, gamma_k, Ggt_stripe[0], nu+nx, 0);
            // std::cout << "computing factorization" << std::endl;
            // PrintNpArray(Ggt_stripe[0], "Ggt_stripe before factorization");
            // std::cout << "(gamma_k = " << gamma_k << ")" << std::endl;
            if (write_factorization_file){
                gamma_k_values[k] = gamma_k;
                gecp(nu, gamma_k, Ggt_stripe[0], 0, 0, Ggt_eq[k], 0, 0);
            }
            lu_fact_transposed(gamma_k, nu + nx + 1, nu, rank_k, Ggt_stripe[0], Pl[k], Pr[k], lu_fact_tol);
            if (write_factorization_file){
                rank_k_values[k] = rank_k;
                gecp(nu, gamma_k, Ggt_stripe[0], 0, 0, LU[k], 0, 0);
            }
            // std::cout << "Pl: "; for (int i = 0; i < rank_k; i++){ std::cout << Pl[k][i] << " ";}std::cout << std::endl;
            // std::cout << "Pr: "; for (int i = 0; i < rank_k; i++){ std::cout << Pr[k][i] << " ";}std::cout << std::endl;
            // std::cout << "rank_" << k << " = " << rank_k << std::endl;
            
            rho[k] = rank_k;
            if (gamma_k - rank_k > 0)
            {
                // transfer eq's to next stage
                if (gamma_k - rank_k > nx){
                    if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                    return LinsolReturnFlag::NOFULL_RANK;
                }
                getr(nx + 1, gamma_k - rank_k, Ggt_stripe[0], nu, rank_k, Hh[k], 0, 0);
            }
            if (rank_k > 0)
            {
                // Ggt_tilde_k <- Ggt_stripe[rho_k:nu+nx+1, :rho] L-T (note that this is slightly
                // different from the implementation)
                trsm_rlnn(nu - rank_k + nx + 1, rank_k, -1.0, Ggt_stripe[0], 0, 0, Ggt_stripe[0],
                          rank_k, 0, Ggt_tilde[k], 0, 0);
                // the following command copies the top block matrix (LU) to the bottom because it
                // it needed later
                gecp(rank_k, gamma_k, Ggt_stripe[0], 0, 0, Ggt_tilde[k], nu - rank_k + nx + 1, 0);
                // permutations
                trtr_l(nu + nx, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0,
                       0); // copy lower part of RSQ to upper part
                Pr[k].apply_on_rows(rank_k, &RSQrqt_tilde[k].mat()); // TODO make use of symmetry
                Pr[k].apply_on_cols(rank_k, &RSQrqt_tilde[k].mat());
                // GL <- Ggt_tilde_k @ RSQ[:rho,:nu+nx] + RSQrqt[rho:nu+nx+1, rho:] (with
                // RSQ[:rho,:nu+nx] = RSQrqt[:nu+nx,:rho]^T) GEMM_NT(nu - rank_k + nx + 1, nu + nx,
                // rank_k, 1.0, Ggt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0, 1.0, RSQrqt_tilde_p
                // + k, rank_k, 0, GgLt[0], 0, 0); split up because valgrind was giving invalid read
                // errors when C matrix has nonzero row offset GgLt[0].print();
                gecp(nu - rank_k + nx + 1, nu + nx, RSQrqt_tilde[k], rank_k, 0, GgLt[0], 0, 0);
                gemm_nt(nu - rank_k + nx + 1, nu + nx, rank_k, 1.0, Ggt_tilde[k], 0, 0,
                        RSQrqt_tilde[k], 0, 0, 1.0, GgLt[0], 0, 0, GgLt[0], 0, 0);
                // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] +
                // GgLt[rank_k:, :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
                syrk_ln_mn(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt[0], 0, 0,
                           Ggt_tilde[k], 0, 0, 1.0, GgLt[0], 0, rank_k, RSQrqt_hat[0], 0, 0);
                // GEMM_NT(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt[0], 0, 0,
                // Ggt_tilde[k], 0, 0, 1.0, GgLt[0], 0, rank_k, RSQrqt_hat[0], 0, 0);
                RSQrq_hat_curr_p = &RSQrqt_hat[0];
            }
            else
            {
                RSQrq_hat_curr_p = &RSQrqt_tilde[k];
            }
            if (k > 0){
                // GuGx_tilde = GuGx * T_r^-1
                gecp(nunxm1, nu, hessian.GuGx[k-1], 0, 0, GuGx_tilde[k-1], 0, 0);
                // PrintNpArray(GuGx_tilde[k-1], "GuGx_tilde before permutation");
                Pr[k].apply_on_cols(rank_k, &GuGx_tilde[k-1].mat());
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                gemm_nt(nunxm1, nu - rank_k, rank_k, 1.0, GuGx_tilde[k-1], 0, 0, Ggt_tilde[k], 0, 0,
                        1.0, GuGx_tilde[k-1], 0, rank_k, GuGx_tilde[k-1], 0, rank_k);
                
                // FuFx = FuFx + GuGx * [Hx_tilde; 0]
                gecp(nunxm1, nx, hessian.FuFx[k-1], 0, 0, FuFx_underbar[k-1], 0, 0);
                // std::cout << "-----------------------------------" << std::endl;
                // std::cout << "testing FuFx and RSQrqt updates" << std::endl;
                // PrintNpArray(GuGx_tilde[k-1], "GuGx_tilde");
                // PrintNpArray(FuFx_underbar[k-1], "FuFx_underbar_before");
                // PrintNpArray(Ggt_tilde[k], "Ggt_tilde");
                // std::cout << "nu = " << nu << std::endl;
                // std::cout << "rank_k = " << rank_k << std::endl;
                // std::cout << "nunxm1 = " << nunxm1 << std::endl;
                // std::cout << "nx = " << nx << std::endl;
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}            
                gemm_nt(nunxm1, nx, rank_k, 1.0, GuGx_tilde[k-1], 0, 0, 
                        Ggt_tilde[k], nu - rank_k, 0, 1.0, FuFx_underbar[k-1], 0, 0, FuFx_underbar[k-1], 0, 0);
                // PrintNpArray(FuFx_underbar[k-1], "FuFx_underbar_after");

                // [r^T q^T] = [r^T q^T] + [h_tilde^T, 0] * GuGx_tilde^T
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                // PrintNpArray(Ggt_tilde[k], "Ggt_tilde");
                // PrintNpArray(RSQrqt_underbar[k-1], "RSQrqt_underbar_before");
                gemm_nt(1, nunxm1, rank_k, 1.0,  Ggt_tilde[k], nu - rank_k + nx, 0, 
                        GuGx_tilde[k-1], 0, 0, 1.0, RSQrqt_underbar[k-1], nunxm1, 0, RSQrqt_underbar[k-1], nunxm1, 0);
                // PrintNpArray(RSQrqt_underbar[k-1], "RSQrqt_underbar_after");
                // std::cout << "-----------------------------------" << std::endl;
            }
        }
        // PrintNpArray(RSQrqt_underbar[0], "RSQrqt_underbar[0]");
        // PrintNpArray(RSQrqt_underbar[1], "RSQrqt_underbar[1]");
        // std::cout << " === SCHUR === " << std::endl;
        //////// SCHUR
        {
            if (nu - rank_k > 0)
            {
                // DLlt_k = [chol(R_hatk); Llk@chol(R_hatk)^-T]
                // PrintNpArray(RSQrq_hat_curr_p[0], "RSQrq_hat");
                if (write_factorization_file){
                    gecp(nu-rank_k, nu-rank_k, RSQrq_hat_curr_p[0], 0, 0, R_shur[k], 0, 0);
                }
                potrf_l_mn(nu - rank_k + nx + 1, nu - rank_k, RSQrq_hat_curr_p[0], 0, 0, Llt[k], 0,
                           0);
                // PrintNpArray(Llt[k], "shur[" + std::to_string(k) + "]");
                // PrintNpArray(RSQrq_hat_curr_p[0], "RSQrq_hat");
                // PrintNpArray(Llt[k], "Llt");
                // PrintNpArray(GuGx_tilde[k-1], "GuGx_tilde");
                if (!check_reg(nu - rank_k, &Llt[k].mat(), 0, 0))
                    return LinsolReturnFlag::INDEFINITE;
                // Pp_k = Qq_hatk - L_k^T @ Ll_k
                // SYRK_LN_MN(nx+1, nx, nu-rank_k, -1.0,Llt_p+k, nu-rank_k,0, Llt_p+k,
                // nu-rank_k,0, 1.0, RSQrq_hat_curr[0], nu-rank_k, nu-rank_k,Pp+k,0,0); // feature
                // not implmented yet
                gecp(nx + 1, nu - rank_k, Llt[k], nu - rank_k, 0, Llt_shift[0], 0,
                     0); // needless operation because feature not implemented yet
                // SYRK_LN_MN(nx + 1, nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0,
                // 0, 1.0, RSQrq_hat_curr[0], nu - rank_k, nu - rank_k, Ppt[k], 0, 0);
                gecp(nx + 1, nx, RSQrq_hat_curr_p[0], nu - rank_k, nu - rank_k, Ppt[k], 0, 0);
                syrk_ln_mn(nx + 1, nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0, 0,
                           1.0, Ppt[k], 0, 0, Ppt[k], 0, 0);
                // next steps are for better accuracy
                if (increased_accuracy)
                {
                    // copy eta
                    getr(nu - rank_k, gamma_k - rank_k, Ggt_stripe[0], rank_k, rank_k,
                         Ggt_stripe[0], 0, 0);
                    // blasfeo_print_dmat(gamma_k-rank_k, nu-rank_k, Ggt_stripe[0], 0,0);
                    // eta L^-T
                    trsm_rltn(gamma_k - rank_k, nu - rank_k, 1.0, Llt[k], 0, 0, Ggt_stripe[0], 0, 0,
                              Ggt_stripe[0], 0, 0);
                    // ([S^T \\ r^T] L^-T) @ (L^-1 eta^T)
                    // (eta L^-T) @ ([S^T \\ r^T] L^-T)^T
                    gemm_nt(gamma_k - rank_k, nx + 1, nu - rank_k, -1.0, Ggt_stripe[0], 0, 0,
                            Llt[k], nu - rank_k, 0, 1.0, Hh[k], 0, 0, Hh[k], 0, 0);
                    // keep (L^-1 eta^T) for forward recursion
                    getr(gamma_k - rank_k, nu - rank_k, Ggt_stripe[0], 0, 0, Ggt_tilde[k], 0,
                         rank_k);
                }
                if (k > 0){
                    // GuGx_hat = GuGx * L^-1
                    if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                    gecp(nunxm1, nu, GuGx_tilde[k-1], 0, 0, GuGx_hat[k-1], 0, 0);
                    trsm_rltn(nunxm1, nu - rank_k, 1.0, Llt[k], 0, 0, GuGx_hat[k-1], 0, rank_k, GuGx_hat[k-1], 0, rank_k);

                    // RSQrqt = RSQrqt - GuGx^T L^-T L^-1 GuGx
                    if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                    // std::cout << "#-----------------------------------" << std::endl;
                    // std::cout << "#testing RSQrqt_bar update (also copy RSQrqt_hat, Llt and GuGxtilde definition)" << std::endl;
                    // std::cout << "nx = " << nx << std::endl;
                    // std::cout << "nu = " << nu << std::endl;
                    // std::cout << "nunxm1 = " << nunxm1 << std::endl;
                    // std::cout << "rank_k = " << rank_k << std::endl;
                    // PrintNpArray(RSQrqt_underbar[k-1], "RSQrqt_underbar_before");
                    // PrintNpArray(RSQrqt_tilde[k], "RSQrqt_tilde");
                    gemm_nt(nunxm1, nunxm1, nu - rank_k, -1.0, GuGx_hat[k-1], 0, rank_k, GuGx_hat[k-1], 0, rank_k, 
                            1.0, RSQrqt_underbar[k-1], 0, 0, RSQrqt_underbar[k-1], 0, 0);
                    if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                    trsm_rltn(1, nu - rank_k, 1.0, Llt[k], 0, 0, RSQrq_hat_curr_p[0], nu - rank_k + nx, 0, v_r_tilde[0], 0, 0);
                    // PrintNpArray(v_r_tilde[0], "v_r_tilde");
                    // PrintNpArray(hessian.GuGx[k-1], "GuGx");
                    // PrintNpArray(GuGx_tilde[k-1], "GuGx_tilde");
                    // PrintNpArray(GuGx_hat[k-1], "GuGx_hat");
                    // PrintNpArray(RSQrqt_underbar[k-1], "RSQrqt_underbar_intermediate");
                    // MatRealAllocated temp(1, nunxm1);
                    // gemm_nt(1, nunxm1, nu - rank_k, -1.0, v_r_tilde[0], 0, 0, GuGx_hat[k-1], 0, rank_k, 
                    //         1.0, temp, 0, 0, temp, 0, 0);
                    // PrintNpArray(temp, "temp");
                    gemm_nt(1, nunxm1, nu - rank_k, -1.0, v_r_tilde[0], 0, 0, GuGx_hat[k-1], 0, rank_k,
                            1.0, RSQrqt_underbar[k-1], nunxm1, 0, RSQrqt_underbar[k-1], nunxm1, 0);
                    // PrintNpArray(RSQrqt_underbar[k-1], "RSQrqt_underbar_after");

                    // FuFx = FuFx - GuGx_hat * L
                    // PrintNpArray(FuFx_underbar[k-1], "FuFx_underbar_before");
                    // PrintNpArray(Llt[k], "Llt");
                    if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                    gemm_nt(nunxm1, nx, nu - rank_k, -1.0, GuGx_hat[k-1], 0, rank_k, Llt[k], nu - rank_k, 0,
                            1.0, FuFx_underbar[k-1], 0, 0, FuFx_underbar[k-1], 0, 0);
                    // PrintNpArray(FuFx_underbar[k-1], "FuFx_underbar_after");
                    // std::cout << "-----------------------------------" << std::endl;
                }
            }
            else
            {
                gecp(nx + 1, nx, RSQrq_hat_curr_p[0], 0, 0, Ppt[k], 0, 0);
            }
            trtr_l(nx, Ppt[k], 0, 0, Ppt[k], 0, 0);
        }
        // PrintNpArray(RSQrqt_underbar[0], "RSQrqt_underbar[0]");
        // PrintNpArray(RSQrqt_underbar[1], "RSQrqt_underbar[1]");
    }
    // std::cout << " === FIRST STAGE === " << std::endl;
    if (print_initial_stage){
        PrintNpArray(Ppt[0], "Ppt");
        PrintNpArray(Hh[0], "Hh");
    }
    rankI = 0;
    //////// FIRST_STAGE
    {
        const Index nx = info.dims.number_of_states[0];
        Index gamma_I = gamma[0] - rho[0];
        if (gamma_I > nx)
        {
            if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
            return LinsolReturnFlag::NOFULL_RANK;
        }
        if (gamma_I > 0)
        {
            getr(gamma_I, nx + 1, Hh[0], 0, 0, HhIt[0], 0, 0); // transposition may be avoided
            // HhIt[0].print();
            lu_fact_transposed(gamma_I, nx + 1, nx, rankI, HhIt[0], PlI[0], PrI[0], lu_fact_tol);
            if (rankI < gamma_I){
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                return LinsolReturnFlag::NOFULL_RANK;
            }
            // PpIt_tilde <- Ggt[rankI:nx+1, :rankI] L-T (note that this is slightly different from
            // the implementation)
            trsm_rlnn(nx - rankI + 1, rankI, -1.0, HhIt[0], 0, 0, HhIt[0], rankI, 0, GgIt_tilde[0],
                      0, 0);
            // permutations
            PrI[0].apply_on_rows(rankI, &Ppt[0].mat()); // TODO make use of symmetry
            PrI[0].apply_on_cols(rankI, &Ppt[0].mat());
            // // GL <- GgIt_tilde @ Pp[:rankI,:nx] + Ppt[rankI:nx+1, rankI:] (with Pp[:rankI,:nx] =
            // Ppt[:nx,:rankI]^T) GEMM_NT(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde[0], 0, 0,
            // Ppt[0], 0, 0, 1.0, Ppt[0], rankI, 0, GgLIt[0], 0, 0); split up because valgrind was
            // giving invalid read errors when C matrix has nonzero row offset
            gecp(nx - rankI + 1, nx, Ppt[0], rankI, 0, GgLIt[0], 0, 0);
            gemm_nt(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde[0], 0, 0, Ppt[0], 0, 0, 1.0,
                    GgLIt[0], 0, 0, GgLIt[0], 0, 0);
            // // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] + GgLt[rank_k:,
            // :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
            syrk_ln_mn(nx - rankI + 1, nx - rankI, rankI, 1.0, GgLIt[0], 0, 0, GgIt_tilde[0], 0, 0,
                       1.0, GgLIt[0], 0, rankI, PpIt_hat[0], 0, 0);
            // TODO skipped if nx-rankI = 0
            potrf_l_mn(nx - rankI + 1, nx - rankI, PpIt_hat[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx - rankI, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
        else
        {
            rankI = 0;
            potrf_l_mn(nx + 1, nx, Ppt[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
    }
    // std::cout << " === FORWARD FIRST STAGE === " << std::endl;
    ////// FORWARD_SUBSTITUTION:
    // first stage
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_u = info.offsets_primal_u[0];
        const Index offs_x = info.offsets_primal_x[0];
        const Index offs_g = info.offsets_g_eq_path[0];
        // calculate xIb
        rowex(nx - rankI, -1.0, LlIt[0], nx - rankI, 0, x, offs_x + rankI);
        // assume TRSV_LTN allows aliasing, this is the case in normal BLAS
        trsv_ltn(nx - rankI, LlIt[0], 0, 0, x, offs_x + rankI, x, offs_x + rankI);
        // calculate xIa
        rowex(rankI, 1.0, GgIt_tilde[0], nx - rankI, 0, x, offs_x);
        // assume aliasing is possible for last two elements
        gemv_t(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, x, offs_x + rankI, 1.0, x, offs_x, x,
               offs_x);
        //// lag
        rowex(rankI, -1.0, Ppt[0], nx, 0, eq_mult, offs_g);
        // assume aliasing is possible for last two elements
        gemv_t(nx, rankI, -1.0, Ppt[0], 0, 0, x, offs_x, 1.0, eq_mult, offs_g, eq_mult, offs_g);

        // U^-T
        trsv_lnn(rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        // L^-T
        trsv_unu(rankI, rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        PlI[0].apply_inverse(rankI, &eq_mult.vec(), offs_g);
        PrI[0].apply_inverse(rankI, &x.vec(), offs_x);
    }
    // other stages
    if (print_initial_stage){
        PrintNpArray(x, "x_first_stage");
        PrintNpArray(eq_mult, "eq_mult_first_stage");
    }
    // std::cout << "x initial solution:\n" << x << std::endl;
    // std::cout << "eq_mult initial solution:\n" << eq_mult << std::endl;
    // std::cout << " === FORWARD OTHER STAGES === " << std::endl;
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index rho_k = rho[k];
        const Index numrho_k = nu - rho_k;
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index gammamrho_k = gamma[k] - rho[k];
        const Index gamma_k = gamma[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (numrho_k > 0)
        {
            /// calculate ukb_tilde
            // -Lkxk - lk
            rowex(numrho_k, -1.0, Llt[k], numrho_k + nx, 0, x, offs + rho_k);
            if (increased_accuracy)
            {
                gemv_n(nu - rho_k, gamma_k - rho_k, -1.0, Ggt_tilde[k], 0, rho_k, eq_mult, offs_g_k,
                       1.0, x, offs + rho_k, x, offs + rho_k);
            }
            // assume aliasing of last two eliments is allowed
            gemv_t(nx, numrho_k, -1.0, Llt[k], numrho_k, 0, x, offs_x, 1.0, x, offs + rho_k, x,
                   offs + rho_k);
            trsv_ltn(numrho_k, Llt[k], 0, 0, x, offs + rho_k, x, offs + rho_k);

            // + GuGxt [uk-1, xk-1]
            if (k > 0){
                const Index nunxm1 = info.dims.number_of_controls[k-1] + info.dims.number_of_states[k-1];
                // std::cout << "----------------------------------------" << std::endl;
                // std::cout << "test ukb_tilde update" << std::endl;
                // PrintNpArray(hessian.GuGx[k-1], "GuGx");
                // PrintNpArray(GuGx_hat[k-1], "GuGx_hat");
                // PrintNpArray(Llt[k], "Llt");
                // std::cout << "nunxm1 = " << nunxm1 << std::endl;
                // std::cout << "nu = " << nu << std::endl;
                // std::cout << "rho_k = " << rho_k << std::endl;
                // PrintNpArray(x, info.offsets_primal_u[k-1], nunxm1, "ukxk");
                // PrintNpArray(x, info.offsets_primal_u[k], nu, "uk_before");
                trsm_rlnn(nunxm1, numrho_k, 1.0, Llt[k], 0, 0, GuGx_hat[k-1], 0, rho_k, GuGx_hat[k-1], 0, rho_k);
                // PrintNpArray(GuGx_hat[k-1], "GuGx_hat_intermediate");
                gemv_t(nunxm1, nu - rho_k, -1.0, GuGx_hat[k-1], 0, rho_k, x, info.offsets_primal_u[k-1], 
                       1.0, x, offs + rho_k, x, offs + rho_k);
                // PrintNpArray(x, info.offsets_primal_u[k], nu, "uk_after");
                // std::cout << "----------------------------------------" << std::endl;
            }
        }
        /// calcualate uka_tilde
        if (rho_k > 0)
        {
            rowex(rho_k, 1.0, Ggt_tilde[k], numrho_k + nx, 0, x, offs);
            gemv_t(nx + numrho_k, rho_k, 1.0, Ggt_tilde[k], 0, 0, x, offs + rho_k, 1.0, x, offs, x,
                   offs);
            // calculate lamda_tilde_k
            // copy vk to right location
            veccp(gammamrho_k, eq_mult, offs_g_k, v_tmp[0], 0);
            veccp(gammamrho_k, v_tmp[0], 0, eq_mult, offs_g_k + rho_k);
            rowex(rho_k, -1.0, RSQrqt_tilde[k], nu + nx, 0, eq_mult, offs_g_k);
            // assume aliasing of last two eliments is allowed
            gemv_t(nu + nx, rho_k, -1.0, RSQrqt_tilde[k], 0, 0, x, offs, 1.0, eq_mult, offs_g_k,
                   eq_mult, offs_g_k);
            if (k > 0){
                const Index nunxm1 = info.dims.number_of_controls[k-1] + info.dims.number_of_states[k-1];
                // std::cout << "eq_mult before:\n" << eq_mult << std::endl;
                // PrintNpArray(GuGx_tilde[k-1], "GuGx_tilde");
                // PrintNpArray(x, info.offsets_primal_u[k-1], nunxm1, "ukxk");
                gemv_t(nunxm1, rho_k, -1.0, GuGx_tilde[k-1], 0, 0, x, info.offsets_primal_u[k-1], 1.0, 
                       eq_mult, offs_g_k, eq_mult, offs_g_k);
                // std::cout << "eq_mult after:\n" << eq_mult << std::endl;
            }

            // nu-rank_k+nx,0
            // needless copy because feature not implemented yet in trsv_lnn
            gecp(rho_k, gamma_k, Ggt_tilde[k], nu - rho_k + nx + 1, 0, AL[0], 0, 0);
            // U^-T
            trsv_lnn(rho_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            // L^-T
            trsv_unu(rho_k, gamma_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            Pl[k].apply_inverse(rho_k, &eq_mult.vec(), offs_g_k);
            Pr[k].apply_inverse(rho_k, &x.vec(), offs);
            // std::cout << "eq_mult after after:\n" << eq_mult << std::endl;
        }
        // we've already covered this in pre-processing
        // VecRealAllocated eq_mult_copy(eq_mult);
        // for (int i = 0; i < eq_mult.m(); i++){ eq_mult_copy(i) = eq_mult(i);}
        // if (ng_ineq > 0)
        // {
        //     gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
        //            eq_mult, offs_eq_ineq);
        //     eq_mult.block(ng_ineq, offs_eq_ineq) =
        //         eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        //     // if (k == 6){
        //     // PrintNpArray(jacobian.Gg_ineqt[k], "\nGg_ineqt[" + std::to_string(k) + "]", nu + nx + 1, ng_ineq);
        //     // PrintNpArray(D_s, offs_slack, ng_ineq, "D_s[" + std::to_string(k) + "]");
        //     // PrintNpArray(g, offs_eq_ineq, ng_ineq, "g[" + std::to_string(k) + "]");
        //     // PrintNpArray(x, offs, nu + nx, "x[" + std::to_string(k) + "]");
        //     // PrintNpArray(eq_mult_copy, offs_eq_ineq, ng_ineq, "[" + std::to_string(k) + "] eq_mult before ineq regularization");
        //     // std::cout << "nu: " << nu << " nx: " << nx << " ng_ineq: " << ng_ineq << std::endl;
        //     // std::cout << "offs: " << offs << " offs_eq_ineq: " << offs_eq_ineq << " offs_slack: " << offs_slack << std::endl;
        //     // gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
        //     //        eq_mult_copy, offs_eq_ineq);
        //     // eq_mult_copy.block(ng_ineq, offs_eq_ineq) =
        //     //     eq_mult_copy.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        //     // PrintNpArray(eq_mult_copy, offs_eq_ineq, ng_ineq, "[" + std::to_string(k) + "] eq_mult after ineq regularization");
        //     // }
        // }
        if (k != info.dims.K - 1)
        {
            const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offsxp1 = info.offsets_primal_x[k + 1];
            const Index offs_g_kp1 = info.offsets_g_eq_path[k + 1];
            const Index gammamrho_kp1 = gamma[k + 1] - rho[k + 1];
            // calculate xkp1
            rowex(nxp1, 1.0, jacobian.BAbt[k], nu + nx, 0, x, offsxp1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offsxp1, x,
                   offsxp1);
            // calculate lam_dyn xp1
            rowex(nxp1, 1.0, Ppt[k + 1], nxp1, 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offsxp1, 1.0, eq_mult, offs_dyn_eq_k,
                   eq_mult, offs_dyn_eq_k);
            gemv_t(gammamrho_kp1, nxp1, 1.0, Hh[k + 1], 0, 0, eq_mult, offs_g_kp1, 1.0, eq_mult,
                   offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
            
            // std::cout << "----------------------------------------" << std::endl;
            // std::cout << "test pi update" << std::endl;
            // PrintNpArray(eq_mult, offs_dyn_eq_k, nxp1, "eq_mult");
            // PrintNpArray(hessian.FuFx[k], "FuFx");
            // PrintNpArray(x, offs, nu + nx, "x");
            gemv_t(nu + nx, nxp1, 1.0, FuFx_underbar[k], 0, 0, x, offs, 1.0, 
                   eq_mult, offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
            // PrintNpArray(eq_mult, offs_dyn_eq_k, nxp1, "eq_mult");
            // std::cout << "----------------------------------------" << std::endl;
        }
    }

    if (write_factorization_file){
        PrintFactorizationInfo(info, rank_k_values, Pl, Pr, LU, gamma_k_values, Ggt_eq, Llt, R_shur);
    }

    return LinsolReturnFlag::SUCCESS;
}
LinsolReturnFlag ModifiedAugSystemSolver::solve(const ProblemInfo &info,
                                           Jacobian<ImplicitOcpType> &jacobian, Hessian<ImplicitOcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_eq,
                                           const VecRealView &D_s, const VecRealView &f,
                                           const VecRealView &g, VecRealView &x,
                                           VecRealView &eq_mult)
{
    MatRealView *RSQrq_hat_curr_p;
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_ineq_k = info.offsets_slack[k];
        const Index offset_u = info.offsets_primal_u[k];
        const Index offset_eq_k = info.offsets_eq[k];
        const Index offset_g_eq_k = info.offsets_g_eq_path[k];
        const Index offset_g_ineq_k = info.offsets_g_eq_slack[k];
        // const fatrop_int offs_g_ineq_k = offs_g_ineq_p[k];
        //////// SUBSDYN
        if (k == info.dims.K - 1)
        {
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            gecp(nx + nu + 1, nu + nx, hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        else
        {
            const Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            // AL <- [BAb]^T_k P_kp1
            rowin(nxp1, 1.0, g, offset_eq_dyn, jacobian.BAbt[k], nu + nx, 0);
            gemm_nt(nu + nx + 1, nxp1, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Ppt[k + 1], 0, 0, 0.0,
                    AL[0], 0, 0, AL[0], 0, 0);
            // AL[-1,:] <- AL[-1,:] + p_kp1^T
            gead(1, nxp1, 1.0, Ppt[k + 1], nxp1, 0, AL[0], nx + nu, 0);
            // RSQrqt_stripe <- AL[BA] + RSQrqt
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            syrk_ln_mn(nu + nx + 1, nu + nx, nxp1, 1.0, AL[0], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                       hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);

            // Add second order dynamics contribution                
            gemm_nt(nu + nx + 1, nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, FuFx_underbar[k], 0, 0, 1.0,
                    RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
            gemm_nt(nu + nx, nu + nx, nxp1, 1.0, FuFx_underbar[k], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                    RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        // Covered in pre-processing
        // // equality penalty
        // {
        //     rowin(ng, 1.0, g, offset_g_eq_k, jacobian.Gg_eqt[k], nu + nx, 0);
        //     gecp(nu + nx + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
        //     for (Index i = 0; i < ng; i++)
        //     {
        //         Scalar scaling_factor = 1.0 / D_eq(offset_eq_k + i);
        //         colsc(nu + nx + 1, scaling_factor, Ggt_stripe[0], 0, i);
        //     }
        //     // add the penalty
        //     syrk_ln_mn(nu + nx + 1, nu + nx, ng, 1.0, Ggt_stripe[0], 0, 0, jacobian.Gg_eqt[k], 0, 0,
        //                1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        // }
        // // inequalities + inertia correction
        // {
        //     if (ng_ineq > 0)
        //     {
        //         rowin(ng_ineq, 1.0, g, offset_g_ineq_k, jacobian.Gg_ineqt[k], nu + nx, 0);
        //         gecp(nu + nx + 1, ng_ineq, jacobian.Gg_ineqt[k], 0, 0, Ggt_ineq_temp[0], 0, 0);
        //         for (Index i = 0; i < ng_ineq; i++)
        //         {
        //             Scalar scaling_factor = 1.0 / D_s(offs_ineq_k + i);
        //             colsc(nu + nx + 1, scaling_factor, Ggt_ineq_temp[0], 0, i);
        //         }
        //         // add the penalty
        //         syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, Ggt_ineq_temp[0], 0, 0,
        //                    jacobian.Gg_ineqt[k], 0, 0, 1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k],
        //                    0, 0);
        //     }
        //     // inertia correction
        //     diaad(nu + nx, 1.0, D_x, offset_u, RSQrqt_tilde[k], 0, 0);
        // }

        //////// TRANSFORM_AND_SUBSEQ
        {
            RSQrq_hat_curr_p = &RSQrqt_tilde[k];
        }
        //////// SCHUR
        {
            // DLlt_k = [chol(R_hatk); Llk@chol(R_hatk)^-T]
            potrf_l_mn(nu + nx + 1, nu, *RSQrq_hat_curr_p, 0, 0, Llt[k], 0, 0);
            if (!check_reg(nu, &Llt[k].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
            // Pp_k = Qq_hatk - L_k^T @ Ll_k
            // SYRK_LN_MN(nx+1, nx, nu-rank_k, -1.0,Llt_p+k, nu-rank_k,0, Llt_p+k, nu-rank_k,0, 1.0,
            // RSQrq_hat_curr_p, nu-rank_k, nu-rank_k,Pp+k,0,0); // feature not implmented yet
            gecp(nx + 1, nu, Llt[k], nu, 0, Llt_shift[0], 0,
                 0); // needless operation because feature not implemented yet
            syrk_ln_mn(nx + 1, nx, nu, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0, 0, 1.0,
                       *RSQrq_hat_curr_p, nu, nu, Ppt[k], 0, 0);

            if (k > 0){
                // GuGx_hat = GuGx * L^-1
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                const Index nunxm1 = info.dims.number_of_controls[k-1] + info.dims.number_of_states[k-1];
                gecp(nunxm1, nu, hessian.GuGx[k-1], 0, 0, GuGx_hat[k-1], 0, 0);
                trsm_rltn(nunxm1, nu, 1.0, Llt[k], 0, 0, GuGx_hat[k-1], 0, 0, GuGx_hat[k-1], 0, 0);

                // RSQrqt = RSQrqt - GuGx^T L^-T L^-1 GuGx
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                gemm_nt(nunxm1, nunxm1, nu, -1.0, GuGx_hat[k-1], 0, 0, GuGx_hat[k-1], 0, 0, 
                        1.0, RSQrqt_underbar[k-1], 0, 0, RSQrqt_underbar[k-1], 0, 0);
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                trsm_rltn(1, nu, 1.0, Llt[k], 0, 0, RSQrq_hat_curr_p[0], nu + nx, 0, v_r_tilde[0], 0, 0);
                gemm_nt(1, nunxm1, nu, -1.0, v_r_tilde[0], 0, 0, GuGx_hat[k-1], 0, 0,
                        1.0, RSQrqt_underbar[k-1], nunxm1, 0, RSQrqt_underbar[k-1], nunxm1, 0);

                // FuFx = FuFx - GuGx_hat * L
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                gemm_nt(nunxm1, nx, nu, -1.0, GuGx_hat[k-1], 0, 0, Llt[k], nu, 0,
                        1.0, FuFx_underbar[k-1], 0, 0, FuFx_underbar[k-1], 0, 0);
            }

        }
        trtr_l(nx, Ppt[k], 0, 0, Ppt[k], 0, 0);
    }
    //////// FIRST_STAGE
    {
        const Index nx = info.dims.number_of_states[0];
        {
            potrf_l_mn(nx + 1, nx, Ppt[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
    }
    ////// FORWARD_SUBSTITUTION:
    // first stage
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_x = info.offsets_primal_x[0];
        // calculate xIb
        rowex(nx, -1.0, LlIt[0], nx, 0, x, offs_x);
        // assume TRSV_LTN allows aliasing, this is the case in normal BLAS
        trsv_ltn(nx, LlIt[0], 0, 0, x, offs_x, x, offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        rowex(nu, -1.0, Llt[k], nu + nx, 0, x, offs);
        gemv_t(nx, nu, -1.0, Llt[k], nu, 0, x, offs_x, 1.0, x, offs, x, offs);
        trsv_ltn(nu, Llt[k], 0, 0, x, offs, x, offs);
        // + GuGxt [uk-1, xk-1]
        if (k > 0){
            const Index nunxm1 = info.dims.number_of_controls[k-1] + info.dims.number_of_states[k-1];
            trsm_rlnn(nunxm1, nu, 1.0, Llt[k], 0, 0, GuGx_hat[k-1], 0, 0, GuGx_hat[k-1], 0, 0);
            gemv_t(nunxm1, nu, -1.0, GuGx_hat[k-1], 0, 0, x, info.offsets_primal_u[k-1], 
                    1.0, x, offs, x, offs);
        }
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offs_x_p1 = info.offsets_primal_x[k + 1];
            const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
            // calculate xkp1
            rowex(nxp1, 1.0, jacobian.BAbt[k], nu + nx, 0, x, offs_x_p1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offs_x_p1, x,
                   offs_x_p1);
            // calculate lam_dyn xp1
            rowex(nxp1, 1.0, Ppt[k + 1], nxp1, 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offs_x_p1, 1.0, eq_mult, offs_dyn_eq_k,
                   eq_mult, offs_dyn_eq_k);
            
            gemv_t(nu + nx, nxp1, 1.0, FuFx_underbar[k], 0, 0, x, offs, 1.0, 
                   eq_mult, offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
        }
        // Covered in post-processing
        // const Index ng = info.dims.number_of_eq_constraints[k];
        // const Index offs_g_eq_k = info.offsets_g_eq_path[k];
        // const Index offs_eq_k = info.offsets_eq[k];
        // if (ng > 0)
        // {
        //     gemv_t(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, x, offs, 1.0, g, offs_g_eq_k,
        //            eq_mult, offs_g_eq_k);
        //     eq_mult.block(ng, offs_g_eq_k) =
        //         eq_mult.block(ng, offs_g_eq_k) / D_eq.block(ng, offs_eq_k);
        // }
        // const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        // const Index offs_slack = info.offsets_slack[k];
        // const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        // if (ng_ineq > 0)
        // {
        //     gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
        //            eq_mult, offs_eq_ineq);
        //     eq_mult.block(ng_ineq, offs_eq_ineq) =
        //         eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        // }
    }
    return LinsolReturnFlag::SUCCESS;
}

LinsolReturnFlag ModifiedAugSystemSolver::solve_rhs(const ProblemInfo &info,
                                               const Jacobian<ImplicitOcpType> &jacobian,
                                               const Hessian<ImplicitOcpType> &hessian,
                                               const VecRealView &D_s, const VecRealView &f,
                                               const VecRealView &g, VecRealView &x,
                                               VecRealView &eq_mult)
{
    VecRealView *v_RSQrq_hat_curr_p;
    Index rank_k;
    /////////////// recursion ///////////////

    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offset_ineq_k = info.offsets_slack[k];
        const Index offs_g_ineq_k = info.offsets_g_eq_slack[k];
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index offs = info.offsets_primal_u[k];
        //         //////// SUBSDYN
        Index gamma_k;
        if (k == info.dims.K - 1)
        {
            gamma_k = ng;
            gamma[k] = gamma_k;
            veccp(ng, g, offs_g_k, v_Ggt_stripe[0], 0);
            veccp(nu + nx, f, offs, v_RSQrqt_tilde[k], 0);
        }
        else
        {
            const Index offs_dyn_k = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index Hp1_size = gamma[k + 1] - rho[k + 1];
            gamma_k = Hp1_size + ng;
            gemv_n(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, g, offs_dyn_k, 0.0, v_AL[0], 0, v_AL[0], 0);
            axpy(nxp1, 1.0, v_Ppt[k + 1], 0, v_AL[0], 0, v_AL[0], 0);
            gemv_n(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, v_AL[0], 0, 1.0, f, offs,
                   v_RSQrqt_tilde[k], 0);
            if (gamma_k > 0)
            {
                if (ng > 0)
                {
                    veccp(ng, g, offs_g_k, v_Ggt_stripe[0], 0);
                }
                if (Hp1_size > 0)
                {
                    gemv_n(Hp1_size, nxp1, 1.0, Hh[k + 1], 0, 0, g, offs_dyn_k, 0.0,
                           v_Ggt_stripe[0], ng, v_Ggt_stripe[0], ng);
                    axpy(Hp1_size, 1.0, v_Hh[k + 1], 0, v_Ggt_stripe[0], ng, v_Ggt_stripe[0], ng);
                }
            }
            else
            {
                rank_k = 0;
                v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
            }
        }
        if (ng_ineq > 0)
        {
            for (Index i = 0; i < ng_ineq; i++)
            {
                Scalar scaling_factor = D_s(offset_ineq_k + i);
                Scalar grad_barrier = g(offs_g_ineq_k + i);
                v_Ggt_ineq_temp[0](i) = grad_barrier / scaling_factor;
            }
            gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, v_Ggt_ineq_temp[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        {
            rank_k = rho[k];
            gecp(rank_k, gamma_k, Ggt_tilde[k], nu - rank_k + nx + 1, 0, Ggt_stripe[0], 0, 0);
            Pl[k].apply(rank_k, &v_Ggt_stripe[0].vec(), 0);
            trsv_utu(rank_k, Ggt_stripe[0], 0, 0, v_Ggt_stripe[0], 0, v_Ggt_stripe[0], 0);
            gemv_t(rank_k, gamma_k - rank_k, -1.0, Ggt_stripe[0], 0, rank_k, v_Ggt_stripe[0], 0,
                   1.0, v_Ggt_stripe[0], rank_k, v_Ggt_stripe[0], rank_k);

            if (gamma_k - rank_k > 0)
            {
                veccp(gamma_k - rank_k, v_Ggt_stripe[0], rank_k, v_Hh[k], 0);
            }
            if (rank_k > 0)
            {
                veccpsc(rank_k, -1.0, v_Ggt_stripe[0], 0, v_Ggt_tilde[k], 0);
                trsv_ltn(rank_k, Ggt_stripe[0], 0, 0, v_Ggt_tilde[k], 0, v_Ggt_tilde[k], 0);
                Pr[k].apply(rank_k, &v_RSQrqt_tilde[k].vec(), 0);
                veccp(nu + nx, v_RSQrqt_tilde[k], 0, v_GgLt[0], 0);
                gemv_n(nu + nx, rank_k, 1.0, RSQrqt_tilde[k], 0, 0, v_Ggt_tilde[k], 0, 1.0,
                       v_GgLt[0], 0, v_GgLt[0], 0);
                gemv_n(nu + nx - rank_k, rank_k, 1.0, Ggt_tilde[k], 0, 0, v_GgLt[0], 0, 1.0,
                       v_GgLt[0], rank_k, v_RSQrqt_hat[0], 0);
                v_RSQrq_hat_curr_p = &v_RSQrqt_hat[0];
            }
            else
            {
                v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
            }
        }
        //         //////// SCHUR
        {
            if (nu - rank_k > 0)
            {
                trsv_lnn(nu - rank_k, Llt[k], 0, 0, *v_RSQrq_hat_curr_p, 0, v_Llt[k], 0);
                gecp(nx + 1, nu - rank_k, Llt[k], nu - rank_k, 0, Llt_shift[0], 0, 0);
                veccp(nu - rank_k, v_Llt[k], 0, v_Llt_shift[0], 0);
                veccp(nx, *v_RSQrq_hat_curr_p, nu - rank_k, v_Ppt[k], 0);
                gemv_n(nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, v_Llt_shift[0], 0, 1.0, v_Ppt[k],
                       0, v_Ppt[k], 0);
                if (increased_accuracy)
                {
                    gemv_t(nu - rank_k, gamma_k - rank_k, -1.0, Ggt_tilde[k], 0, rank_k, v_Llt[k],
                           0, 1.0, v_Hh[k], 0, v_Hh[k], 0);
                }
            }
            else
            {
                veccp(nx, *v_RSQrq_hat_curr_p, 0, v_Ppt[k], 0);
            }
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        Index gamma_I = gamma[0] - rho[0];
        if (gamma_I > 0)
        {
            veccp(gamma_I, v_Hh[0], 0, v_HhIt[0], 0);
            PlI[0].apply(rankI, &v_HhIt[0].vec(), 0);
            trsv_utu(rankI, HhIt[0], 0, 0, v_HhIt[0], 0, v_HhIt[0], 0);
            gemv_t(rankI, gamma_I - rankI, -1.0, HhIt[0], 0, rankI, v_HhIt[0], 0, 1.0, v_HhIt[0],
                   rankI, v_HhIt[0], rankI);
            veccpsc(rankI, -1.0, v_HhIt[0], 0, v_GgIt_tilde[0], 0);
            trsv_ltn(rankI, HhIt[0], 0, 0, v_GgIt_tilde[0], 0, v_GgIt_tilde[0], 0);
            PrI[0].apply(rankI, &v_Ppt[0].vec(), 0);
            veccp(nx, v_Ppt[0], 0, v_GgLIt[0], 0);
            gemv_n(nx, rankI, 1.0, Ppt[0], 0, 0, v_GgIt_tilde[0], 0, 1.0, v_GgLIt[0], 0, v_GgLIt[0],
                   0);
            gemv_n(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, v_GgLIt[0], 0, 1.0, v_GgLIt[0],
                   rankI, v_PpIt_hat[0], 0);
            trsv_lnn(nx - rankI, LlIt[0], 0, 0, v_PpIt_hat[0], 0, v_LlIt[0], 0);
        }
        else
        {
            trsv_lnn(nx, LlIt[0], 0, 0, v_Ppt[0], 0, v_LlIt[0], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_u = info.offsets_primal_u[0];
        const Index offs_x = info.offsets_primal_x[0];
        const Index offs_g = info.offsets_g_eq_path[0];
        veccpsc(nx - rankI, -1.0, v_LlIt[0], 0, x, offs_x + rankI);
        trsv_ltn(nx - rankI, LlIt[0], 0, 0, x, offs_x + rankI, x, offs_x + rankI);
        veccp(rankI, v_GgIt_tilde[0], 0, x, offs_x);
        gemv_t(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, x, offs_x + rankI, 1.0, x, offs_x, x,
               offs_x);
        veccpsc(rankI, -1.0, v_Ppt[0], 0, eq_mult, offs_g);
        gemv_t(nx, rankI, -1.0, Ppt[0], 0, 0, x, nu, 1.0, eq_mult, offs_g, eq_mult, offs_g);
        trsv_lnn(rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        trsv_unu(rankI, rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        PlI[0].apply_inverse(rankI, &eq_mult.vec(), offs_g);
        PrI[0].apply_inverse(rankI, &x.vec(), offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {

        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index rho_k = rho[k];
        const Index numrho_k = nu - rho_k;
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index gammamrho_k = gamma[k] - rho[k];
        const Index gamma_k = gamma[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (numrho_k > 0)
        {
            veccpsc(numrho_k, -1.0, v_Llt[k], 0, x, offs + rho_k);
            if (increased_accuracy)
            {
                gemv_n(nu - rho_k, gamma_k - rho_k, -1.0, Ggt_tilde[k], 0, rho_k, eq_mult, offs_g_k,
                       1.0, x, offs + rho_k, x, offs + rho_k);
            }
            gemv_t(nx, numrho_k, -1.0, Llt[k], numrho_k, 0, x, offs_x, 1.0, x, offs + rho_k, x,
                   offs + rho_k);
            trsv_ltn(numrho_k, Llt[k], 0, 0, x, offs + rho_k, x, offs + rho_k);
        }
        //         /// calcualate uka_tilde
        if (rho_k > 0)
        {
            // ROWEX(rho_k, 1.0, Ggt_tilde[k], numrho_k + nx, 0, ux[0], offs);
            veccp(rho_k, v_Ggt_tilde[k], 0, x, offs);
            gemv_t(nx + numrho_k, rho_k, 1.0, Ggt_tilde[k], 0, 0, x, offs + rho_k, 1.0, x, offs, x,
                   offs);
            veccp(gammamrho_k, eq_mult, offs_g_k, v_tmp[0], 0);
            veccp(gammamrho_k, v_tmp[0], 0, eq_mult, offs_g_k + rho_k);
            veccpsc(rho_k, -1.0, v_RSQrqt_tilde[k], 0, eq_mult, offs_g_k);
            gemv_t(nu + nx, rho_k, -1.0, RSQrqt_tilde[k], 0, 0, x, offs, 1.0, eq_mult, offs_g_k,
                   eq_mult, offs_g_k);
            gecp(rho_k, gamma_k, Ggt_tilde[k], nu - rho_k + nx + 1, 0, AL[0], 0, 0);
            trsv_lnn(rho_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            trsv_unu(rho_k, gamma_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            Pl[k].apply_inverse(rho_k, &eq_mult.vec(), offs_g_k);
            Pr[k].apply_inverse(rho_k, &x.vec(), offs);
        }
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        }
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offsxp1 = info.offsets_primal_x[k + 1];
            const Index offs_g_kp1 = info.offsets_g_eq_path[k + 1];
            const Index offs_dyn_k = info.offsets_g_eq_dyn[k];
            const Index gammamrho_kp1 = gamma[k + 1] - rho[k + 1];
            veccp(nxp1, g, offs_dyn_k, x, offsxp1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offsxp1, x,
                   offsxp1);
            veccp(nxp1, v_Ppt[k + 1], 0, eq_mult, offs_dyn_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offsxp1, 1.0, eq_mult, offs_dyn_k, eq_mult,
                   offs_dyn_k);
            gemv_t(gammamrho_kp1, nxp1, 1.0, Hh[k + 1], 0, 0, eq_mult, offs_g_kp1, 1.0, eq_mult,
                   offs_dyn_k, eq_mult, offs_dyn_k);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}
LinsolReturnFlag ModifiedAugSystemSolver::solve_rhs(const ProblemInfo &info,
                                               const Jacobian<ImplicitOcpType> &jacobian,
                                               const Hessian<ImplicitOcpType> &hessian,
                                               const VecRealView &D_eq, const VecRealView &D_s,
                                               const VecRealView &f, const VecRealView &g,
                                               VecRealView &x, VecRealView &eq_mult)
{
    VecRealView *v_RSQrq_hat_curr_p;
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index offs_ux_k = info.offsets_primal_u[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_g_dyn = info.offsets_g_eq_dyn[k];
        const Index offs_g_eq = info.offsets_g_eq_path[k];
        const Index offs_ge_eq_ineq = info.offsets_g_eq_slack[k];
        //     //////// SUBSDYN
        if (k == info.dims.K - 1)
        {
            veccp(nu + nx, f, offs_ux_k, v_RSQrqt_tilde[k], 0);
        }
        else
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            gemv_n(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, g, offs_g_dyn, 0.0, v_AL[0], 0, v_AL[0], 0);
            axpy(nxp1, 1.0, v_Ppt[k + 1], 0, v_AL[0], 0, v_AL[0], 0);
            gemv_n(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, v_AL[0], 0, 1.0, f, offs_ux_k,
                   v_RSQrqt_tilde[k], 0);
        }
        if (ng > 0)
        {
            const Index offs_eq_k = info.offsets_eq[k];
            for (Index i = 0; i < ng; i++)
            {
                Scalar scaling_factor = D_eq(offs_eq_k + i);
                v_Ggt_stripe[0](i) = g(offs_g_eq + i) / scaling_factor;
            }
            gemv_n(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, v_Ggt_stripe[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        if (ng_ineq > 0)
        {
            const Index offs_ineq_k = info.offsets_slack[k];
            for (Index i = 0; i < ng_ineq; i++)
            {
                Scalar scaling_factor = D_s(offs_ineq_k + i);
                v_Ggt_ineq_temp[0](i) = g(offs_ge_eq_ineq + i) / scaling_factor;
            }
            gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, v_Ggt_ineq_temp[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        {
            v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
        }
        {
            trsv_lnn(nu, Llt[k], 0, 0, *v_RSQrq_hat_curr_p, 0, v_Llt[k], 0);
            veccp(nu, v_Llt[k], 0, v_Llt_shift[0], 0);
            gemv_n(nx, nu, -1.0, Llt[k], nu, 0, v_Llt_shift[0], 0, 1.0, v_RSQrqt_tilde[k], nu,
                   v_Ppt[k], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        {
            trsv_lnn(nx, LlIt[0], 0, 0, v_Ppt[0], 0, v_LlIt[0], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_x = info.offsets_primal_x[0];
        veccpsc(nx, -1.0, v_LlIt[0], 0, x, offs_x);
        trsv_ltn(nx, LlIt[0], 0, 0, x, offs_x, x, offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
        veccpsc(nu, -1.0, v_Llt[k], 0, x, offs);
        gemv_t(nx, nu, -1.0, Llt[k], nu, 0, x, offs_x, 1.0, x, offs, x, offs);
        trsv_ltn(nu, Llt[k], 0, 0, x, offs, x, offs);
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offs_x_p1 = info.offsets_primal_x[k + 1];
            veccp(nxp1, g, offs_dyn_eq_k, x, offs_x_p1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offs_x_p1, x,
                   offs_x_p1);
            veccp(nxp1, v_Ppt[k + 1], 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offs_x_p1, 1.0, eq_mult,
                   offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
        }
    }
    // // calculate lam_eq xk
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index offs_eq = info.offsets_eq[k];
        if (ng > 0)
        {
            gemv_t(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, x, offs, 1.0, g, offs_g_k,
                   eq_mult, offs_g_k);
            eq_mult.block(ng, offs_g_k) =
                eq_mult.block(ng, offs_g_k) / D_eq.block(ng, offs_eq);
        }
    }

    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_gineq_k = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_gineq_k,
                   eq_mult, offs_gineq_k);
            eq_mult.block(ng_ineq, offs_gineq_k) =
                eq_mult.block(ng_ineq, offs_gineq_k) / D_s.block(ng_ineq, offs_slack);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}

void ModifiedAugSystemSolver::set_performance_mode(bool set){
    if (set){
        print_debug_lines = false;
        print_initial_stage = false;
        write_factorization_file = false;
    }
}







































































void PrintLuInfo(MatRealAllocated const &JBAbt, MatRealAllocated const &Jt, 
                 PermutationMatrix &Pl, PermutationMatrix &Pr){

        int nx_next = JBAbt.n();
        std::cout << "JBAbt: " << std::endl << JBAbt << std::endl;

        PrintNpArray(Jt, "Jt");

        std::cout << "Pl: "; for (int i = 0; i < nx_next; i++){std::cout << Pl[i] << "  ";} std::cout << std::endl;
        std::cout << "Pr: "; for (int i = 0; i < nx_next; i++){std::cout << Pr[i] << "  ";} std::cout << std::endl;

        // print L and U matrix
        std::cout << "L = np.array([\n";
        for (int i = 0; i < nx_next; i++){
            std::cout << "\t[";
            for (int j = 0; j < nx_next; j++){
                if (i > j){
                    std::cout << JBAbt(j,i);
                } else if (i == j){
                    std::cout << 1.0;
                } else {
                    std::cout << 0.0;
                }
                if (j < nx_next - 1){
                    std::cout << ", ";
                }
            }
            std::cout << "]";
            if (i < nx_next){
                std::cout << ",\n";
            }
        }
        std::cout << "\n])" << std::endl;
        std::cout << "U = np.array([\n";
        for (int i = 0; i < nx_next; i++){
            std::cout << "\t[";
            for (int j = 0; j < nx_next; j++){
                if (i <= j){
                    std::cout << JBAbt(j,i);
                } else {
                    std::cout << 0.0;
                }
                if (j < nx_next - 1){
                    std::cout << ", ";
                }
            }
            std::cout << "]";
            if (i < nx_next){
                std::cout << ",\n";
            }
        }
        std::cout << "\n])" << std::endl;
}

MatRealAllocated GetKKT(const ProblemInfo &info,
                        Jacobian<ImplicitOcpType> &jacobian,
                        Hessian<ImplicitOcpType> &hessian,
                        bool identity_jacobian=false){
    int nb_primal = info.number_of_primal_variables;
    int nb_eq = info.number_of_eq_constraints;
    MatRealAllocated full_kkt_matrix = MatRealAllocated(
        nb_primal + nb_eq, nb_primal + nb_eq);
    MatRealAllocated full_matrix_jacobian = MatRealAllocated(nb_eq, nb_primal);
    MatRealAllocated full_matrix_hessian = MatRealAllocated(nb_primal, nb_primal);

    for (int k = 0; k < info.dims.K-1; k++){
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index offs_ux = info.offsets_primal_u[k];
        Index offs_x_next = info.offsets_primal_x[k + 1];
        Index offs_u_next = info.offsets_primal_u[k + 1];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index nu_next = info.dims.number_of_controls[k + 1];
        Index offs_eq_dyn = info.offsets_g_eq_dyn[k];
        full_matrix_jacobian.block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
            transpose(jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0));
        full_matrix_hessian.block(nx_next, nu + nx, offs_x_next, offs_ux) = 
            transpose(hessian.FuFx[k]);
        full_matrix_hessian.block(nu + nx, nx_next, offs_ux, offs_x_next) =
            hessian.FuFx[k];
        full_matrix_hessian.block(nu_next, nu + nx, offs_u_next, offs_ux) = 
            transpose(hessian.GuGx[k]);
        full_matrix_hessian.block(nu + nx, nu_next, offs_ux, offs_u_next) =
            hessian.GuGx[k];
        if (identity_jacobian){
            for (int i = 0; i < nx_next; i++){
                full_matrix_jacobian(offs_eq_dyn + i, offs_x_next + i) = -1;    
            }
        } else {
            full_matrix_jacobian.block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                transpose(jacobian.Jt[k]);
        }
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
    for (Index k = 0; k < info.dims.K; k++)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
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
    // PrintNpArray(full_kkt_matrix, "KKT");

    return full_kkt_matrix;
}

void VerifyIntermediateSolution(const ProblemInfo &info,
                                Jacobian<ImplicitOcpType> &jacobian,
                                Hessian<ImplicitOcpType> &hessian,
                                VecRealView &x, VecRealView& mult,
                                VecRealView &f, VecRealView g){
    VecRealAllocated solution_grad = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated solution_g = VecRealAllocated(info.number_of_eq_constraints);

    hessian.apply_on_right(info, x, 0.0, solution_grad, solution_grad);
    jacobian.transpose_apply_on_right(info, mult, 1.0, solution_grad, solution_grad, true);

    jacobian.apply_on_right(info, x, 0.0, solution_g, solution_g, true);

    double max_diff_grad = 0.0;
    for (int i = 0; i < info.number_of_primal_variables; i++){
        std::cout << std::setw(10) << std::setprecision(5) << solution_grad(i) << "\t-\t" << -f(i) << std::endl;
        max_diff_grad = std::max(max_diff_grad, std::abs(solution_grad(i) + f(i)));
    }
    std::cout << "MAX DIFF GRAD: " << max_diff_grad << std::endl;

    std::cout << "------------" << std::endl;
    double max_diff_g = 0.0;
    for (int i = 0; i < info.number_of_eq_constraints; i++){
        std::cout << std::setw(10) << std::setprecision(5) << solution_g(i) << "\t-\t" << -g(i) << std::endl;
        max_diff_g = std::max(max_diff_g, std::abs(solution_g(i) + g(i)));
    }
    std::cout << "MAX DIFF G: " << max_diff_g << std::endl;
}




AugSystemSolver<ImplicitOcpType>::AugSystemSolver(const ProblemInfo &info) : ModifiedAugSystemSolver(info)
{
    // Initialize additional members specific to ImplicitOcpType if needed
    int max_nx = *std::max_element(info.dims.number_of_states.begin(), info.dims.number_of_states.end());
    int max_nu = *std::max_element(info.dims.number_of_controls.begin(), info.dims.number_of_controls.end());
    int max_ng = *std::max_element(info.dims.number_of_eq_constraints.begin(), info.dims.number_of_eq_constraints.end());
    
    number_of_states = info.dims.number_of_states;
    number_of_controls = info.dims.number_of_controls;
    number_of_eq_constraints = info.dims.number_of_eq_constraints;
    number_of_ineq_constraints = info.dims.number_of_ineq_constraints;
    
    f_copy.emplace_back(info.number_of_primal_variables);
    g_copy.emplace_back(info.number_of_eq_constraints);
    D_x_copy.emplace_back(info.number_of_primal_variables);
    D_s_copy.emplace_back(info.number_of_slack_variables);
    D_eq_copy.emplace_back(info.number_of_eq_constraints);
    x_copy.emplace_back(info.number_of_primal_variables);
    eq_mult_copy.emplace_back(info.number_of_eq_constraints + info.number_of_slack_variables);

    int dim = std::max(max_nu + max_nx + 1, max_ng);
    scratch = std::make_unique<MatRealAllocated>(dim + 8, dim);
    JBAbt.emplace_back(dim + max_nx, max_nx);
    JBAbt_modified.emplace_back(dim + max_nx, max_nx);
};

LinsolReturnFlag AugSystemSolver<ImplicitOcpType>::solve(const ProblemInfo &info,
                                           Jacobian<ImplicitOcpType> &jacobian, Hessian<ImplicitOcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_s,
                                           const VecRealView &f, const VecRealView &g,
                                           VecRealView &x, VecRealView &eq_mult)
{
    if (print_debug) {std::cout << "AugSystemSolver<ImplicitOcpType> solve start" << std::endl;}
    // copy the rhs since they are altered during preprocessing and are needed for checking the solution
    auto start = std::chrono::high_resolution_clock::now();
    veccp(info.number_of_primal_variables, f, 0, f_copy[0], 0);
    veccp(info.number_of_eq_constraints, g, 0, g_copy[0], 0);
    veccp(info.number_of_primal_variables, D_x, 0, D_x_copy[0], 0);
    veccp(info.number_of_slack_variables, D_s, 0, D_s_copy[0], 0);

    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    ProblemInfo modified_info = PreProcess(info, jacobian, hessian, f_copy[0], g_copy[0], &(D_x_copy[0]), nullptr, &(D_s_copy[0]));
    stop = std::chrono::high_resolution_clock::now();
    duration_preprocess = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    LinsolReturnFlag flag = ModifiedAugSystemSolver::solve(modified_info, jacobian, hessian, D_x_copy[0], D_s_copy[0], f_copy[0], g_copy[0], x, eq_mult);

    if (write_preprocessing_file){
        PrintPreProcessNpInfo(info, modified_info, hessian, jacobian, x, eq_mult, D_x);
    }

    if (print_preprocessed_system){
        std::cout << "KKT matrix:" << std::endl;
        PrintNpArray(GetKKT(modified_info, jacobian, hessian, true), "KKT");
        VecRealAllocated full_rhs = VecRealAllocated(modified_info.number_of_primal_variables + modified_info.number_of_eq_constraints);
        for (Index i = 0; i < info.number_of_primal_variables; ++i){full_rhs(i) = (f_copy[0])(i) + (D_x_copy[0])(i)*x(i);}
        for (Index i = 0; i < info.number_of_eq_constraints; ++i){full_rhs(info.number_of_primal_variables + i) = (g_copy[0])(i);}
        for (Index i = 0; i < info.number_of_slack_variables; ++i){
            full_rhs(info.number_of_primal_variables + info.offset_g_eq_slack + i) -= D_s(i) * eq_mult(info.offset_g_eq_slack + i);
        }
        PrintNpArray(full_rhs, "rhs");
        std::cout << "obtained x:" << std::endl << x << std::endl;
        std::cout << "obtained eq_mult:" << std::endl << eq_mult << std::endl;
    }
    if (verify_preprocessed_solution){
        VerifyIntermediateSolution(modified_info, jacobian, hessian, x, eq_mult, f_copy[0], g_copy[0]);    
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult, &D_s, nullptr, g);
    stop = std::chrono::high_resolution_clock::now();
    duration_postprocess = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    if (print_final_solution){
        std::string file = "final_solution.py";
        std::ofstream o(file);
        o << "import numpy as np\n\n";
        MatRealAllocated x_mat(info.number_of_primal_variables, 1);
        for (int i = 0; i < info.number_of_primal_variables; i++){
            x_mat(i,0) = x(i);
        }
        PrintNpArray(x_mat, "x", info.number_of_primal_variables, 1, true, o);
        MatRealAllocated eq_mult_mat(info.number_of_eq_constraints, 1);
        for (int i = 0; i < info.number_of_eq_constraints; i++){
            eq_mult_mat(i,0) = eq_mult(i);
        }
        PrintNpArray(eq_mult_mat, "eq_mult", info.number_of_eq_constraints, 1, true, o);
    }
    if (print_debug) {std::cout << "AugSystemSolver<ImplicitOcpType> solve end" << std::endl;}
    return flag;
}
LinsolReturnFlag AugSystemSolver<ImplicitOcpType>::solve(const ProblemInfo &info,
                                           Jacobian<ImplicitOcpType> &jacobian, Hessian<ImplicitOcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_eq,
                                           const VecRealView &D_s, const VecRealView &f,
                                           const VecRealView &g, VecRealView &x,
                                           VecRealView &eq_mult)
{
    if (print_debug) {std::cout << "AugSystemSolver<ImplicitOcpType> solve start" << std::endl;}
    // copy the rhs since they are altered during preprocessing and are needed for checking the solution
    auto start = std::chrono::high_resolution_clock::now();
    veccp(info.number_of_primal_variables, f, 0, f_copy[0], 0);
    veccp(info.number_of_eq_constraints, g, 0, g_copy[0], 0);
    veccp(info.number_of_primal_variables, D_x, 0, D_x_copy[0], 0);
    veccp(info.number_of_eq_constraints, D_eq, 0, D_eq_copy[0], 0);
    veccp(info.number_of_slack_variables, D_s, 0, D_s_copy[0], 0);

    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // return LinsolReturnFlag::SUCCESS;
    ProblemInfo modified_info = PreProcess(info, jacobian, hessian, f_copy[0], g_copy[0], &D_x_copy[0], &D_eq_copy[0], &D_s_copy[0]);
    start = std::chrono::high_resolution_clock::now();
    LinsolReturnFlag flag = ModifiedAugSystemSolver::solve(modified_info, jacobian, hessian, D_x_copy[0], D_eq_copy[0], D_s_copy[0], f_copy[0], g_copy[0], x, eq_mult);
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult, &D_s, &D_eq, g);
    if (print_debug) {std::cout << "AugSystemSolver<ImplicitOcpType> solve end" << std::endl;}
    return flag;
}

LinsolReturnFlag AugSystemSolver<ImplicitOcpType>::solve_rhs(const ProblemInfo &info,
                                               Jacobian<ImplicitOcpType> &jacobian,
                                               Hessian<ImplicitOcpType> &hessian,
                                               const VecRealView &D_s, const VecRealView &f,
                                               const VecRealView &g, VecRealView &x,
                                               VecRealView &eq_mult)
{
    auto start = std::chrono::high_resolution_clock::now();
    veccp(info.number_of_primal_variables, f, 0, f_copy[0], 0);
    veccp(info.number_of_eq_constraints, g, 0, g_copy[0], 0);
    veccp(info.number_of_slack_variables, D_s, 0, D_s_copy[0], 0);
    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // return LinsolReturnFlag::SUCCESS;
    ProblemInfo modified_info = PreProcess(info, jacobian, hessian, f_copy[0], g_copy[0], nullptr, nullptr, &D_s_copy[0]);
    start = std::chrono::high_resolution_clock::now();
    LinsolReturnFlag flag = ModifiedAugSystemSolver::solve_rhs(modified_info, jacobian, hessian, D_s, f_copy[0], g_copy[0], x, eq_mult);
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult, &D_s, nullptr, g);
    return flag;
}
LinsolReturnFlag AugSystemSolver<ImplicitOcpType>::solve_rhs(const ProblemInfo &info,
                                               Jacobian<ImplicitOcpType> &jacobian,
                                               Hessian<ImplicitOcpType> &hessian,
                                               const VecRealView &D_eq, const VecRealView &D_s,
                                               const VecRealView &f, const VecRealView &g,
                                               VecRealView &x, VecRealView &eq_mult)
{
    auto start = std::chrono::high_resolution_clock::now();
    veccp(info.number_of_primal_variables, f, 0, f_copy[0], 0);
    veccp(info.number_of_eq_constraints, g, 0, g_copy[0], 0);
    veccp(info.number_of_eq_constraints, D_eq, 0, D_eq_copy[0], 0);
    veccp(info.number_of_slack_variables, D_s, 0, D_s_copy[0], 0);

    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // return LinsolReturnFlag::SUCCESS;
    ProblemInfo modified_info = PreProcess(info, jacobian, hessian, f_copy[0], g_copy[0], nullptr, &D_eq_copy[0], &D_s_copy[0]);
    start = std::chrono::high_resolution_clock::now();
    LinsolReturnFlag flag = ModifiedAugSystemSolver::solve_rhs(modified_info, jacobian, hessian, D_eq_copy[0], D_s_copy[0], f_copy[0], g_copy[0], x, eq_mult);
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult, &D_s, &D_eq, g);
    return flag;
}

void AugSystemSolver<ImplicitOcpType>::set_performance_mode(bool set){
    if (set){
        print_debug = false;
        print_preprocessed_system = false;
        write_preprocessing_file = false;
        verify_preprocessed_solution = false;
        print_final_solution = false;
        ModifiedAugSystemSolver::set_performance_mode(true);
    }
}

ProblemInfo AugSystemSolver<ImplicitOcpType>::PreProcess(const ProblemInfo &info,
                                                  Jacobian<ImplicitOcpType> &jacobian,
                                                  Hessian<ImplicitOcpType> &hessian,
                                                  VecRealView &f,
                                                  VecRealView &g,
                                                  VecRealView* D_x,
                                                  VecRealView* D_eq,
                                                  VecRealView* D_s)
{
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::PreProcess start" << std::endl;}

    // GENERAL VERSION
    auto start = std::chrono::high_resolution_clock::now();
    jacobian.PreProcess(info, f, g);
    auto stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_jac = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::jacobian::PreProcess done" << std::endl;}

    start = std::chrono::high_resolution_clock::now();
    hessian.PreProcess(info, jacobian, f, g);
    stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_hess = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::hessian::PreProcess done" << std::endl;}

    start = std::chrono::high_resolution_clock::now();
    int K = info.dims.K;
    bool USE_NEW_REGULARIZATION_TREATMENT = true;
    if (USE_NEW_REGULARIZATION_TREATMENT){
    // Deal with regularization terms
    for (int k = 0; k < K; ++k){
        const Index nx = number_of_states[k];
        const Index nx_next = number_of_states[k + 1];
        const Index nu = number_of_controls[k];
        const Index nu_next = number_of_controls[k + 1];
        const Index ng = number_of_eq_constraints[k];
        const Index ng_ineq = number_of_ineq_constraints[k];
        const Index offset_eq_k = info.offsets_g_eq_path[k];
        const Index offs_ineq_k = info.offsets_slack[k];
        const Index offset_g_ineq_k = info.offsets_g_eq_slack[k];
        const Index offset_u = info.offsets_primal_u[k];

        trtr_l(nu + nx, hessian.RSQrqt[k], 0, 0, hessian.RSQrqt[k], 0, 0); // copy lower part of RSQ to upper part
        if (D_eq != nullptr)
        // equality penalty
        {
            for (Index i = 0; i < ng; i++)
            {
                Scalar scaling_factor = 1.0 / (*D_eq)(offset_eq_k + i);
                colsc(nu + nx + 1, scaling_factor, jacobian.Gg_eqt[k], 0, i);
            }
            // add the penalty to hessian
            syrk_ln_mn(nu + nx + 1, nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, jacobian.Gg_eqt_original[k], 0, 0,
                       1.0, hessian.RSQrqt[k], 0, 0, hessian.RSQrqt[k], 0, 0);
            // add the penalty to rhs
            gemv_n(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, g, offset_eq_k, 1.0, f, offset_u, f, offset_u);

            vecse(ng, 1.0, *D_eq, offset_eq_k);
        }
        if (D_s != nullptr)
        // inequalities + inertia correction
        {
            if (ng_ineq > 0)
            {
                for (Index i = 0; i < ng_ineq; i++)
                {
                    Scalar scaling_factor = 1.0 / (*D_s)(offs_ineq_k + i);
                    colsc(nu + nx + 1, scaling_factor, jacobian.Gg_ineqt[k], 0, i);
                }
                // add the penalty to hessian
                syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0,
                           jacobian.Gg_ineqt_original[k], 0, 0, 1.0, hessian.RSQrqt[k], 0, 0, hessian.RSQrqt[k],
                           0, 0);
                // add the penalty to rhs
                gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, g, offset_g_ineq_k, 1.0, f, offset_u, f, offset_u);

                vecse(ng_ineq, 1.0, *D_s, offs_ineq_k);
            }
        }
        if (D_x != nullptr)
        {
        // inertia correction
        diaad(nu + nx, 1.0, *D_x, offset_u, hessian.RSQrqt[k], 0, 0);
        vecse(nu + nx, 0.0, *D_x, offset_u);
        }
    }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_regularization = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Pre-process 
    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < K-1; ++k){
        int nx = number_of_states[k];
        int nx_next = number_of_states[k + 1];
        int nu = number_of_controls[k];
        int nu_next = number_of_controls[k + 1];

        // construct JABbt-matrix
        int rank;
        // std::vector<MatRealAllocated> JBAbt = {MatRealAllocated(nx_next + nu + nx + 1, nx_next)};
        // std::vector<MatRealAllocated> JBAbt_modified = {MatRealAllocated(nx_next + nu + nx + 1, nx_next)};
        auto inner_start = std::chrono::high_resolution_clock::now();
        gecp(nx_next, nx_next, jacobian.Jt[k], 0, 0, JBAbt[0], 0, 0);
        gecp(nu + nx + 1, nx_next, jacobian.BAbt[k], 0, 0, JBAbt[0], nx_next, 0);
        gecp(nx_next + nu + nx + 1, nx_next, JBAbt[0], 0, 0, JBAbt_modified[0], 0, 0);
        auto inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_copies += std::chrono::duration_cast<std::chrono::microseconds>(inner_stop - inner_start);

        // decompose J matrix
        inner_start = std::chrono::high_resolution_clock::now();
        lu_fact_transposed(nx_next, nx_next + nu + nx + 1, nx_next, rank, JBAbt[0], jacobian.Pl_pre[k], jacobian.Pr_pre[k], lu_fact_tol);
        gecp(rank, rank, JBAbt[0], 0, 0, jacobian.U1t[k], 0, 0);
        // PrintLuInfo(JBAbt, jacobian.Jt[k], jacobian.Pl_pre[k], jacobian.Pr_pre[k]);
        inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_decomp += std::chrono::duration_cast<std::chrono::microseconds>(inner_stop - inner_start);
        inner_start = std::chrono::high_resolution_clock::now();

        if (nu + nx < info.dims.number_of_eq_constraints[k] + nx_next - rank){
            // there will be more constraints at this stage than can be
            // satisfied using the constrols and the states.
            // likely, the problem is ill-defined
            throw std::runtime_error("The problem seems to be ill-defined "
                "since the number of constraints at stage " + std::to_string(k) + 
                " (" + std::to_string(info.dims.number_of_eq_constraints[k]) + " + " + std::to_string(nx_next-rank) + ") "
                "exceeds the number of controls and states (" + std::to_string(nu) + " + " + std::to_string(nx) + ") "
                "that can be used to satisfy them.");
        }
        // extend permutation matrix to include nu_next part
        PermutationMatrix Pr_extended = PermutationMatrix(nu_next + rank);
        for (int i = 0; i < rank; i++){ Pr_extended[nu_next + i] = nu_next + jacobian.Pr_pre[k][i];}

        // Modify dynamics jacobian
        jacobian.Pl_pre[k].apply_on_cols(rank, &(JBAbt_modified[0]).mat());                                              // * P_l
        trsm_runu(nx_next + nu + nx + 1, nx_next, 1.0, JBAbt[0], 0, 0, JBAbt_modified[0], 0, 0, JBAbt_modified[0], 0, 0);    // * L^-1
        trsm_rlnn(nx_next + nu + nx + 1, rank, -1.0, JBAbt[0], 0, 0, JBAbt_modified[0], 0, 0, JBAbt_modified[0], 0, 0);      // * U^-1
        gecp(nu + nx + 1, nx_next, JBAbt_modified[0], nx_next, 0, jacobian.BAbt[k], 0, 0);
        jacobian.Pr_pre[k].apply_on_rows(rank, &(JBAbt_modified[0]).mat());                                              // * P_r
        gecp(nx_next-rank, rank, JBAbt_modified[0], rank, 0, jacobian.U1U2t[k], 0, 0);

        // other hessian contribution
        trtr_l(nu_next + nx_next, hessian.RSQrqt[k+1], 0, 0, hessian.RSQrqt[k+1], 0, 0);
        if (!USE_NEW_REGULARIZATION_TREATMENT){
        if (D_x != nullptr){
            // consider regularization already here
            diaad(nu_next + nx_next, 1.0, *D_x, info.offsets_primal_u[k+1], hessian.RSQrqt[k+1], 0, 0);
            // make sure to not consider them later on again
            vecsc(nu_next + nx_next, 0.0, *D_x, info.offsets_primal_u[k+1]);

            // don't skip k == 0
            if (k == 0){
                diaad(nu + nx, 1.0, *D_x, info.offsets_primal_u[k], hessian.RSQrqt[k], 0, 0);
                vecsc(nu + nx, 0.0, *D_x, info.offsets_primal_u[k]);
            }
        }
        }
        inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_scale1 += std::chrono::duration_cast<std::chrono::microseconds>(inner_stop - inner_start);
        inner_start = std::chrono::high_resolution_clock::now();

        // right-multiply right part with Dr^-1
        Pr_extended.apply_on_rows(nu_next + rank, &hessian.RSQrqt[k+1].mat());
        gemm_nn(nx_next - rank, nu_next + nx_next, rank, 1.0, jacobian.U1U2t[k], 0, 0, hessian.RSQrqt[k+1], nu_next, 0, 1.0, 
                hessian.RSQrqt[k+1], nu_next + rank, 0, hessian.RSQrqt[k+1], nu_next + rank, 0);
        // left-multiply bottom part with Dr^-T
        Pr_extended.apply_on_cols(nu_next + rank, &hessian.RSQrqt[k+1].mat());
        gemm_nt(nu_next + nx_next + 1, nx_next - rank, rank, 1.0, hessian.RSQrqt[k+1], 0, nu_next, jacobian.U1U2t[k], 0, 0, 1.0, 
                hessian.RSQrqt[k+1], 0, nu_next + rank, hessian.RSQrqt[k+1], 0, nu_next + rank);


        // hessian contribution of dynamics
        jacobian.Pr_pre[k].apply_on_cols(rank, &hessian.FuFx[k].mat());
        gemm_nt(nu + nx, nx_next - rank, rank, 1.0, hessian.FuFx[k], 0, 0, jacobian.U1U2t[k], 0, 0, 1.0, hessian.FuFx[k], 0, rank, hessian.FuFx[k], 0, rank);
            
        // right multiply with Dr^-1
        if (k < K - 2){
            int nx_next_next = number_of_states[k + 2];
            // dynamics jacobian
            Pr_extended.apply_on_rows(nu_next + rank, &jacobian.BAbt[k+1].mat());
            gemm_nn(nx_next - rank, nx_next_next, rank, 1.0, jacobian.U1U2t[k], 0, 0, jacobian.BAbt[k+1], nu_next, 0, 1.0, 
                    jacobian.BAbt[k+1], nu_next + rank, 0, jacobian.BAbt[k+1], nu_next + rank, 0);
            // dynamics hessian
            Pr_extended.apply_on_rows(nu_next + rank, &hessian.FuFx[k+1].mat());
            gemm_nn(nx_next - rank, nx_next_next, rank, 1.0, jacobian.U1U2t[k], 0, 0, hessian.FuFx[k+1], nu_next, 0, 1.0, hessian.FuFx[k+1], nu_next + rank, 0, hessian.FuFx[k+1], nu_next + rank, 0);
        }

        // equality constraints
        Pr_extended.apply_on_rows(nu_next + rank, &jacobian.Gg_eqt[k+1].mat());
        gemm_nn(nx_next - rank, info.dims.number_of_eq_constraints[k+1], rank, 1.0, jacobian.U1U2t[k], 0, 0, jacobian.Gg_eqt[k+1], nu_next, 0, 1.0, jacobian.Gg_eqt[k+1], nu_next + rank, 0, jacobian.Gg_eqt[k+1], nu_next + rank, 0);
        // inequality constraints
        // Pr_extended.apply_on_rows(nu_next + rank, &jacobian.Gg_ineqt[k+1].mat());
        // gemm_nn(nx_next - rank, info.dims.number_of_ineq_constraints[k+1], rank, 1.0, jacobian.U1U2t[k], 0, 0, jacobian.Gg_ineqt[k+1], nu_next, 0, 1.0, jacobian.Gg_ineqt[k+1], nu_next + rank, 0, jacobian.Gg_ineqt[k+1], nu_next + rank, 0);
        inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_scale2 += std::chrono::duration_cast<std::chrono::microseconds>(inner_stop - inner_start);

        inner_start = std::chrono::high_resolution_clock::now();
        // Move undefined states to controls
        if (rank < nx_next){
            gecp(nu + nx, nx_next - rank, hessian.FuFx[k], 0, rank, hessian.GuGx[k], 0, nu_next);
            TreatStatesAsInputs(nu_next, nx_next, rank, hessian.RSQrqt[k+1], true);
            TreatStatesAsInputs(nu_next, nx_next, rank, hessian.RSQrqt[k+1]);
            TreatStatesAsInputs(nu_next, nx_next, rank, jacobian.Gg_eqt[k+1], true);
            // TreatStatesAsInputs(nu_next, nx_next, rank, jacobian.Gg_ineqt[k+1], true);
            if (k < K - 2){
                TreatStatesAsInputs(nu_next, nx_next, rank, jacobian.BAbt[k+1], true);
                TreatStatesAsInputs(nu_next, nx_next, rank, hessian.FuFx[k+1], true);
            }

            // treat some of the dynamics constraints as path constraints
            gecp(nu + nx + 1, nx_next - rank, jacobian.BAbt[k], 0, rank, jacobian.Gg_eqt[k], 0, info.dims.number_of_eq_constraints[k]);

            // update dimensions
            int sk = nx_next - rank;
            number_of_controls[k + 1] += sk;
            number_of_eq_constraints[k] += sk;
            number_of_states[k + 1] = rank;
        }
        inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_permutation += std::chrono::duration_cast<std::chrono::microseconds>(inner_stop - inner_start);

        // store info
        inner_start = std::chrono::high_resolution_clock::now();
        jacobian.J_ranks[k] = rank;
        // jacobian.Jt_LU[k] = JBAbt[0];
        gecp(nx_next, nx_next, JBAbt[0], 0, 0, jacobian.Jt_LU[k], 0, 0);
        inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_store += std::chrono::duration_cast<std::chrono::microseconds>(inner_stop - inner_start);
    }
    stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_decomposition += std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    ProblemInfo modified_info(ProblemDims(K, number_of_controls, number_of_states, number_of_eq_constraints, number_of_ineq_constraints));
    stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_info += std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // modify right-hand side
    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < K; ++k){
        int nx = modified_info.dims.number_of_states[k];
        int nu = modified_info.dims.number_of_controls[k];

        // modify f
        // for (int i = 0; i < nu + nx; i++){
        //     f(modified_info.offsets_primal_u[k] + i) = hessian.RSQrqt[k](nu + nx, i);
        // }
        rowex(nu + nx, 1.0, hessian.RSQrqt[k], nu + nx, 0, f, modified_info.offsets_primal_u[k]);

        // modify g
        // for (int i = 0; i < modified_info.dims.number_of_eq_constraints[k]; i++){
        //     g(modified_info.offsets_g_eq_path[k] + i) = jacobian.Gg_eqt[k](nu + nx, i);
        // }
        rowex(modified_info.dims.number_of_eq_constraints[k], 1.0, jacobian.Gg_eqt[k], nu + nx, 0, g, modified_info.offsets_g_eq_path[k]);

        if (k < K - 1){
            int nx_next = modified_info.dims.number_of_states[k + 1];
            // for (int i = 0; i < nx_next; i++){
            //     g(modified_info.offsets_g_eq_dyn[k] + i) = jacobian.BAbt[k](nu + nx, i);
            // }
            rowex(nx_next, 1.0, jacobian.BAbt[k], nu + nx, 0, g, modified_info.offsets_g_eq_dyn[k]); 
        }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_modify_rhs = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::PreProcess done" << std::endl;}   
    return modified_info;
}

void AugSystemSolver<ImplicitOcpType>::PostProcess(const ProblemInfo &info,
                                                   const ProblemInfo &modified_info,
                                                   Jacobian<ImplicitOcpType> &jacobian,
                                                   Hessian<ImplicitOcpType> &hessian,
                                                   VecRealView &x, VecRealView &eq_mult,
                                                   const VecRealView* D_s, 
                                                   const VecRealView* D_eq,
                                                   const VecRealView &g){   
    // GENERAL VERSION
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::PostProcess start" << std::endl;}
    // VecRealAllocated x_copy(x.m());
    // for (int i = 0; i < x.m(); i++){
    //     x_copy(i) = x(i);
    // }
    // VecRealAllocated eq_mult_copy(eq_mult.m());
    // for (int i = 0; i < eq_mult.m(); i++){
    //     eq_mult_copy(i) = eq_mult(i);
    // }
    veccp(x.m(), x, 0, x_copy[0], 0);
    veccp(eq_mult.m(), eq_mult, 0, eq_mult_copy[0], 0);

    for (int k = 0; k < info.dims.K; ++k){
        Index nu = info.dims.number_of_controls[k];
        Index nu_mod = modified_info.dims.number_of_controls[k];
        Index s = (k < info.dims.K - 1) ? info.dims.number_of_states[k + 1] - jacobian.J_ranks[k] : 0;
        Index s_states = (k > 0) ? info.dims.number_of_states[k] - jacobian.J_ranks[k-1] : 0;

        // controls (plain copy)
        // for (int i = 0; i < nu; i++){
        //     x(info.offsets_primal_u[k] + i) = (x_copy[0])(modified_info.offsets_primal_u[k] + i);
        // }
        veccp(nu, x_copy[0], modified_info.offsets_primal_u[k], x, info.offsets_primal_u[k]);

        // states (copy existing states, and append additional states treated as controls)
        if (k > 0){
            // for (int i = 0; i < jacobian.J_ranks[k-1]; i++){
            //     x(info.offsets_primal_x[k] + i) = (x_copy[0])(modified_info.offsets_primal_x[k] + i);
            // }
            veccp(jacobian.J_ranks[k-1], x_copy[0], modified_info.offsets_primal_x[k], x, info.offsets_primal_x[k]);
            // for (int i = 0; i < s_states; i++){
            //     x(info.offsets_primal_x[k] + jacobian.J_ranks[k-1] + i) = (x_copy[0])(modified_info.offsets_primal_u[k] + nu + i);
            // }
            veccp(s_states, x_copy[0], modified_info.offsets_primal_u[k] + nu, x, info.offsets_primal_x[k] + jacobian.J_ranks[k-1]);
        } else {
            // for (int i = 0; i < info.dims.number_of_states[k]; i++){
            //     x(info.offsets_primal_x[k] + i) = (x_copy[0])(modified_info.offsets_primal_x[k] + i);
            // }
            veccp(info.dims.number_of_states[k], x_copy[0], modified_info.offsets_primal_x[k], x, info.offsets_primal_x[k]);
        }
        if (k < info.dims.K - 1){
            // dynamics (copy existing dynamics, and append additional path constraints)
            // for (int i = 0; i < jacobian.J_ranks[k]; i++){
            //     eq_mult(info.offsets_g_eq_dyn[k] + i) = (eq_mult_copy[0])(modified_info.offsets_g_eq_dyn[k] + i);
            // }
            veccp(jacobian.J_ranks[k], eq_mult_copy[0], modified_info.offsets_g_eq_dyn[k], eq_mult, info.offsets_g_eq_dyn[k]);
            // for (int i = 0; i < s; i++){
            //     eq_mult(info.offsets_g_eq_dyn[k] + jacobian.J_ranks[k] + i) = 
            //         (eq_mult_copy[0])(modified_info.offsets_g_eq_path[k] + info.dims.number_of_eq_constraints[k] + i);
            // }
            veccp(s, eq_mult_copy[0], modified_info.offsets_g_eq_path[k] + info.dims.number_of_eq_constraints[k], eq_mult, info.offsets_g_eq_dyn[k] + jacobian.J_ranks[k]);
        }

        // equality path constraints (plain copy)
        // for (int i = 0; i < info.dims.number_of_eq_constraints[k]; i++){
        //     eq_mult(info.offsets_g_eq_path[k] + i) = (eq_mult_copy[0])(modified_info.offsets_g_eq_path[k] + i);
        // }
        veccp(info.dims.number_of_eq_constraints[k], eq_mult_copy[0], modified_info.offsets_g_eq_path[k], eq_mult, info.offsets_g_eq_path[k]);

        // inequality path constraints (plain copy)
        // for (int i = 0; i < info.dims.number_of_ineq_constraints[k]; i++){
        //     eq_mult(info.offsets_g_eq_slack[k] + i) = (eq_mult_copy[0])(modified_info.offsets_g_eq_slack[k] + i);
        // }
        veccp(info.dims.number_of_ineq_constraints[k], eq_mult_copy[0], modified_info.offsets_g_eq_slack[k], eq_mult, info.offsets_g_eq_slack[k]);
        
        // scale states and dynamics multipliers
        if (k > 0){
            // if (k < info.dims.K){
            gemv_t(info.dims.number_of_states[k] - jacobian.J_ranks[k-1], info.dims.number_of_states[k], 1.0, 
                jacobian.U1U2t[k-1], 0, 0, 
                x, info.offsets_primal_x[k] + jacobian.J_ranks[k-1], 1.0, 
                x, info.offsets_primal_x[k], 
                x, info.offsets_primal_x[k]);
            jacobian.Pr_pre[k-1].apply_inverse(jacobian.J_ranks[k-1], &x.vec(), info.offsets_primal_x[k]);
            // } 

            // U^-T * 
            // PrintNpArray(eq_mult, info.offsets_g_eq_dyn[k-1], info.dims.number_of_states[k], "\n[" + std::to_string(k) + "] eq_mult before");
            trsv_lnn(jacobian.J_ranks[k-1], jacobian.U1t[k-1], 0, 0, eq_mult, info.offsets_g_eq_dyn[k-1], eq_mult, info.offsets_g_eq_dyn[k-1]);
            vecsc(jacobian.J_ranks[k-1], -1.0, eq_mult, info.offsets_g_eq_dyn[k-1]);
            // L^-T *
            trsv_unu(info.dims.number_of_states[k], info.dims.number_of_states[k], jacobian.Jt_LU[k-1], 0, 0, eq_mult, info.offsets_g_eq_dyn[k-1], eq_mult, info.offsets_g_eq_dyn[k-1]);
            // Pl * 
            jacobian.Pl_pre[k-1].apply_inverse(jacobian.J_ranks[k-1], &eq_mult.vec(), info.offsets_g_eq_dyn[k-1]);
            // std::cout << "eq_mult after:\n" << eq_mult << std::endl;
            // PrintNpArray(eq_mult, info.offsets_g_eq_dyn[k-1], info.dims.number_of_states[k], "[" + std::to_string(k) + "] eq_mult after");
        }
    }

    jacobian.ResetPreProcess(info);
    hessian.ResetPreProcess(info, jacobian);

    // Consider constraint regularizations
    if (D_s != nullptr){
        for (int k = 0; k < info.dims.K; ++k){
            const Index nu = info.dims.number_of_controls[k];
            const Index nx = info.dims.number_of_states[k];
            const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
            const Index offs = info.offsets_primal_u[k];
            const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
            const Index offs_slack = info.offsets_slack[k];
            // PrintNpArray(jacobian.Gg_ineqt[k], "\nGg_ineqt[" + std::to_string(k) + "]", nu + nx + 1, ng_ineq);
            // PrintNpArray(*D_s, offs_slack, ng_ineq, "D_s[" + std::to_string(k) + "]");
            // PrintNpArray(g, offs_eq_ineq, ng_ineq, "g[" + std::to_string(k) + "]");
            // PrintNpArray(x, offs, nu + nx, "x[" + std::to_string(k) + "]");
            // PrintNpArray(eq_mult, offs_eq_ineq, ng_ineq, "[" + std::to_string(k) + "] eq_mult before ineq regularization");
            // std::cout << "nu: " << nu << " nx: " << nx << " ng_ineq: " << ng_ineq << std::endl;
            // std::cout << "offs: " << offs << " offs_eq_ineq: " << offs_eq_ineq << " offs_slack: " << offs_slack << std::endl;
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / (*D_s).block(ng_ineq, offs_slack);
            // PrintNpArray(eq_mult, offs_eq_ineq, ng_ineq, "[" + std::to_string(k) + "] eq_mult after ineq regularization");
        }
    }
    if (D_eq != nullptr){
        for (int k = 0; k < info.dims.K; ++k){
            const Index nu = info.dims.number_of_controls[k];
            const Index nx = info.dims.number_of_states[k];
            const Index offs = info.offsets_primal_u[k];
            const Index ng = info.dims.number_of_eq_constraints[k];
            const Index offs_g_eq_k = info.offsets_g_eq_path[k];
            const Index offs_eq_k = info.offsets_eq[k];            
            if (ng > 0)
            {
                gemv_t(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, x, offs, 1.0, g, offs_g_eq_k,
                    eq_mult, offs_g_eq_k);
                eq_mult.block(ng, offs_g_eq_k) =
                    eq_mult.block(ng, offs_g_eq_k) / (*D_eq).block(ng, offs_eq_k);
            }
        }
    }
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::PostProcess done" << std::endl;}
}

void AugSystemSolver<ImplicitOcpType>::TreatStatesAsInputs(Index nu_next, Index nx_next, Index rank, MatRealAllocated& A, bool rows){
    // std::cout << "Matrix before shifting states to inputs:" << std::endl;
    // std::cout << A << std::endl;
    if (!rows){
        // copy nu_next + nx_next columns to scratch
        gecp(A.m(), nu_next + nx_next, A, 0, 0, *scratch, 0, 0);

        // put last nx_next - rank columns in front
        gecp(A.m(), nx_next - rank, *scratch, 0, nu_next + rank, A, 0, nu_next);

        // shift back the first rank columns
        gecp(A.m(), rank, *scratch, 0, nu_next, A, 0, nu_next + nx_next - rank);

        // insert copied columns
        // gecp(A.m(), nx_next - rank, *scratch, 0, 0, A, 0, nu_next);
    } else {
        // copy nu_next + nx_next rows to scratch
        gecp(nu_next + nx_next, A.n(), A, 0, 0, *scratch, 0, 0);

        // put last nx_next - rank rows in front
        gecp(nx_next - rank, A.n(), *scratch, nu_next + rank, 0, A, nu_next, 0);

        // shift back the first rank rows
        gecp(rank, A.n(), *scratch, nu_next, 0, A, nu_next + nx_next - rank, 0);

        // insert copied rows
        // gecp(nx_next - rank, A.n(), *scratch, 0, 0, A, 0, 0);
    }
    // std::cout << "Matrix after shifting states to inputs:" << std::endl;
    // std::cout << A << std::endl;
}