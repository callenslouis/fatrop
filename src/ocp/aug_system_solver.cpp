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
        RSQrqt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Ggt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Llt.emplace_back(nu + nx + 1, nu);
    }

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
};

LinsolReturnFlag ModifiedAugSystemSolver::solve(const ProblemInfo &info,
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
LinsolReturnFlag ModifiedAugSystemSolver::solve(const ProblemInfo &info,
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

LinsolReturnFlag ModifiedAugSystemSolver::solve_rhs(const ProblemInfo &info,
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
LinsolReturnFlag ModifiedAugSystemSolver::solve_rhs(const ProblemInfo &info,
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







































































void PrintNpArray(MatRealAllocated const &A, std::string name){
    std::cout << name << " = np.array([\n\t";
    for (int i = 0; i < A.m(); i++){
        std::cout << "[";
        for (int j = 0; j < A.n(); j++){
            std::cout << A(i,j);
            if (j < A.n() - 1){ std::cout << ",";}
            std::cout << " ";
        }
        std::cout << "],\n\t";        
    }
    std::cout << "])" << std::endl;
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

MatRealAllocated GetKKT(const ProblemInfo &info,
                        Jacobian<ImplicitOcpType> &jacobian,
                        Hessian<ImplicitOcpType> &hessian){
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
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offs_eq_dyn = info.offsets_g_eq_dyn[k];
        full_matrix_jacobian.block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
            transpose(jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0));
        full_matrix_jacobian.block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
            transpose(jacobian.Jt[k]);
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
    PrintNpArray(full_kkt_matrix, "KKT");

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
    jacobian.transpose_apply_on_right(info, mult, 1.0, solution_grad, solution_grad);

    jacobian.apply_on_right(info, x, 0.0, solution_g, solution_g);

    for (int i = 0; i < info.number_of_primal_variables; i++){
        std::cout << solution_grad(i) << "\t-\t" << f(i) << std::endl;
    }

    std::cout << "------------" << std::endl;
    for (int i = 0; i < info.number_of_eq_constraints; i++){
        std::cout << solution_g(i) << "\t-\t" << g(i) << std::endl;
    }
}




AugSystemSolver<ImplicitOcpType>::AugSystemSolver(const ProblemInfo &info) : ModifiedAugSystemSolver(info)
{
    // Initialize additional members specific to ImplicitOcpType if needed
    int max_nx = *std::max_element(info.dims.number_of_states.begin(), info.dims.number_of_states.end());
    int max_nu = *std::max_element(info.dims.number_of_controls.begin(), info.dims.number_of_controls.end());
    int max_ng = *std::max_element(info.dims.number_of_eq_constraints.begin(), info.dims.number_of_eq_constraints.end());
    int dim = std::max(max_nu + max_nx + 1, max_ng);

    scratch = std::make_unique<MatRealAllocated>(dim, dim);
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
    VecRealAllocated f_copy(info.number_of_primal_variables);
    VecRealAllocated g_copy(info.number_of_eq_constraints);
    for (int i = 0; i < info.number_of_primal_variables; i++){
        f_copy(i) = f(i);
    }
    for (int i = 0; i < info.number_of_eq_constraints; i++){
        g_copy(i) = g(i);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    ProblemInfo modified_info = PreProcess(info, jacobian, hessian, f_copy, g_copy);
    start = std::chrono::high_resolution_clock::now();
    // LinsolReturnFlag flag = AugSystemSolver<OcpType>::solve(info, jacobian, hessian, D_x, D_s, f_copy, g_copy, x, eq_mult);
    LinsolReturnFlag flag = ModifiedAugSystemSolver::solve(modified_info, jacobian, hessian, D_x, D_s, f_copy, g_copy, x, eq_mult);
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult);
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
    VecRealAllocated f_copy(info.number_of_primal_variables);
    VecRealAllocated g_copy(info.number_of_eq_constraints);
    for (int i = 0; i < info.number_of_primal_variables; i++){
        f_copy(i) = f(i);
    }
    for (int i = 0; i < info.number_of_eq_constraints; i++){
        g_copy(i) = g(i);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // return LinsolReturnFlag::SUCCESS;
    ProblemInfo modified_info = PreProcess(info, jacobian, hessian, f_copy, g_copy);
    start = std::chrono::high_resolution_clock::now();
    // LinsolReturnFlag flag = AugSystemSolver<OcpType>::solve(info, jacobian, hessian, D_x, D_eq, D_s, f_copy, g_copy, x, eq_mult);
    LinsolReturnFlag flag = ModifiedAugSystemSolver::solve(modified_info, jacobian, hessian, D_x, D_eq, D_s, f_copy, g_copy, x, eq_mult);
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult);
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
    VecRealAllocated f_copy(info.number_of_primal_variables);
    VecRealAllocated g_copy(info.number_of_eq_constraints);
    for (int i = 0; i < info.number_of_primal_variables; i++){
        f_copy(i) = f(i);
    }
    for (int i = 0; i < info.number_of_eq_constraints; i++){
        g_copy(i) = g(i);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // return LinsolReturnFlag::SUCCESS;
    ProblemInfo modified_info = PreProcess(info, jacobian, hessian, f_copy, g_copy);
    start = std::chrono::high_resolution_clock::now();
    // LinsolReturnFlag flag = AugSystemSolver<OcpType>::solve_rhs(info, jacobian, hessian, D_s, f_copy, g_copy, x, eq_mult);
    LinsolReturnFlag flag = ModifiedAugSystemSolver::solve_rhs(modified_info, jacobian, hessian, D_s, f_copy, g_copy, x, eq_mult);
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult);
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
    VecRealAllocated f_copy(info.number_of_primal_variables);
    VecRealAllocated g_copy(info.number_of_eq_constraints);
    for (int i = 0; i < info.number_of_primal_variables; i++){
        f_copy(i) = f(i);
    }
    for (int i = 0; i < info.number_of_eq_constraints; i++){
        g_copy(i) = g(i);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // return LinsolReturnFlag::SUCCESS;
    ProblemInfo modified_info = PreProcess(info, jacobian, hessian, f_copy, g_copy);
    start = std::chrono::high_resolution_clock::now();
    // LinsolReturnFlag flag = AugSystemSolver<OcpType>::solve_rhs(info, jacobian, hessian, D_eq, D_s, f_copy, g_copy, x, eq_mult);
    LinsolReturnFlag flag = ModifiedAugSystemSolver::solve_rhs(modified_info, jacobian, hessian, D_eq, D_s, f_copy, g_copy, x, eq_mult);
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult);
    return flag;
}

ProblemInfo AugSystemSolver<ImplicitOcpType>::PreProcess(const ProblemInfo &info,
                                                  Jacobian<ImplicitOcpType> &jacobian,
                                                  Hessian<ImplicitOcpType> &hessian,
                                                  VecRealView &f,
                                                  VecRealView &g)
{
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::PreProcess start" << std::endl;}

    // GENERAL VERSION
    jacobian.PreProcess(info, f, g);
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::jacobian::PreProcess done" << std::endl;}
    hessian.PreProcess(info, jacobian, f, g);
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::hessian::PreProcess done" << std::endl;}

    int K = info.dims.K;
    std::vector<Index> number_of_states = info.dims.number_of_states;
    std::vector<Index> number_of_controls = info.dims.number_of_controls;
    std::vector<Index> number_of_eq_constraints = info.dims.number_of_eq_constraints;
    std::vector<Index> number_of_ineq_constraints = info.dims.number_of_ineq_constraints;
    for (int k = 0; k < K-1; ++k){
        int nx = number_of_states[k];
        int nx_next = number_of_states[k + 1];
        int nu = number_of_controls[k];
        int nu_next = (k < K-2) ? number_of_controls[k + 1] : 0;

        // construct JABbt-matrix
        int rank;
        MatRealAllocated JBAbt = MatRealAllocated(nx_next + nu + nx + 1, nx_next);
        MatRealAllocated JBAbt_modified = MatRealAllocated(nx_next + nu + nx + 1, nx_next);
        gecp(nx_next, nx_next, jacobian.Jt[k], 0, 0, JBAbt, 0, 0);
        gecp(nu + nx + 1, nx_next, jacobian.BAbt[k], 0, 0, JBAbt, nx_next, 0);
        gecp(nx_next + nu + nx + 1, nx_next, JBAbt, 0, 0, JBAbt_modified, 0, 0);

        // decompose J matrix
        lu_fact_transposed(nx_next, nx_next + nu + nx + 1, nx_next, rank, JBAbt, jacobian.Pl_pre[k], jacobian.Pr_pre[k], lu_fact_tol);
        gecp(rank, rank, JBAbt, 0, 0, jacobian.U1t[k], 0, 0);
        // if (rank < nx_next && k == K-2){
        //     throw std::runtime_error("Undefined states detected in last stage. An additional termainal stage should be added, this is not implemented yet.");
        // }
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
        // std::cout << "Pr_extended: " << std::endl;
        // for (int i = 0; i < nu_next + rank; i++){std::cout << Pr_extended[i] << " ";}
        // std::cout << std::endl;

        // Modify dynamics jacobian
        trsm_rlnn(nx_next + nu + nx + 1, rank, -1.0, JBAbt, 0, 0, JBAbt, 0, 0, JBAbt_modified, 0, 0);
        gecp(nu + nx + 1, nx_next, JBAbt_modified, nx_next, 0, jacobian.BAbt[k], 0, 0);
        gecp(nx_next-rank, rank, JBAbt_modified, rank, 0, jacobian.U1U2t[k], 0, 0);

        // other hessian contribution
        trtr_l(nu_next + nx_next, hessian.RSQrqt[k+1], 0, 0, hessian.RSQrqt[k+1], 0, 0); // copy lower part of RSQ to upper part
        // right-multiply second-column part with Dr^-1
        Pr_extended.apply_on_cols(nu_next + rank, &hessian.RSQrqt[k+1].mat());
        Pr_extended.apply_on_rows(nu_next + rank, &hessian.RSQrqt[k+1].mat());

        gemm_nn(nx_next - rank, nu_next + nx_next, rank, 1.0, JBAbt_modified, rank, 0, hessian.RSQrqt[k+1], nu_next, 0, 1.0, 
                hessian.RSQrqt[k+1], rank, 0, hessian.RSQrqt[k+1], rank, 0);
        // right-multiply S
        gemm_nn(nx_next - rank, nu_next, rank, 1.0, JBAbt_modified, rank, 0, hessian.RSQrqt[k+1], 0, nu_next, 1.0, 
                hessian.RSQrqt[k+1], rank, nu_next, hessian.RSQrqt[k+1], rank, nu_next);
        // left-multiply bottom part with Dl^-T
        gemm_nt(nu_next + nx_next + 1, nx_next - rank, rank, 1.0, hessian.RSQrqt[k+1], 0, nu_next, JBAbt_modified, rank, 0, 1.0, 
                hessian.RSQrqt[k+1], 0, nu_next + rank, hessian.RSQrqt[k+1], 0, nu_next + rank);
        // left-multiply S^T
        gemm_nt(nx_next, nx_next - rank, rank, 1.0, hessian.RSQrqt[k+1], nu_next, 0, JBAbt_modified, rank, 0, 1.0, 
                hessian.RSQrqt[k+1], nu_next, rank, hessian.RSQrqt[k+1], nu_next, rank);

        // hessian contribution of dynamics
        jacobian.Pr_pre[k].apply_on_cols(rank, &hessian.FuFxt[k].mat());
        gemm_nn(nx_next - rank, nx, rank, 1.0, JBAbt_modified, rank, 0, hessian.FuFxt[k], 0, 0, 1.0, hessian.FuFxt[k], rank, 0, hessian.FuFxt[k], rank, 0);
            
        // right multiply with Dr^-1
        if (k < K - 2){
            int nx_next_next = number_of_states[k + 2];
            // dynamics jacobian
            Pr_extended.apply_on_rows(nu_next + rank, &jacobian.BAbt[k+1].mat());
            gemm_nn(nx_next - rank, nx_next_next, rank, 1.0, JBAbt_modified, rank, 0, jacobian.BAbt[k+1], 0, nu_next, 1.0, 
                    jacobian.BAbt[k+1], rank, nu_next, jacobian.BAbt[k+1], rank, nu_next);
            // dynamics hessian
            Pr_extended.apply_on_cols(nu_next + rank, &hessian.FuFxt[k+1].mat());
            gemm_nn(nx_next - rank, nx_next_next, rank, 1.0, JBAbt_modified, rank, 0, hessian.FuFxt[k+1], 0, nu_next, 1.0, 
                    hessian.FuFxt[k+1], rank, nu_next, hessian.FuFxt[k+1], rank, nu_next);
        }
        // equality constraints
        Pr_extended.apply_on_rows(nu_next + rank, &jacobian.Gg_eqt[k+1].mat());
        gemm_nn(nx_next - rank, info.dims.number_of_eq_constraints[k+1], rank, 1.0, JBAbt_modified, rank, 0, jacobian.Gg_eqt[k+1], 0, nu_next, 1.0, jacobian.Gg_eqt[k+1], rank, nu_next, jacobian.Gg_eqt[k+1], rank, nu_next);
        // inequality constraints
        Pr_extended.apply_on_rows(nu_next + rank, &jacobian.Gg_ineqt[k+1].mat());
        gemm_nn(nx_next - rank, info.dims.number_of_ineq_constraints[k+1], rank, 1.0, JBAbt_modified, rank, 0, jacobian.Gg_ineqt[k+1], 0, nu_next, 1.0, jacobian.Gg_ineqt[k+1], rank, nu_next, jacobian.Gg_ineqt[k+1], rank, nu_next);


        // Move undefined states to controls
        if (rank < nx_next){
            gecp(nx_next - rank, nu + nx, hessian.FuFxt[k], rank, 0, hessian.GuGxt[k], nu_next, 0);
            TreatStatesAsInputs(nu_next, nx_next, rank, hessian.RSQrqt[k+1], true);
            TreatStatesAsInputs(nu_next, nx_next, rank, hessian.RSQrqt[k+1]);
            TreatStatesAsInputs(nu_next, nx_next, rank, jacobian.Gg_eqt[k+1], true);
            TreatStatesAsInputs(nu_next, nx_next, rank, jacobian.Gg_ineqt[k+1], true);
            if (k < K - 2){
                TreatStatesAsInputs(nu_next, nx_next, rank, jacobian.BAbt[k+1], true);
            }

            // treat some of the dynamics constraints as path constraints
            gecp(nu + nx + 1, nx_next - rank, jacobian.BAbt[k], 0, rank, jacobian.Gg_eqt[k], 0, info.dims.number_of_eq_constraints[k]);

            // update dimensions
            int sk = nx_next - rank;
            number_of_controls[k + 1] += sk;
            number_of_eq_constraints[k] += sk;
            number_of_states[k + 1] = rank;
        }

        // store info
        jacobian.J_ranks.push_back(rank);
        jacobian.Jt_LU[k] = JBAbt;
    }
    ProblemInfo modified_info(ProblemDims(K, number_of_controls, number_of_states, number_of_eq_constraints, number_of_ineq_constraints));

    // modify right-hand side
    VecRealAllocated f_original = VecRealAllocated(info.number_of_primal_variables);
    for (int i = 0; i < info.number_of_primal_variables; i++){
        f_original(i) = f(i);
    }
    VecRealAllocated g_original = VecRealAllocated(info.number_of_eq_constraints);
    for (int i = 0; i < info.number_of_eq_constraints; i++){
        g_original(i) = g(i);
    }
    for (int k = 0; k < K; ++k){
        int nx = modified_info.dims.number_of_states[k];
        int nu = modified_info.dims.number_of_controls[k];

        // modify f
        for (int i = 0; i < nu + nx; i++){
            f(modified_info.offsets_primal_u[k] + i) = hessian.RSQrqt[k](nu + nx, i);
        }

        // modify g
        for (int i = 0; i < modified_info.dims.number_of_eq_constraints[k]; i++){
            g(modified_info.offsets_g_eq_path[k] + i) = jacobian.Gg_eqt[k](nu + nx, i);
        }
        if (k < K - 1){
            int nx_next = modified_info.dims.number_of_states[k + 1];
            for (int i = 0; i < nx_next; i++){
                g(modified_info.offsets_g_eq_dyn[k] + i) = jacobian.BAbt[k](nu + nx, i);
            }
        }
    }

    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::PreProcess done" << std::endl;}
    return modified_info;
}

void AugSystemSolver<ImplicitOcpType>::PostProcess(const ProblemInfo &info,
                                                   const ProblemInfo &modified_info,
                                                   Jacobian<ImplicitOcpType> &jacobian,
                                                   Hessian<ImplicitOcpType> &hessian,
                                                   VecRealView &x, VecRealView &eq_mult){
    // GENERAL VERSION
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::PostProcess start" << std::endl;}
    VecRealAllocated x_copy(x.m());
    for (int i = 0; i < x.m(); i++){
        x_copy(i) = x(i);
    }
    VecRealAllocated eq_mult_copy(eq_mult.m());
    for (int i = 0; i < eq_mult.m(); i++){
        eq_mult_copy(i) = eq_mult(i);
    }

    for (int k = 0; k < info.dims.K; ++k){
        Index nu = info.dims.number_of_controls[k];
        Index nu_mod = modified_info.dims.number_of_controls[k];
        Index s = (k < info.dims.K - 1) ? info.dims.number_of_states[k + 1] - jacobian.J_ranks[k] : 0;
        Index s_states = (k > 0) ? info.dims.number_of_states[k] - jacobian.J_ranks[k-1] : 0;

        // controls (plain copy)
        for (int i = 0; i < nu; i++){
            x(info.offsets_primal_u[k] + i) = x_copy(modified_info.offsets_primal_u[k] + i);
        }

        // states (copy existing states, and append additional states treated as controls)
        if (k > 0){
            for (int i = 0; i < jacobian.J_ranks[k-1]; i++){
                x(info.offsets_primal_x[k] + i) = x_copy(modified_info.offsets_primal_x[k] + i);
            }
            for (int i = 0; i < s_states; i++){
                x(info.offsets_primal_x[k] + jacobian.J_ranks[k-1] + i) = x_copy(modified_info.offsets_primal_u[k] + nu + i);
            }
        } else {
            for (int i = 0; i < info.dims.number_of_states[k]; i++){
                x(info.offsets_primal_x[k] + i) = x_copy(modified_info.offsets_primal_x[k] + i);
            }
        }
        if (k < info.dims.K - 1){
            // dynamics (copy existing dynamics, and append additional path constraints)
            for (int i = 0; i < jacobian.J_ranks[k]; i++){
                eq_mult(info.offsets_g_eq_dyn[k] + i) = eq_mult_copy(modified_info.offsets_g_eq_dyn[k] + i);
            }
            for (int i = 0; i < s; i++){
                eq_mult(info.offsets_g_eq_dyn[k] + jacobian.J_ranks[k] + i) = 
                    eq_mult_copy(modified_info.offsets_g_eq_path[k] + info.dims.number_of_eq_constraints[k] + i);
            }
        }

        // equality path constraints (plain copy)
        for (int i = 0; i < info.dims.number_of_eq_constraints[k]; i++){
            eq_mult(info.offsets_g_eq_path[k] + i) = eq_mult_copy(modified_info.offsets_g_eq_path[k] + i);
        }

        // inequality path constraints (plain copy)
        for (int i = 0; i < info.dims.number_of_ineq_constraints[k]; i++){
            eq_mult(info.offsets_g_eq_slack[k] + i) = eq_mult_copy(modified_info.offsets_g_eq_slack[k] + i);
        }
        
        // scale states and dynamics multipliers
        if (k > 0){
            gemv_t(info.dims.number_of_states[k] - jacobian.J_ranks[k-1], info.dims.number_of_states[k], 1.0, 
                jacobian.Jt_LU[k-1], 0, jacobian.J_ranks[k-1], 
                x, info.offsets_primal_x[k] + jacobian.J_ranks[k-1], 1.0, 
                x, info.offsets_primal_x[k], 
                x, info.offsets_primal_x[k]);
            jacobian.Pr_pre[k-1].apply(jacobian.J_ranks[k-1], &x.vec(), info.offsets_primal_x[k]);

            trsv_lnn(jacobian.J_ranks[k-1], jacobian.U1t[k-1], 0, 0, eq_mult, info.offsets_g_eq_dyn[k-1],
                eq_mult, info.offsets_g_eq_dyn[k-1]);
            vecsc(jacobian.J_ranks[k-1], -1.0, eq_mult, info.offsets_g_eq_dyn[k-1]);            
        }
    }

    for (int k = 0; k < info.dims.K; k++){
        // std::cout << "inputs original: ";
        // for (int i = 0; i < info.dims.number_of_controls[k]; i++){
        //     std::cout << x(info.offsets_primal_u[k] + i) << "\t";
        // }
        // std::cout << "\ninputs modified: ";
        // for (int i = 0; i < modified_info.dims.number_of_controls[k]; i++){
        //     std::cout << x_copy(modified_info.offsets_primal_u[k] + i) << "\t";
        // }
        // std::cout << std::endl;
        // std::cout << "\nstates original: ";
        // for (int i = 0; i < info.dims.number_of_states[k]; i++){
        //     std::cout << x(info.offsets_primal_x[k] + i) << "\t";
        // }
        // std::cout << "\nstates modified: ";
        // for (int i = 0; i < modified_info.dims.number_of_states[k]; i++){
        //     std::cout << x_copy(modified_info.offsets_primal_x[k] + i) << "\t";
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;

        // if (k < info.dims.K - 1){
        //     std::cout << "[" << k << "] dynamics constraint:       " << std::endl;
        //     for (int i = 0; i < info.dims.number_of_states[k+1]; i++){
        //         std::cout << eq_mult(info.offsets_g_eq_dyn[k] + i) << "\n";
        //     }
        //     std::cout << "\n[" << k << "] dynamics constraint (mod): " << std::endl;
        //     for (int i = 0; i < modified_info.dims.number_of_states[k+1]; i++){
        //         std::cout << eq_mult_copy(modified_info.offsets_g_eq_dyn[k] + i) << "\n";
        //     }
        // }
        // std::cout << "\n[" << k << "] eq path constraint:       " << std::endl;
        // for (int i = 0; i < info.dims.number_of_eq_constraints[k]; i++){
        //     std::cout << eq_mult(info.offsets_g_eq_path[k] + i) << "\n";
        // }
        // std::cout << "\n[" << k << "] eq path constraint (mod): " << std::endl;
        // for (int i = 0; i < modified_info.dims.number_of_eq_constraints[k]; i++){
        //     std::cout << eq_mult_copy(modified_info.offsets_g_eq_path[k] + i) << "\n";
        // }
        // std::cout << std::endl;
        
    }
    // for (int k = 0; k < x.m(); k++){
    //     std::cout << x(k) << "\t" << x_copy(k) << std::endl;
    // }

    jacobian.ResetPreProcess(info);
    hessian.ResetPreProcess(info, jacobian);
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