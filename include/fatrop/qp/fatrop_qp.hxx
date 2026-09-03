//
// Copyright (C) 2026 Lander Vanroye, KU Leuven
//
// INTERNAL header. Not part of the installed public API (only .hpp files are
// installed). Contains the template definitions for MehrotraQpAlgorithm /
// MehrotraQpBuilder declared in fatrop/qp/fatrop_qp.hpp. The OcpType
// instantiation lives in src/ocp/mehrotra_qp.cpp -- user code never needs to
// include this file.
//
#ifndef __fatrop_fatrop_qp_hxx__
#define __fatrop_fatrop_qp_hxx__

#include "fatrop/qp/fatrop_qp.hpp"

#include "fatrop/common/fwd.hpp"
#include "fatrop/common/options.hpp"
#include "fatrop/common/printing.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/ip_algorithm/fwd.hpp"
#include "fatrop/ip_algorithm/ip_algorithm.hpp"
#include "fatrop/ip_algorithm/ip_data.hpp"
#include "fatrop/ip_algorithm/ip_data.hxx"
#include "fatrop/ip_algorithm/ip_eq_mult_initializer.hpp"
#include "fatrop/ip_algorithm/ip_eq_mult_initializer.hxx"
#include "fatrop/ip_algorithm/ip_initializer.hpp"
#include "fatrop/ip_algorithm/ip_initializer.hxx"
#include "fatrop/ip_algorithm/ip_iterate.hpp"
#include "fatrop/ip_algorithm/ip_iterate.hxx"
#include "fatrop/ip_algorithm/ip_nlp_orig.hpp"
#include "fatrop/ip_algorithm/ip_nlp_orig.hxx"
#include "fatrop/ip_algorithm/ip_timings.hpp"
#include "fatrop/ip_algorithm/pd_solver_orig.hpp"
#include "fatrop/ip_algorithm/pd_system_orig.hpp"
#include "fatrop/linear_algebra/fwd.hpp"
#include "fatrop/linear_algebra/linear_solver.hpp"
#include "fatrop/linear_algebra/linear_solver.hxx"
#include "fatrop/linear_algebra/linear_solver_return_flags.hpp"
#include "fatrop/linear_algebra/vector.hpp"
#include "fatrop/nlp/fwd.hpp"
#include "fatrop/nlp/nlp.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/pd_solver_orig.hpp"
#include "fatrop/ocp/pd_system_orig.hpp"
#include "fatrop/ocp/problem_info.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>

namespace fatrop
{
    // ---------------------------------------------------------------------------
    // MehrotraQpAlgorithm  -- implementations of the class declared in
    // fatrop/qp/fatrop_qp.hpp.
    // ---------------------------------------------------------------------------

    template <typename ProblemType>
    MehrotraQpAlgorithm<ProblemType>::MehrotraQpAlgorithm(
        const IpDataSp &ipdata, const IpNlpOrigSp &nlp_orig, const PdSolverSp &pd_solver,
        const InitializerSp &initializer, const EqMultInitSp &eq_mult_initializer)
        : ipdata_(ipdata), nlp_orig_(nlp_orig), pd_solver_(pd_solver), initializer_(initializer),
          eq_mult_initializer_(eq_mult_initializer),
          rhs_x_(ipdata->current_iterate().nlp()->nlp_dims().number_of_tangent_variables),
          rhs_s_(ipdata->current_iterate().nlp()->nlp_dims().number_of_ineq_constraints),
          rhs_g_(ipdata->current_iterate().nlp()->nlp_dims().number_of_eq_constraints),
          rhs_cl_(ipdata->current_iterate().nlp()->nlp_dims().number_of_ineq_constraints),
          rhs_cu_(ipdata->current_iterate().nlp()->nlp_dims().number_of_ineq_constraints),
          ds_aff_(ipdata->current_iterate().nlp()->nlp_dims().number_of_ineq_constraints),
          dzl_aff_(ipdata->current_iterate().nlp()->nlp_dims().number_of_ineq_constraints),
          dzu_aff_(ipdata->current_iterate().nlp()->nlp_dims().number_of_ineq_constraints)
    {
    }

    template <typename ProblemType> void MehrotraQpAlgorithm<ProblemType>::reset()
    {
        ipdata_->reset(false);
        initializer_->reset();
        eq_mult_initializer_->reset();
        iteration_ = 0;
        mu_ = 1.0;
        delta_w_last_ = 0.;
    }

    template <typename ProblemType>
    const ProblemInfo<ProblemType> &MehrotraQpAlgorithm<ProblemType>::info() const
    {
        return ipdata_->current_iterate().info();
    }

    template <typename ProblemType>
    const VecRealView &MehrotraQpAlgorithm<ProblemType>::solution_primal() const
    {
        return ipdata_->current_iterate().primal_x();
    }

    template <typename ProblemType>
    const VecRealView &MehrotraQpAlgorithm<ProblemType>::solution_dual() const
    {
        return ipdata_->current_iterate().dual_eq();
    }

    template <typename ProblemType> void MehrotraQpAlgorithm<ProblemType>::recompute_mu()
    {
        IpIterate<ProblemType> &it = ipdata_->current_iterate();
        const Index n_bounds = it.number_of_bounds();
        if (n_bounds == 0)
        {
            // pure-equality QP -- no barrier, mu can be 0
            it.set_mu(0.);
            mu_ = 0.;
            return;
        }
        const Scalar sum_compl = sum(it.complementarity_l()) + sum(it.complementarity_u());
        const Scalar new_mu = std::max(sum_compl / static_cast<Scalar>(n_bounds), 0.);
        it.set_mu(new_mu);
        mu_ = new_mu;
    }

    template <typename ProblemType>
    Scalar MehrotraQpAlgorithm<ProblemType>::compute_mu_aff(
        Scalar alpha_pr, Scalar alpha_du, const VecRealView &ds_aff,
        const VecRealView &dzl_aff, const VecRealView &dzu_aff) const
    {
        IpIterate<ProblemType> &it =
            const_cast<IpData<ProblemType> *>(ipdata_.get())->current_iterate();
        const Index n = it.primal_s().m();
        const auto &lb = it.lower_bounded();
        const auto &ub = it.upper_bounded();
        const VecRealView &dl = it.delta_lower();
        const VecRealView &du = it.delta_upper();
        const VecRealView &zl = it.dual_bounds_l();
        const VecRealView &zu = it.dual_bounds_u();
        Scalar sum_aff = 0.;
        Index count = 0;
        for (Index i = 0; i < n; ++i)
        {
            if (lb[i])
            {
                const Scalar sl_new = dl(i) + alpha_pr * ds_aff(i);
                const Scalar zl_new = zl(i) + alpha_du * dzl_aff(i);
                sum_aff += sl_new * zl_new;
                count++;
            }
            if (ub[i])
            {
                const Scalar su_new = du(i) - alpha_pr * ds_aff(i);
                const Scalar zu_new = zu(i) + alpha_du * dzu_aff(i);
                sum_aff += su_new * zu_new;
                count++;
            }
        }
        if (count == 0)
            return 0.;
        return sum_aff / static_cast<Scalar>(count);
    }

    template <typename ProblemType>
    LinsolReturnFlag MehrotraQpAlgorithm<ProblemType>::compute_predictor_corrector_step(
        Scalar &alpha_pr_out, Scalar &alpha_du_out, Scalar &sigma_out)
    {
        IpIterate<ProblemType> &curr = ipdata_->current_iterate();
        const Scalar mu = std::max(curr.mu(), mu_min_);

        // -------- assemble linear system at the current iterate --------------
        // RHS for stationarity / feasibility blocks (identical for affine and corrector)
        rhs_x_ = curr.dual_infeasibility_x();
        rhs_s_ = curr.dual_infeasibility_s();
        rhs_g_ = curr.constr_viol();
        // Affine complementarity: target = 0  -> rhs = compl (no -mu shift)
        rhs_cl_ = curr.complementarity_l();
        rhs_cu_ = curr.complementarity_u();
        
        // Baseline: no inertia/dual regularization for a convex QP -- its
        // reduced Hessian is normally PD, so the one-shot factorization below
        // is the fast path. When reg_enabled_ is set, an INDEFINITE/
        // NOFULL_RANK pivot failure instead triggers a retry with growing
        // Hessian/dual regularization (same scheme as the NLP solver's
        // IpSearchDirImpl, see ip_search_dir.hxx) rather than aborting.
        curr.set_Dx(curr.primal_damping());
        curr.set_De(VecRealScalar(curr.De().m(), 0.));
        curr.set_De_is_zero(true);

        Scalar delta_w = 0.;
        Scalar delta_c = 0.;
        bool first_try_delta_w = true;
        Index indefinite_tries = 0;

        // -------- predictor: factorize + solve (retried under regularization) --
        LinsolReturnFlag ret_aff;
        for (Index try_ = 0;; ++try_)
        {
            // std::cout << "try: " << try_ << ", delta_w: " << delta_w << ", delta_c: " << delta_c
            //           << std::endl;
            curr.set_Dx(VecRealScalar(curr.Dx().m(), delta_w) + curr.primal_damping());
            curr.set_De(VecRealScalar(curr.De().m(), delta_c));
            curr.set_De_is_zero(delta_c == 0.);

            LinearSystem<PdSystemType<ProblemType>> ls(
                curr.info(), curr.jacobian(), curr.hessian(), curr.Dx(), curr.De_is_zero(),
                curr.De(), curr.delta_lower(), curr.delta_upper(), curr.dual_bounds_l(),
                curr.dual_bounds_u(), rhs_x_, rhs_s_, rhs_g_, rhs_cl_, rhs_cu_);

            {
                ScopedTimer _t(ipdata_->timing_statistics().compute_search_dir,
                               ipdata_->timing_statistics());
                ret_aff = pd_solver_->solve_in_place(ls);
            }

            bool solved;
            switch (ret_aff)
            {
            case LinsolReturnFlag::SUCCESS:
            case LinsolReturnFlag::ITREF_MAX_ITER:
            case LinsolReturnFlag::ITREF_INCREASE:
                solved = true;
                break;
            case LinsolReturnFlag::INDEFINITE:
            case LinsolReturnFlag::NOFULL_RANK:
                solved = false;
                break;
            default:
                return ret_aff;
            }
            if (solved)
                break;
            // Disabled (default): bail out exactly like the original
            // one-shot behavior.
            if (!reg_enabled_ || try_ >= reg_max_tries_ - 1)
                return ret_aff;

            if (ret_aff == LinsolReturnFlag::INDEFINITE)
            {
                ++indefinite_tries;
                // delta_w is injected before the per-stage equality nullspace
                // projection (aug_system_solver.cpp); on a shooting OCP that
                // projection is oblique and can absorb an arbitrarily large
                // delta_w without the factored block's pivots improving. So
                // delta_wmax_ caps delta_w growth, it does not gate whether we
                // keep retrying: once delta_w alone has had a few tries (or
                // would need to exceed delta_wmax_ to keep growing -- e.g.
                // because delta_w_last_ carried over a large value from a
                // previous outer iteration), engage delta_c instead. Nonzero
                // delta_c switches the recursion to the penalty-based
                // equality elimination, where regularization lands directly
                // on the factored diagonal rather than through that
                // projection.
                const Scalar next_delta_w =
                    first_try_delta_w
                        ? (delta_w_last_ == 0. ? delta_w0_
                                                : std::max(delta_w_last_ * kappa_wmin_, delta_wmin_))
                        : (delta_w_last_ == 0. ? kappa_wplusem_ * delta_w : kappa_wplus_ * delta_w);
                if (delta_c == 0. &&
                    (indefinite_tries >= delta_w_tries_before_c_ || next_delta_w > delta_wmax_))
                {
                    delta_c = delta_c_stripe_ * std::pow(mu, kappa_c_);
                }
                else
                {
                    delta_w = next_delta_w;
                    first_try_delta_w = false;
                }
            }
            else // NOFULL_RANK
            {
                delta_c = delta_c_stripe_ * std::pow(mu, kappa_c_);
            }
        }
        if (delta_w > 0.)
            delta_w_last_ = delta_w;
        curr.search_dir_info().inertia_correction_primal = delta_w;
        curr.search_dir_info().inertia_correction_dual = delta_c;

        // Rebuild ls against the (possibly regularized) Dx/De that succeeded;
        // the corrector below reuses this cached factorization rather than
        // refactoring, so it must see the exact same Dx/De/De_is_zero.
        LinearSystem<PdSystemType<ProblemType>> ls(
            curr.info(), curr.jacobian(), curr.hessian(), curr.Dx(), curr.De_is_zero(),
            curr.De(), curr.delta_lower(), curr.delta_upper(), curr.dual_bounds_l(),
            curr.dual_bounds_u(), rhs_x_, rhs_s_, rhs_g_, rhs_cl_, rhs_cu_);

        // After solve_in_place, the rhs vectors hold the (affine) solution.
        // Stash the slices we need for the corrector RHS.
        ds_aff_ = rhs_s_;
        dzl_aff_ = rhs_cl_;
        dzu_aff_ = rhs_cu_;

        // -------- compute the Mehrotra centering parameter --------------------
        const Scalar alpha_aff_pr = curr.maximum_step_size_primal(1.0, ds_aff_);
        const Scalar alpha_aff_du = curr.maximum_step_size_dual(1.0, dzl_aff_, dzu_aff_);
        const Scalar mu_aff =
            compute_mu_aff(alpha_aff_pr, alpha_aff_du, ds_aff_, dzl_aff_, dzu_aff_);

        Scalar sigma = 0.;
        if (mu > 0.)
            sigma = std::pow(mu_aff / mu, 3.);
        sigma = std::min(std::max(sigma, 0.), sigma_max_);
        const Scalar sigma_mu = sigma * mu;
        sigma_out = sigma;
        last_mu_aff_ = mu_aff;

        // -------- corrector: rebuild RHS, reuse factorization -----------------
        const auto &lb = curr.lower_bounded();
        const auto &ub = curr.upper_bounded();

        rhs_x_ = curr.dual_infeasibility_x();
        rhs_s_ = curr.dual_infeasibility_s();
        rhs_g_ = curr.constr_viol();
        rhs_cl_ = if_else(lb,
                          curr.complementarity_l() + ds_aff_ * dzl_aff_ -
                              VecRealScalar(ds_aff_.m(), sigma_mu),
                          VecRealScalar(rhs_cl_.m(), 0.));
        rhs_cu_ = if_else(ub,
                          curr.complementarity_u() - ds_aff_ * dzu_aff_ -
                              VecRealScalar(ds_aff_.m(), sigma_mu),
                          VecRealScalar(rhs_cu_.m(), 0.));

        LinsolReturnFlag ret_cor;
        {
            ScopedTimer _t(ipdata_->timing_statistics().compute_search_dir,
                           ipdata_->timing_statistics());
            ret_cor = pd_solver_->solve_in_place_rhs(ls);
        }
        switch (ret_cor)
        {
        case LinsolReturnFlag::SUCCESS:
        case LinsolReturnFlag::ITREF_MAX_ITER:
        case LinsolReturnFlag::ITREF_INCREASE:
            break;
        default:
            return ret_cor;
        }

        // Now rhs_* hold the corrector (full) direction. Store on iterate.
        curr.set_delta_primal_x(rhs_x_);
        curr.set_delta_primal_s(rhs_s_);
        curr.set_delta_dual_eq(rhs_g_);
        curr.set_delta_dual_bounds_l(rhs_cl_);
        curr.set_delta_dual_bounds_u(rhs_cu_);

        const Scalar tau = std::max(tau_min_, 1. - mu);
        alpha_pr_out = curr.maximum_step_size_primal(tau);
        alpha_du_out = curr.maximum_step_size_dual(tau);
        return ret_cor;
    }

    template <typename ProblemType>
    void MehrotraQpAlgorithm<ProblemType>::apply_step(Scalar alpha_pr, Scalar alpha_du)
    {
        IpIterate<ProblemType> &curr = ipdata_->current_iterate();
        IpIterate<ProblemType> &trial = ipdata_->trial_iterate();

        // Primal x via NLP retraction (Euclidean by default, manifold-aware otherwise)
        trial.set_primal_x_from_step(curr.primal_x(), curr.delta_primal_x(), alpha_pr);
        trial.set_primal_s(curr.primal_s() + alpha_pr * curr.delta_primal_s());
        trial.set_dual_eq(curr.dual_eq() + alpha_du * curr.delta_dual_eq());
        trial.set_dual_bounds_l(curr.dual_bounds_l() + alpha_du * curr.delta_dual_bounds_l());
        trial.set_dual_bounds_u(curr.dual_bounds_u() + alpha_du * curr.delta_dual_bounds_u());
        trial.set_mu(curr.mu());

        // Step bookkeeping (for output)
        curr.step_info().alpha_primal = alpha_pr;
        curr.step_info().alpha_dual = alpha_du;
        curr.step_info().step_length =
            std::max({norm_inf(curr.delta_primal_x()), norm_inf(curr.delta_primal_s()),
                      norm_inf(curr.delta_dual_eq()), norm_inf(curr.delta_dual_bounds_l()),
                      norm_inf(curr.delta_dual_bounds_u())});

        ipdata_->accept_trial_iterate();
    }

    template <typename ProblemType>
    void MehrotraQpAlgorithm<ProblemType>::print_header() const
    {
        if (!verbose_)
            return;
        PRINT_ITERATIONS << std::setw(4) << "iter" << " " << std::setw(12) << "objective" << " "
                         << std::setw(9) << "inf_pr" << " " << std::setw(9) << "inf_du" << " "
                         << std::setw(9) << "inf_co" << " " << std::setw(9) << "mu" << " "
                         << std::setw(9) << "mu_aff" << " " << std::setw(7) << "sigma" << " "
                         << std::setw(9) << "alpha_pr" << " " << std::setw(9) << "alpha_du"
                         << std::endl;
    }

    template <typename ProblemType>
    void MehrotraQpAlgorithm<ProblemType>::print_iteration(
        Scalar inf_pr, Scalar inf_du, Scalar inf_compl, Scalar mu_aff, Scalar sigma,
        Scalar alpha_pr, Scalar alpha_du) const
    {
        if (!verbose_)
            return;
        IpIterate<ProblemType> &it =
            const_cast<IpData<ProblemType> *>(ipdata_.get())->current_iterate();
        PRINT_ITERATIONS << std::setw(4) << iteration_ << "  " << std::setw(12) << std::scientific
                         << std::setprecision(4) << it.obj_value() << " " << std::setw(9)
                         << std::scientific << std::setprecision(2) << inf_pr << " " << std::setw(9)
                         << std::scientific << std::setprecision(2) << inf_du << " " << std::setw(9)
                         << std::scientific << std::setprecision(2) << inf_compl << " "
                         << std::setw(9) << std::scientific << std::setprecision(2) << mu_ << " "
                         << std::setw(9) << std::scientific << std::setprecision(2) << mu_aff
                         << " " << std::setw(7) << std::fixed << std::setprecision(3) << sigma
                         << " " << std::setw(9) << std::scientific << std::setprecision(2)
                         << alpha_pr << " " << std::setw(9) << std::scientific
                         << std::setprecision(2) << alpha_du << std::endl;
    }

    template <typename ProblemType>
    IpSolverReturnFlag MehrotraQpAlgorithm<ProblemType>::optimize()
    {
        reset();
        ipdata_->timing_statistics().full_algorithm.start();

        {
            ScopedTimer _t(ipdata_->timing_statistics().initialization,
                           ipdata_->timing_statistics());
            initializer_->initialize();
        }

        // Initial mu reflects current complementarity, so the very first
        // sigma = (mu_aff/mu)^3 is meaningful.
        recompute_mu();

        print_header();

        IpSolverReturnFlag ret = IpSolverReturnFlag::Unknown;
        while (true)
        {
            ipdata_->get_nlp()->callback(*ipdata_);

            const Scalar inf_pr = norm_inf(ipdata_->current_iterate().constr_viol());
            const Scalar inf_du =
                std::max(norm_inf(ipdata_->current_iterate().dual_infeasibility_x()),
                         norm_inf(ipdata_->current_iterate().dual_infeasibility_s()));
            const Scalar inf_compl =
                std::max(norm_inf(ipdata_->current_iterate().complementarity_l()),
                         norm_inf(ipdata_->current_iterate().complementarity_u()));

            // Convergence: KKT residuals all below tol (mu drives compl, the
            // other two come from constr_viol and dual_infeas).
            if (inf_pr <= constr_viol_tol_ && inf_du <= tol_ && inf_compl <= tol_)
            {
                if (verbose_)
                    print_iteration(inf_pr, inf_du, inf_compl, 0., 0., 0., 0.);
                ret = IpSolverReturnFlag::Success;
                break;
            }
            if (iteration_ >= max_iter_)
            {
                if (verbose_)
                    print_iteration(inf_pr, inf_du, inf_compl, 0., 0., 0., 0.);
                ret = IpSolverReturnFlag::MaxIterExceeded;
                break;
            }

            Scalar alpha_pr = 0., alpha_du = 0., sigma = 0.;
            
            LinsolReturnFlag sd_ret =
                compute_predictor_corrector_step(alpha_pr, alpha_du, sigma);
            if (sd_ret == LinsolReturnFlag::UNKNOWN ||
                sd_ret == LinsolReturnFlag::NAN_SOLUTION ||
                sd_ret == LinsolReturnFlag::INDEFINITE ||
                sd_ret == LinsolReturnFlag::NOFULL_RANK)
            {
                ret = IpSolverReturnFlag::ErrorInStepComputation;
                std::cout << "Error in step computation. linsol return flag is "; PrintSdReturnFlag(sd_ret);
                break;
            }

            print_iteration(inf_pr, inf_du, inf_compl, last_mu_aff_, sigma, alpha_pr, alpha_du);

            apply_step(alpha_pr, alpha_du);

            // Update mu from the new complementarity, then keep z away from
            // ill-conditioned values via the same kappa_sigma trick the NLP
            // solver uses. Cheap if the bounds are well-behaved.
            recompute_mu();
            ipdata_->current_iterate().modify_dual_bounds(std::max(mu_, mu_min_));
            recompute_mu();

            ++iteration_;
            ipdata_->set_iteration_number(iteration_);
        }

        ipdata_->timing_statistics().full_algorithm.pause();
        return ret;
    }

    template <typename ProblemType>
    void MehrotraQpAlgorithm<ProblemType>::register_options(OptionRegistry &registry)
    {
        registry.register_option("max_iter", &MehrotraQpAlgorithm<ProblemType>::set_max_iter, this);
        registry.register_option("tolerance", &MehrotraQpAlgorithm<ProblemType>::set_tolerance, this);

        // Inertia correction (off by default -- see set_reg_enabled).
        registry.register_option("qp_reg_enabled", &MehrotraQpAlgorithm<ProblemType>::set_reg_enabled, this);
        registry.register_option("qp_delta_w0", &MehrotraQpAlgorithm<ProblemType>::set_delta_w0, this);
        registry.register_option("qp_delta_wmin", &MehrotraQpAlgorithm<ProblemType>::set_delta_wmin, this);
        registry.register_option("qp_delta_wmax", &MehrotraQpAlgorithm<ProblemType>::set_delta_wmax, this);
        registry.register_option("qp_kappa_wmin", &MehrotraQpAlgorithm<ProblemType>::set_kappa_wmin, this);
        registry.register_option("qp_kappa_wplus", &MehrotraQpAlgorithm<ProblemType>::set_kappa_wplus, this);
        registry.register_option("qp_kappa_wplusem", &MehrotraQpAlgorithm<ProblemType>::set_kappa_wplusem, this);
        registry.register_option("qp_kappa_c", &MehrotraQpAlgorithm<ProblemType>::set_kappa_c, this);
        registry.register_option("qp_delta_c_stripe", &MehrotraQpAlgorithm<ProblemType>::set_delta_c_stripe, this);
        registry.register_option("qp_reg_max_tries", &MehrotraQpAlgorithm<ProblemType>::set_reg_max_tries, this);
        registry.register_option("qp_delta_w_tries_before_c", &MehrotraQpAlgorithm<ProblemType>::set_delta_w_tries_before_c, this);
    }

    // ---------------------------------------------------------------------------
    // MehrotraQpBuilder
    // ---------------------------------------------------------------------------

    template <typename ProblemType>
    MehrotraQpBuilder<ProblemType>::MehrotraQpBuilder(const std::shared_ptr<Nlp<ProblemType>> &nlp)
    {
        nlp_orig_ = std::make_shared<IpNlpOrig<ProblemType>>(nlp);
    }

    template <typename ProblemType>
    std::shared_ptr<MehrotraQpAlgorithm<ProblemType>> MehrotraQpBuilder<ProblemType>::build()
    {
        // IpData manages the iterates (current/trial), the ProblemInfo, and
        // owns the bounds + Hessian/Jacobian storage. Same wiring as the NLP
        // builder, minus the components we are replacing (search dir, line
        // search, mu update, convergence check, iteration output, restoration).
        ipdata_ = std::make_shared<IpData<ProblemType>>(nlp_orig_);
        if (options_registry_)
            ipdata_->register_options(*options_registry_);

        const auto &ocp_dims = nlp_orig_->problem_dims();
        problem_info_ = std::make_shared<ProblemInfo<ProblemType>>(ocp_dims);
        aug_system_solver_ = std::make_shared<AugSystemSolver<ProblemType>>(*problem_info_);
        if (options_registry_)
            aug_system_solver_->register_options(*options_registry_);
        pd_solver_ =
            std::make_shared<PdSolverOrig<ProblemType>>(*problem_info_, aug_system_solver_);

        eq_mult_initializer_ =
            std::make_shared<IpEqMultInitializer<ProblemType>>(ipdata_, pd_solver_);
        if (options_registry_)
            options_registry_->register_options(*eq_mult_initializer_);

        initializer_ =
            std::make_shared<IpInitializer<ProblemType>>(ipdata_, eq_mult_initializer_);
        if (options_registry_)
            options_registry_->register_options(*initializer_);

        nlp_orig_->set_timing_statistics(&ipdata_->timing_statistics());
        if (options_registry_)
            options_registry_->register_options(*nlp_orig_);

        algorithm_ = std::make_shared<MehrotraQpAlgorithm<ProblemType>>(
            ipdata_, nlp_orig_, pd_solver_, initializer_, eq_mult_initializer_);

        if (options_registry_)
        {
            options_registry_->register_option<Index>(
                "print_level", &PrintLevelManager::set_print_level);
            options_registry_->register_options(*algorithm_);
        }
        return algorithm_;
    }

} // namespace fatrop

#endif // __fatrop_fatrop_qp_hxx__
