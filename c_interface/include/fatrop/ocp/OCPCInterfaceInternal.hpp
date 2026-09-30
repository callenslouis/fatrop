#ifndef __fatrop_ocp_solver_ocp_c_interface_internal_hpp__
#define __fatrop_ocp_solver_ocp_c_interface_internal_hpp__
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/fwd.hpp"
#include "fatrop/linear_algebra/matrix.hpp"
#include "fatrop/linear_algebra/vector.hpp"
#include "fatrop/nlp/nlp.hpp"
namespace fatrop
{
#include "fatrop/ocp/OCPCInterface.h"
}
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/problem_info.hpp"
namespace fatrop
{
    class FatropOcpCMapping : public Nlp<OcpType>
    {
    public:
        FatropOcpCInterface *ocp;
        FatropOcpCDims s;
        FatropOcpCMapping(FatropOcpCInterface *ocp);
        const NlpDims &nlp_dims() const override;
        const ProblemDims<OcpType> &problem_dims() const override;
        Index eval_lag_hess(const ProblemInfo<OcpType> &info, const Scalar objective_scale,
                            const VecRealView &primal_x, const VecRealView &primal_s,
                            const VecRealView &lam, Hessian<OcpType> &hess) override;
        Index eval_constr_jac(const ProblemInfo<OcpType> &info, const VecRealView &primal_x,
                              const VecRealView &primal_s, Jacobian<OcpType> &jac) override;
        bool has_constant_hessian() const override { return ocp->has_constant_hessian != 0; }
        bool has_constant_jacobian() const override { return ocp->has_constant_jacobian != 0; }
        Index eval_constraint_violation(const ProblemInfo<OcpType> &info,
                                        const VecRealView &primal_x, const VecRealView &primal_s,
                                        VecRealView &res) override;
        Index eval_objective_gradient(const ProblemInfo<OcpType> &info,
                                      const Scalar objective_scale, const VecRealView &primal_x,
                                      const VecRealView &primal_s, VecRealView &grad_x,
                                      VecRealView &grad_s) override;
        Index eval_objective(const ProblemInfo<OcpType> &info, const Scalar objective_scale,
                             const VecRealView &primal_x, const VecRealView &primal_s,
                             Scalar &res) override;
        Index get_bounds(const ProblemInfo<OcpType> &info, VecRealView &lower_bounds,
                         VecRealView &upper_bounds) override;
        Index get_initial_primal(const ProblemInfo<OcpType> &info, VecRealView &primal_x) override;
        void get_primal_damping(const ProblemInfo<OcpType> &info, VecRealView &damping) override;
        void apply_jacobian_s_transpose(const ProblemInfo<OcpType> &info,
                                        const VecRealView &multipliers, const Scalar alpha,
                                        const VecRealView &y, VecRealView &out) override;
        /// Marks the constant QP data as stale, so it is re-evaluated on first use. Must be
        /// called before each solve, since the data behind the callbacks may have changed.
        void invalidate_constant_data() { constant_data_valid_ = false; }

    private:
        /// Evaluates the constant QP data (see below) if it is not valid yet.
        void ensure_constant_data();

        ProblemDims<OcpType> ocp_dims_;
        NlpDims nlp_dims_;
        Index K_;
        std::vector<MAT> matrix_buffer_[3];

        // Constant QP data, used when has_constant_hessian/has_constant_jacobian is set. The
        // callbacks are affine in the primal point and multipliers, so evaluating them once at
        // zero gives the constant blocks, with the constant terms in the right-hand-side row:
        //   RSQrqt0_[k] = [H_k; q_k^T], so grad_k = H_k [u;x] + q_k,
        //   BAbt0_[k] = [B_k^T; A_k^T; b_k^T], so b_k(x,u) = [B A] [u;x] + b_k - x_{k+1},
        // and similarly for Gg_eqt0_[k] and Gg_ineqt0_[k]. obj0_[k] is the objective at zero.
        std::vector<MatRealAllocated> RSQrqt0_, BAbt0_, Gg_eqt0_, Gg_ineqt0_;
        std::vector<Scalar> obj0_;
        /// Zeros, used as primal point and multipliers when evaluating the constant data.
        std::vector<Scalar> zeros_;
        /// Scratch vector of size max_k(nu[k] + nx[k]).
        VecRealAllocated work_;
        bool constant_data_valid_ = false;
        /// Incremented each time the constant data is re-evaluated, see Hessian::valid_data_id.
        Index constant_data_id_ = -1;
    };
}
#endif // __fatrop_ocp_solver_ocp_c_interface_internal_hpp__