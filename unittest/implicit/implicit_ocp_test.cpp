
//
// Copyright (c) Lander Vanroye, KU Leuven
//
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"

#include "fatrop/common/timing.hpp"
#include "fatrop/common/options.hpp"
#include "fatrop/ip_algorithm/ip_alg_builder.hpp"
#include "fatrop/ip_algorithm/ip_algorithm.hpp"
#include "fatrop/ip_algorithm/ip_data.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/nlp_ocp.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "ocp_interface_generator.hpp"

#include "json/single_include/nlohmann/json.hpp"

#include <fstream>

using namespace fatrop;
using json = nlohmann::json;


// example problem 2D point mass
// states: [x, y, vx, vy]
// inputs: [fx, fy]
// dynamics: [xk+1 = xk + dt*vxk+1, yk+1 = yk + dt*vyk+1, vxk+1 = vxk + dt*fx/m, vyk+1 = vyk + dt*fy/m]
// cost: fx^2 + fy^2
// constraints:
//  at k = 0: x = 0, y = 0, vx = 0, vy = 0
//  at k = K: x = 1, y = 1, vx = 0, vy = 0
class ImplicitOcpTestProblem : public ImplicitOcpAbstract
{
public:
    virtual Index get_nx(const Index k) const { return 4; }

    virtual Index get_nu(const Index k) const
    {
        if (k == 0)
        {
            return 2;
        }
        else if (k == K_ - 1)
        {
            return 0;
        }
        else
        {
            return 2;
        }
    }
    virtual Index get_ng(const Index k) const
    {
        if (k == 0)
        {
            return 4;
        }
        else if (k == K_ - 1)
        {
            return 4;
        }
        else
        {
            return 0;
        }
    };

    virtual Index get_ng_ineq(const Index k) const { return k == K_ - 1 ? 0 : 2; };
    virtual Index get_horizon_length() const { return K_; };
    virtual Index eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k,
                            const Scalar *states_k, MAT *res, const Index k)
    {
        // set zero
        blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
        // Matrix B
        // [  0,    0  ]
        // [  0,    0  ]
        // [ dt/m,  0  ]
        // [  0,   dt/m ]

        // Matrix A
        // [ 1,  0,   0,  0  ]
        // [ 0,  1,   0,  0  ]
        // [ 0,  0,   1,  0  ]
        // [ 0,  0,   0,  1  ]

        blasfeo_matel_wrap(res, 0, 2) = dt_ / m_;
        blasfeo_matel_wrap(res, 1, 3) = dt_ / m_;

        blasfeo_diare_wrap(4, 1.0, res, 2, 0);
        if (MAKE_EXPLICIT){
            blasfeo_matel_wrap(res, 4, 0) = dt_;
            blasfeo_matel_wrap(res, 5, 1) = dt_;
        }
        return 0;
    }
    virtual Index eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k,
                                const Scalar *states_k, const Scalar *lam_dyn_k,
                                const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k, MAT *res,
                                const Index k)
    {
        // set zero
        blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
        // Matrix R
        // [ 2,  0 ]
        // [ 0,  2 ]
        if (k < K_ - 1)
        {
            blasfeo_diare_wrap(2, *objective_scale*2.0, res, 0, 0);
        }
        return 0;
    };
    virtual Index eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res,
                            const Index k)
    {
        // set zero
        blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
        if (k == 0)
            blasfeo_diare_wrap(4, 1.0, res, 2, 0);
        if (k == K_ - 1)
            blasfeo_diare_wrap(4, 1.0, res, 0, 0);
        return 0;
    }
    virtual Index eval_Ggt_ineq(const Scalar *inputs_k, const Scalar *states_k, MAT *res,
                                const Index k)
    {
        if (k == K_ - 1)
            return 0;
        // set zero
        blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
        blasfeo_matel_wrap(res, 0, 0) = 1.0;
        blasfeo_matel_wrap(res, 1, 1) = 1.0;
        return 0;
    };
    virtual Index eval_b(const Scalar *states_kp1, const Scalar *inputs_k,
                            const Scalar *states_k, Scalar *res, const Index k)
    {
        res[0] = -states_kp1[0] + states_k[0] + dt_ * states_k[2];
        res[1] = -states_kp1[1] + states_k[1] + dt_ * states_k[3];
        res[2] = -states_kp1[2] + states_k[2] + dt_ * inputs_k[0] / m_;
        res[3] = -states_kp1[3] + states_k[3] + dt_ * inputs_k[1] / m_;
        return 0;
    }

    virtual Index eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
                            const Index k)
    {
        if (k == 0)
        {
            res[0] = states_k[0];
            res[1] = states_k[1];
            res[2] = states_k[2];
            res[3] = states_k[3];
        }
        else if (k == K_ - 1)
        {
            res[0] = states_k[0] - 1;
            res[1] = states_k[1] - 1;
            res[2] = states_k[2];
            res[3] = states_k[3];
        }
        return 0;
    };
    virtual Index eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
                                const Index k)
    {
        if (k == K_ - 1)
            return 0;
        res[0] = inputs_k[0];
        res[1] = inputs_k[1];
        return 0;
    };
    virtual Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k,
                            const Scalar *states_k, Scalar *res, const Index k)
    {
        if (k == K_ - 1)
        {
            res[0] = 0.;
            res[1] = 0.;
            res[2] = 0.;
            res[3] = 0.;
        }
        else
        {
            res[0] = 2 * objective_scale[0] * inputs_k[0];
            res[1] = 2 * objective_scale[0] * inputs_k[1];
            res[2] = 0.;
            res[3] = 0.;
            res[4] = 0.;
            res[5] = 0.;
            res[6] = 0.;
        }
        return 0;
    }
    virtual Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k,
                            const Scalar *states_k, Scalar *res, const Index k)
    {
        if (k == K_ - 1)
        {
            *res = 0.;
        }
        else
        {
            *res = objective_scale[0] *
                    (inputs_k[0] * inputs_k[0] + inputs_k[1] * inputs_k[1]);
        }
        return 0;
    }
    virtual Index get_bounds(Scalar *lower, Scalar *upper, const Index k) const
    {
        if (k == K_ - 1)
            return 0;
        lower[0] = -2;
        upper[0] = 2;
        lower[1] = -2;
        upper[1] = 2;
        return 0;
    }

    virtual Index get_initial_xk(Scalar *xk, const Index k) const
    {
        xk[0] = 0*0.01 * k;
        xk[1] = 0*0.02 * k;
        xk[2] = 0*0.03 * k;
        xk[3] = 0*0.04 * k;
        return 0;
    };
    virtual Index get_initial_uk(Scalar *uk, const Index k) const
    {
        if (k == K_ - 1)
            return 0;
        uk[0] = 0*0.05 * k;
        uk[1] = 0*0.06 * k;
        return 0;
    };
    virtual ~ImplicitOcpTestProblem() = default;

    virtual Index eval_Jt(const Scalar *states_kp1, const Scalar *inputs_k,
                            const Scalar *states_k, MAT *res, const Index k){
        blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
        for (Index i = 0; i < res->m; i++){
            blasfeo_matel_wrap(res, i, i) = -1.0;
        }
        if (!MAKE_EXPLICIT){
            blasfeo_matel_wrap(res, 2, 0) = dt_;
            blasfeo_matel_wrap(res, 3, 1) = dt_;
        }
        return 0;
    };

    virtual Index eval_Jt_inv(const Scalar *states_kp1, const Scalar *inputs_k,
                                const Scalar *states_k, MAT *res, const Index k){
        blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
        for (Index i = 0; i < res->m; i++){
            blasfeo_matel_wrap(res, i, i) = -1.0;
        }
        if (!MAKE_EXPLICIT){
            blasfeo_matel_wrap(res, 2, 0) = -dt_;
            blasfeo_matel_wrap(res, 3, 1) = -dt_;
        }
        return 0;
    };

    virtual Index eval_FuFxt(const Scalar *inputs_k, const Scalar *states_k, 
                                const Scalar *states_kp1, MAT *res, const Index k){
        blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
        if (!MAKE_EXPLICIT){
            // nothing in this case
        }
        return 0;
    };
    

private:
    const Index K_ = 100;
    const Scalar m_ = 1.0;
    const Scalar dt_ = 0.05;

    bool MAKE_EXPLICIT = false;
};


int main(int argc, char **argv)
{
    OcpInterfaceGenerator generator;
    auto tp = std::make_shared<TestProblem>(generator.PrepareHolonomic(1, 2));

    OptionRegistry options;
    // IpAlgBuilder<ImplicitOcpType> builder(std::make_shared<ImplicitNlpOcp>(std::make_shared<ImplicitOcpTestProblem>()));
    IpAlgBuilder<ImplicitOcpType> builder(std::make_shared<ImplicitNlpOcp>(tp));

    std::shared_ptr<IpAlgorithm<ImplicitOcpType>> ipalg = builder.with_options_registry(&options).build();
    std::cout << options << std::endl;
    for(int i =0; i < 1; i++)
    {
        Timer timer;
        timer.start();
        IpSolverReturnFlag ret = ipalg->optimize();
        std::cout << "Elapsed time: " << timer.stop() << std::endl;
        auto data = builder.get_ipdata();
        std::cout << "Return flag: " << int(ret) << std::endl;
        std::cout << "Return flag == success: " << (ret == IpSolverReturnFlag::Success)
                  << std::endl;
        std::cout << data->timing_statistics() << std::endl;

        json result;
        result["metadata"] = json::object();
        result["metadata"]["return_flag"] = int(ret);
        json timing_stats;
        timing_stats["initialization"] = data->timing_statistics().initialization.elapsed();
        timing_stats["compute search dir"] = data->timing_statistics().compute_search_dir.elapsed();
        timing_stats["eval objective"] = data->timing_statistics().eval_objective.elapsed();
        timing_stats["eval gradient"] = data->timing_statistics().eval_gradient.elapsed();
        timing_stats["eval constraint violation"] = data->timing_statistics().eval_constraint_violation.elapsed();
        timing_stats["eval hessian"] = data->timing_statistics().eval_hessian.elapsed();
        timing_stats["eval jacobian"] = data->timing_statistics().eval_jacobian.elapsed();
        timing_stats["rest time"] = data->timing_statistics().compute_rest_time();
        timing_stats["function evaluation"] = data->timing_statistics().compute_function_evaluation();
        timing_stats["fatrop"] = data->timing_statistics().compute_fatrop();
        timing_stats["total"] = data->timing_statistics().full_algorithm.elapsed();
        result["metadata"]["timing_statistics"] = timing_stats;

        result["dt"] = 0.05;
        result["K"] = data->info().dims.K;
        result["states"] = json::array();
        for (int k = 0; k < data->info().dims.K; k++)
        {
            json state = json::array();
            for (int i = 0; i < data->info().dims.number_of_states[k]; i++)
            {
                state.push_back(data->current_iterate().primal_x()(data->info().offsets_primal_x[k] + i));
            }
            result["states"].push_back(state);
        }
        result["inputs"] = json::array();
        for (int k = 0; k < data->info().dims.K-1; k++)
        {
            json input = json::array();
            for (int i = 0; i < data->info().dims.number_of_controls[k]; i++)
            {
                input.push_back(data->current_iterate().primal_x()(data->info().offsets_primal_u[k] + i));
            }
            result["inputs"].push_back(input);
        }

        // dump
        std::ofstream file("implicit_ocp_result.json");
        if (file.is_open())
        {
            file << result.dump(4); // pretty print with 4 spaces
            file.close();
            std::cout << "Results saved to implicit_ocp_result.json" << std::endl;
        }
    }
    return 0;
}
