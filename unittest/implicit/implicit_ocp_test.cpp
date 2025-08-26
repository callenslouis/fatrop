
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

template <typename T>
double get_average(const std::vector<T>& v){
    double sum = 0.0;
    for (const auto& val : v) {sum += val;}
    return sum / v.size();
}

template <typename ProblemType>
json add_json_data(std::shared_ptr<IpData<ProblemType>> data, std::string problem_type, int n, int control_level)
{
    json my_json;
    my_json["metadata"] = json::object();
    my_json["metadata"]["timing_statistics"] = json::object();
    my_json["metadata"]["timing_statistics"]["initialization"] = data->timing_statistics().initialization.elapsed();
    my_json["metadata"]["timing_statistics"]["compute search dir"] = data->timing_statistics().compute_search_dir.elapsed();
    my_json["metadata"]["timing_statistics"]["eval objective"] = data->timing_statistics().eval_objective.elapsed();
    my_json["metadata"]["timing_statistics"]["eval gradient"] = data->timing_statistics().eval_gradient.elapsed();
    my_json["metadata"]["timing_statistics"]["eval constraint violation"] = data->timing_statistics().eval_constraint_violation.elapsed();
    my_json["metadata"]["timing_statistics"]["eval hessian"] = data->timing_statistics().eval_hessian.elapsed();
    my_json["metadata"]["timing_statistics"]["eval jacobian"] = data->timing_statistics().eval_jacobian.elapsed();
    my_json["metadata"]["timing_statistics"]["rest time"] = data->timing_statistics().compute_rest_time();
    my_json["metadata"]["timing_statistics"]["function evaluation"] = data->timing_statistics().compute_function_evaluation();
    my_json["metadata"]["timing_statistics"]["fatrop"] = data->timing_statistics().compute_fatrop();
    my_json["metadata"]["timing_statistics"]["total"] = data->timing_statistics().full_algorithm.elapsed();
    my_json["metadata"]["iterations"] = data->iteration_number();

    my_json["problem type"] = problem_type;
    my_json["solver"] = "FATROP";
    my_json["K"] = data->info().dims.K;
    my_json["nx"] = get_average(data->info().dims.number_of_states);
    my_json["nu"] = get_average(data->info().dims.number_of_controls);
    my_json["ng"] = get_average(data->info().dims.number_of_eq_constraints);
    my_json["ng_ineq"] = get_average(data->info().dims.number_of_ineq_constraints);
    my_json["ocp problem"] = json::object();
    my_json["ocp problem"]["name"] = "holonomic";
    my_json["ocp problem"]["number of dimensions"] = n;
    my_json["ocp problem"]["control level"] = control_level;
    my_json["ocp problem"]["dt"] = 0.05;

    my_json["states"] = json::array();
    for (int k = 0; k < data->info().dims.K; k++)
    {
        json state = json::array();
        for (int i = 0; i < data->info().dims.number_of_states[k]; i++)
        {
            state.push_back(data->current_iterate().primal_x()(data->info().offsets_primal_x[k] + i));
        }
        my_json["states"].push_back(state);
    }
    my_json["inputs"] = json::array();
    for (int k = 0; k < data->info().dims.K-1; k++)
    {
        json input = json::array();
        for (int i = 0; i < data->info().dims.number_of_controls[k]; i++)
        {
            input.push_back(data->current_iterate().primal_x()(data->info().offsets_primal_u[k] + i));
        }
        my_json["inputs"].push_back(input);
    }

    return my_json;
};

int main(int argc, char **argv)
{
    int file_counter = 0;
    // create a directory ocp_results
    int temp = system("mkdir -p ocp_results");

    /*
    // int n = 7;
    // int control_level = 3;
    for (int n = 1; n < 8; n++){
    for (int control_level = 1; control_level < 3; control_level++){
    HolonomicInterfaceGenerator generator(n, control_level);
    */
   int n = 1;
   int control_level = 0;
   TruckTrailerInterfaceGenerator generator(n);

    std::cout << "creating implicit test problem" << std::endl;
    auto tp_impl = std::make_shared<ImplicitTestProblem>(generator.PrepareImplicit());
    std::cout << "creating explicit test problem" << std::endl;
    auto tp_expl = std::make_shared<ExplicitTestProblem>(generator.PrepareExplicit());
    std::cout << "creating reformulated test problem" << std::endl;
    auto tp_reform = std::make_shared<ExplicitTestProblem>(generator.PrepareReformulated());

    OptionRegistry options;
    IpAlgBuilder<ImplicitOcpType> builder_impl(std::make_shared<ImplicitNlpOcp>(tp_impl));
    IpAlgBuilder<OcpType> builder_expl(std::make_shared<NlpOcp>(tp_expl));
    IpAlgBuilder<OcpType> builder_reform(std::make_shared<NlpOcp>(tp_reform));    

    std::shared_ptr<IpAlgorithm<ImplicitOcpType>> ipalg_impl = builder_impl.with_options_registry(&options).build();
    std::shared_ptr<IpAlgorithm<OcpType>> ipalg_expl = builder_expl.with_options_registry(&options).build();
    std::shared_ptr<IpAlgorithm<OcpType>> ipalg_reform = builder_reform.with_options_registry(&options).build();
    
    
    for(int i = 0; i < 1; i++)
    {       
        // IMPLICIT
        std::cout << "solving implicit test problem" << std::endl;
        Timer timer_impl;
        timer_impl.start();
        IpSolverReturnFlag ret_impl = ipalg_impl->optimize();
        std::cout << "Elapsed time: " << timer_impl.stop() << std::endl;
        auto data_impl = builder_impl.get_ipdata();
        json result_impl = add_json_data(data_impl, "implicit", n, control_level);
        std::ofstream file("ocp_results/ocp_result_" + std::to_string(file_counter) + ".json");
        if (file.is_open())
        {
            file << result_impl.dump(4);
            file.close();
            file_counter++;
        }

        // EXPLICIT
        std::cout << "solving explicit test problem" << std::endl;
        Timer timer_expl;
        timer_expl.start();
        IpSolverReturnFlag ret_expl = ipalg_expl->optimize();
        std::cout << "Elapsed time: " << timer_expl.stop() << std::endl;
        auto data_expl = builder_expl.get_ipdata();
        json result_expl = add_json_data(data_expl, "explicit", n, control_level);
        std::ofstream file2("ocp_results/ocp_result_" + std::to_string(file_counter) + ".json");
        if (file2.is_open())
        {
            file2 << result_expl.dump(4);
            file2.close();
            file_counter++;
        }

        // REFORMULATED
        std::cout << "solving reformulated test problem" << std::endl;
        Timer timer_reform;
        timer_reform.start();
        IpSolverReturnFlag ret_reform = ipalg_reform->optimize();
        std::cout << "Elapsed time: " << timer_reform.stop() << std::endl;
        auto data_reform = builder_reform.get_ipdata();
        json result_reform = add_json_data(data_reform, "reformulated", n, control_level);
        std::ofstream file3("ocp_results/ocp_result_" + std::to_string(file_counter) + ".json");
        if (file3.is_open())
        {
            file3 << result_reform.dump(4);
            file3.close();
            file_counter++;
        }
    }
    // }
    // }
    
    return 0;
}
