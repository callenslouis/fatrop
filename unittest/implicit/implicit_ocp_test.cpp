
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
#include "generators/ocp_interface_generator.hpp"
#include "generators/holonomic_generator.hpp"
#include "generators/truck_trailer_generator.hpp"

#include "json/single_include/nlohmann/json.hpp"

#include <fstream>

using namespace fatrop;
using json = nlohmann::json;

template <typename T>
double get_average(const std::vector<T>& v){
    double sum = 0.0;
    for (const auto& val : v) {sum += val;}
    return sum / v.size();
}

template <typename ProblemType>
json add_json_data(std::shared_ptr<IpData<ProblemType>> data, std::string problem_type)
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

    std::unique_ptr<InterfaceGenerator> generator;
    std::string gen_type = "holonomic";
    std::string file_name_appendix;

    if (argc > 1 && std::string(argv[1]) == "truck_trailer"){
        std::cout << "Solving truck_trailer problem" << std::endl;

        gen_type = "truck_trailer";

        int n_trailers = 3;
        if (argc > 2){
            n_trailers = std::stoi(argv[2]);
        }

        generator = std::make_unique<TruckTrailerInterfaceGenerator>(n_trailers);
        file_name_appendix = "n_" + std::to_string(n_trailers);

    } else {
        std::cout << "Solving holonomic problem" << std::endl;
        int n = 3;
        int control_level = 2;

        if (argc > 2){
            n = std::stoi(argv[2]);
        }
        if (argc > 3){
            control_level = std::stoi(argv[3]);
        }

        generator = std::make_unique<HolonomicInterfaceGenerator>(n, control_level);
        file_name_appendix = "n_" + std::to_string(n) + "cl_" + std::to_string(control_level);
    }

    std::cout << "prepare implicit" << std::endl;
    auto tp_impl = std::make_shared<ImplicitTestProblem>(generator->PrepareImplicit());
    auto tp_expl = std::make_shared<ExplicitTestProblem>(generator->PrepareExplicit());
    auto tp_reform = std::make_shared<ExplicitTestProblem>(generator->PrepareReformulated());

    OptionRegistry options;
    IpAlgBuilder<ImplicitOcpType> builder_impl(std::make_shared<ImplicitNlpOcp>(tp_impl));
    IpAlgBuilder<OcpType> builder_expl(std::make_shared<NlpOcp>(tp_expl));
    IpAlgBuilder<OcpType> builder_reform(std::make_shared<NlpOcp>(tp_reform));    

    std::shared_ptr<IpAlgorithm<ImplicitOcpType>> ipalg_impl = builder_impl.with_options_registry(&options).build();
    std::shared_ptr<IpAlgorithm<OcpType>> ipalg_expl = builder_expl.with_options_registry(&options).build();
    std::shared_ptr<IpAlgorithm<OcpType>> ipalg_reform = builder_reform.with_options_registry(&options).build();
    
    for(int i = 0; i < 1; i++)
    {       
        // EXPLICIT
        std::cout << "solving explicit test problem" << std::endl;
        Timer timer_expl; timer_expl.start();
        IpSolverReturnFlag ret_expl = ipalg_expl->optimize();
        std::cout << "Elapsed time: " << timer_expl.stop() << std::endl;
        auto data_expl = builder_expl.get_ipdata();
        json result_expl = add_json_data(data_expl, "explicit");
        result_expl["generator_data"] = generator->GetJsonData();
        std::ofstream file2("ocp_results/ocp_result_" + gen_type + "_" + file_name_appendix + "_" + std::to_string(file_counter) + ".json");
        if (file2.is_open())
        {
            file2 << result_expl.dump(4);
            file2.close();
            file_counter++;
        }

        // IMPLICIT
        std::cout << "solving implicit test problem" << std::endl;
        Timer timer_impl; timer_impl.start();
        IpSolverReturnFlag ret_impl = ipalg_impl->optimize();
        std::cout << "Elapsed time: " << timer_impl.stop() << std::endl;
        auto data_impl = builder_impl.get_ipdata();
        json result_impl = add_json_data(data_impl, "implicit");
        result_impl["generator_data"] = generator->GetJsonData();
        std::ofstream file("ocp_results/ocp_result_" + gen_type + "_" + file_name_appendix + "_" + std::to_string(file_counter) + ".json");
        if (file.is_open())
        {
            file << result_impl.dump(4);
            file.close();
            file_counter++;
        }

        // REFORMULATED
        std::cout << "solving reformulated test problem" << std::endl;
        Timer timer_reform; timer_reform.start();
        IpSolverReturnFlag ret_reform = ipalg_reform->optimize();
        std::cout << "Elapsed time: " << timer_reform.stop() << std::endl;
        auto data_reform = builder_reform.get_ipdata();
        json result_reform = add_json_data(data_reform, "reformulated");
        result_reform["generator_data"] = generator->GetJsonData();
        std::ofstream file3("ocp_results/ocp_result_" + gen_type + "_" + file_name_appendix + "_" + std::to_string(file_counter) + ".json");
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
