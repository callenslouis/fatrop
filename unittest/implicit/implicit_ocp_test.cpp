
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
#include "generators/bycicle_generator.hpp"
#include "generators/example_static_generator.hpp"
#include "generators/show_interface_output.hpp"
#include "generators/n_link_planar_robot.hpp"

#include "json/single_include/nlohmann/json.hpp"

#include <fstream>
#include <tuple>

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

void SolveProblem(std::unique_ptr<InterfaceGenerator> &generator){
    auto tp_impl = std::make_shared<ImplicitTestProblem>(generator->PrepareImplicit());
    auto tp_reform = std::make_shared<ExplicitTestProblem>(generator->PrepareReformulated());
    auto tp_expl = std::make_shared<ExplicitTestProblem>(generator->PrepareExplicit());
    ImplicitTestProblem tp_interface_impl = *tp_impl;
    ExplicitTestProblem tp_interface_expl = *tp_expl;
    ExplicitTestProblem tp_interface_reform = *tp_reform;
    // show_implicit_interface_output(tp_interface_impl, "output_interface_implicit.txt");
    // show_interface_output(tp_interface_expl, "output_interface_explicit.txt");
    // show_interface_output(tp_interface_reform, "output_interface_reformulated.txt");

    std::cout << "Generated test problems" << std::endl;
    std::string gen_type = generator->GetInterfaceName();
    std::string file_name_appendix = generator->GetFileNameAppendix();

    OptionRegistry options;
    IpAlgBuilder<ImplicitOcpType> builder_impl(std::make_shared<ImplicitNlpOcp>(tp_impl));
    IpAlgBuilder<OcpType> builder_expl(std::make_shared<NlpOcp>(tp_expl));
    IpAlgBuilder<OcpType> builder_reform(std::make_shared<NlpOcp>(tp_reform));    

    std::shared_ptr<IpAlgorithm<ImplicitOcpType>> ipalg_impl = builder_impl.with_options_registry(&options).build();
    std::shared_ptr<IpAlgorithm<OcpType>> ipalg_expl = builder_expl.with_options_registry(&options).build();
    std::shared_ptr<IpAlgorithm<OcpType>> ipalg_reform = builder_reform.with_options_registry(&options).build();
    std::cout << "built ip algorithms" << std::endl;

    json result_expl, result_reform, result_impl;
    bool solved_expl = false, solved_reform = false, solved_impl = false;
    
    // REFORMULATED
    try{
        std::cout << "solving reformulated test problem" << std::endl;
        Timer timer_reform; timer_reform.start();
        IpSolverReturnFlag ret_reform = ipalg_reform->optimize();
        std::cout << "Elapsed time: " << timer_reform.stop() << std::endl;
        auto data_reform = builder_reform.get_ipdata();
        result_reform = add_json_data(data_reform, "reformulated");
        result_reform["generator_data"] = generator->GetJsonData();
        std::ofstream file3("ocp_results/ocp_result_reformulated_" + gen_type + "_" + file_name_appendix + ".json");
        if (file3.is_open())
        {
            file3 << result_reform.dump(4);
            file3.close();
        }
        solved_reform = true;
    } catch (std::exception& e){
        std::cout << "Exception caught during reformulated solve: " << e.what() << std::endl;
    }

    // EXPLICIT
    try{
        std::cout << "solving explicit test problem" << std::endl;
        Timer timer_expl; timer_expl.start();
        IpSolverReturnFlag ret_expl = ipalg_expl->optimize();
        std::cout << "Elapsed time: " << timer_expl.stop() << std::endl;
        auto data_expl = builder_expl.get_ipdata();
        result_expl = add_json_data(data_expl, "explicit");
        result_expl["generator_data"] = generator->GetJsonData();
        std::ofstream file2("ocp_results/ocp_result_explicit_" + gen_type + "_" + file_name_appendix + ".json");
        if (file2.is_open())
        {
            file2 << result_expl.dump(4);
            file2.close();
        }
        solved_expl = true;
    } catch (std::exception& e){
        std::cout << "Exception caught during explicit solve: " << e.what() << std::endl;
    }

    // IMPLICIT
    try{
        std::cout << "solving implicit test problem" << std::endl;
        Timer timer_impl; timer_impl.start();
        IpSolverReturnFlag ret_impl = ipalg_impl->optimize();
        std::cout << "Elapsed time: " << timer_impl.stop() << std::endl;
        auto data_impl = builder_impl.get_ipdata();
        result_impl = add_json_data(data_impl, "implicit");
        result_impl["generator_data"] = generator->GetJsonData();
        std::ofstream file("ocp_results/ocp_result__implicit_" + gen_type + "_" + file_name_appendix + ".json");
        if (file.is_open())
        {
            file << result_impl.dump(4);
            file.close();
        }
        solved_impl = true;
    } catch (std::exception& e){
        std::cout << "Exception caught during implicit solve: " << e.what() << std::endl;
    }

    std::cout << "Finished solving problem" << std::endl;
    std::cout << "nb iterations: " << std::endl;
    if (solved_expl){std::cout << "\texplicit:     " << result_expl["metadata"]["iterations"] << std::endl;}
    if (solved_reform){std::cout << "\treformulated: " << result_reform["metadata"]["iterations"] << std::endl;}
    if (solved_impl){std::cout << "\timplicit:     " << result_impl["metadata"]["iterations"] << std::endl;}

    std::cout << "t_total: " << std::endl;
    if (solved_expl){std::cout << "\texplicit:     " << result_expl["metadata"]["timing_statistics"]["total"] << std::endl;}
    if (solved_reform){std::cout << "\treformulated: " << result_reform["metadata"]["timing_statistics"]["total"] << std::endl;}
    if (solved_impl){std::cout << "\timplicit:     " << result_impl["metadata"]["timing_statistics"]["total"] << std::endl;}

    std::cout << "t_func: " << std::endl;
    if (solved_expl){std::cout << "\texplicit:     " << result_expl["metadata"]["timing_statistics"]["function evaluation"] << std::endl;}
    if (solved_reform){std::cout << "\treformulated: " << result_reform["metadata"]["timing_statistics"]["function evaluation"] << std::endl;}
    if (solved_impl){std::cout << "\timplicit:     " << result_impl["metadata"]["timing_statistics"]["function evaluation"] << std::endl;}

    std::cout << "t_fatrop: " << std::endl;
    if (solved_expl){std::cout << "\texplicit:     " << result_expl["metadata"]["timing_statistics"]["fatrop"] << std::endl;}
    if (solved_reform){std::cout << "\treformulated: " << result_reform["metadata"]["timing_statistics"]["fatrop"] << std::endl;}
    if (solved_impl){std::cout << "\timplicit:     " << result_impl["metadata"]["timing_statistics"]["fatrop"] << std::endl;}

}

void SolveSingleProblemTruckTrailer(int n_trailers){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<TruckTrailerInterfaceGenerator>(n_trailers);
    SolveProblem(generator);
}
void SolveAllTruckTrailer(){
    for (int n_trailers = 0; n_trailers <= 10; n_trailers++){
        SolveSingleProblemTruckTrailer(n_trailers);
    }
}

void SolveSingleProblemBycicle(){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<BycicleGenerator>();
    SolveProblem(generator);
}

void SolveSingleProblemExampleStatic(){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<ExampleStaticGenerator>();
    SolveProblem(generator);
}

void SolveSingleProblemHolonomic(int n, int control_level){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<HolonomicInterfaceGenerator>(n, control_level);
    SolveProblem(generator);
}
void SolveAllHolonomic(){
    for (int n = 1; n <= 7; n++){
        for (int control_level = n == 1 ? 2 : 1; control_level <= 4; control_level++){
            SolveSingleProblemHolonomic(n, control_level);
        }
    }
}

void SolveSingleProblemPlanarRobot(int n_links){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<PlanarRobot>(n_links);
    SolveProblem(generator);
}
void SolveAllPlanarRobot(){
    for (int n_links = 1; n_links <= 5; n_links++){
        SolveSingleProblemPlanarRobot(n_links);
    }
}


int main(int argc, char **argv)
{
    // // create a directory ocp_results
    // int temp = system("mkdir -p ocp_results");

    if (argc < 3){
        std::cout << "Please provide the following arguments to this executable:" << std::endl;
        std::cout << "\t\"single\" or \"all\"" << std::endl;
        std::cout << "\tproblem name (truck_trailer, bycicle, example_static, holonomic)" << std::endl;
        std::cout << "in case of a single problem, also provide the parameters desired (optionally)" << std::endl;
        return 0;
    }

    if (std::string(argv[1]) == "single"){
        std::unique_ptr<InterfaceGenerator> generator;

        if (std::string(argv[2]) == "truck_trailer"){
            int n_trailers = 1;
            if (argc > 3){ n_trailers = std::stoi(argv[3]);}
            generator = std::make_unique<TruckTrailerInterfaceGenerator>(n_trailers);

        } else if (std::string(argv[2]) == "bycicle"){
            generator = std::make_unique<BycicleGenerator>();
        } else if (std::string(argv[2]) == "example_static"){
            generator = std::make_unique<ExampleStaticGenerator>();
        } else if (std::string(argv[2]) == "holonomic"){
            int n = 3;
            int control_level = 2;

            if (argc > 3){ n = std::stoi(argv[3]);}
            if (argc > 4){ control_level = std::stoi(argv[4]);}

            generator = std::make_unique<HolonomicInterfaceGenerator>(n, control_level);
        } else if (std::string(argv[2]) == "planar_robot"){
            int n_links = 3;
            if (argc > 3){ n_links = std::stoi(argv[3]);}
            generator = std::make_unique<PlanarRobot>(n_links);
        } else {
            std::cout << "Second argument should be either \"truck_trailer\", \"bycicle\", \"example_static\" or \"holonomic\" when first argument is \"single\"" << std::endl;
            return 0;
        }

        SolveProblem(generator);

    } else if (std::string(argv[1]) == "all"){
        if (std::string(argv[2]) == "truck_trailer"){
            SolveAllTruckTrailer();
        } else if (std::string(argv[2]) == "holonomic"){
            SolveAllHolonomic();
        } else if (std::string(argv[2]) == "planar_robot"){
            SolveAllPlanarRobot();
        } else {
            std::cout << "Second argument should be either \"truck_trailer\", \"holonomic\" or \"planar_robot\" when first argument is \"all\"" << std::endl;
            return 0;
        }

    } else {
        std::cout << "First argument should be either \"single\" or \"all\"" << std::endl;
        return 0;
    }

    return 0;
}
