#ifndef __SOLAR_RECEIVER_REACTOR_GENERATOR_HPP__
#define __SOLAR_RECEIVER_REACTOR_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include "ocp_interface_generator.hpp"
#include <json/include/nlohmann/json.hpp>

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;
using json = nlohmann::json;

class SolarReceiverReactorGenerator : public InterfaceGenerator {
    public:
        // Constructor
        SolarReceiverReactorGenerator(int n){
            n_ = n;

            // load metadata
            std::ifstream file("../../unittest/implicit/generators/batch_reactor_model/meta_data.json");
            json meta_data;
            file >> meta_data;
            T_ = meta_data["T"];
            dt_ = meta_data["dt"];
            K_ = T_ / dt_;
            nx_ = meta_data["nx"];
            nu_ = meta_data["nu"];
            dt_ = meta_data["dt"];
            file.close();

            // load functions
            std::string folder = "../../unittest/implicit/generators/solar_receiver_reactor_model/";
            eval_objk_ = Function::load(folder + "eval_objk.casadi");
            eval_objK_ = Function::load(folder + "eval_objK.casadi");
            eval_g0_ = Function::load(folder + "eval_g0.casadi");
            eval_gk_ = Function::load(folder + "eval_gk.casadi");
            eval_gk_ineq_ = Function::load(folder + "eval_gk_ineq.casadi");
            eval_gK_ = Function::load(folder + "eval_gK.casadi");
            eval_gK_ineq_ = Function::load(folder + "eval_gK_ineq.casadi");
            rhs_ = Function::load(folder + "rhs.casadi");
            expl_dyn_ = Function::load(folder + "expl_dyn.casadi");
            expl_dyn_rk2_ = Function::load(folder + "expl_dyn_rk2.casadi");
            impl_dyn_ = Function::load(folder + "impl_dyn.casadi");
            impl_dyn_trap_ = Function::load(folder + "impl_dyn_trap.casadi");

            // set initialization //
            x_init_ = std::vector<std::vector<double>>(K_+1, std::vector<double>(nx_, 0.0));
            u_init_ = std::vector<std::vector<double>>(K_, std::vector<double>(nu_, 0.0));
            for (int k = 0; k < K_+1; k++){
                for (int i = 0; i < nx_; i++){ x_init_[k][i] = x0_[i];}
            }
            for (int k = 0; k < K_; k++){ u_init_[k][0] = u0_;}
        };

        virtual ImplicitTestProblem PrepareImplicit(){           
            return ImplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    impl_dyn_trap_);
        }

        virtual ExplicitTestProblem PrepareExplicit(){           
            return ExplicitTestProblem(
                    K_, nx_, nu_, 
                    x_init_, u_init_,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_,
                    eval_gk_ineq_, eval_gK_ineq_, expl_dyn_rk2_);
        }

        virtual ExplicitTestProblem PrepareReformulated(){
            MX zk = MX::sym("zk", nx_);
            MX uk_aug = vertcat(uk_, zk);

            MXVector ukxk = {uk_, xk_};
            Function eval_objk = Function("eval_objk", {uk_aug, xk_}, {eval_objk_(ukxk)[0]});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {eval_gk_ineq_(ukxk)[0]});
            Function eval_gK = Function("eval_gK", {xk_}, {eval_gK_(xk_)[0]});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {eval_gK_ineq_(xk_)[0]});
            
            Function eval_g0 = Function("eval_g0", {uk_aug, xk_}, {vertcat(eval_g0_(ukxk)[0], impl_dyn_trap_)});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_}, {vertcat(eval_gk_(ukxk)[0], xk_ + dt_*rhs_(MXVector{uk_, zk})[0] - zk)});
            Function eval_dynamics_equation_reformulated = Function("eval_dynamics_equation", {uk_aug, xk_}, {zk});

            std::vector<std::vector<double>> u_init(K_-1, std::vector<double>(nu_ + nx_, 0.0));
            for (int k = 0; k < K_-1; k++){
                for (int i = 0; i < nu_; i++){
                    u_init[k][i] = u_init_[k][i];
                }
                for (int i = 0; i < nx_; i++){
                    u_init[k][nu_ + i] = x_init_[k][i];
                }
            }

            return ExplicitTestProblem(
                    K_, nx_, nu_+nx_, 
                    x_init_, u_init,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk, eval_objK_, eval_gk, eval_g0, eval_gK,
                    eval_gk_ineq, eval_gK_ineq,
                    eval_dynamics_equation_reformulated);
        }

        virtual json GetJsonData(){
            json j;
            j["problem_name"] = "solar_receiver_reactor";
            j["n"] = n_;
            j["K"] = K_;
            j["nx"] = nx_;
            j["nu"] = nu_;
            j["dt"] = dt_;
            return j;
        }

        virtual std::string GetInterfaceName(){ return "solar_receiver_reactor";};
        virtual std::string GetFileNameAppendix(){return std::to_string(n_);};

    private:
        int K_;
        double T_;
        double dt_;

        int n_;

        int nx_ = 18;
        int nu_ = 1;
        
        MX xk_ = MX::sym("xk", nx_);
        MX uk_ = MX::sym("uk", nu_);
        MX xkp_ = MX::sym("xkp", nx_);
        
        Function eval_objk_;
        Function eval_objK_;

        Function eval_g0_;
        Function eval_gk_;
        Function eval_gk_ineq_;
        Function eval_gK_;
        Function eval_gK_ineq_;
        Function rhs_;
        Function expl_dyn_;
        Function expl_dyn_rk2_;
        Function impl_dyn_;
        Function impl_dyn_trap_;

        std::vector<double> x0 = {6.79982544e-01, 3.42209761e-01, 
            9.44749463e-03, 2.80015274e-01, 5.75566459e-01, 8.66733442e-01, 
            4.00021820e-02, 8.22237799e-02, 1.23819063e-01, 3.11737855e+02, 
            3.11947172e+02, 3.12186673e+02, 5.39943204e+02, 5.41843654e+02, 
            5.44006506e+02, 5.27211235e+02, 5.30084664e+02, 5.37091843e+02};
        double u0 = 0.5186015644624212;
        std::vector<double> lb_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 298, 298, 298, 450, 450, 450, 450, 450, 450};
        std::vector<double> ub_ = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
        std::vector<double> lb_K_ = {};
        std::vector<double> ub_K_ = {};

        std::vector<std::vector<double>> x_init_;
        std::vector<std::vector<double>> u_init_;
};

#endif