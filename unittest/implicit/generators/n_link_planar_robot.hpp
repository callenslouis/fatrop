#ifndef __N_LINK_GENERATOR_HPP__
#define __N_LINK_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include "ocp_interface_generator.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

// see: "On 𝑛-Link Planar Revolute Robot: Motion Equations and New Properties"
class PlanarRobot : public InterfaceGenerator {
    public:
        // Constructor
        // n: number of links
        PlanarRobot(int n){
            // define parameters
            K_ = 40;
            dt_ = 0.05;

            n_ = n;

            nx_ = 2*n_;
            nu_ = n_;

            // define start and endpoints
            start_ = std::vector<double>(nx_, 0.0);
            end_ = std::vector<double>(nx_, 0.0);
            for (int i = 0; i < n_; i++){
                start_[i] = -3.14/2;
                end_[i] = 3.14/2;
            }

            // define variables
            th_ = MX::sym("theta", n_);     
            thd_ = MX::sym("theta_dot", n_);
            uk_ = MX::sym("uk", n);
            xk_ = vertcat(th_, thd_);
            thp_ = MX::sym("theta_plus", n_);
            thdp_ = MX::sym("theta_dot_plus", n_);
            xkp_ = vertcat(thp_, thdp_);
            
            // define dynamics basics
            Rinv_ = DM::zeros(n_, n_);
            for (int i = 0; i < n_; i++){
                for (int j = 0; j < n_; j++){
                    if (j < i){
                        Rinv_(i,j) = -1;
                    } else if (j == i){
                        Rinv_(i,j) = 1;
                    } else {
                        Rinv_(i,j) = 0;
                    }
                }
            }
            set_M(); set_C(); set_G();

            // set bounds
            lb_ = std::vector<double>(nu_+n_+1, uk_min_);
            ub_ = std::vector<double>(nu_+n_+1, uk_max_);
            for (int i = 0; i < n_ + 1; i++){
                lb_[i] = -100;
                ub_[i] = 100;
            }
            lb_K_ = {};
            ub_K_ = {};

            // set initialization
            x_init_ = std::vector<std::vector<double>>(K_+1, std::vector<double>(nx_, 0.0));
            u_init_ = std::vector<std::vector<double>>(K_, std::vector<double>(nu_, 0.0));
            for (int k = 0; k < K_+1; k++){
                for (int i = 0; i < n_; i++){
                    x_init_[k][i] = start_[i] + (end_[i] - start_[i]) * k / K_;
                }
            }
        };

        virtual ImplicitTestProblem PrepareImplicit(){
            Function eval_objk = Function("eval_objk", {uk_, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {MX::zeros(0,1)});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_ - end_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX rhs = vertcat(thdp_, mtimes(inv(eval_M_(thp_)[0]), mtimes(Rinv_, uk_) - mtimes(eval_C_(xkp_)[0], thdp_) - eval_G_(thp_)[0]));
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {uk_, xk_, xkp_}, {xk_ + dt_*rhs - xkp_});

            return ImplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK, eval_gk_ineq, eval_gK_ineq,
                    eval_dynamics_equation_implicit);
        }

        virtual ExplicitTestProblem PrepareExplicit(){
            Function eval_objk = Function("eval_objk", {uk_, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {MX::zeros(0,1)});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {vertcat(th_, uk_)});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_ - end_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX rhs = vertcat(thd_, mtimes(inv(eval_M_(th_)[0]), mtimes(Rinv_, uk_) - mtimes(eval_C_(xk_)[0], thd_) - eval_G_(th_)[0]));
            Function eval_dynamics_equation_explicit = Function("eval_dynamics_equation", {uk_, xk_}, {xk_ + dt_*rhs});

            return ExplicitTestProblem(
                    K_, nx_, nu_, 
                    x_init_, u_init_,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK,
                    eval_gk_ineq, eval_gK_ineq, eval_dynamics_equation_explicit);
        }

        virtual ExplicitTestProblem PrepareReformulated(){
            MX zk = MX::sym("zk", nx_);
            MX uk_aug = vertcat(uk_, zk);

            Function eval_objk = Function("eval_objk", {uk_aug, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {uk_});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_ - end_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX rhs = vertcat(thd_, mtimes(inv(eval_M_(th_)[0]), mtimes(Rinv_, uk_) - mtimes(eval_C_(xk_)[0], thd_) - eval_G_(th_)[0]));

            Function eval_g0 = Function("eval_g0", {uk_aug, xk_}, {vertcat(xk_ - start_, xk_ + dt_*rhs - zk)});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_}, {xk_ + dt_*rhs - zk});
            Function eval_dynamics_equation_reformulated = Function("eval_dynamics_equation", {uk_aug, xk_}, {zk});

            return ExplicitTestProblem(
                    K_, nx_, nu_+nx_, 
                    x_init_,
                    std::vector<std::vector<double>>(K_, std::vector<double>(nu_ + nx_, 0.0)), 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK,
                    eval_gk_ineq, eval_gK_ineq,
                    eval_dynamics_equation_reformulated);
        }

        virtual json GetJsonData(){
            json j;
            j["problem_name"] = "planar_robot";
            j["K"] = K_;
            j["nx"] = nx_;
            j["nu"] = nu_;
            j["dt"] = dt_;
            j["n"] = n_;
            j["m"] = m_;
            j["l"] = l_;
            j["lc"] = lc_;
            j["J"] = J;
            j["uk_min"] = uk_min_;
            j["uk_max"] = uk_max_;
            j["start"] = start_;
            j["end"] = end_;
            return j;
        }

        virtual std::string GetInterfaceName(){ return "planar_robot";};
        virtual std::string GetFileNameAppendix(){return "nb_links_" + std::to_string(n_);};

    private:
        double get_alpha(int i, int j){
            if (i == j){
                double val = J + m_*lc_*lc_;
                for (int k = j+1; k <= n_; k++){
                    val += m_*l_*l_;
                }
                return val;
            } else {
                double val = m_*lc_*l_;
                for (int k = std::max(i,j)+1; k <= n_; k++){
                    val += m_*l_*l_;
                }
                return val;
            }
        }

        double get_beta(int i){
            if (i == n_){
                return m_*lc_*g;
            } else {
                double val = m_*lc_*g;
                for (int k = i+1; k <= n_; k++){
                    val += m_*l_*g;
                }
                return val;
            }
        }

        void set_M(){
            MX M = MX::zeros(n_, n_);
            for (int i = 0; i < n_; i++){
                for (int j = 0; j < n_; j++){
                    M(i,j) = get_alpha(i,j)*cos(th_(j)-th_(i));
                }
            }
            eval_M_ = Function("eval_M", {th_}, {M});
        }

        void set_C(){
            MX C = MX::zeros(n_, n_);
            for (int i = 0; i < n_; i++){
                for (int j = 0; j < n_; j++){
                    C(i,j) = -get_alpha(i, j)*sin(th_(j)-th_(i))*thd_(j);
                }
            }
            eval_C_ = Function("eval_C", {xk_}, {C});
        }

        void set_G(){
            MX G = MX::zeros(n_, 1);
            for (int i = 0; i < n_; i++){
                G(i) = -get_beta(i)*sin(th_(i));
            }
            eval_G_ = Function("eval_G", {th_}, {G});
        }

        int K_ = 100;
        double dt_ = 0.05;

        double m_ = 1.0;
        double l_ = 1.0;
        double lc_ = l_/2;
        double J = 1.0;
        double g = 9.81;

        double uk_min_ = -10;
        double uk_max_ = 10;

        MX th_;
        MX thd_;
        MX uk_;
        MX xk_;
        MX thp_;
        MX thdp_;
        MX xkp_;
        DM Rinv_;

        Function eval_M_;
        Function eval_C_;
        Function eval_G_;

        int n_;
        int nx_;
        int nu_;

        std::vector<double> start_;
        std::vector<double> end_;

        std::vector<double> lb_;
        std::vector<double> ub_;
        std::vector<double> lb_K_;
        std::vector<double> ub_K_;

        std::vector<std::vector<double>> x_init_;
        std::vector<std::vector<double>> u_init_;
};

#endif