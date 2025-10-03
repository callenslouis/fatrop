#ifndef __QUADRUPED_GENERATOR_HPP__
#define __QUADRUPED_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include "ocp_interface_generator.hpp"
#include "quadruped_helper.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

class QuadrupedGenerator : public InterfaceGenerator {
    public:
        // Constructor
        QuadrupedGenerator(){
            // define params
            standing_stance_.insert(standing_stance_.end(), 
                    standing_body_pos_.begin(), standing_body_pos_.end());
            standing_stance_.insert(standing_stance_.end(),
                    standing_body_quat_.begin(), standing_body_quat_.end());
            standing_stance_.insert(standing_stance_.end(),
                    standing_leg_q_.begin(), standing_leg_q_.end());

            for (int i = 0; i < nq_; i++){start_[i] = standing_stance_[i];}
            start_[nq_] = push_vx_;
            start_[nq_ + 1] = push_vy_;

            // define initial guess
            x_init_ = std::vector<std::vector<double>>(K_+1, std::vector<double>(nx_, 0.0));
            for (int k = 0; k < K_+1; k++){
                for (int i = 0; i < nq_; i++){
                    x_init_[k][i] = standing_stance_[i];
                }
            }
            u_init_ = std::vector<std::vector<double>>(K_, std::vector<double>(nu_, 0.0));
            for (int k = 0; k < K_; k++){
                u_init_[k] = {
                    -4.27557097,   5.91086644,  17.19719031,  -3.27040327,  
                    -4.81348097, -12.73962232,   3.67687864,   4.22256038,  
                    10.28272374,   5.82689115,  -3.42997252, -14.4580922
                };
            }

            // define functions
            eval_objk_ = Function("eval_objk", {uk_, xk_}, 
                {1e1*sumsqr(base_pos_ - standing_body_pos_) + 
                 1e3*sumsqr(base_quat_ - standing_body_quat_) +
                 1e0*sumsqr(v_(Slice(0,3)))});
            eval_objK_ = Function("eval_objK", {xk_}, 
                {1e3*sumsqr(v_) + 
                 1e3*(sumsqr(base_quat_ - standing_body_quat_) + 
                      sumsqr(leg_q_ - standing_leg_q_))});
            
            eval_g0_ = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            
            MX z = MX::zeros(0,1);
            eval_gk_ = Function("eval_gk", {uk_, xk_}, {z});
            eval_gK_ = Function("eval_gK", {xk_}, {z});
            eval_gk_ineq_ = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            eval_gK_ineq_ = Function("eval_gK_ineq", {xk_}, {z});

            std::cout << "initializing Pinocchio-Casadi model..." << std::endl;
            pc_ = std::make_unique<PinocchioCasadi>(dt_);
            std::cout << "done!" << std::endl;
            // pc_.SimulateFalling();
        };

        virtual ImplicitTestProblem PrepareImplicit(){
            MXVector in = {q_, v_, uk_};
            MXVector out = pc_->discrete_fn(in);
            MX qnext = out[0];
            MX vnext = out[1];
            Function eval_dynamics_equation_implicit = 
                Function("eval_dynamics_equataion", {uk_, xk_, xkp_}, {vertcat(qnext - q_p_, vnext - v_p_)});

            std::cout << "Creating implicit test problem" << std::endl;
            return ImplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    eval_dynamics_equation_implicit);
        }

        virtual ExplicitTestProblem PrepareExplicit(){
            MXVector in = {q_, v_, uk_};
            MXVector out = pc_->discrete_fn(in);
            MX qnext = out[0];
            MX vnext = out[1];
            Function eval_dynamics_equation_explicit = Function("eval_dynamics_equation", {uk_, xk_}, {vertcat(qnext, vnext)});

            return ExplicitTestProblem(
                    K_, nx_, nu_, 
                    x_init_, u_init_,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_,
                    eval_gk_ineq_, eval_gK_ineq_, eval_dynamics_equation_explicit);
        }

        virtual ExplicitTestProblem PrepareReformulated(){
            MX zk = MX::sym("zk", nx_);
            MX uk_aug = vertcat(uk_, zk);
            MXVector ukxk = {uk_, xk_};
            eval_objk_(ukxk);

            Function eval_objk = Function("eval_objk", {uk_aug, xk_}, {eval_objk_(ukxk)[0]});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {eval_gk_ineq_(ukxk)[0]});

            MXVector in = {q_, v_, uk_};
            MXVector out = pc_->discrete_fn(in);
            MX qnext = out[0];
            MX vnext = out[1];
            Function eval_g0 = Function("eval_g0", {uk_aug, xk_}, {vertcat(eval_g0_(ukxk)[0], vertcat(qnext, vnext) - zk)});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_}, {vertcat(qnext, vnext) - zk});

            Function eval_dynamics_equation_reformulated = Function("eval_dynamics_equation", {uk_aug, xk_}, {zk});

            std::vector<std::vector<double>> u_init(K_, std::vector<double>(nu_ + nx_, 0.0));
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
                    eval_objk, eval_objK_, eval_gk, eval_g0, eval_gK_,
                    eval_gk_ineq, eval_gK_ineq_,
                    eval_dynamics_equation_reformulated);
        }

        virtual json GetJsonData(){
            json j;
            j["problem_name"] = "quadruped";
            j["K"] = K_;
            j["nx"] = nx_;
            j["nu"] = nu_;
            j["dt"] = dt_;
            j["uk_min"] = uk_min_;
            j["uk_max"] = uk_max_;
            j["start"] = start_;
            return j;
        }

    virtual std::string GetInterfaceName(){ return "quadruped";};
    virtual std::string GetFileNameAppendix(){return "";};

        std::unique_ptr<PinocchioCasadi> pc_;
    private:
        int K_ = 10 + 0*100;
        int nq_ = 3 + 4 + 12;
        int nx_ = 2*nq_ - 1;
        int nu_ = 12;
        double dt_ = 0.02;
        double uk_min_ = -50;
        double uk_max_ = 50;

        double push_vx_ = 0*1.5;
        double push_vy_ = 0*2.0;


        MX base_pos_ = MX::sym("base_pos", 3);
        MX base_vel_ = MX::sym("base_vel", 3);
        MX base_quat_ = MX::sym("base_quat", 4);
        MX base_omega_ = MX::sym("base_omega", 3);
        MX leg_q_ = MX::sym("leg_q", 12);
        MX leg_v_ = MX::sym("leg_qdot", 12);
        MX q_ = vertcat(base_pos_, base_quat_, leg_q_);
        MX v_ = vertcat(base_vel_, base_omega_, leg_v_);

        MX base_pos_p_ = MX::sym("base_pos_p", 3);
        MX base_vel_p_ = MX::sym("base_vel_p", 3);
        MX base_quat_p_ = MX::sym("base_quat_p", 4);
        MX base_omega_p_ = MX::sym("base_omega_p", 3);
        MX leg_q_p_ = MX::sym("leg_q_p", 12);
        MX leg_v_p_ = MX::sym("leg_qdot_p", 12);
        MX q_p_ = vertcat(base_pos_p_, base_quat_p_, leg_q_p_);
        MX v_p_ = vertcat(base_vel_p_, base_omega_p_, leg_v_p_);

        MX xk_ = vertcat(q_, v_);
        MX uk_ = MX::sym("uk", nu_);
        MX xkp_ = vertcat(q_p_, v_p_);

        std::vector<double> standing_body_pos_ = {0, 0, 0.5292};
        std::vector<double> standing_body_quat_ = {0, 0, 0, 1};
        std::vector<double> standing_leg_q_ = {-0.1, 0.7, -1, 0.1, 0.7, -1,
                                               -0.1, -0.7, -1, 0.1, -0.7, -1};
        std::vector<double> standing_stance_;
        std::vector<double> start_ = std::vector<double>(nx_, 0.0);

        std::vector<double> lb_ = std::vector<double>(nu_, uk_min_);
        std::vector<double> ub_ = std::vector<double>(nu_, uk_max_);
        std::vector<double> lb_K_ = {};
        std::vector<double> ub_K_ = {};

        std::vector<std::vector<double>> x_init_;
        std::vector<std::vector<double>> u_init_;

        Function eval_objk_;
        Function eval_objK_;
        Function eval_g0_;
        Function eval_gk_;
        Function eval_gK_;
        Function eval_gk_ineq_;
        Function eval_gK_ineq_;
};

#endif