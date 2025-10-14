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
            original_standing_stance_.insert(original_standing_stance_.end(), 
                    standing_body_pos_.begin(), standing_body_pos_.end());
            original_standing_stance_.insert(original_standing_stance_.end(),
                    standing_body_quat_.begin(), standing_body_quat_.end());    
            original_standing_stance_.insert(original_standing_stance_.end(),
                    standing_leg_q_.begin(), standing_leg_q_.end());

            std::vector<double> original_standing_body_pos_ = standing_body_pos_;
            standing_body_pos_[2] += 0.05;
            standing_stance_.insert(standing_stance_.end(), 
                    standing_body_pos_.begin(), standing_body_pos_.end());
            standing_stance_.insert(standing_stance_.end(),
                    standing_body_quat_.begin(), standing_body_quat_.end());
            standing_stance_.insert(standing_stance_.end(),
                    standing_leg_q_.begin(), standing_leg_q_.end());

            for (int i = 0; i < nq_; i++){start_[i] = standing_stance_[i];}
            start_[nq_] = 0*push_vx_;
            start_[nq_ + 1] = 0*push_vy_;

            // define initial guess
            x_init_ = std::vector<std::vector<double>>(K_+1, std::vector<double>(nx_, 0.0));
            for (int k = 0; k < K_+1; k++){
                for (int i = 0; i < nq_; i++){
                    x_init_[k][i] = original_standing_stance_[i];
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
                {1 * 1e1*sumsqr(base_pos_ - original_standing_body_pos_) + 
                 1 * 1e3*sumsqr(base_quat_ - standing_body_quat_) +
                 1 * 1e1*sumsqr(v_(Slice(0,3))) +
                 1 * 1e0*sumsqr(v_) +
                 1 * 1e5*(sumsqr(uk_) + 0*sumsqr(xk_))});
            eval_objK_ = Function("eval_objK", {xk_}, //{0});
                {1 * 1e3*sumsqr(v_) + 
                 1 * 1e3*(sumsqr(base_quat_ - standing_body_quat_) + 
                      sumsqr(leg_q_ - standing_leg_q_)) +
                      0 * sumsqr(xk_)});
            
            
            MX z = MX::zeros(0,1);
            eval_g0_ = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            // eval_g0_ = Function("eval_g0", {uk_, xk_}, {z});
            eval_gk_ = Function("eval_gk", {uk_, xk_}, {z});
            eval_gK_ = Function("eval_gK", {xk_}, {z});
            eval_gk_ineq_ = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            // eval_gk_ineq_ = Function("eval_gk_ineq", {uk_, xk_}, {z});
            eval_gK_ineq_ = Function("eval_gK_ineq", {xk_}, {z});

            // pc_ = std::make_unique<PinocchioCasadi>(dt_);
            // pc_.SimulateFalling();

            // load the casadi dynamics functions
            expl_dyn_ = Function::load("quadruped_explicit_integrator.casadi");
            impl_dyn_ = Function::load("quadruped_implicit_integrator.casadi");
        };

        virtual ImplicitTestProblem PrepareImplicit(){
            return ImplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    impl_dyn_);
        }

        virtual ExplicitTestProblem PrepareExplicit(){
            // normal appraoch
            // MXVector in = {q_, v_, uk_};
            // MXVector out = pc_->discrete_fn(in);
            // MX qnext = out[0];
            // MX vnext = out[1];
            // Function eval_dynamics_equation_explicit = Function("eval_dynamics_equation", {uk_, xk_}, {vertcat(qnext, vnext)});

            // no dynamics
            // MX xnext = MX(nx_, 1);
            // xnext(Slice(0, nx_ - nu_)) = xk_(Slice(0, nx_-nu_));
            // xnext(Slice(nx_ - nu_, nx_)) = xk_(Slice(nx_-nu_, nx_)) + uk_;
            // Function eval_dynamics_equation_explicit = Function("eval_dynamics_equation", {uk_, xk_}, {xnext});

            // test dynamics by evaluating and printing
            // DM q_test = DM(nq_, 1);
            // DM v_test = DM(nq_-1, 1);
            // DM u_test = DM(nu_, 1);
            // for (int i = 0; i < nq_; i++){q_test(i) = 1 - 0.1*i + 0.02*i*i;}
            // for (int i = 0; i < nq_-1; i++){v_test(i) = -2 + 0.524*i*i*i;}
            // for (int i = 0; i < nu_; i++){u_test(i) = 3.141592*(0.5 - i);}
            // DMVector out_test = eval_dynamics_equation_explicit(DMVector{u_test, vertcat(q_test, v_test)});
            // std::cout << "test dynamics: " << out_test[0].T() << std::endl;
            // std::cout << "acc_func: " << pc_->acc_func(DMVector{q_test, v_test, u_test})[0] << std::endl;

            return ExplicitTestProblem(
                    K_, nx_, nu_, 
                    x_init_, u_init_,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_,
                    eval_gk_ineq_, eval_gK_ineq_, expl_dyn_);
        }

        virtual ExplicitTestProblem PrepareReformulated(){
            MX zk = MX::sym("zk", nx_);
            MX uk_aug = vertcat(uk_, zk);
            MXVector ukxk = {uk_, xk_};

            Function eval_objk = Function("eval_objk", {uk_aug, xk_}, {eval_objk_(ukxk)[0]});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {eval_gk_ineq_(ukxk)[0]});

            MXVector in = {uk_, xk_, zk};
            MXVector out = impl_dyn_(in);
            MX xnext = out[0];
            Function eval_g0 = Function("eval_g0", {uk_aug, xk_}, {vertcat(eval_g0_(ukxk)[0], xnext - zk)});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_}, {xnext - zk});

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

        void SolveOptiInstance(){
            MXVector in = {uk_, vertcat(q_, v_)};
            // MXVector out = pc_->discrete_fn(in);
            MXVector out = expl_dyn_(in);
            MX xnext = out[0];
            Function eval_dynamics_equation_explicit = Function("eval_dynamics_equation", {uk_, xk_}, {xnext});

            Opti opti = Opti();
            int N = K_-1;

            std::vector<MX> qq_list = {};
            std::vector<MX> vv_list = {};
            std::vector<MX> uu_list = {};
            for (int k = 0; k < N + 1; k++){
                qq_list.push_back(opti.variable(nq_));
                vv_list.push_back(opti.variable(nq_-1));
                if (k < N){
                    uu_list.push_back(opti.variable(nu_));
                }
            }
            MX qq = vertcat(qq_list);
            MX vv = vertcat(vv_list);
            MX uu = vertcat(uu_list);

            if (eval_g0_.sparsity_out(0).size1() > 0){
                opti.subject_to(eval_g0_(MXVector{uu_list[0], vertcat(qq_list[0], vv_list[0])})[0] == DM::zeros(eval_g0_.sparsity_out(0).size1()));
            }
            
            for (int k = 0; k < N; k++){
                MX qk = qq_list[k];
                MX vk = vv_list[k];
                MX uk = uu_list[k];

                MX x = vertcat(qk, vk);
                MX x_next = eval_dynamics_equation_explicit(MXVector{uk, x})[0];
                MX q_next = x_next(Slice(0, nq_));
                MX v_next = x_next(Slice(nq_, nx_));

                opti.subject_to(qq_list[k+1] == q_next);
                opti.subject_to(vv_list[k+1] == v_next);
                // no dynamics
                // MX v_next_no_dyn = vk;
                // for (int i = 0; i < nu_; i++){
                //     v_next_no_dyn(nq_-1 - nu_ + i) = vk(nq_-1 - nu_ + i) + uk(i);
                // }
                // opti.subject_to(qq_list[k+1] == qq_list[k]);
                // opti.subject_to(vv_list[k+1] == v_next_no_dyn);

                if (eval_gk_ineq_.sparsity_out(0).size1() > 0){
                    opti.subject_to(uk_min_ <= (eval_gk_ineq_(MXVector{uk, x})[0] <= uk_max_));
                }
            }

            MX obj = 0;
            for (int k = 0; k < N; k++){
                MX qk = qq_list[k];
                MX vk = vv_list[k];
                MX uk = uu_list[k];
                MX x = vertcat(qk, vk);
                obj += eval_objk_(MXVector{uk, x})[0];
            }
            MX qk = qq_list[N];
            MX vk = vv_list[N];
            MX x = vertcat(qk, vk);
            obj += eval_objK_(x)[0];
            opti.minimize(obj);

            for (int k = 0; k < N+1; k++){
                for (int i = 0; i < nq_; i++){
                    opti.set_initial(qq_list[k](i), x_init_[0][i]);
                }
                if (k < N){
                    for (int i = 0; i < nu_; i++){
                        opti.set_initial(uu_list[k](i), u_init_[0][i]);
                    }
                }
            }

            // define options
            Dict casadi_opts;
            Dict solver_opts;
            casadi_opts["structure_detection"] = "auto";
            // casadi_opts["fatrop.print_level"] = 12;
            casadi_opts["fatrop.mu_init"] = 0.1;
            solver_opts["max_iter"] = 1000;
            // solver_opts["mu_init"] = 0.1;
            // solver_opts["print_level"] = 7;

            opti.solver("fatrop", casadi_opts, solver_opts);

            opti.solve();
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

        // std::unique_ptr<PinocchioCasadi> pc_;
    private:
        int K_ = 100;
        int nq_ = 3 + 4 + 12;
        int nx_ = 2*nq_ - 1;
        int nu_ = 12;
        double dt_ = 0.02;
        double uk_min_ = -100;
        double uk_max_ = 100;

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
        std::vector<double> standing_leg_q_ = {-0.1, 0.7, -1, -0.1, -0.7, 1,
                                                0.1, 0.7, -1, 0.1, -0.7, 1};
        std::vector<double> standing_stance_;
        std::vector<double> original_standing_stance_;
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

        Function expl_dyn_;
        Function impl_dyn_;
};

#endif