#ifndef FATROP_IMPLICIT_TEST_PROBLEM_HPP
#define FATROP_IMPLICIT_TEST_PROBLEM_HPP

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

/////////////////////////////////////
/////////////////////////////////////
////                             ////
////  IMPLICIT OCP TEST PROBLEM  ////
////                             ////
/////////////////////////////////////
/////////////////////////////////////
class ImplicitTestProblem : public ImplicitOcpAbstract{
    public:
        ImplicitTestProblem(Index K, const Index nx, const Index nu, 
            std::vector<std::vector<double>> x_init, 
            std::vector<std::vector<double>> u_init, 
            std::vector<double> g_ineq_lb, std::vector<double> g_ineq_ub, 
            std::vector<double> g_ineq_K_lb, std::vector<double> g_ineq_K_ub, 
            Function eval_objk, Function eval_objK, Function eval_gk, Function eval_g0, 
            Function eval_gK, Function eval_gk_ineq, Function eval_gK_ineq,
            Function eval_dynamics_equation){
            K_ = K;
            nx_ = nx;
            nu_ = nu;
            x_init_ = x_init;
            u_init_ = u_init;
            g_ineq_lb_ = g_ineq_lb;
            g_ineq_ub_ = g_ineq_ub;
            g_ineq_K_lb_ = g_ineq_K_lb;
            g_ineq_K_ub_ = g_ineq_K_ub;
            eval_objk_ = eval_objk;
            eval_objK_ = eval_objK;
            eval_gk_ = eval_gk;
            eval_g0_ = eval_g0;
            eval_gK_ = eval_gK;
            eval_gk_ineq_ = eval_gk_ineq;
            eval_gK_ineq_ = eval_gK_ineq;
            eval_dynamics_equation_ = eval_dynamics_equation;

            // Initialize derived functions
            MX xk = MX::sym("xk", nx_);
            MX uk = MX::sym("uk", nu_);
            MX xkp = MX::sym("xkp", nx_);
            MXVector ukxk = {uk, xk};
            MXVector ukxkxkp = {uk, xk, xkp};
            MX lam_dyn_k = MX::sym("lam_dyn_k", nx_);
            MX lam_eq_k = MX::sym("lam_eq_k", eval_gk_.sparsity_out(0).size1());
            MX lam_eq_0 = MX::sym("lam_eq_0", eval_g0_.sparsity_out(0).size1());
            MX lam_eq_K = MX::sym("lam_eq_K", eval_gK_.sparsity_out(0).size1());
            MX lam_ineq_k = MX::sym("lam_eq_k", eval_gk_ineq_.sparsity_out(0).size1());
            MX lam_ineq_K = MX::sym("lam_eq_K", eval_gK_ineq_.sparsity_out(0).size1());
            MX obj_scale = MX::sym("obj_scale", 1);

            grad_ = Function("grad", {uk, xk}, 
                {transpose(jacobian(eval_objk_(ukxk)[0], vertcat(uk, xk)))});
            grad_K_ = Function("gradK", {xk}, 
                {transpose(jacobian(eval_objK_(xk)[0], xk))});
            BAbt_ = Function("BAbt", {uk, xk, xkp}, 
                {transpose(horzcat(
                    jacobian(eval_dynamics_equation_(ukxkxkp)[0], uk),      // B
                    jacobian(eval_dynamics_equation_(ukxkxkp)[0], xk)       // A
                ))});
            Ggt_ = Function("Ggt", {uk, xk}, {transpose(jacobian(eval_gk_(ukxk)[0], vertcat(uk, xk)))});
            GgKt_ = Function("GgKt", {xk}, {transpose(jacobian(eval_gK_(xk)[0], xk))});
            Gg0t_ = Function("Gg0t", {uk, xk}, {transpose(jacobian(eval_g0_(ukxk)[0], vertcat(uk, xk)))});
            Ggt_ineq_ = Function("Ggt_ineq", {uk, xk}, {transpose(jacobian(eval_gk_ineq_(ukxk)[0], vertcat(uk, xk)))});
            GgKt_ineq_ = Function("GgKt_ineq", {xk}, {transpose(jacobian(eval_gK_ineq_(xk)[0], xk))});
            b_ = Function("b", {uk, xk, xkp}, {eval_dynamics_equation_(ukxkxkp)[0]});
            Jt_ = Function("Jt", {uk, xk, xkp}, 
                {transpose(
                    jacobian(eval_dynamics_equation_(ukxkxkp)[0], xkp)
                )});
            Jt_inv_ = Function("Jt_inv", {uk, xk, xkp},
                {inv(Jt_(ukxkxkp)[0])});
            FuFxt_ = Function("FuFxt", {uk, xk, xkp, lam_dyn_k}, 
                {jacobian(                                                                      // (nu + nx) x nx
                    transpose(                                                                  // (nu + nx) x 1
                        jacobian(mtimes(transpose(lam_dyn_k), b_(ukxkxkp)[0]), vertcat(uk, xk)) // 1 x (nu + nx)
                    ), xkp)
                });


            // construct lagrangian (containing uk, xk and potentially xkp)
            MX lagrangian_k = obj_scale*eval_objk_(ukxk)[0] + \
                mtimes(transpose(lam_dyn_k), b_(ukxkxkp)[0]) + \
                mtimes(transpose(lam_eq_k), eval_gk_(ukxk)[0]) + \
                mtimes(transpose(lam_ineq_k), eval_gk_ineq_(ukxk)[0]);
            MX lagrangian_0 = obj_scale*eval_objk_(ukxk)[0] + \
                mtimes(transpose(lam_dyn_k), b_(ukxkxkp)[0]) + \
                mtimes(transpose(lam_eq_0), eval_g0_(ukxk)[0]) + \
                mtimes(transpose(lam_ineq_k), eval_gk_ineq_(ukxk)[0]);
            MX lagrangian_K = obj_scale*eval_objK_(xk)[0] + \
                mtimes(transpose(lam_eq_K), eval_gK_(xk)[0]) + \
                mtimes(transpose(lam_ineq_K), eval_gK_ineq_(xk)[0]);
            lag_hess_k_ = Function("lag_hess_k", {uk, xk, xkp, lam_dyn_k, lam_eq_k, lam_ineq_k, obj_scale}, 
                {transpose(hessian(lagrangian_k, vertcat(uk, xk)))});
            lag_hess_0_ = Function("lag_hess_0", {uk, xk, xkp, lam_dyn_k, lam_eq_0, lam_ineq_k, obj_scale}, 
                {transpose(hessian(lagrangian_0, vertcat(uk, xk)))});
            lag_hess_K_ = Function("lag_hess_K", {xk, xkp, lam_dyn_k, lam_eq_K, lam_ineq_K, obj_scale}, 
                {transpose(hessian(lagrangian_K, xk))});

            dyn_hess_kp_ = Function("dyn_hess_kp", {uk, xk, xkp, lam_dyn_k}, 
                {transpose(hessian(mtimes(transpose(lam_dyn_k), b_(ukxkxkp)[0]), xkp))});

            // update sparsities
            BAbt_sp_ = BAbt_.sparsity_out(0);
            lag_hess_k_sp_ = lag_hess_k_.sparsity_out(0);
            lag_hess_0_sp_ = lag_hess_0_.sparsity_out(0);
            lag_hess_K_sp_ = lag_hess_K_.sparsity_out(0);
            dyn_hess_kp_sp_ = dyn_hess_kp_.sparsity_out(0);
            Ggt_sp_ = Ggt_.sparsity_out(0);
            GgKt_sp_ = GgKt_.sparsity_out(0);
            Gg0t_sp_ = Gg0t_.sparsity_out(0);
            Ggt_ineq_sp_ = Ggt_ineq_.sparsity_out(0);
            GgKt_ineq_sp_ = GgKt_ineq_.sparsity_out(0);
            Jt_sp_ = Jt_.sparsity_out(0);
            Jt_inv_sp_ = Jt_inv_.sparsity_out(0);
            FuFxt_sp_ = FuFxt_.sparsity_out(0);
        };

        virtual Index get_nx(const Index k) const { return nx_;}
        virtual Index get_nu(const Index k) const { return (k == K_-1) ? 0 : nu_;}
        virtual Index get_ng(const Index k) const
        {
            if (k == 0) {
                return eval_g0_.sparsity_out(0).size1();
            } else if (k == K_ - 1) {
                return eval_gK_.sparsity_out(0).size1();
            } else {
                return eval_gk_.sparsity_out(0).size1();
            }
        };
        virtual Index get_ng_ineq(const Index k) const {
            if (k == K_ - 1) { return eval_gK_ineq_.sparsity_out(0).size1();}
            return eval_gk_ineq_.sparsity_out(0).size1();
        };
        virtual Index get_horizon_length() const { return K_; };
        virtual Index eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k,
                                const Scalar *states_k, MAT *res, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            if (k == K_ - 1){
                throw std::runtime_error("Error in eval_BAbt: cannot evaluate BAbt at final stage");
            }
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            std::vector<double*> arg_out = {&scratch_[0]};
            BAbt_(arg_in, arg_out);

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < BAbt_sp_.size2(); j++){
                for (int i = 0; i < BAbt_sp_.size1(); i++){
                    if (i > res->m || j > res->n){
                        throw std::runtime_error("Error in eval_BAbt: trying to write outside of matrix bounds");
                    }
                    if (BAbt_.sparsity_out(0).has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        }
        virtual Index eval_RSQrqt(const Scalar *objective_scale, 
                                    const Scalar *inputs_km1,
                                    const Scalar *states_km1,
                                    const Scalar *inputs_k,
                                    const Scalar *states_k, 
                                    const Scalar *states_kp1,
                                    const Scalar *lam_dyn_k,
                                    const Scalar *lam_dyn_km1,
                                    const Scalar *lam_eq_k, 
                                    const Scalar *lam_eq_ineq_k, 
                                    MAT *res, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);

            // contribution of obj(uk, xk), equality and inequality constraints
            // and f(uk, xk, xkp) = 0
            Function lag_hess = (k == 0) ? lag_hess_0_ : (k == K_ - 1) ? lag_hess_K_ : lag_hess_k_;
            Sparsity lag_hess_sp = (k == 0) ? lag_hess_0_sp_ : (k == K_ - 1) ? lag_hess_K_sp_ : lag_hess_k_sp_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k, states_kp1, lam_dyn_k, lam_eq_k, lam_eq_ineq_k, objective_scale}:
                std::vector<const double*>{inputs_k, states_k, states_kp1, lam_dyn_k, lam_eq_k, lam_eq_ineq_k, objective_scale};
            
            std::vector<double*> arg_out = {&scratch_[0]};
            lag_hess(arg_in, arg_out);

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < lag_hess_sp.size2(); j++){
                for (int i = 0; i < lag_hess_sp.size1(); i++){
                    if (i > res->m || j > res->n){
                        throw std::runtime_error("Error in eval_RSQrqt: trying to write outside of matrix bounds");
                    }
                    if (lag_hess_sp.has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }

            // contribution of f(uk-1, xk-1, xk) = 0
            if (k > 0){
                std::vector<const double*> arg_in_dyn = {inputs_km1, states_km1, states_k, lam_dyn_km1};
                std::vector<double*> arg_out_dyn = {&scratch_[0]};
                dyn_hess_kp_(arg_in_dyn, arg_out_dyn);

                // store nonzeros in the matrix
                int scratch_ptr_dyn = 0;
                for (int j = 0; j < dyn_hess_kp_sp_.size2(); j++){
                    for (int i = 0; i < dyn_hess_kp_sp_.size1(); i++){
                        if (i > get_nu(k) + res->m || j > get_nu(k) + res->n){
                            throw std::runtime_error("Error in eval_RSQrqt (2): trying to write outside of matrix bounds");
                        }
                        if (dyn_hess_kp_sp_.has_nz(i, j)) {
                            blasfeo_matel_wrap(res, get_nu(k) + i, get_nu(k) + j) += scratch_[scratch_ptr_dyn];
                            scratch_ptr_dyn++;
                        } else {
                            blasfeo_matel_wrap(res, get_nu(k) + i, get_nu(k) + j) += 0.0;
                        }
                    }
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };
        virtual Index eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res,
                                const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);            
            Function G = (k == 0) ? Gg0t_ : (k == K_ - 1) ? GgKt_ : Ggt_;
            Sparsity G_sp = (k == 0) ? Gg0t_sp_ : (k == K_ - 1) ? GgKt_sp_ : Ggt_sp_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k} : 
                std::vector<const double*>{inputs_k, states_k};
            
            std::vector<double*> arg_out = {&scratch_[0]};
            G(arg_in, arg_out);
           
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < G_sp.size2(); j++){
                for (int i = 0; i < G_sp.size1(); i++){
                    if (i > res->m || j > res->n){
                        throw std::runtime_error("Error in eval_Ggt: trying to write outside of matrix bounds");
                    }
                    if (G_sp.has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }
            
            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        }
        virtual Index eval_Ggt_ineq(const Scalar *inputs_k, const Scalar *states_k, MAT *res,
                                    const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            Function Ggt_ineq = (k == K_ - 1) ? GgKt_ineq_ : Ggt_ineq_;
            Sparsity Ggt_ineq_sp = (k == K_ - 1) ? GgKt_ineq_sp_ : Ggt_ineq_sp_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k} : 
                std::vector<const double*>{inputs_k, states_k};
            
            std::vector<double*> arg_out = {&scratch_[0]};
            Ggt_ineq(arg_in, arg_out);
            
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < Ggt_ineq_sp.size2(); j++){
                for (int i = 0; i < Ggt_ineq_sp.size1(); i++){
                    if (i > res->m || j > res->n){
                        throw std::runtime_error("Error in eval_Ggt_ineq: trying to write outside of matrix bounds");
                    }
                    if (Ggt_ineq_sp.has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };
        virtual Index eval_b(const Scalar *states_kp1, const Scalar *inputs_k,
                                const Scalar *states_k, Scalar *res, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            std::vector<double*> arg_out = {res};
            eval_dynamics_equation_(arg_in, arg_out);
            
            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                std::cout << "b: ";
                for (Index i = 0; i < get_nx(k); ++i) {
                    std::cout << res[i] << " ";
                }
                std::cout<<std::endl;
            }
            return 0;
        }

        virtual Index eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
                                const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k} : 
                std::vector<const double*>{inputs_k, states_k};

            std::vector<double*> arg_out = {res};
            if (k == 0){
                eval_g0_(arg_in, arg_out);
            } else if (k == K_ - 1){
                eval_gK_(arg_in, arg_out);
            } else {
                eval_gk_(arg_in, arg_out);
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                std::cout << "g: ";
                for (Index i = 0; i < get_ng(k); ++i) {
                    std::cout << res[i] << " ";
                }
                std::cout<<std::endl;
            }
            return 0;
        };
        virtual Index eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
                                    const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            Function g_ineq = (k == K_ - 1) ? eval_gK_ineq_ : eval_gk_ineq_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k} : 
                std::vector<const double*>{inputs_k, states_k};

            std::vector<double*> arg_out = {res};
            g_ineq(arg_in, arg_out);
            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                std::cout << "g_ineq: ";
                for (Index i = 0; i < g_ineq.sparsity_out(0).size1(); ++i) {
                    std::cout << res[i] << " ";
                }
                std::cout<<std::endl;
            }
            return 0;
        };
        virtual Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k,
                                const Scalar *states_k, Scalar *res, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            Function grad = (k == K_ - 1) ? grad_K_ : grad_;
            std::vector<const double*> arg_in = (k == K_ - 1) ?
                std::vector<const double*>{states_k} :
                std::vector<const double*>{inputs_k, states_k};

            std::vector<double*> arg_out = {res};
            grad(arg_in, arg_out);

            for (Index i = 0; i < get_nu(k) + get_nx(k); ++i) {
                res[i] *= (*objective_scale);
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                std::cout << "grad: ";
                for (Index i = 0; i < get_nu(k) + get_nx(k); ++i) {
                    std::cout << res[i] << " ";
                }
                std::cout<<std::endl;
            }
            return 0;
        }
        virtual Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k,
                                const Scalar *states_k, Scalar *res, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            Function obj = (k == K_ - 1) ? eval_objK_ : eval_objk_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k} :
                std::vector<const double*>{inputs_k, states_k};
            
            std::vector<double*> arg_out = {res};
            obj(arg_in, arg_out);
            res[0] *= (*objective_scale);        
            return 0;
        }
        virtual Index get_bounds(Scalar *lower, Scalar *upper, const Index k) const
        {
            if (k == K_ - 1){ 
                for (Index i = 0; i < eval_gK_ineq_.sparsity_out(0).size1(); ++i) {
                    lower[i] = g_ineq_K_lb_[i];
                    upper[i] = g_ineq_K_ub_[i];
                }    
                return 0;
            } else {
                for (Index i = 0; i < eval_gk_ineq_.sparsity_out(0).size1(); ++i) {
                    lower[i] = g_ineq_lb_[i];
                    upper[i] = g_ineq_ub_[i];
                }
                return 0;
            }
            return 0;
        }

        virtual Index get_initial_xk(Scalar *xk, const Index k) const
        {
            for (Index i = 0; i < get_nx(k); ++i) {
                xk[i] = x_init_[k][i];
            }
            return 0;
        };
        virtual Index get_initial_uk(Scalar *uk, const Index k) const
        {
            for (Index i = 0; i < get_nu(k); ++i) {
                uk[i] = u_init_[k][i];
            }
            return 0;
        };
        virtual ~ImplicitTestProblem() = default;

        virtual Index eval_Jt(const Scalar *states_kp1, const Scalar *inputs_k,
                                const Scalar *states_k, MAT *res, const Index k){
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            if (k == K_ - 1){
                throw std::runtime_error("[ocp_interface_generator]: Jt is not defined for the last stage.");
            }
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            std::vector<double*> arg_out = {&scratch_[0]};
            Jt_(arg_in, arg_out);
            
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < Jt_sp_.size2(); j++){
                for (int i = 0; i < Jt_sp_.size1(); i++){
                    if (i > res->m || j > res->n){
                        throw std::runtime_error("Error in eval_Jt: trying to write outside of matrix bounds");
                    }
                    if (Jt_.sparsity_out(0).has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };

        virtual Index eval_Jt_inv(const Scalar *states_kp1, const Scalar *inputs_k,
                                    const Scalar *states_k, MAT *res, const Index k){
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            if (k == K_ - 1){
                throw std::runtime_error("[ocp_interface_generator]: Jt_inv is not defined for the last stage.");
            }
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            std::vector<double*> arg_out = {&scratch_[0]};
            Jt_inv_(arg_in, arg_out);
            
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < Jt_inv_sp_.size2(); j++){
                for (int i = 0; i < Jt_inv_sp_.size1(); i++){
                    if (i > res->m || j > res->n){
                        throw std::runtime_error("Error in eval_Jt_inv: trying to write outside of matrix bounds");
                    }
                    if (Jt_inv_.sparsity_out(0).has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };

        virtual Index eval_FuFxt(const Scalar *inputs_k, const Scalar *states_k, 
                                 const Scalar *states_kp1, 
                                 const Scalar *lam_dyn_k,MAT *res, const Index k){
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            if (k == K_ - 1){
                throw std::runtime_error("[ocp_interface_generator]: FuFxt is not defined for the last stage.");
            }
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1, lam_dyn_k};
            std::vector<double*> arg_out = {&scratch_[0]};
            FuFxt_(arg_in, arg_out);
            

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < FuFxt_sp_.size2(); j++){
                for (int i = 0; i < FuFxt_sp_.size1(); i++){
                    if (i > res->m || j > res->n){
                        throw std::runtime_error("Error in eval_FuFxt: trying to write outside of matrix bounds");
                    }
                    if (FuFxt_sp_.has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };
    

    private:
        bool DEBUG_PRINT = false;

        // user-provided info
        Index K_;
        Index nx_;
        Index nu_;

        std::vector<std::vector<double>> x_init_;
        std::vector<std::vector<double>> u_init_;
        std::vector<double> g_ineq_lb_;
        std::vector<double> g_ineq_ub_;
        std::vector<double> g_ineq_K_lb_;
        std::vector<double> g_ineq_K_ub_;

        Function eval_objk_;
        Function eval_objK_;
        Function eval_gk_;
        Function eval_g0_;
        Function eval_gK_;
        Function eval_gk_ineq_;
        Function eval_gK_ineq_;
        Function eval_dynamics_equation_;

        // deduced info
        Function grad_;
        Function grad_K_;
        Function BAbt_;
        Function lag_hess_k_;
        Function lag_hess_0_;
        Function lag_hess_K_;
        Function dyn_hess_kp_;
        Function Ggt_;
        Function GgKt_;
        Function Gg0t_;
        Function Ggt_ineq_;
        Function GgKt_ineq_;
        Function b_;
        Function Jt_;
        Function Jt_inv_;
        Function FuFxt_;

        Sparsity BAbt_sp_;
        Sparsity lag_hess_k_sp_;
        Sparsity lag_hess_0_sp_;
        Sparsity lag_hess_K_sp_;
        Sparsity dyn_hess_kp_sp_;
        Sparsity Ggt_sp_;
        Sparsity GgKt_sp_;
        Sparsity Gg0t_sp_;
        Sparsity Ggt_ineq_sp_;
        Sparsity GgKt_ineq_sp_;
        Sparsity Jt_sp_;
        Sparsity Jt_inv_sp_;
        Sparsity FuFxt_sp_;

        // scratch space
        std::vector<double> scratch_ = std::vector<double>(1000, 0.0); // Adjust size as needed
        std::vector<double> scratch2_ = std::vector<double>(1000, 0.0); // Adjust size as needed
};  // IMPLICIT OCP TEST PROBLEM

#endif //FATROP_IMPLICIT_TEST_PROBLEM_HPP