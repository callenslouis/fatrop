#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

class TestProblem : public ImplicitOcpAbstract{
    public:
        TestProblem(Index K, const Index nx, const Index nu, 
            std::vector<std::vector<double>> x_init, 
            std::vector<std::vector<double>> u_init, std::vector<double> g_ineq_lb, std::vector<double> g_ineq_ub, Function eval_objk, Function eval_gk, 
            Function eval_g0, Function eval_gK, Function eval_gk_ineq, 
            Function eval_dynamics_equation){
            K_ = K;
            nx_ = nx;
            nu_ = nu;
            x_init_ = x_init;
            u_init_ = u_init;
            g_ineq_lb_ = g_ineq_lb;
            g_ineq_ub_ = g_ineq_ub;
            eval_objk_ = eval_objk;
            eval_gk_ = eval_gk;
            eval_g0_ = eval_g0;
            eval_gK_ = eval_gK;
            eval_gk_ineq_ = eval_gk_ineq;
            eval_dynamics_equation_ = eval_dynamics_equation;

            // Initialize derived functions
            MX xk = MX::sym("xk", nx_);
            MX uk = MX::sym("uk", nu_);
            MX xkp = MX::sym("xkp", nx_);
            MXVector ukxk = {uk, xk};
            MXVector ukxkxkp = {uk, xk, xkp};
            MX lam_dyn_k = MX::sym("lam_dyn_k", nx_);
            MX lam_eq_k = MX::sym("lam_eq_k", eval_gk_.n_out() > 0 ? eval_gk_.sparsity_out(0).size1() : 0);
            MX lam_eq_0 = MX::sym("lam_eq_0", eval_g0_.n_out() > 0 ? eval_g0_.sparsity_out(0).size1() : 0);
            MX lam_eq_K = MX::sym("lam_eq_K", eval_gK_.n_out() > 0 ? eval_gK_.sparsity_out(0).size1() : 0);
            MX lam_ineq_k = MX::sym("lam_eq_k", eval_gk_ineq_.n_out() > 0 ? eval_gk_ineq_.sparsity_out(0).size1() : 0);
            MX obj_scale = MX::sym("obj_scale", 1);

            grad_ = Function("grad", {uk, xk}, 
                {transpose(jacobian(eval_objk_(ukxk)[0], vertcat(uk, xk)))});
            BAbt_ = Function("BAbt", {uk, xk, xkp}, 
                {transpose(horzcat(
                    jacobian(eval_dynamics_equation_(ukxkxkp)[0], uk),      // B
                    jacobian(eval_dynamics_equation_(ukxkxkp)[0], xk)       // A
                ))});
            Ggt_ = eval_gk_.n_out() > 0 ? 
                Function("Ggt", {uk, xk}, {transpose(jacobian(eval_gk_(ukxk)[0], vertcat(uk, xk)))})
                : Function("Ggt", {uk, xk}, {MX::zeros(nu_ + nx_, 0)});
            GgKt_ = eval_gK_.n_out() > 0 ? 
                Function("GgKt", {uk, xk}, {transpose(jacobian(eval_gK_(ukxk)[0], vertcat(uk, xk)))})
                : Function("GgKt", {uk, xk}, {MX::zeros(nu_ + nx_, 0)});
            Gg0t_ = eval_g0_.n_out() > 0 ? 
                Function("Gg0t", {uk, xk}, {transpose(jacobian(eval_g0_(ukxk)[0], vertcat(uk, xk)))})
                : Function("Gg0t", {uk, xk}, {MX::zeros(nu_ + nx_, 0)});
            Ggt_ineq_ = Function("Ggt_ineq", {uk, xk}, 
                {jacobian(eval_gk_ineq_(ukxk)[0], vertcat(uk, xk))});
            b_ = Function("b", {uk, xk, xkp}, {eval_dynamics_equation_(ukxkxkp)[0]});
            Jt_ = Function("Jt", {uk, xk, xkp}, 
                {transpose(
                    jacobian(eval_dynamics_equation_(ukxkxkp)[0], xkp)
                )});
            Jt_inv_ = Function("Jt_inv", {uk, xk, xkp},
                {inv(Jt_(ukxkxkp)[0])});
            FuFxt_ = Function("FuFxt", {uk, xk, xkp},
                {transpose(
                    horzcat(
                        jacobian(jacobian(eval_dynamics_equation_(ukxkxkp)[0], xkp), uk),
                        jacobian(jacobian(eval_dynamics_equation_(ukxkxkp)[0], xkp), xk)
                    )
                )});


            // construct lagrangian (containing uk, xk and potentially xkp)
            MX lagrangian_k = obj_scale*eval_objk_(ukxk)[0] + \
                mtimes(transpose(lam_dyn_k), b_(ukxkxkp)[0]) + \
                mtimes(transpose(lam_eq_k), eval_gk_(ukxk)[0]) + \
                mtimes(transpose(lam_ineq_k), eval_gk_ineq_(ukxk)[0]);
            MX lagrangian_0 = obj_scale*eval_objk_(ukxk)[0] + \
                mtimes(transpose(lam_dyn_k), b_(ukxkxkp)[0]) + \
                mtimes(transpose(lam_eq_0), eval_g0_(ukxk)[0]) + \
                mtimes(transpose(lam_ineq_k), eval_gk_ineq_(ukxk)[0]);
            MX lagrangian_K = obj_scale*eval_objk_(ukxk)[0] + \
                mtimes(transpose(lam_eq_K), eval_gK_(ukxk)[0]) + \
                mtimes(transpose(lam_ineq_k), eval_gk_ineq_(ukxk)[0]);
            lag_hess_k_ = Function("lag_hess_k", {xk, uk, xkp, lam_dyn_k, lam_eq_k, lam_ineq_k, obj_scale}, 
                {transpose(hessian(lagrangian_k, vertcat(uk, xk))),     // RSQ 
                 transpose(hessian(lagrangian_k, xkp))});               // RSQ[k+1]
            lag_hess_0_ = Function("lag_hess_0", {xk, uk, xkp, lam_dyn_k, lam_eq_0, lam_ineq_k, obj_scale}, 
                {transpose(hessian(lagrangian_0, vertcat(uk, xk))),     // RSQ
                 transpose(hessian(lagrangian_0, xkp))});               // RSQ[k+1]
            lag_hess_K_ = Function("lag_hess_K", {xk, uk, xkp, lam_dyn_k, lam_eq_K, lam_ineq_k, obj_scale}, 
                {transpose(hessian(lagrangian_K, vertcat(uk, xk))),
                 MX::zeros(0,0)});

            // update sparsities
            BAbt_sp_ = BAbt_.sparsity_out(0);
            lag_hess_k_sp_ = lag_hess_k_.sparsity_out(0);
            lag_hess_0_sp_ = lag_hess_0_.sparsity_out(0);
            lag_hess_K_sp_ = lag_hess_K_.sparsity_out(0);
            Ggt_sp_ = Ggt_.sparsity_out(0);
            GgKt_sp_ = GgKt_.sparsity_out(0);
            Gg0t_sp_ = Gg0t_.sparsity_out(0);
            Ggt_ineq_sp_ = Ggt_ineq_.sparsity_out(0);
            Jt_sp_ = Jt_.sparsity_out(0);
            Jt_inv_sp_ = Jt_inv_.sparsity_out(0);
            FuFxt_sp_ = FuFxt_.sparsity_out(0);
        };

        virtual Index get_nx(const Index k) const { return nx_;}
        virtual Index get_nu(const Index k) const { return nu_;}
        virtual Index get_ng(const Index k) const
        {
            if (k == 0) {
                return eval_g0_.n_out() > 0 ? eval_g0_.sparsity_out(0).size1() : 0;
            } else if (k == K_ - 1) {
                return eval_gK_.n_out() > 0 ? eval_gK_.sparsity_out(0).size1() : 0;
            } else {
                return eval_gk_.n_out() > 0 ? eval_gk_.sparsity_out(0).size1() : 0;
            }
        };
        virtual Index get_ng_ineq(const Index k) const {
            if (k == K_ - 1) { return 0;}
            return eval_gk_ineq_.sparsity_out(0).size1();
        };
        virtual Index get_horizon_length() const { return K_; };
        virtual Index eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k,
                                const Scalar *states_k, MAT *res, const Index k)
        {
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            std::vector<double*> arg_out = {&scratch_[0]};
            BAbt_(arg_in, arg_out);

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < BAbt_sp_.size2(); j++){
                for (int i = 0; i < BAbt_sp_.size1(); i++){
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
        virtual Index eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k,
                                    const Scalar *states_k, const Scalar *states_kp1,
                                    const Scalar *lam_dyn_k,
                                    const Scalar *lam_eq_k, 
                                    const Scalar *lam_eq_ineq_k, MAT *res,
                                    MAT *res_next, const Index k)
        {
            Function lag_hess = (k == 0) ? lag_hess_0_ : (k == K_ - 1) ? lag_hess_K_ : lag_hess_k_;
            Sparsity lag_hess_sp = (k == 0) ? lag_hess_0_sp_ : (k == K_ - 1) ? lag_hess_K_sp_ : lag_hess_k_sp_;
            
            std::vector<const double*> arg_in = {states_k, inputs_k, states_kp1, lam_dyn_k, lam_eq_k, lam_eq_ineq_k, objective_scale};
            std::vector<double*> arg_out = {&scratch_[0], &scratch2_[0]};
            lag_hess(arg_in, arg_out);

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < lag_hess_sp.size2(); j++){
                for (int i = 0; i < lag_hess_sp.size1(); i++){
                    if (lag_hess_sp.has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) += scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) += 0.0;
                    }
                }
            }

            // store RSQ[k+1] in res_next
            if (k < K_ - 1){
                Sparsity lag_hess_sp_next = lag_hess.sparsity_out(1);
                scratch_ptr = 0;
                for (int j = 0; j < lag_hess_sp_next.size2(); j++){
                    for (int i = 0; i < lag_hess_sp_next.size1(); i++){
                        if (lag_hess_sp_next.has_nz(i, j)) {
                            blasfeo_matel_wrap(res_next, get_nu(k+1) + i, get_nu(k+1) + j) += scratch2_[scratch_ptr];
                            scratch_ptr++;
                        } else {
                            blasfeo_matel_wrap(res_next, i, j) += 0.0;
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
            if (k == 0 && eval_g0_.n_out() == 0){
                return 0; // No Gg0t for the first step
            } else if (k == K_ - 1 && eval_gK_.n_out() == 0){
                return 0; // No GgKt for the last step
            } else if (k > 0 && k < K_ - 1 && eval_gk_.n_out() == 0){
                return 0; // No Ggt for intermediate steps
            }
            
            Function G = (k == 0) ? Gg0t_ : (k == K_ - 1) ? GgKt_ : Ggt_;
            Sparsity G_sp = (k == 0) ? Gg0t_sp_ : (k == K_ - 1) ? GgKt_sp_ : Ggt_sp_;

            std::vector<const double*> arg_in = {inputs_k, states_k};
            std::vector<double*> arg_out = {&scratch_[0]};
            G(arg_in, arg_out);
           
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < G_sp.size2(); j++){
                for (int i = 0; i < G_sp.size1(); i++){
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
            if (k == K_ - 1 || Ggt_ineq_.n_out() == 0){
                return 0;
            }
            std::vector<const double*> arg_in = {inputs_k, states_k};
            std::vector<double*> arg_out = {&scratch_[0]};
            Ggt_ineq_(arg_in, arg_out);
            
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < Ggt_ineq_sp_.size2(); j++){
                for (int i = 0; i < Ggt_ineq_sp_.size1(); i++){
                    if (Ggt_ineq_.sparsity_out(0).has_nz(i, j)) {
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
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            std::vector<double*> arg_out = {res};
            eval_dynamics_equation_(arg_in, arg_out);
            return 0;
        }

        virtual Index eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
                                const Index k)
        {
            std::vector<const double*> arg_in = {inputs_k, states_k};
            std::vector<double*> arg_out = {res};
            if (k == 0){
                eval_g0_(arg_in, arg_out);
            } else if (k == K_ - 1){
                eval_gK_(arg_in, arg_out);
            } else {
                eval_gk_(arg_in, arg_out);
            }
            return 0;
        };
        virtual Index eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
                                    const Index k)
        {
            if (k == 0 || k == K_ - 1) {
                return 0; // No inequality constraints for the first and last step
            }
            std::vector<const double*> arg_in = {inputs_k, states_k};
            std::vector<double*> arg_out = {res};
            return 0;
        };
        virtual Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k,
                                const Scalar *states_k, Scalar *res, const Index k)
        {
            std::vector<const double*> arg_in = {inputs_k, states_k};
            std::vector<double*> arg_out = {res};
            grad_(arg_in, arg_out);
            for (auto s : scratch_) { s *= (*objective_scale);}
            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                std::cout << "grad: ";
                for (Index i = 0; i < nu_ + nx_; ++i) {
                    std::cout << res[i] << " ";
                }
                std::cout<<std::endl;
            }
            return 0;
        }
        virtual Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k,
                                const Scalar *states_k, Scalar *res, const Index k)
        {
            if (k == K_ - 1)
            {
                res[0] = 0.0; // No objective at the last step
                return 0;
            }
            std::vector<const double*> arg_in = {inputs_k, states_k};
            std::vector<double*> arg_out = {res};
            eval_objk_(arg_in, arg_out);
            res[0] *= (*objective_scale);        
            return 0;
        }
        virtual Index get_bounds(Scalar *lower, Scalar *upper, const Index k) const
        {
            if (k == K_ - 1){ return 0;}
            for (Index i = 0; i < g_ineq_lb_.size(); ++i) {
                lower[i] = g_ineq_lb_[i];
                upper[i] = g_ineq_ub_[i];
            }
            return 0;
        }

        virtual Index get_initial_xk(Scalar *xk, const Index k) const
        {
            for (Index i = 0; i < nx_; ++i) {
                xk[i] = x_init_[k][i];
            }
            return 0;
        };
        virtual Index get_initial_uk(Scalar *uk, const Index k) const
        {
            for (Index i = 0; i < nu_; ++i) {
                uk[i] = u_init_[k][i];
            }
            return 0;
        };
        virtual ~TestProblem() = default;

        virtual Index eval_Jt(const Scalar *states_kp1, const Scalar *inputs_k,
                                const Scalar *states_k, MAT *res, const Index k){
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            std::vector<double*> arg_out = {&scratch_[0]};
            Jt_(arg_in, arg_out);
            
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < Jt_sp_.size2(); j++){
                for (int i = 0; i < Jt_sp_.size1(); i++){
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
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            std::vector<double*> arg_out = {&scratch_[0]};
            Jt_inv_(arg_in, arg_out);
            
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < Jt_inv_sp_.size2(); j++){
                for (int i = 0; i < Jt_inv_sp_.size1(); i++){
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
                                    const Scalar *states_kp1, MAT *res, const Index k){
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            std::vector<double*> arg_out = {&scratch_[0]};
            FuFxt_(arg_in, arg_out);
            
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < FuFxt_sp_.size2(); j++){
                for (int i = 0; i < FuFxt_sp_.size1(); i++){
                    if (FuFxt_.sparsity_out(0).has_nz(i, j)) {
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

        Function eval_objk_;
        Function eval_gk_;
        Function eval_g0_;
        Function eval_gK_;
        Function eval_gk_ineq_;
        Function eval_dynamics_equation_;

        // deduced info
        Function grad_;
        Function BAbt_;
        Function lag_hess_k_;
        Function lag_hess_0_;
        Function lag_hess_K_;
        Function Ggt_;
        Function GgKt_;
        Function Gg0t_;
        Function Ggt_ineq_;
        Function b_;
        Function Jt_;
        Function Jt_inv_;
        Function FuFxt_;

        Sparsity BAbt_sp_;
        Sparsity lag_hess_k_sp_;
        Sparsity lag_hess_0_sp_;
        Sparsity lag_hess_K_sp_;
        Sparsity Ggt_sp_;
        Sparsity GgKt_sp_;
        Sparsity Gg0t_sp_;
        Sparsity Ggt_ineq_sp_;
        Sparsity Jt_sp_;
        Sparsity Jt_inv_sp_;
        Sparsity FuFxt_sp_;

        // scratch space
        std::vector<double> scratch_ = std::vector<double>(1000, 0.0); // Adjust size as needed
        std::vector<double> scratch2_ = std::vector<double>(1000, 0.0); // Adjust size as needed
};


class OcpInterfaceGenerator{
    public:
        // Constructor
        OcpInterfaceGenerator(){};

        // prepare holonomic ocp in n dimensions
        // velocity control: level 1
        // acceleration control: level 2
        // ...
        TestProblem PrepareHolonomic(int n, int control_level){
            int nx = control_level*n;
            int nu = n;
            double dt = 0.05;

            std::vector<double> start = std::vector<double>(nx, 1.0);
            std::vector<double> end = std::vector<double>(nx, 2.0);

            MX xk = MX::sym("xk", nx);
            MX uk = MX::sym("uk", nu);
            MX xkp = MX::sym("xkp", nx);

            Function eval_objk = Function("eval_objk", {uk, xk}, {sumsqr(uk)});
            Function eval_gk = Function("eval_gk", {uk, xk}, {MX::zeros(0,1)});
            Function eval_g0 = Function("eval_g0", {uk, xk}, {xk - start});
            // Function eval_gK = Function("eval_gK", {uk, xk}, {});
            // Function eval_gK = Function("eval_gK", {uk, xk}, {xk - end});
            Function eval_gK = Function("eval_gK", {uk, xk}, {xk - start});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk, xk}, {uk});
            
            // MX temp = MX(nx, 1);
            // for (int i = 0; i < control_level; ++i) {
            //     for (int j = 0; j < n; ++j) {
            //         MX der = (i < control_level - 1) ? xkp((i+1)*n + j) : uk(j);
            //         temp(i*n + j) = xk(i*n + j) + dt*der - xkp(i*n + j);
            //     }
            // }
            // Function eval_dynamics_equation = Function("eval_dynamics_equation", {uk, xk, xkp}, {temp});
            Function eval_dynamics_equation = Function("eval_dynamics_equation", {uk, xk, xkp}, {uk - xkp});

            return TestProblem(
                3, nx, nu, 
                std::vector<std::vector<double>>(100, start), 
                std::vector<std::vector<double>>(100, std::vector<double>(nu, 0.0)), 
                std::vector<double>(nu, -10.0), 
                std::vector<double>(nu, 10.0), 
                eval_objk, eval_gk, eval_g0, eval_gK, eval_gk_ineq, eval_dynamics_equation
            );
        }

    private:
};