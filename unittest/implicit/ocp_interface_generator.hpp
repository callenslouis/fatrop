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

            grad_ = Function("grad", {uk, xk}, 
                {vertcat(transpose(jacobian(eval_objk_(ukxk)[0], uk)),
                        transpose(jacobian(eval_objk_(ukxk)[0], xk)))});
            BAbt_ = Function("BAbt", {uk, xk, xkp}, 
                {transpose(horzcat(
                    jacobian(eval_dynamics_equation_(ukxkxkp)[0], uk),      // B
                    horzcat(
                        jacobian(eval_dynamics_equation_(ukxkxkp)[0], xk),  // A
                        eval_dynamics_equation_(ukxkxkp)[0]                 // b
                    )
                ))});
            RSQrqt_ = Function("RSQrqt", {xk, uk}, 
                {transpose(
                    horzcat(
                        hessian(eval_objk_(ukxk)[0], vertcat(uk, xk)),
                        grad_(ukxk)[0]
                    )
                )});
            Ggt_ = eval_gk_.n_out() > 0 ? 
                Function("Ggt", {uk, xk}, {jacobian(eval_gk_(ukxk)[0], vertcat(uk, xk))})
                : Function("Ggt", {uk, xk}, {MX::zeros(0, nu_ + nx_)});
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

            // update sparsities
            BAbt_sp_ = BAbt_.sparsity_out(0);
            RSQrqt_sp_ = RSQrqt_.sparsity_out(0);
            Ggt_sp_ = Ggt_.sparsity_out(0);
            Ggt_ineq_sp_ = Ggt_ineq_.sparsity_out(0);
            Jt_sp_ = Jt_.sparsity_out(0);
            Jt_inv_sp_ = Jt_inv_.sparsity_out(0);
            FuFxt_sp_ = FuFxt_.sparsity_out(0);
        };

        virtual Index get_nx(const Index k) const { return nx_;}
        virtual Index get_nu(const Index k) const { return nu_;}
        virtual Index get_ng(const Index k) const
        {
            return eval_gk_.n_out() > 0 ? eval_gk_.sparsity_out(0).size1() : 0;
        };
        virtual Index get_ng_ineq(const Index k) const {
            std::cout << "ng_ineq: " << eval_gk_ineq_.sparsity_out(0).size1() << std::endl;
            return eval_gk_ineq_.sparsity_out(0).size1();
        };
        virtual Index get_horizon_length() const { return K_; };
        virtual Index eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k,
                                const Scalar *states_k, MAT *res, const Index k)
        {
            std::cout << __func__ << std::endl;
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
                std::cout << __func__ << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        }
        virtual Index eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k,
                                    const Scalar *states_k, const Scalar *lam_dyn_k,
                                    const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k, MAT *res,
                                    const Index k)
        {
            std::vector<const double*> arg_in = {states_k, inputs_k};
            std::vector<double*> arg_out = {&scratch_[0]};
            RSQrqt_(arg_in, arg_out);

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < RSQrqt_sp_.size2(); j++){
                for (int i = 0; i < RSQrqt_sp_.size1(); i++){
                    if (RSQrqt_.sparsity_out(0).has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };
        virtual Index eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res,
                                const Index k)
        {
            std::cout << __func__ << std::endl;
            if (k == 0 || k == K_ - 1 || Ggt_.n_out() == 0){
                return 0;
            }

            std::vector<const double*> arg_in = {inputs_k, states_k};
            std::vector<double*> arg_out = {&scratch_[0]};
            Ggt_(arg_in, arg_out);
           
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < Ggt_sp_.size2(); j++){
                for (int i = 0; i < Ggt_sp_.size1(); i++){
                    if (Ggt_.sparsity_out(0).has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }
            
            if (DEBUG_PRINT){
                std::cout << __func__ << std::endl;
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
                std::cout << __func__ << std::endl;
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
                std::cout << __func__ << std::endl;
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
                std::cout << __func__ << std::endl;
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
                std::cout << __func__ << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };
    

    private:
        bool DEBUG_PRINT = true;

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
        Function RSQrqt_;
        Function Ggt_;
        Function Ggt_ineq_;
        Function b_;
        Function Jt_;
        Function Jt_inv_;
        Function FuFxt_;

        Sparsity BAbt_sp_;
        Sparsity RSQrqt_sp_;
        Sparsity Ggt_sp_;
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

            std::vector<double> start = std::vector<double>(nx, 0.0);
            std::vector<double> end = std::vector<double>(nx, 1.0);

            MX xk = MX::sym("xk", nx);
            MX uk = MX::sym("uk", nu);
            MX xkp = MX::sym("xkp", nx);

            Function eval_objk = Function("eval_objk", {uk, xk}, {sumsqr(uk)});
            Function eval_gk = Function("eval_gk", {uk, xk}, {});
            Function eval_g0 = Function("eval_g0", {uk, xk}, {xk - start});
            Function eval_gK = Function("eval_gK", {uk, xk}, {xk - end});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk, xk}, {uk});
            
            MX temp = MX(nx, 1);
            for (int i = 0; i < control_level; ++i) {
                for (int j = 0; j < n; ++j) {
                    MX der = (i < control_level - 1) ? xkp((i+1)*n + j) : uk(j);
                    temp(i*n + j) = xk(i*n + j) + dt*der - xkp(i*n + j);
                }
            }
            std::cout << "temp: " << temp << std::endl;
            MX B = jacobian(temp, uk);
            MX A = jacobian(temp, xk);
            MX Jt = jacobian(temp, xkp);
            Sparsity sp_B = B.sparsity();
            Sparsity sp_A = A.sparsity();
            Sparsity sp_Jt = Jt.sparsity();
            std::cout << "B sparsity: " << std::endl;
            for (int i = 0; i < sp_B.size1(); ++i) {
                for (int j = 0; j < sp_B.size2(); ++j) {
                    if (sp_B.has_nz(i, j)) {
                        std::cout << "X ";
                    } else {
                        std::cout << ". ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << "A sparsity: " << std::endl;
            for (int i = 0; i < sp_A.size1(); ++i) {
                for (int j = 0; j < sp_A.size2(); ++j) {
                    if (sp_A.has_nz(i, j)) {
                        std::cout << "X ";
                    } else {
                        std::cout << ". ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << "Jt sparsity: " << std::endl;
            for (int i = 0; i < sp_Jt.size1(); ++i) {
                for (int j = 0; j < sp_Jt.size2(); ++j) {
                    if (sp_Jt.has_nz(i, j)) {
                        std::cout << "X ";
                    } else {
                        std::cout << ". ";
                    }
                }
                std::cout << std::endl;
            }

            Function eval_dynamics_equation = Function("eval_dynamics_equation", {uk, xk, xkp}, {temp});

            return TestProblem(
                100, nx, nu, 
                std::vector<std::vector<double>>(100, start), 
                std::vector<std::vector<double>>(100, std::vector<double>(nu, 0.0)), 
                std::vector<double>(nu, -1.0), 
                std::vector<double>(nu, 1.0), 
                eval_objk, eval_gk, eval_g0, eval_gK, eval_gk_ineq, eval_dynamics_equation
            );
        }

    private:
};