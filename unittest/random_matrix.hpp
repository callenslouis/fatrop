#ifndef __fatrop_unittest_random_matrix_hpp__
#define __fatrop_unittest_random_matrix_hpp__
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include <random>
namespace fatrop
{
    namespace test
    {
        // Function to generate a random double
        Scalar random(Scalar lower_bound = 0.0, Scalar upper_bound = 1.0)
        {
            // Static ensures the generator and distribution are initialized once
            static std::mt19937 gen(0); // Random number engine
            std::uniform_real_distribution<Scalar> dist(lower_bound, upper_bound);
            return dist(gen);
        }
        MatRealAllocated random_matrix(Index rows, Index cols, Scalar lower_bound = 0.0,
                                       Scalar upper_bound = 1.0)
        {
            MatRealAllocated matrix(rows, cols);
            for (Index i = 0; i < rows; ++i)
            {
                for (Index j = 0; j < cols; ++j)
                {
                    matrix(i, j) = random(lower_bound, upper_bound);
                }
            }
            return matrix;
        }
        MatRealAllocated random_spd_matrix(Index m, Scalar lower_bound = 0.0,
                                           Scalar upper_bound = 1.0)
        {
            MatRealAllocated matrix = random_matrix(m, m, lower_bound, upper_bound);
            MatRealAllocated ret(m, m);
            syrk_ln(m, m, 1.0, matrix, 0, 0, matrix, 0, 0, 0.0, ret, 0, 0, ret, 0, 0);
            trtr_l(m, ret, 0, 0, ret, 0, 0);
            return ret;
        }

        MatRealAllocated identity_matrix(Index m, double diagonal_value = 1.0){
            MatRealAllocated matrix(m, m);
            for (Index i = 0; i < m; ++i)
            {
                for (Index j = 0; j < m; ++j)
                {
                    matrix(i, j) = (i == j) ? diagonal_value : 0.0;
                }
            }
            return matrix;
        }

        MatRealAllocated empty_matrix(Index m, Index n)
        {
            MatRealAllocated matrix(m, n);
            for (Index i = 0; i < m; ++i)
            {
                for (Index j = 0; j < n; ++j)
                {
                    matrix(i, j) = 0.0;
                }
            }
            return matrix;
        }

        VecRealAllocated random_vector(Index rows, Scalar lower_bound = 0.0,
                                       Scalar upper_bound = 1.0)
        {
            VecRealAllocated v(rows);
            for (Index i = 0; i < rows; ++i)
            {
                v(i) = random(lower_bound, upper_bound);
            }
            return v;
        }

        int random_int(int lb, int ub){
            return random(static_cast<Scalar>(lb), static_cast<Scalar>(ub));
        }

    MatRealAllocated get_inverse(const MatRealView &A)
    {
        fatrop_dbg_assert(A.m() == A.n() && "Matrix must be square for inversion");
        MatRealAllocated A_inv(A.m(), A.m());
        MatRealAllocated LU(A.m(), A.m());
        blasfeo_dgetrf_np(A.m(), A.m(), const_cast<MAT *>(&A.mat()), 0, 0, &LU.mat(), 0, 0);

        // Solve the system LU * X = I, where I is the identity matrix
        MatRealAllocated I = identity_matrix(A.m());

        // (1) solve L Y = I
        blasfeo_dtrsm_llnu(A.m(), A.m(), 1.0, &LU.mat(), 0, 0, &I.mat(), 0, 0, &A_inv.mat(), 0, 0);
        // (2) solve U X = Y
        blasfeo_dtrsm_lunn(A.m(), A.m(), 1.0, &LU.mat(), 0, 0, &A_inv.mat(), 0, 0, &A_inv.mat(), 0, 0);

        // std::cout << "Inverse of A: \n" << A << "\nis given by \n"
        //           << A_inv << std::endl;

        // check if A_inv contains any NaN or Inf values
        for (Index i = 0; i < A_inv.m(); ++i)
        {
            for (Index j = 0; j < A_inv.n(); ++j)
            {
                if (std::isnan(A_inv(i, j)) || std::isinf(A_inv(i, j)))
                {
                    throw std::runtime_error("Inverse contains NaN or Inf values");
                }
            }
        }

        // check result
        MatRealAllocated I_check = identity_matrix(A.m());
        blasfeo_dgemm_nn(A.m(), A.m(), A.m(), 1.0, 
                        const_cast<MAT *>(&A.mat()), 0, 0, 
                        const_cast<MAT *>(&A_inv.mat()), 0, 0, 0.0, 
                        const_cast<MAT *>(&I_check.mat()), 0, 0,
                        const_cast<MAT *>(&I_check.mat()), 0, 0);
        // std::cout << "identity check: \n"
        //           << I_check << std::endl;

        return A_inv;
    }

    } // namespace test
} // namespace fatrop

#endif // __fatrop_unittest_random_matrix_hpp__