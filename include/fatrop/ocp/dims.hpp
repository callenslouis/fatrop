//
// Copyright (C) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_ocp_dims_hpp__
#define __fatrop_ocp_dims_hpp__

#include "fatrop/context/context.hpp"
#include "fatrop/nlp/dims.hpp"
#include "fatrop/ocp/type.hpp"
#include <vector>

namespace fatrop
{
    /**
     * @brief Represents the dimensions of an Optimal Control Problem (OCP).
     *
     * This structure holds the dimensions of various components in an OCP,
     * including the number of states, controls, and constraints for each stage.
     */
    struct ProblemDims
    {
        /**
         * @brief Constructs an ProblemDims object (lvalue reference version).
         *
         * @param K Number of stages / time steps in the Ocp. This means that there are K state
         * vectors, and K-1 control vectors used in the dynamics.
         * @param nu Vector of number of controls for each stage.
         * @param nx Vector of number of states for each stage.
         * @param ng Vector of number of equality constraints for each stage.
         * @param ng_ineq Vector of number of inequality constraints for each stage.
         */
        ProblemDims(int K, const std::vector<Index> &nu, const std::vector<Index> &nx,
                const std::vector<Index> &ng, const std::vector<Index> &ng_ineq);

        /**
         * @brief Constructs an ProblemDims object (rvalue reference version).
         *
         * @param k Number of stages in the OCP.
         * @param nu Vector of number of controls for each stage.
         * @param nx Vector of number of states for each stage.
         * @param ng Vector of number of equality constraints for each stage.
         * @param ng_ineq Vector of number of inequality constraints for each stage.
         */
        ProblemDims(int K, std::vector<Index> &&nu, std::vector<Index> &&nx, std::vector<Index> &&ng,
                std::vector<Index> &&ng_ineq);

        /**
         * @brief Constructs an ProblemDims object with a given number of dynamics constraints.
         *
         * Only implicit OCPs (ImplicitOcpType) support a number of dynamics constraints that
         * differs from the number of states of the next stage.
         *
         * @param n_dyn Vector of number of dynamics constraints that couple stage k and k+1
         * (K-1 values).
         */
        ProblemDims(int K, const std::vector<Index> &nu, const std::vector<Index> &nx,
                const std::vector<Index> &ng, const std::vector<Index> &ng_ineq,
                const std::vector<Index> &n_dyn);
        void check_problem_dimensions() const;

        const int K;                                  ///< Number of stages in the OCP.
        const std::vector<Index> number_of_controls; ///< Number of controls for each stage.
        const std::vector<Index> number_of_states;    ///< Number of states for each stage.
        const std::vector<Index>
            number_of_eq_constraints; ///< Number of path equality constraints for each stage.
        const std::vector<Index>
            number_of_ineq_constraints; ///< Number of path inequality constraints for each stage.
        const std::vector<Index>
            number_of_dyn_constraints; ///< Number of dynamics constraints between stage k and k+1
                                       ///< (K-1 values, nx[k+1] unless specified).
    };

} // namespace fatrop

#endif //__fatrop_ocp_dims_hpp__
