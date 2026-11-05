#include "fatrop/ocp/dims.hpp"
#include "fatrop/common/exception.hpp"
#include <algorithm>
using namespace fatrop;

namespace
{
  // by default, there is one dynamics constraint for every state of the next stage
  std::vector<Index> default_number_of_dyn_constraints(const std::vector<Index> &nx)
  {
    if (nx.size() < 2)
      return {};
    return std::vector<Index>(nx.begin() + 1, nx.end());
  }
}

ProblemDims::ProblemDims(int K, const std::vector<Index> &nu, const std::vector<Index> &nx,
                 const std::vector<Index> &ng, const std::vector<Index> &ng_ineq)
    : K(K), number_of_controls(nu), number_of_states(nx), number_of_eq_constraints(ng),
      number_of_ineq_constraints(ng_ineq),
      number_of_dyn_constraints(default_number_of_dyn_constraints(nx))
{
  check_problem_dimensions();
}

ProblemDims::ProblemDims(int K, std::vector<Index> &&nu, std::vector<Index> &&nx, std::vector<Index> &&ng,
                 std::vector<Index> &&ng_ineq)
    : K(K), number_of_controls(std::move(nu)), number_of_states(std::move(nx)),
      number_of_eq_constraints(std::move(ng)), number_of_ineq_constraints(std::move(ng_ineq)),
      number_of_dyn_constraints(default_number_of_dyn_constraints(number_of_states))
{
  check_problem_dimensions();
}

ProblemDims::ProblemDims(int K, const std::vector<Index> &nu, const std::vector<Index> &nx,
                 const std::vector<Index> &ng, const std::vector<Index> &ng_ineq,
                 const std::vector<Index> &n_dyn)
    : K(K), number_of_controls(nu), number_of_states(nx), number_of_eq_constraints(ng),
      number_of_ineq_constraints(ng_ineq), number_of_dyn_constraints(n_dyn)
{
  check_problem_dimensions();
}

void ProblemDims::check_problem_dimensions() const {
    // check if the number of controls, states, and constraints are of the correct size
    fatrop_assert_msg(number_of_controls.size() == K, "The number of controls is not of size K.");
    fatrop_assert_msg(number_of_states.size() == K, "The number of states is not of size K.");
    fatrop_assert_msg(number_of_eq_constraints.size() == K, "The number of equality constraints is not of size K.");
    fatrop_assert_msg(number_of_ineq_constraints.size() == K, "The number of inequality constraints is not of size K.");
    fatrop_assert_msg(number_of_dyn_constraints.size() == static_cast<size_t>(std::max(K - 1, 0)),
                      "The number of dynamics constraints is not of size K-1.");
    // iterate over every time step and do some checks
    for (Index i = 0; i < K; i++) {
        fatrop_assert_msg(number_of_controls[i] >= 0, "The number of controls must be non-negative.");
        fatrop_assert_msg(number_of_states[i] >= 0, "The number of states must be non-negative.");
        fatrop_assert_msg(number_of_eq_constraints[i] >= 0, "The number of equality constraints must be non-negative.");
        fatrop_assert_msg(number_of_ineq_constraints[i] >= 0, "The number of inequality constraints must be non-negative.");
        fatrop_assert_msg(number_of_eq_constraints[i] <= number_of_states[i] + number_of_controls[i],
                          "The number of eq constraints (" +
                          std::to_string(number_of_eq_constraints[i]) +
                          ") exceeds the number of variables (" +
                          std::to_string(number_of_states[i] + number_of_controls[i]) + ").");
        if (i < K - 1) {
            fatrop_assert_msg(number_of_dyn_constraints[i] >= 0, "The number of dynamics constraints must be non-negative.");
            fatrop_assert_msg(number_of_dyn_constraints[i] <= number_of_states[i] + number_of_controls[i] + number_of_states[i + 1],
                              "The number of dynamics constraints (" +
                              std::to_string(number_of_dyn_constraints[i]) +
                              ") exceeds the number of variables they depend on (" +
                              std::to_string(number_of_states[i] + number_of_controls[i] + number_of_states[i + 1]) + ").");
        }
    }
}
