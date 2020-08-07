#pragma once

#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"

namespace drake {
namespace fem {

template <typename T>
class LinearSystemSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LinearSystemSolver)

  // TODO(xuchenhan-tri) Abstract out an Objective Class from
  // BackwardEulerObjective.
  LinearSystemSolver(const BackwardEulerObjective<T>& objective)
      : objective_(objective) {}

  virtual ~LinearSystemSolver() {}

  /** Perform linear solve to get x = A⁻¹*rhs. */
  virtual void Solve(VectorX<T>& rhs, VectorX<T>* x) = 0;

  /** Set up the equation A*x = rhs. */
  virtual void SetUp() = 0;

  const BackwardEulerObjective<T>& get_objective() const { return objective_; }
 private:
  const BackwardEulerObjective<T>& objective_;
};
}  // namespace fem
}  // namespace drake