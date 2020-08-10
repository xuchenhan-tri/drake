#pragma once

#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"

namespace drake {
namespace fem {

template <typename T>
class LinearSystemSolver {
 public:
  virtual ~LinearSystemSolver() {}

  /** Perform linear solve to get x = A⁻¹*rhs. */
  virtual void Solve(VectorX<T>& rhs, VectorX<T>* x) = 0;

  /** Set up the equation A*x = rhs. */
  virtual void SetUp() = 0;
};
}  // namespace fem
}  // namespace drake