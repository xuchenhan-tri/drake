#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {

template <typename T>
class LinearSystemSolver {
 public:
  virtual ~LinearSystemSolver() {}

  /** Perform linear solve to get x = A⁻¹*rhs. */
  virtual void Solve(const Eigen::Ref<const VectorX<T>>& rhs,
                     EigenPtr<VectorX<T>> x) = 0;

  /** Set up the equation A*x = rhs. */
  virtual void SetUp() = 0;
};
}  // namespace fem
}  // namespace drake