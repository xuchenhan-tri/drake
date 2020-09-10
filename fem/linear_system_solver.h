#pragma once

#include "drake/common/eigen_types.h"
#include "drake/multibody/solvers/linear_operator.h"

namespace drake {
namespace fem {

template <typename T>
class LinearSystemSolver {
 public:
  virtual ~LinearSystemSolver() {}

  /** Perform linear solve to get x = A⁻¹*rhs. */
  virtual void Solve(const Eigen::Ref<const VectorX<T>>& rhs,
                     EigenPtr<VectorX<T>> x) = 0;

  /** Set up the left-hand-side, A, in the equation A*x = rhs. */
  virtual void SetUp(const multibody::solvers::LinearOperator<T>& lop) = 0;

  virtual int rows() const = 0;
  virtual int cols() const = 0;
};
}  // namespace fem
}  // namespace drake