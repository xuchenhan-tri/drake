#pragma once

#include <memory>

#include <Eigen/Sparse>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

class CholmodSparseMatrix {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CholmodSparseMatrix);

  /* Constructs a CHOLMOD sparse matrix from an Eigen::SparseMatrix. */
  explicit CholmodSparseMatrix(const Eigen::SparseMatrix<double>& matrix);

  ~CholmodSparseMatrix();

  void Factor() const;

  VectorX<double> Solve(const VectorX<double>& rhs) const;

  void Print() const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
