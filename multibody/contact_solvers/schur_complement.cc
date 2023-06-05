#include "drake/multibody/contact_solvers/schur_complement.h"

#include <algorithm>
#include <utility>

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

SchurComplement::SchurComplement(const Block3x3SparseSymmetricMatrix& M,
                                 std::unordered_set<int> D_indices)
    : D_indices_(D_indices.begin(), D_indices.end()) {
  /* Keep D_indices_ and A_indices_ sorted. */
  std::sort(D_indices_.begin(), D_indices_.end());
  A_indices_.reserve(M.block_cols() - D_indices.size());
  for (int j = 0; j < M.block_cols(); ++j) {
    if (D_indices.count(j) == 0) {
      A_indices_.push_back(j);
    }
  }
  const int block_cols = A_indices_.size() + D_indices_.size();
  DRAKE_DEMAND(block_cols * 3 == M.cols());
  const bool success = solver_.CalcSchurComplementAndFactor(M, D_indices, &S_);
  if (!success) {
    throw std::runtime_error(
        "Factorization failed when computing Schur complement. Make sure the "
        "matrix is symmetric positive definite and not ill-conditioned.");
  }
}

VectorX<double> SchurComplement::SolveForX(
    const Eigen::Ref<const VectorX<double>>& y) const {
  DRAKE_DEMAND(y.size() == 3 * ssize(A_indices_));
  if (D_indices_.size() == 0) {
    return VectorX<double>::Zero(0);
  }
  if (A_indices_.size() == 0) {
    return VectorX<double>::Zero(D_indices_.size());
  }

  /* Build `a`, the rhs corrsponding to variable y. */
  const VectorX<double> a = S_ * y;
  /* Build the full rhs. */
  const int block_cols = A_indices_.size() + D_indices_.size();
  VectorX<double> rhs(VectorX<double>::Zero(3 * block_cols));
  for (int i = 0; i < ssize(A_indices_); ++i) {
    rhs.segment<3>(3 * A_indices_[i]) = a.segment<3>(3 * i);
  }
  solver_.SolveInPlace(&rhs);
  /* Extract x from the lhs. */
  VectorX<double> x(3 * D_indices_.size());
  for (int i = 0; i < ssize(D_indices_); ++i) {
    x.segment<3>(3 * i) = rhs.segment<3>(3 * D_indices_[i]);
  }
  return x;
}

VectorX<double> SchurComplement::Solve(
    const Eigen::Ref<const VectorX<double>>& c) const {
  DRAKE_DEMAND(3 * (ssize(A_indices_) + ssize(D_indices_)) == c.size());
  return solver_.Solve(c);
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
