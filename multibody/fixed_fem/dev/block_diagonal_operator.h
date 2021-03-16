#pragma once

#include <string>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/linear_operator.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

/* A LinearOperator that concatenates a vector of other LinearOperators in a
 block diagonal fashion.
 @tparam_nonsymbolic_scalar T. */
template <typename T>
class BlockDiagonalOperator final : public LinearOperator<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BlockDiagonalOperator)

  /* Constructs an operator with given `name` implementing the LinearOperator
   interface for a vector of operators concatenated in a block diagonal fashion.
   This class keeps a reference to the vector of LinearOperators making up the
   blocks, and therefore each LinearOperator block is required to outlive
   this object. */
  BlockDiagonalOperator(const std::string& name,
                        const std::vector<const LinearOperator<T>*>& blocks)
      : LinearOperator<T>(name), blocks_(blocks) {
    for (int i = 0; i < static_cast<int>(blocks_.size()); ++i) {
      DRAKE_DEMAND(blocks_[i] != nullptr);
    }
  }

  ~BlockDiagonalOperator() = default;

  int rows() const final {
    int rows = 0;
    for (const auto* block : blocks_) {
      rows += block->rows();
    }
    return rows;
  }

  int cols() const final {
    int cols = 0;
    for (const auto* block : blocks_) {
      cols += block->cols();
    }
    return cols;
  }

 protected:
  void DoMultiply(const Eigen::Ref<const VectorX<T>>& x,
                  VectorX<T>* y) const final {
    int starting_row = 0;
    int starting_col = 0;
    for (const auto* block : blocks_) {
      DRAKE_DEMAND(block != nullptr);
      const int block_rows = block->rows();
      const int block_cols = block->cols();

      const auto& x_segment = x.segment(starting_col, block_cols);
      VectorX<T> y_segment(block_rows);
      block->Multiply(x_segment, &y_segment);
      y->segment(starting_row, block_rows) = y_segment;

      starting_row += block_rows;
      starting_col += block_cols;
    }
  }

  void DoMultiply(const Eigen::Ref<const Eigen::SparseVector<T>>& x,
                  Eigen::SparseVector<T>* y) const final {
    VectorX<T> dense_x(x);
    VectorX<T> dense_y(dense_x.size());
    DoMultiply(dense_x, &dense_y);
    *y = dense_y.sparseView();
  }

  void DoAssembleMatrix(Eigen::SparseMatrix<T>* A) const final {
    std::vector<Eigen::Triplet<T>> triplets;
    int starting_row = 0;
    int starting_col = 0;
    for (const auto* block : blocks_) {
      DRAKE_DEMAND(block != nullptr);
      const int block_rows = block->rows();
      const int block_cols = block->cols();

      Eigen::SparseMatrix<T> block_matrix(block_rows, block_cols);
      block->AssembleMatrix(&block_matrix);
      AssembleBlockSparseMatrix(block_matrix, starting_row, starting_col,
                                &triplets);

      starting_row += block_rows;
      starting_col += block_cols;
    }
    A->setFromTriplets(triplets.begin(), triplets.end());
  }

 private:
  static void AssembleBlockSparseMatrix(
      const Eigen::SparseMatrix<T>& block_matrix, int starting_row,
      int starting_col, std::vector<Eigen::Triplet<T>>* triplets) {
    using InnerIterator = typename Eigen::SparseMatrix<T>::InnerIterator;
    for (int k = 0; k < block_matrix.outerSize(); ++k)
      for (InnerIterator it(block_matrix, k); it; ++it) {
        triplets->emplace_back(it.row() + starting_row, it.col() + starting_col,
                               it.value());
      }
  }
  const std::vector<const LinearOperator<T>*> blocks_;
};
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::BlockDiagonalOperator)
