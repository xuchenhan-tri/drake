#include "drake/multibody/fixed_fem/dev/symmetric_block_sparse_matrix_impl.h"

#include <utility>

#include "drake/common/autodiff.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T, int block_size>
SymmetricBlockSparseMatrixImpl<T, block_size>::SymmetricBlockSparseMatrixImpl(
    std::vector<std::vector<int>> row_blocks)
    : row_blocks_(std::move(row_blocks)),
      num_column_blocks_(row_blocks_.size()),
      data_(num_column_blocks_),
      num_row_blocks_(num_column_blocks_),
      row_blocks_to_index_(num_column_blocks_,
                           std::vector<int>(num_column_blocks_, -1)) {
  for (int c = 0; c < num_column_blocks_; ++c) {
    num_row_blocks_[c] = row_blocks_[c].size();
    for (int index = 0; index < num_row_blocks_[c]; ++index) {
      const int r = row_blocks_[c][index];
      DRAKE_DEMAND(r <= c);
      row_blocks_to_index_[c][r] = index;
      /* Add two blocks if the block is not on the diagonal (due to symmetry).
       */
      num_blocks_ += (r == c) ? 1 : 2;
    }
  }
  for (int c = 0; c < num_column_blocks_; ++c) {
    data_[c].resize(num_row_blocks_[c],
                    Eigen::Matrix<T, block_size, block_size>::Zero());
  }
}

template <typename T, int block_size>
void SymmetricBlockSparseMatrixImpl<T, block_size>::AddToBlock(
    int i, int j,
    const Eigen::Ref<const Eigen::Matrix<T, block_size, block_size>>& Aij) {
  DRAKE_DEMAND(0 <= i && i <= j && j <= num_column_blocks_);
  const int index = row_blocks_to_index_[j][i];
  DRAKE_DEMAND(index >= 0);
  data_[j][index] += Aij;
}

template <typename T, int block_size>
void SymmetricBlockSparseMatrixImpl<T, block_size>::DoSetZero() {
  for (int c = 0; c < num_column_blocks_; ++c) {
    for (auto& block : data_[c]) {
      block.setZero();
    }
  }
}

template <typename T, int block_size>
void SymmetricBlockSparseMatrixImpl<T, block_size>::DoMultiply(
    const VectorX<T>& x, VectorX<T>* y) const {
  y->setZero();
  for (int c = 0; c < num_column_blocks_; ++c) {
    for (int index = 0; index < num_row_blocks_[c]; ++index) {
      const int r = row_blocks_[c][index];
      y->template segment<block_size>(block_size * r) +=
          data_[c][index] * x.template segment<block_size>(block_size * c);
      if (r != c) {
        y->template segment<block_size>(block_size * c) +=
            data_[c][index].transpose() *
            x.template segment<block_size>(block_size * r);
      }
    }
  }
}

template <typename T, int block_size>
MatrixX<T> SymmetricBlockSparseMatrixImpl<T, block_size>::DoMakeDenseMatrix()
    const {
  MatrixX<T> A = MatrixX<T>::Zero(Base::rows(), cols());
  for (int c = 0; c < num_column_blocks_; ++c) {
    for (int index = 0; index < num_row_blocks_[c]; ++index) {
      const int r = row_blocks_[c][index];
      A.template block<block_size, block_size>(block_size * r, block_size * c) =
          data_[c][index];
      if (r != c) {
        A.template block<block_size, block_size>(
            block_size * c, block_size * r) = data_[c][index].transpose();
      }
    }
  }
  return A;
}

template <typename T, int block_size>
Eigen::SparseMatrix<T>
SymmetricBlockSparseMatrixImpl<T, block_size>::DoMakeEigenSparseMatrix() const {
  Eigen::SparseMatrix<T> A(Base::rows(), cols());
  std::vector<Eigen::Triplet<T>> triplets;
  triplets.reserve(9 * num_blocks_);

  auto add_block_to_triplets =
      [&triplets](
          const Eigen::Ref<const Eigen::Matrix<T, block_size, block_size>>&
              block,
          int block_row, int block_column) {
        for (int i = 0; i < block_size; ++i) {
          for (int j = 0; j < block_size; ++j) {
            triplets.emplace_back(block_size * block_row + i,
                                  block_size * block_column + j, block(i, j));
          }
        }
      };
  for (int c = 0; c < num_column_blocks_; ++c) {
    for (int index = 0; index < num_row_blocks_[c]; ++index) {
      const int r = row_blocks_[c][index];
      add_block_to_triplets(data_[c][index], r, c);
      if (r != c) {
        add_block_to_triplets(data_[c][index].transpose(), c, r);
      }
    }
  }
  A.setFromTriplets(triplets.begin(), triplets.end());
  A.makeCompressed();
  return A;
}

template class SymmetricBlockSparseMatrixImpl<double, 3>;
template class SymmetricBlockSparseMatrixImpl<AutoDiffXd, 3>;

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
