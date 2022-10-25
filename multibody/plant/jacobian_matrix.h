#pragma once

#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace internal {

template <class T>
class Matrix3BlockMatrix {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Matrix3BlockMatrix);

  /* block_row, block_col, and a 3x3 dense matrix. */
  using Triplet = std::tuple<int, int, Matrix3<T>>;

  /* Create a matrix with `row_blocks` * `col_blocks` of 3x3 matrices. */
  Matrix3BlockMatrix(int row_blocks, int col_blocks)
      : row_blocks_(row_blocks), col_blocks_(col_blocks) {}

  void AddTriplet(int block_row, int block_col, Matrix3<T> m) {
    DRAKE_DEMAND(0 <= block_row && block_row < row_blocks_);
    DRAKE_DEMAND(0 <= block_col && block_col < col_blocks_);
    data_.emplace_back(block_row, block_col, std::move(m));
  }

  int rows() const { return row_blocks_ * 3; }
  int cols() const { return col_blocks_ * 3; }

  void LeftMultiplyAndAddTo(const MatrixX<T>& A, MatrixX<T>* y) const {
    DRAKE_DEMAND(A.cols() == rows());
    for (const auto& triplet : data_) {
      const int block_row = std::get<0>(triplet);
      const int block_col = std::get<1>(triplet);
      const Matrix3<T>& m = std::get<2>(triplet);
      y->template middleCols<3>(3 * block_col) +=
          A.template middleCols<3>(3 * block_row) * m;
    }
  }

  MatrixX<T> MakeDenseMatrix() const {
    MatrixX<T> result = MatrixX<T>::Zero(rows(), cols());
    for (const auto& triplet : data_) {
      const int block_row = std::get<0>(triplet);
      const int block_col = std::get<1>(triplet);
      const Matrix3<T>& m = std::get<2>(triplet);
      result.template block<3, 3>(3 * block_row, 3 * block_col) = m;
    }
    return result;
  }

  const std::vector<Triplet>& get_triplets() const { return data_; }

 private:
  std::vector<Triplet> data_;
  int row_blocks_{};
  int col_blocks_{};
};

template <class T>
class JacobianBlock {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(JacobianBlock);

  JacobianBlock() : JacobianBlock(MatrixX<T>::Zero(0, 0)) {}

  explicit JacobianBlock(Matrix3BlockMatrix<T> data)
      : data_(std::move(data)), is_dense_(false) {}

  // NOLINTNEXTLINE(runtime/explicit)
  JacobianBlock(MatrixX<T> data) : data_(std::move(data)), is_dense_(true) {}

  /* We need the static_cast here because Eigen's rows() and cols() are long. */
  int rows() const {
    return std::visit([](auto&& arg) { return static_cast<int>(arg.rows()); },
                      data_);
  }

  int cols() const {
    return std::visit([](auto&& arg) { return static_cast<int>(arg.cols()); },
                      data_);
  }

  /* Performs *y += A * M, where M is `this` matrix. */
  void LeftMultiplyAndAddTo(const MatrixX<T>& A, MatrixX<T>* y) const {
    if (is_dense_) {
      const MatrixX<T>& matrix = std::get<MatrixX<T>>(data_);
      *y += A * matrix;
    } else {
      const Matrix3BlockMatrix<T>& matrix =
          std::get<Matrix3BlockMatrix<T>>(data_);
      matrix.LeftMultiplyAndAddTo(A, y);
    }
  }

  bool is_dense() const { return is_dense_; }

  /* Functions to satisfy existing tests. */
  MatrixX<T> MakeDenseMatrix() const {
    if (is_dense_) {
      return std::get<MatrixX<T>>(data_);
    }
    return std::get<Matrix3BlockMatrix<T>>(data_).MakeDenseMatrix();
  }

  operator MatrixX<T>() const { return MakeDenseMatrix(); }

  bool operator==(const JacobianBlock<T>& other) const {
    return this->MakeDenseMatrix() == other.MakeDenseMatrix();
  }

 private:
  template <class U>
  friend JacobianBlock<U> StackJacobianBlocks(
      const std::vector<JacobianBlock<U>>& blocks);
  std::variant<MatrixX<T>, Matrix3BlockMatrix<T>> data_;
  bool is_dense_{};
};

template <typename T>
JacobianBlock<T> StackJacobianBlocks(
    const std::vector<JacobianBlock<T>>& blocks) {
  if (blocks.empty()) {
    return {};
  }

  const bool is_dense = blocks[0].is_dense();
  const int cols = blocks[0].cols();
  int rows = 0;
  for (const auto& b : blocks) {
    /* Don't allow mixing dense and sparse matrices.*/
    DRAKE_THROW_UNLESS(is_dense == b.is_dense());
    DRAKE_THROW_UNLESS(cols == b.cols());
    rows += b.rows();
  }

  if (is_dense) {
    MatrixX<T> result(rows, cols);
    int row_offset = 0;
    for (const auto& b : blocks) {
      result.middleRows(row_offset, b.rows()) = std::get<MatrixX<T>>(b.data_);
      row_offset += b.rows();
    }
    return JacobianBlock<T>(std::move(result));
  }

  /* If this is a stack of sparse 3x3 blocks, then the total number of rows and
   cols are multiples of 3. */
  DRAKE_DEMAND(rows % 3 == 0);
  DRAKE_DEMAND(cols % 3 == 0);
  const int row_blocks = rows / 3;
  const int col_blocks = cols / 3;
  int block_row_offset = 0;
  Matrix3BlockMatrix<T> result(row_blocks, col_blocks);
  for (const auto& b : blocks) {
    const Matrix3BlockMatrix<T>& entry =
        std::get<Matrix3BlockMatrix<T>>(b.data_);
    for (const typename Matrix3BlockMatrix<T>::Triplet& t :
         entry.get_triplets()) {
      const int block_row = std::get<0>(t) + block_row_offset;
      const int block_col = std::get<1>(t);
      const Matrix3<T>& m = std::get<2>(t);
      result.AddTriplet(block_row, block_col, m);
    }
    block_row_offset += entry.rows() / 3;
  }
  return JacobianBlock<T>(std::move(result));
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake
