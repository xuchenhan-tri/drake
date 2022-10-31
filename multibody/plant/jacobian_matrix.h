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

/* Block sparse matrix with 3*3 blocks. The size of the matrix is
 3*row_blocks-by-3*col_blocks. The blocks are sorted according to block row
 indices. We use M to denote `this` matrix throughout this class. */
template <class T>
class Matrix3BlockMatrix {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Matrix3BlockMatrix);

  /* block_row, block_col, and a 3x3 dense matrix. */
  using Triplet = std::tuple<int, int, Matrix3<T>>;

  /* Create a matrix with `row_blocks` * `col_blocks` of 3x3 matrices. */
  Matrix3BlockMatrix(int row_blocks, int col_blocks)
      : row_blocks_(row_blocks), col_blocks_(col_blocks) {}

  /* @pre block_row >= all existing block rows. */
  void AddTriplet(int block_row, int block_col, Matrix3<T> m) {
    DRAKE_DEMAND(0 <= block_row && block_row < row_blocks_);
    DRAKE_DEMAND(0 <= block_col && block_col < col_blocks_);
    DRAKE_DEMAND(data_.empty() || std::get<0>(data_.back()) <= block_row);
    data_.emplace_back(block_row, block_col, std::move(m));
  }

  int rows() const { return row_blocks_ * 3; }
  int cols() const { return col_blocks_ * 3; }

  /* Performs *y += M * x. */
  void MultiplyAndAddTo(const Eigen::Ref<const VectorX<T>>& x,
                        EigenPtr<VectorX<T>> y) const {
    DRAKE_DEMAND(x.size() == cols());
    for (const auto& triplet : data_) {
      const int block_row = std::get<0>(triplet);
      const int block_col = std::get<1>(triplet);
      const Matrix3<T>& m = std::get<2>(triplet);
      y->template segment<3>(3 * block_row) +=
          m * x.template segment<3>(3 * block_col);
    }
  }

  /* Performs *y += A * M. */
  void LeftMultiplyAndAddTo(const Eigen::Ref<const MatrixX<T>>& A,
                            EigenPtr<MatrixX<T>> y) const {
    DRAKE_DEMAND(A.cols() == rows());
    for (const auto& triplet : data_) {
      const int block_row = std::get<0>(triplet);
      const int block_col = std::get<1>(triplet);
      const Matrix3<T>& m = std::get<2>(triplet);
      y->template middleCols<3>(3 * block_col) +=
          A.template middleCols<3>(3 * block_row) * m;
    }
  }

  /* Performs *y += Mᵀ * A, where A is dense. */
  void TransposeMultiplyAndAddTo(const Eigen::Ref<const MatrixX<T>>& A,
                                 EigenPtr<MatrixX<T>> y) const {
    DRAKE_DEMAND(rows() == A.rows());
    for (const auto& triplet : data_) {
      const int block_row = std::get<0>(triplet);
      const int block_col = std::get<1>(triplet);
      const Matrix3<T>& m = std::get<2>(triplet);
      y->template middleRows<3>(3 * block_col) +=
          m.transpose() * A.template middleRows<3>(3 * block_row);
    }
  }

  /* Performs *y += Mᵀ * A, where A is also 3x3 block sparse. */
  void TransposeMultiplyAndAddTo(const Matrix3BlockMatrix<T>& A,
                                 EigenPtr<MatrixX<T>> y) const {
    DRAKE_DEMAND(rows() == A.rows());
    DRAKE_DEMAND(y->rows() == this->cols());
    DRAKE_DEMAND(y->cols() == A.cols());

    if (A.data_.empty() || this->data_.empty()) {
      return;
    }

    /* We are performing *y_ij += M_ki * A_kj in this function. For each ij
     entry in y, we need to sum over row indices, k, where M_ki and A_kj both
     have non-zero blocks. We do this by looping over row indices. For each k,
     we find all blocks of A and M that have k as row index. Then we loop over
     their column indices (i for M and j for A), perform the dense
     multiplication, and add to the corresponding entry in y. */

    /* Find the `A_start` and `A_end` such that
     the blocks in A with flat index inside internal [A_start, A_end) all have
     the same row index, `A_row`. */
    int A_start = 0;
    int A_row = std::get<0>(A.data_[A_start]);
    int A_index = A_start + 1;
    while (A_index < A.num_blocks() && std::get<0>(A.data_[A_index]) == A_row) {
      ++A_index;
    }
    int A_end = A_index;

    for (int m = 0; m < num_blocks(); ++m) {
      int M_row = std::get<0>(data_[m]);
      if (M_row > A_row) {
        /* If we are here, it means that we have exhausted blocks in M that will
         multiply with blocks in A with row index `A_row`. It's time to find the
         next interval [A_start, A_end) such that blocks in A with flat index
         falling in this interval has the same block row index (aka the next
         `A_row`). Since blocks in A and M are sorted according to block row
         indexes, we can skip intervals [A_start, A_end) where `A_row < M_row`,
         because we have already gone past all possible M blocks that can
         multiply with those A blocks. */
        while (A_index < A.num_blocks() &&
               std::get<0>(A.data_[A_index]) < M_row) {
          ++A_index;
        }
        if (A_index == A.num_blocks()) {
          /* If there's no more A with `A_row >= M_row`, we've done all the
           * multiplication we need to do. */
          return;
        } else {
          /* Otherwise, update the internal [A_start, A_end) as well as `A_row`.
           */
          A_start = A_index;
          A_row = std::get<0>(A.data_[A_start]);
          while (A_index < A.num_blocks() &&
                 std::get<0>(A.data_[A_index]) == A_row) {
            ++A_index;
          }
          A_end = A_index;
        }
      }

      /* Skip blocks of M until either
       1. M_row > A_row, then we need to find the next interval with the same
          A_row (see above), or
       2. M_row == A_row, then we can actually start doing some multiplications
          (see below). */
      if (M_row < A_row) continue;

      DRAKE_DEMAND(M_row == A_row);
      /* *y_ij += M_ki * A_kj. The col index of M, i, is the row index of y.
       The col index of A, j, is the col index of y. In addition, don't forget
       to transpose the blocks of M. */
      const Matrix3<T>& Mt = std::get<2>(data_[m]).transpose();
      const int i = std::get<1>(data_[m]);
      for (int a = A_start; a < A_end; ++a) {
        const int j = std::get<1>(A.data_[a]);
        y->template block<3, 3>(3 * i, 3 * j) += Mt * std::get<2>(A.data_[a]);
      }
    }
  }

  /* Returns M * scale.asDiagonal() * M.transpose() */
  MatrixX<T> MultiplyByScaledTranspose(const VectorX<T>& scale) const {
    /* We need to sum M_ik * scale_k * M_jk and the sum is over column index.
     For efficiency, we first build a map from block column index to a vector of
     flat index for all blocks.  */
    std::unordered_map<int, std::vector<int>> col_to_flat;
    for (int flat = 0; flat < num_blocks(); ++flat) {
      const int col = std::get<1>(data_[flat]);
      col_to_flat[col].emplace_back(flat);
    }

    MatrixX<T> result = MatrixX<T>::Zero(rows(), rows());
    /* We use the notation m1_ik * scale_k * m2_jk. */
    for (const auto& t1 : data_) {
      const int i = std::get<0>(t1);
      const int k = std::get<1>(t1);
      const Matrix3<T>& m1 = std::get<2>(t1);
      for (int flat : col_to_flat[k]) {
        const auto& t2 = data_[flat];
        const int j = std::get<0>(t2);
        DRAKE_ASSERT(k == std::get<1>(t2));
        const Matrix3<T>& m2 = std::get<2>(t2);
        const auto scale_block = scale.template segment<3>(3 * k);
        result.template block<3, 3>(3 * i, 3 * j) +=
            m1 * scale_block.asDiagonal() * m2.transpose();
      }
    }
    return result;
  }

  /* For debugging. */
  MatrixX<T> MakeDenseMatrix() const {
    MatrixX<T> result = MatrixX<T>::Zero(rows(), cols());
    for (const auto& triplet : data_) {
      const int block_row = std::get<0>(triplet);
      const int block_col = std::get<1>(triplet);
      const Matrix3<T>& m = std::get<2>(triplet);
      result.template block<3, 3>(3 * block_row, 3 * block_col) += m;
    }
    return result;
  }

  const std::vector<Triplet>& get_triplets() const { return data_; }

  int num_blocks() const { return data_.size(); }

 private:
  std::vector<Triplet> data_;
  int row_blocks_{};
  int col_blocks_{};
};

/* Data structure to store the jacobian matrix for a particular tree and patch.
 It's eventually stored as a block in the block sparse jacobian matrix and
 thus the name. It can either be a dense Eigen matrix or a block sparse 3x3
 matrix. We use J to denote `this` jacobian block throughout this class. */
template <class T>
class JacobianBlock {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(JacobianBlock);

  /* Default constructs an empty JacobianBlock as a dense 0x0 matrix. */
  JacobianBlock() : JacobianBlock(MatrixX<T>::Zero(0, 0)) {}

  explicit JacobianBlock(Matrix3BlockMatrix<T> data)
      : data_(std::move(data)), is_dense_(false) {}

  explicit JacobianBlock(MatrixX<T> data)
      : data_(std::move(data)), is_dense_(true) {}

  /* We need the static_cast here because Eigen's rows() and cols() are long. */
  int rows() const {
    return std::visit([](auto&& arg) { return static_cast<int>(arg.rows()); },
                      data_);
  }

  int cols() const {
    return std::visit([](auto&& arg) { return static_cast<int>(arg.cols()); },
                      data_);
  }

  /* Performs *y += A * J. */
  void LeftMultiplyAndAddTo(const Eigen::Ref<const MatrixX<T>>& A,
                            EigenPtr<MatrixX<T>> y) const {
    if (is_dense_) {
      const MatrixX<T>& matrix = std::get<MatrixX<T>>(data_);
      *y += A * matrix;
      return;
    }
    const Matrix3BlockMatrix<T>& matrix =
        std::get<Matrix3BlockMatrix<T>>(data_);
    matrix.LeftMultiplyAndAddTo(A, y);
  }

  /* Performs *y += J * x. */
  void MultiplyAndAddTo(const Eigen::Ref<const VectorX<T>>& x,
                        EigenPtr<VectorX<T>> y) const {
    if (is_dense()) {
      const MatrixX<T>& J = std::get<MatrixX<T>>(data_);
      *y += J * x;
      return;
    }
    const Matrix3BlockMatrix<T>& J = std::get<Matrix3BlockMatrix<T>>(data_);
    J.MultiplyAndAddTo(x, y);
  }

  /* Performs *y += Jᵀ * A. */
  void TransposeMultiplyAndAddTo(const Eigen::Ref<const MatrixX<T>>& A,
                                 EigenPtr<MatrixX<T>> y) const {
    if (is_dense()) {
      const MatrixX<T>& J = std::get<MatrixX<T>>(data_);
      *y += J.transpose() * A;
      return;
    }
    const Matrix3BlockMatrix<T>& J = std::get<Matrix3BlockMatrix<T>>(data_);
    J.TransposeMultiplyAndAddTo(A, y);
  }

  void TransposeMultiplyAndAddTo(const JacobianBlock<T>& A,
                                 EigenPtr<MatrixX<T>> y) const {
    DRAKE_DEMAND(y != nullptr);
    DRAKE_DEMAND(rows() == A.rows());
    DRAKE_DEMAND(rows() == A.rows());
    DRAKE_DEMAND(cols() == y->rows());
    DRAKE_DEMAND(A.cols() == y->cols());

    if (A.is_dense()) {
      const MatrixX<T>& A_matrix = std::get<MatrixX<T>>(A.data_);
      this->TransposeMultiplyAndAddTo(Eigen::Ref<const MatrixX<T>>(A_matrix),
                                      y);
      return;
    }
    if (this->is_dense()) {
      const MatrixX<T>& J = std::get<MatrixX<T>>(this->data_);
      A.LeftMultiplyAndAddTo(J.transpose(), y);
      return;
    }
    /* A and J both sparse. */
    const Matrix3BlockMatrix<T>& J =
        std::get<Matrix3BlockMatrix<T>>(this->data_);
    const Matrix3BlockMatrix<T>& A_matrix =
        std::get<Matrix3BlockMatrix<T>>(A.data_);
    J.TransposeMultiplyAndAddTo(A_matrix, y);
  }

  /* Computes G*J where G is a block diagonal matrix with the diagonal blocks
   specified as a vector of dense matrices. In particular, the diagonal blocks
   are the [G_start, G_end] entries in the given `G`. */
  JacobianBlock<T> LeftMultiplyByBlockDiagonal(const std::vector<MatrixX<T>>& G,
                                               int G_start, int G_end) const {
    DRAKE_DEMAND(G_start >= 0);
    DRAKE_DEMAND(G_end >= G_start);
    DRAKE_DEMAND(static_cast<int>(G.size()) > G_end);
    /* Verify that the sizes of G and this Jacobian is compatible. */
    int G_size = 0;
    bool is_G_block_3_by_3 = true;
    for (int i = G_start; i <= G_end; ++i) {
      DRAKE_DEMAND(G[i].rows() == G[i].cols());
      G_size += G[i].rows();
      if (G[i].rows() != 3) {
        is_G_block_3_by_3 = false;
      }
    }
    DRAKE_DEMAND(G_size == rows());

    if (is_dense()) {
      const MatrixX<T>& J = std::get<MatrixX<T>>(data_);
      MatrixX<T> GJ(rows(), cols());
      int row_offset = 0;
      for (int index = G_start; index <= G_end; ++index) {
        const int num_rows = G[index].rows();
        GJ.middleRows(row_offset, num_rows).noalias() =
            G[index] * J.middleRows(row_offset, num_rows);
        row_offset += num_rows;
      }
      return JacobianBlock<T>(std::move(GJ));
    }

    /* J is sparse. We abort if not all blocks of G is 3x3 and we can't easily
     exploit the sparsity of the Jacobian. */
    DRAKE_DEMAND(is_G_block_3_by_3);
    const Matrix3BlockMatrix<T>& J = std::get<Matrix3BlockMatrix<T>>(data_);
    Matrix3BlockMatrix<T> GJ(J.rows() / 3, J.cols() / 3);
    for (const auto& t : J.get_triplets()) {
      const int block_row = std::get<0>(t);
      const int block_col = std::get<1>(t);
      const Matrix3<T>& m = std::get<2>(t);
      GJ.AddTriplet(block_row, block_col, G[G_start + block_row] * m);
    }
    return JacobianBlock<T>(std::move(GJ));
  }

  /* Returns J * scale.asDiagonal() * J.transpose() */
  MatrixX<T> MultiplyByScaledTranspose(const VectorX<T>& scale) const {
    DRAKE_DEMAND(cols() == scale.size());
    if (is_dense()) {
      const MatrixX<T>& J = std::get<MatrixX<T>>(data_);
      return J * scale.asDiagonal() * J.transpose();
    }
    const Matrix3BlockMatrix<T>& J = std::get<Matrix3BlockMatrix<T>>(data_);
    return J.MultiplyByScaledTranspose(scale);
  }

  bool is_dense() const { return is_dense_; }

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
