#include "drake/multibody/fixed_fem/dev/permute_tangent_matrix.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
namespace {

using Eigen::MatrixXd;
const int kNumDofs = 9;
/* Returns an arbitrary tangent matrix for testing purpose. */
MatrixXd MakeTangentMatrix() {
  const int rows = kNumDofs;
  const int cols = kNumDofs;
  MatrixXd A(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      A(i, j) = cols * i + j;
    }
  }
  return A;
}

/* Verify that PermuteTangentMatrix provides the expected answer as
hand-calculated solution on a small problem. The problem consists of 3 vertices
such that vertex 0 and 2 are participating in contact and vertex 1 is not. */
GTEST_TEST(PermuteTangentMatrix, AnalyticTest) {
  const MatrixXd A = MakeTangentMatrix();
  const Eigen::SparseMatrix<double> A_sparse = A.sparseView();
  const std::vector<int> vertex_mapping = {0, 2, 1};
  const int num_participating = 2;
  const BlockTangentMatrix<double> permutated_A =
      PermuteTangentMatrix(A_sparse, vertex_mapping, num_participating);

  MatrixXd participating_block(6, 6);
  /* We use "i-j block" to denote the submatrix in the tangent matrix whose rows
   correspond to the i-th vertex and whose cols correspond to the j-th vertex.
  */
  /* The new 0-0 block is the same as the old 0-0 block. */
  participating_block.topLeftCorner<3, 3>() = A.topLeftCorner<3, 3>();
  /* The new 0-1 block is the same as the old 0-2 block. */
  participating_block.topRightCorner<3, 3>() = A.topRightCorner<3, 3>();
  /* The new 1-0 block is the same as the old 2-0 block. */
  participating_block.bottomLeftCorner<3, 3>() = A.bottomLeftCorner<3, 3>();
  /* The new 1-1 block is the same as the old 2-2 block. */
  participating_block.bottomRightCorner<3, 3>() = A.bottomRightCorner<3, 3>();
  MatrixXd off_diagonal_block(6, 3);
  /* The new 0-2 block is the same as the old 0-1 block. */
  off_diagonal_block.topRows<3>() = A.block<3, 3>(0, 3);
  /* The new 1-2 block is the same as the old 2-1 block. */
  off_diagonal_block.bottomRows<3>() = A.block<3, 3>(6, 3);
  /* The new 2-2 block is the same as the old 1-1 block. */
  MatrixXd nonparticipating_block = A.block<3, 3>(3, 3);

  EXPECT_TRUE(
      CompareMatrices(permutated_A.participating_block, participating_block));
  EXPECT_TRUE(
      CompareMatrices(permutated_A.off_diagonal_block, off_diagonal_block));
  EXPECT_TRUE(CompareMatrices(permutated_A.nonparticipating_block,
                              nonparticipating_block));
}
}  // namespace
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
