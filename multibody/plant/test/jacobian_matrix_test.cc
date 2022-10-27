#include "drake/multibody/plant/jacobian_matrix.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;

/* Create two versions of M, one dense one sparse, and test y += A * M for each.
 */
GTEST_TEST(JacobianMatrixTest, LeftMultiplyAndAddTo) {
  Matrix3d M00;
  M00 << 1, 2, 3, 5, 3, 6, 2, 5, 7;
  Matrix3d M12;
  M12 << 0, 2, 0, 1, 8, 2, 6, 2, 2;

  Matrix6<double> A;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      A(i, j) = 3 * i + 4 * j;
    }
  }

  /* y's are the destinations. */
  MatrixXd y(6, 9);
  y(0, 1) = 0.72;
  y(4, 4) = 0.172;
  MatrixXd y2(y);

  Matrix3BlockMatrix<double> sparse(2, 3);
  sparse.AddTriplet(0, 0, M00);
  sparse.AddTriplet(1, 2, M12);

  MatrixXd dense = MatrixXd::Zero(6, 9);
  dense.topLeftCorner<3, 3>() = M00;
  dense.bottomRightCorner<3, 3>() = M12;

  JacobianBlock<double> dense_jacobian(std::move(dense));
  JacobianBlock<double> sparse_jacobian(std::move(sparse));

  EXPECT_EQ(dense_jacobian.rows(), 6);
  EXPECT_EQ(dense_jacobian.cols(), 9);
  EXPECT_EQ(sparse_jacobian.rows(), 6);
  EXPECT_EQ(sparse_jacobian.cols(), 9);

  dense_jacobian.LeftMultiplyAndAddTo(A, &y);
  sparse_jacobian.LeftMultiplyAndAddTo(A, &y2);

  EXPECT_TRUE(CompareMatrices(y, y2));
}

GTEST_TEST(JacobianMatrixTest, TransposeAndRightMultiply) {
  Matrix3d M00;
  M00 << 1, 2, 3, 5, 3, 6, 2, 5, 7;
  Matrix3d M12;
  M12 << 0, 2, 0, 1, 8, 2, 6, 2, 2;

  Matrix6<double> A;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      A(i, j) = 3 * i + 4 * j;
    }
  }

  Matrix3BlockMatrix<double> sparse(2, 3);
  sparse.AddTriplet(0, 0, M00);
  sparse.AddTriplet(1, 2, M12);

  MatrixXd dense = MatrixXd::Zero(6, 9);
  dense.topLeftCorner<3, 3>() = M00;
  dense.bottomRightCorner<3, 3>() = M12;

  JacobianBlock<double> dense_jacobian(dense);
  EXPECT_EQ(dense_jacobian.TransposeAndRightMultiply(A), dense.transpose() * A);

  JacobianBlock<double> sparse_jacobian(sparse);
  EXPECT_EQ(sparse_jacobian.TransposeAndRightMultiply(A),
            sparse.TransposeAndRightMultiply(A));

  EXPECT_EQ(dense_jacobian.TransposeAndRightMultiply(A),
            sparse_jacobian.TransposeAndRightMultiply(A));
}

GTEST_TEST(JacobianMatrixTest, TransposeAndRightMultiplyByJacobianBlock) {
  Matrix3d M0;
  M0 << 1, 2, 3, 5, 3, 6, 2, 5, 7;
  Matrix3d M1;
  M1 << 0, 2, 0, 1, 8, 2, 6, 2, 2;
  Matrix3d M2;
  M2 << 1, 2, 2, 1, 8, 2, 6, 2, 2;

  Matrix3BlockMatrix<double> matrix(3, 2);
  matrix.AddTriplet(0, 0, M1);
  matrix.AddTriplet(0, 1, M1);
  matrix.AddTriplet(2, 1, M1);

  const JacobianBlock<double> A(matrix);
  const JacobianBlock<double> B(matrix);

  const MatrixXd A_dense = A.MakeDenseMatrix();
  const MatrixXd B_dense = B.MakeDenseMatrix();
  const MatrixXd expected_result = A_dense.transpose() * B_dense;
  // sparse sparse
  EXPECT_EQ(A.TransposeAndRightMultiply(B).MakeDenseMatrix(), expected_result);
  std::cout << "sparse sparse pass" << std::endl;
  // sparse dense
  EXPECT_EQ(A.TransposeAndRightMultiply(JacobianBlock<double>(B_dense))
                .MakeDenseMatrix(),
            A_dense.transpose() * B_dense);
  std::cout << "sparse dense pass" << std::endl;
  // dense dense
  EXPECT_EQ(JacobianBlock<double>(A_dense)
                .TransposeAndRightMultiply(JacobianBlock<double>(B_dense))
                .MakeDenseMatrix(),
            A_dense.transpose() * B_dense);
  std::cout << "dense dense pass" << std::endl;
  // dense sparse
  EXPECT_EQ(JacobianBlock<double>(A_dense)
                .TransposeAndRightMultiply(B)
                .MakeDenseMatrix(),
            A_dense.transpose() * B_dense);
  std::cout << "dense sparse pass" << std::endl;
}

GTEST_TEST(JacobianMatrixTest, LeftMultiplyByBlockDiagonal) {
  Matrix3d M00;
  M00 << 1, 2, 3, 5, 3, 6, 2, 5, 7;
  Matrix3d M12;
  M12 << 0, 2, 0, 1, 8, 2, 6, 2, 2;

  Matrix6<double> A;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      A(i, j) = 3 * i + 4 * j;
    }
  }

  Matrix3BlockMatrix<double> sparse(2, 3);
  sparse.AddTriplet(0, 0, M00);
  sparse.AddTriplet(1, 2, M12);

  MatrixXd dense = MatrixXd::Zero(6, 9);
  dense.topLeftCorner<3, 3>() = M00;
  dense.bottomRightCorner<3, 3>() = M12;

  JacobianBlock<double> dense_jacobian(dense);
  JacobianBlock<double> sparse_jacobian(sparse);

  std::vector<MatrixXd> G1;
  G1.emplace_back(A);
  const MatrixXd expected_result1 = A * dense;
  const MatrixXd dense_result1 =
      dense_jacobian.LeftMultiplyByBlockDiagonal(G1, 0, 0);
  const MatrixXd sparse_result1 =
      sparse_jacobian.LeftMultiplyByBlockDiagonal(G1, 0, 0);

  EXPECT_EQ(expected_result1, dense_result1);
  EXPECT_EQ(expected_result1, sparse_result1);

  MatrixXd A2(6, 6);
  A2.setZero();
  A2.topLeftCorner<3, 3>() = A.topLeftCorner<3, 3>();
  A2.bottomRightCorner<3, 3>() = A.bottomRightCorner<3, 3>();
  const MatrixXd expected_result2 = A2 * dense;
  std::vector<MatrixXd> G2;
  G2.emplace_back(Matrix3d(A2.topLeftCorner<3, 3>()));
  G2.emplace_back(Matrix3d(A2.bottomRightCorner<3, 3>()));
  const MatrixXd dense_result2 =
      dense_jacobian.LeftMultiplyByBlockDiagonal(G2, 0, 1);
  const MatrixXd sparse_result2 =
      sparse_jacobian.LeftMultiplyByBlockDiagonal(G2, 0, 1);
  EXPECT_EQ(expected_result2, dense_result2);
  EXPECT_EQ(expected_result2, sparse_result2);
}

GTEST_TEST(JacobianMatrixTest, StackDenseMatrix) {
  Matrix6<double> A1;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      A1(i, j) = 3 * i + 4 * j;
    }
  }
  Matrix6<double> A2;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      A2(i, j) = 30 * i + 40 * j;
    }
  }

  std::vector<JacobianBlock<double>> blocks;
  blocks.emplace_back(A1);
  blocks.emplace_back(A2);

  const JacobianBlock<double> stack = StackJacobianBlocks(blocks);
  MatrixXd expected_result(12, 6);
  expected_result.topRows<6>() = A1;
  expected_result.bottomRows<6>() = A2;
  EXPECT_EQ(stack.MakeDenseMatrix(), expected_result);
}

GTEST_TEST(JacobianMatrixTest, StackSparseMatrix) {
  Matrix3d M00;
  M00 << 1, 2, 3, 5, 3, 6, 2, 5, 7;
  Matrix3d M12;
  M12 << 0, 2, 0, 1, 8, 2, 6, 2, 2;

  Matrix3d N01 = M00;
  Matrix3d N10 = M12;
  Matrix3d N22 = Matrix3d::Identity();

  Matrix3BlockMatrix<double> sparse_M(2, 3);
  sparse_M.AddTriplet(0, 0, M00);
  sparse_M.AddTriplet(1, 2, M12);
  MatrixXd dense_M = sparse_M.MakeDenseMatrix();

  Matrix3BlockMatrix<double> sparse_N(3, 3);
  sparse_N.AddTriplet(0, 1, N01);
  sparse_N.AddTriplet(1, 0, N10);
  sparse_N.AddTriplet(2, 2, N22);
  MatrixXd dense_N = sparse_N.MakeDenseMatrix();

  std::vector<JacobianBlock<double>> sparse_blocks;
  sparse_blocks.emplace_back(sparse_M);
  sparse_blocks.emplace_back(sparse_N);
  const JacobianBlock<double> sparse_stack = StackJacobianBlocks(sparse_blocks);

  std::vector<JacobianBlock<double>> dense_blocks;
  dense_blocks.emplace_back(dense_M);
  dense_blocks.emplace_back(dense_N);
  const JacobianBlock<double> dense_stack = StackJacobianBlocks(dense_blocks);
  EXPECT_EQ(sparse_stack, dense_stack);
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
