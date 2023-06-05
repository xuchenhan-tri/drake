#include "drake/multibody/contact_solvers/block_sparse_cholesky_solver.h"

#include <memory>
#include <numeric>
#include <utility>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/unused.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;

/* Makes an arbitrary SPD matrix with the following nonzero pattern.

  X X | O O O | O O O O | O O O
  X X | O O O | O O O O | O O O
  --- | ----- |---------| -----
  O O | X X X | X X X X | X X X
  O O | X X X | X X X X | X X X
  O O | X X X | X X X X | X X X
  --- | ----- |---------| -----
  O O | X X X | X X X X | O O O
  O O | X X X | X X X X | O O O
  O O | X X X | X X X X | O O O
  O O | X X X | X X X X | O O O
  --- | ----- |---------| -----
  O O | X X X | O O O O | X X X
  O O | X X X | O O O O | X X X
  O O | X X X | O O O O | X X X

 The scaling factor can be used to control the values of each nonzero entry. */
Eigen::Matrix<double, 12, 12> MakeSpdMatrix(double scale) {
  Eigen::Matrix<double, 12, 12> A;
  for (int i = 0; i < 12; ++i) {
    for (int j = 0; j < 12; ++j) {
      A(i, j) = 0.03 * i + 0.02 * j;
    }
  }
  Eigen::Matrix<double, 12, 12> I = Eigen::Matrix<double, 12, 12>::Identity();
  /* Add a large diagonal entry to ensure diagonal dominance. */
  Eigen::Matrix<double, 12, 12> result = scale * (A * A.transpose() + 12 * I);
  /* Zeroing off-diagonal entries of a diagonally dominant matrix does not
   affect diagonal dominance, which in turns implies SPDness. */
  result.block<10, 2>(2, 0).setZero();
  result.block<2, 10>(0, 2).setZero();
  result.block<3, 4>(9, 5).setZero();
  result.block<4, 3>(5, 9).setZero();
  return result;
}

/* Makes an arbitrary SPD sparse matrix. */
BlockSparseSymmetricMatrix MakeSparseSpdMatrix(double scale = 1.0) {
  std::vector<std::vector<int>> sparsity;
  sparsity.emplace_back(std::vector<int>{0});
  sparsity.emplace_back(std::vector<int>{1, 2, 3});
  sparsity.emplace_back(std::vector<int>{2});
  sparsity.emplace_back(std::vector<int>{3});
  std::vector<int> block_sizes = {2, 3, 4, 3};
  BlockSparsityPattern block_pattern(block_sizes, sparsity);

  BlockSparseSymmetricMatrix A(std::move(block_pattern));
  const std::vector<int>& starting_cols = A.starting_cols();
  const Eigen::Matrix<double, 12, 12> dense_A = MakeSpdMatrix(scale);
  std::vector<std::pair<int, int>> nonzero_lower_triangular_blocks{
      {0, 0}, {1, 1}, {2, 1}, {2, 2}, {3, 1}, {3, 3}};
  for (const auto& [a, b] : nonzero_lower_triangular_blocks) {
    A.AddToBlock(a, b,
                 dense_A.block(starting_cols[a], starting_cols[b],
                               block_sizes[a], block_sizes[b]));
  }
  return A;
}

GTEST_TEST(BlockSparseCholeskySolverTest, Solve) {
  BlockSparseCholeskySolver<MatrixXd> solver;
  BlockSparseSymmetricMatrix A = MakeSparseSpdMatrix();
  MatrixX<double> dense_A = A.MakeDenseMatrix();
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseCholeskySolver<MatrixXd>::SolverMode::kEmpty);
  solver.SetMatrix(A);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseCholeskySolver<MatrixXd>::SolverMode::kAnalyzed);
  bool success = solver.Factor();
  EXPECT_TRUE(success);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseCholeskySolver<MatrixXd>::SolverMode::kFactored);

  const VectorXd b1 = VectorXd::LinSpaced(A.cols(), 0.0, 1.0);
  const VectorXd x1 = solver.Solve(b1);
  const VectorXd expected_x1 = dense_A.llt().solve(b1);
  EXPECT_TRUE(CompareMatrices(x1, expected_x1, 1e-13));

  /* Solve for a different right hand side without refactoring. */
  const VectorXd b2 = VectorXd::LinSpaced(A.cols(), 0.0, 10.0);
  const VectorXd x2 = solver.Solve(b2);
  const VectorXd expected_x2 = dense_A.llt().solve(b2);
  EXPECT_TRUE(CompareMatrices(x2, expected_x2, 1e-13));

  /* SolveInPlace variant. */
  const VectorXd b3 = VectorXd::LinSpaced(A.cols(), 7.0, 8.0);
  const VectorXd expected_x3 = dense_A.llt().solve(b3);
  VectorXd x3 = b3;
  solver.SolveInPlace(&x3);
  EXPECT_TRUE(CompareMatrices(x3, expected_x3, 1e-13));

  /* Update the matrix with different numeric values but the same sparsity
   pattern. */
  BlockSparseSymmetricMatrix A2 = MakeSparseSpdMatrix(10);
  MatrixX<double> dense_A2 = A2.MakeDenseMatrix();
  solver.UpdateMatrix(A2);
  success = solver.Factor();
  EXPECT_TRUE(success);
  const VectorXd b4 = VectorXd::LinSpaced(A2.cols(), 0.0, 10.0);
  const VectorXd x4 = solver.Solve(b4);
  const VectorXd expected_x4 = dense_A2.llt().solve(b4);
  EXPECT_TRUE(CompareMatrices(x4, expected_x4, 1e-13));
}

GTEST_TEST(BlockSparseCholeskySolverTest, SolveFailureDueToNonSpdness) {
  std::vector<std::vector<int>> sparsity;
  sparsity.emplace_back(std::vector<int>{0});
  std::vector<int> block_sizes = {4};
  BlockSparsityPattern block_pattern(block_sizes, sparsity);
  BlockSparseSymmetricMatrix A(std::move(block_pattern));
  A.AddToBlock(0, 0, -Matrix4d::Identity());

  BlockSparseCholeskySolver<MatrixXd> solver;
  solver.SetMatrix(A);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseCholeskySolver<MatrixXd>::SolverMode::kAnalyzed);
  const bool success = solver.Factor();
  EXPECT_FALSE(success);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseCholeskySolver<MatrixXd>::SolverMode::kEmpty);
}

GTEST_TEST(BlockSparseCholeskySolverTest, FactorBeforeSetMatrixThrows) {
  BlockSparseCholeskySolver<MatrixXd> solver;
  EXPECT_THROW(unused(solver.Factor()), std::exception);
}

GTEST_TEST(BlockSparseCholeskySolverTest, SolveBeforeFactorThrows) {
  BlockSparseCholeskySolver<MatrixXd> solver;
  BlockSparseSymmetricMatrix A = MakeSparseSpdMatrix();
  solver.SetMatrix(A);
  VectorXd b = VectorXd::LinSpaced(A.cols(), 0.0, 10.0);
  EXPECT_THROW(solver.Solve(b), std::exception);
  EXPECT_THROW(solver.SolveInPlace(&b), std::exception);
}

GTEST_TEST(BlockSparseCholeskySolverTest, PermutationMatrix) {
  BlockSparseCholeskySolver<MatrixXd> solver;
  BlockSparseSymmetricMatrix A = MakeSparseSpdMatrix();
  const MatrixXd A_dense = A.MakeDenseMatrix();
  solver.SetMatrix(A);
  /* Trying to get L before factorization is an exception. */
  EXPECT_THROW(solver.L(), std::exception);
  const bool success = solver.Factor();
  EXPECT_TRUE(success);
  const MatrixXd L = solver.L().MakeDenseMatrix();
  const Eigen::PermutationMatrix<Eigen::Dynamic> P =
      solver.CalcPermutationMatrix();
  const MatrixXd lhs = L * L.transpose();
  const MatrixXd rhs = P * A_dense * P.transpose();
  EXPECT_TRUE(CompareMatrices(lhs, rhs, 1e-14));
}

GTEST_TEST(BlockSparseCholeskySolverTest, PermutationMatrixPrecondition) {
  BlockSparseCholeskySolver<MatrixXd> solver;
  BlockSparseSymmetricMatrix A = MakeSparseSpdMatrix();
  /* CalcPermutationMatrix() before setting the matrix throws. */
  EXPECT_THROW(solver.CalcPermutationMatrix(), std::exception);
  /* After setting the matrix, CalcPermutatoinMatrix() returns the same result
   before and after factorization. */
  solver.SetMatrix(A);
  const Eigen::PermutationMatrix<Eigen::Dynamic> P0 =
      solver.CalcPermutationMatrix();
  const bool success = solver.Factor();
  EXPECT_TRUE(success);
  const Eigen::PermutationMatrix<Eigen::Dynamic> P1 =
      solver.CalcPermutationMatrix();
  EXPECT_EQ(MatrixXi(P0), MatrixXi(P1));
}

GTEST_TEST(BlockSparseCholeskySolverTest, SolverModeAfterMove) {
  BlockSparseCholeskySolver<MatrixXd> solver;
  BlockSparseSymmetricMatrix A = MakeSparseSpdMatrix();
  solver.SetMatrix(A);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseCholeskySolver<MatrixXd>::SolverMode::kAnalyzed);
  BlockSparseCholeskySolver<MatrixXd> new_solver(std::move(solver));
  EXPECT_EQ(new_solver.solver_mode(),
            BlockSparseCholeskySolver<MatrixXd>::SolverMode::kAnalyzed);
  /* The mode of the old solver resets to kEmpty. */
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseCholeskySolver<MatrixXd>::SolverMode::kEmpty);
}

GTEST_TEST(BlockSparseCholeskySolverTest, CalcSchurComplementAndFactor) {
  BlockSparseCholeskySolver<MatrixXd> solver;
  BlockSparseSymmetricMatrix M = MakeSparseSpdMatrix();
  const int kNumBlocks = 4;
  MatrixXd schur_complement;
  /* All blocks are eliminated. */
  {
    std::vector<int> eliminated_blocks(kNumBlocks);
    std::iota(eliminated_blocks.begin(), eliminated_blocks.end(), 0);
    const bool success = solver.CalcSchurComplementAndFactor(
        M,
        std::unordered_set<int>(eliminated_blocks.begin(),
                                eliminated_blocks.end()),
        &schur_complement);
    EXPECT_TRUE(success);
    EXPECT_EQ(schur_complement, MatrixXd::Zero(0, 0));
    EXPECT_EQ(solver.solver_mode(),
              BlockSparseCholeskySolver<MatrixXd>::SolverMode::kFactored);
  }
  /* None of the blocks is eliminated. */
  {
    const bool success = solver.CalcSchurComplementAndFactor(
        M, std::unordered_set<int>(), &schur_complement);
    EXPECT_TRUE(success);
    EXPECT_TRUE(CompareMatrices(schur_complement, M.MakeDenseMatrix()));
    EXPECT_EQ(solver.solver_mode(),
              BlockSparseCholeskySolver<MatrixXd>::SolverMode::kFactored);
  }
  /* Some of the blocks are eliminated. */
  {
    std::unordered_set<int> eliminated_blocks = {1, 3};
    const bool success = solver.CalcSchurComplementAndFactor(
        M, eliminated_blocks, &schur_complement);
    EXPECT_TRUE(success);
    const MatrixXd dense = M.MakeDenseMatrix();
    MatrixXd A = MatrixXd::Zero(6, 6);
    A.topLeftCorner(2, 2) = dense.topLeftCorner(2, 2);
    A.bottomRightCorner(4, 4) = dense.block<4, 4>(5, 5);
    MatrixXd D = MatrixXd::Zero(6, 6);
    D.topLeftCorner(3, 3) = dense.block<3, 3>(2, 2);
    D.topRightCorner(3, 3) = dense.block<3, 3>(2, 9);
    D.bottomLeftCorner(3, 3) = dense.block<3, 3>(9, 2);
    D.bottomRightCorner(3, 3) = dense.block<3, 3>(9, 9);
    MatrixXd B = MatrixXd::Zero(6, 6);
    B.topRightCorner(3, 4) = dense.block<3, 4>(2, 5);
    MatrixXd Mhat = MatrixXd::Zero(12, 12);
    Mhat.topLeftCorner(6, 6) = D;
    Mhat.bottomRightCorner(6, 6) = A;
    Mhat.topRightCorner(6, 6) = B;
    Mhat.bottomLeftCorner(6, 6) = B.transpose();
    MatrixXd expected_schur_complement = A - B.transpose() * D.llt().solve(B);
    EXPECT_TRUE(
        CompareMatrices(schur_complement, expected_schur_complement, 1e-14));
    EXPECT_EQ(solver.solver_mode(),
              BlockSparseCholeskySolver<MatrixXd>::SolverMode::kFactored);
  }
}

/* In this test, we make a graph with 8 vertices, {0, 1, ..., 7}, such that
 odd indexed vertices belong to V1 and even indexed vertices belong to V2.
 Within V1 and V2, the block sparsity pattern looks like
    X X | O O O | O O O O | O O O
    X X | O O O | O O O O | O O O
    --- | ----- |---------| -----
    O O | X X X | X X X X | O O O
    O O | X X X | X X X X | O O O
    O O | X X X | X X X X | O O O
    --- | ----- |---------| -----
    O O | X X X | X X X X | O O O
    O O | X X X | X X X X | O O O
    O O | X X X | X X X X | O O O
    O O | X X X | X X X X | O O O
    --- | ----- |---------| -----
    O O | O O O | O O O O | X X X
    O O | O O O | O O O O | X X X
    O O | O O O | O O O O | X X X
 The expected elimination ordering for this block sparsity pattern is
 [0, 3, 2, 1] from pen and paper calculation. 2 is eliminated before 1 because
 when 0 and 3 are eliminated, the degree of 2 is 3 and the degree of 1 is 4.

 The global to local index mapping looks like
  0->0, 2->1, 4->2, 6->3
  1->0, 3->1, 5->2, 7->3.
 Because vertices in V1 appear first in the resulting ordering, the final result
 should be [1, 7, 5, 3, 0, 6, 4, 2].
 We arbitrarily add edges across V1 and V2 (4-5, 4-7, 0-7, 0-5, 2-1) but they do
 not affect the result. */
GTEST_TEST(BlockSparseCholeskySolverTest, ConcatenateMdOrderingWithinGroup) {
  std::vector<std::vector<int>> sparsity;
  sparsity.emplace_back(std::vector<int>{0, 5, 7});
  sparsity.emplace_back(std::vector<int>{1, 2});
  sparsity.emplace_back(std::vector<int>{2, 4});
  sparsity.emplace_back(std::vector<int>{3, 5});
  sparsity.emplace_back(std::vector<int>{4, 5, 7});
  sparsity.emplace_back(std::vector<int>{5});
  sparsity.emplace_back(std::vector<int>{6});
  sparsity.emplace_back(std::vector<int>{7});
  std::vector<int> block_sizes = {2, 2, 3, 3, 4, 4, 3, 3};
  BlockSparsityPattern block_pattern(block_sizes, sparsity);
  const std::unordered_set<int> V1 = {1, 3, 5, 7};
  const std::vector<int> result =
      ConcatenateMdOrderingWithinGroup(block_pattern, V1);
  EXPECT_EQ(result, std::vector<int>({1, 7, 5, 3, 0, 6, 4, 2}));
}

}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
