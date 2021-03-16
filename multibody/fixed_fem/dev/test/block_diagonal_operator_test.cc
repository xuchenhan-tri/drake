#include "drake/multibody/fixed_fem/dev/block_diagonal_operator.h"

#include <memory>

#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/multibody/contact_solvers/sparse_linear_operator.h"
#include "drake/multibody/fixed_fem/dev/eigen_conjugate_gradient_solver.h"
#include "drake/multibody/fixed_fem/dev/inverse_operator.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

using SparseMatrixd = Eigen::SparseMatrix<double>;
using SparseVectord = Eigen::SparseVector<double>;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using fixed_fem::internal::EigenConjugateGradientSolver;
/* Rows and cols of the test matrix A, which is half of the size of the block
 diagonal operator. */
constexpr int kD = 4;
const double kTol = std::numeric_limits<double>::epsilon();

/* Returns an arbitrary SPD matrix. */
MatrixXd MakeTestMatrix() {
  MatrixXd B(kD, kD);
  // clang-format off
  B << 1, 2, 0, 2,
       0, 0, 9, 0,
       5, 0, 0, 1,
       4, 5, 0, 0;
  // clang-format on
  return B.transpose() * B + MatrixXd::Identity(kD, kD);
}

class BlockDiagonalOperatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    A_ = MakeTestMatrix();
    A_sparse_ = A_.sparseView();
    block1_ =
        std::make_unique<SparseLinearOperator<double>>("block1", &A_sparse_);
    cg_ = std::make_unique<EigenConjugateGradientSolver<double>>(block1_.get());
    block2_ = std::make_unique<InverseOperator<double>>("block2", cg_.get());
    std::vector<const LinearOperator<double>*> operators = {block1_.get(),
                                                            block2_.get()};
    block_diagonal_operator_ = std::make_unique<BlockDiagonalOperator<double>>(
        "block_operator", operators);
  }

  const MatrixXd& get_A() const { return A_; }

  MatrixXd A_;
  SparseMatrixd A_sparse_;
  std::unique_ptr<SparseLinearOperator<double>> block1_;
  std::unique_ptr<EigenConjugateGradientSolver<double>> cg_;
  std::unique_ptr<InverseOperator<double>> block2_;
  std::unique_ptr<BlockDiagonalOperator<double>> block_diagonal_operator_;
};

TEST_F(BlockDiagonalOperatorTest, Construction) {
  EXPECT_EQ(block_diagonal_operator_->name(), "block_operator");
  EXPECT_EQ(block_diagonal_operator_->rows(), 2 * kD);
  EXPECT_EQ(block_diagonal_operator_->cols(), 2 * kD);
}

TEST_F(BlockDiagonalOperatorTest, MultiplyDense) {
  VectorXd y1(kD);
  VectorXd y2(kD);
  VectorXd y(2 * kD);
  const VectorXd x1 = VectorXd::LinSpaced(kD, 0.0, 1.0);
  const VectorXd x2 = VectorXd::LinSpaced(kD, 0.0, 2.0);
  VectorXd x(2 * kD);
  x << x1, x2;
  block_diagonal_operator_->Multiply(x, &y);
  block1_->Multiply(x1, &y1);
  block2_->Multiply(x2, &y2);

  VectorXd y_expected(2 * kD);
  y_expected << y1, y2;
  EXPECT_TRUE(CompareMatrices(y, y_expected, kTol));
}

TEST_F(BlockDiagonalOperatorTest, MultiplyByTransposeDense) {
  VectorXd y(2 * kD);
  const VectorXd x = VectorXd::LinSpaced(2 * kD, 0.0, 1.0);
  DRAKE_EXPECT_THROWS_MESSAGE(
      block_diagonal_operator_->MultiplyByTranspose(x, &y), std::exception,
      "DoMultiplyByTranspose\\(\\): Instance 'block_operator' of type "
      "'drake::multibody::contact_solvers::internal::BlockDiagonalOperator<"
      "double>' must provide an implementation.");
}

TEST_F(BlockDiagonalOperatorTest, MultiplySparse) {
  const VectorXd x1_dense = VectorXd::LinSpaced(kD, 0.0, 1.0);
  const VectorXd x2_dense = VectorXd::LinSpaced(kD, 0.0, 2.0);
  VectorXd x_dense(2 * kD);
  x_dense << x1_dense, x2_dense;

  const SparseVectord x1 = x1_dense.sparseView();
  const SparseVectord x2 = x2_dense.sparseView();
  const SparseVectord x = x_dense.sparseView();

  SparseVectord y1(kD);
  SparseVectord y2(kD);
  SparseVectord y(2 * kD);
  block_diagonal_operator_->Multiply(x, &y);
  block1_->Multiply(x1, &y1);
  block2_->Multiply(x2, &y2);

  VectorXd y_expected(2 * kD);
  y_expected << VectorXd(y1), VectorXd(y2);
  EXPECT_TRUE(CompareMatrices(VectorXd(y), y_expected, kTol));
}

TEST_F(BlockDiagonalOperatorTest, MultiplyByTransposeSparse) {
  SparseVectord y(2 * kD);
  const SparseVectord x = VectorXd::LinSpaced(2 * kD, 0.0, 1.0).sparseView();
  DRAKE_EXPECT_THROWS_MESSAGE(
      block_diagonal_operator_->MultiplyByTranspose(x, &y), std::exception,
      "DoMultiplyByTranspose\\(\\): Instance 'block_operator' of type "
      "'drake::multibody::contact_solvers::internal::BlockDiagonalOperator<"
      "double>' must provide an implementation.");
}

TEST_F(BlockDiagonalOperatorTest, AssembleMatrix) {
  /* Assemble the block diagnomal matrix. */
  SparseMatrixd block_diagonal_matrix_sparse(2 * kD, 2 * kD);
  block_diagonal_operator_->AssembleMatrix(&block_diagonal_matrix_sparse);
  const MatrixXd block_diagonal_matrix(block_diagonal_matrix_sparse);

  /* Assemble the individual blocks. */
  SparseMatrixd block1_matrix_sparse(kD, kD);
  SparseMatrixd block2_matrix_sparse(kD, kD);
  block1_->AssembleMatrix(&block1_matrix_sparse);
  block2_->AssembleMatrix(&block2_matrix_sparse);
  MatrixXd expected_block_diagonal_matrix = MatrixXd::Zero(2 * kD, 2 * kD);
  expected_block_diagonal_matrix.topLeftCorner(kD, kD) =
      MatrixXd(block1_matrix_sparse);
  expected_block_diagonal_matrix.bottomRightCorner(kD, kD) =
      MatrixXd(block2_matrix_sparse);
  /* The two matrices should be equal exactly. */
  EXPECT_TRUE(CompareMatrices(block_diagonal_matrix,
                              expected_block_diagonal_matrix, 0));
}
}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
