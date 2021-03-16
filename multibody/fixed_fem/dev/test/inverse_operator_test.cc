#include "drake/multibody/fixed_fem/dev/inverse_operator.h"

#include <memory>

#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/multibody/contact_solvers/sparse_linear_operator.h"
#include "drake/multibody/fixed_fem/dev/eigen_conjugate_gradient_solver.h"

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
/* Rows and cols of the test matrix. */
constexpr int kD = 4;
const double kTol = 1e-14;

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

class InverseOperatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    A_ = MakeTestMatrix();
    A_sparse_ = A_.sparseView();
    A_op_ = std::make_unique<SparseLinearOperator<double>>("A", &A_sparse_);
    cg_ = std::make_unique<EigenConjugateGradientSolver<double>>(A_op_.get());
    Ainv_ = std::make_unique<InverseOperator<double>>("Ainv", cg_.get());
  }

  const MatrixXd& get_A() const { return A_; }

  MatrixXd A_;
  SparseMatrixd A_sparse_;
  std::unique_ptr<SparseLinearOperator<double>> A_op_;
  std::unique_ptr<EigenConjugateGradientSolver<double>> cg_;
  std::unique_ptr<InverseOperator<double>> Ainv_;
};

TEST_F(InverseOperatorTest, Construction) {
  EXPECT_EQ(Ainv_->name(), "Ainv");
  EXPECT_EQ(Ainv_->rows(), kD);
  EXPECT_EQ(Ainv_->cols(), kD);
}

TEST_F(InverseOperatorTest, MultiplyDense) {
  VectorXd y(kD);
  const VectorXd x = VectorXd::LinSpaced(kD, 0.0, 1.0);
  Ainv_->Multiply(x, &y);

  const MatrixXd& A = get_A();
  const VectorXd y_expected = A.fullPivLu().solve(x);
  EXPECT_TRUE(CompareMatrices(y, y_expected, kTol));
}

TEST_F(InverseOperatorTest, MultiplyByTransposeDense) {
  VectorXd y(kD);
  const VectorXd x = VectorXd::LinSpaced(kD, 0.0, 1.0);
  DRAKE_EXPECT_THROWS_MESSAGE(
      Ainv_->MultiplyByTranspose(x, &y), std::exception,
      "DoMultiplyByTranspose\\(\\): Instance 'Ainv' of type "
      "'drake::multibody::contact_solvers::internal::InverseOperator<double>' "
      "must provide an implementation.");
}

TEST_F(InverseOperatorTest, MultiplySparse) {
  SparseVectord y(kD);
  const SparseVectord x = VectorXd::LinSpaced(kD, 0.0, 1.0).sparseView();
  Ainv_->Multiply(x, &y);

  const MatrixXd& A = get_A();
  const VectorXd y_expected = A.fullPivLu().solve(VectorXd(x));
  EXPECT_TRUE(CompareMatrices(VectorXd(y), y_expected, kTol));
}

TEST_F(InverseOperatorTest, MultiplyByTransposeSparse) {
  SparseVectord y(kD);
  const SparseVectord x = VectorXd::LinSpaced(kD, 0.0, 1.0).sparseView();
  DRAKE_EXPECT_THROWS_MESSAGE(
      Ainv_->MultiplyByTranspose(x, &y), std::exception,
      "DoMultiplyByTranspose\\(\\): Instance 'Ainv' of type "
      "'drake::multibody::contact_solvers::internal::InverseOperator<double>' "
      "must provide an implementation.");
}

TEST_F(InverseOperatorTest, AssembleMatrix) {
  SparseMatrixd Ainv_matrix_sparse(kD, kD);
  Ainv_->AssembleMatrix(&Ainv_matrix_sparse);
  MatrixXd expected_Ainv_matrix =
      A_.fullPivLu().solve(MatrixXd::Identity(kD, kD));
  EXPECT_TRUE(CompareMatrices(MatrixXd(Ainv_matrix_sparse),
                              expected_Ainv_matrix, kTol));
}
}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
