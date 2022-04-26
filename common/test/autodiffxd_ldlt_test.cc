#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace test {
namespace {

GTEST_TEST(AutoDiffXdLdltTest, derivatives) {
  using Eigen::VectorXd;
  using std::cout;
  using std::endl;

  constexpr int kDerivatives = 5;
  constexpr int kNumDofs = 2;
  VectorX<AutoDiffXd> b(kNumDofs);
  b << 0, -6.00689e-17;

  VectorXd db0(kDerivatives);
  db0 << 0, 0, 11, -0.5, 0.1;
  VectorXd db1(kDerivatives);
  db1 << 0, 0.4905, -0.5, 0.25, 0;
  b[0].derivatives() = db0;
  b[1].derivatives() = db1;

  cout << "db: " << endl;
  cout << math::ExtractGradient(b) << endl;
  cout << endl;

  Matrix2<AutoDiffXd> A;
  A << 11, -0.5, -0.5, 0.25;

  // VectorXd dA10(kDerivatives);
  // dA10 << 0, -6.12323e-17, 0, 0, 0;
  // VectorXd dA01 = dA10;
  // A(0, 0).derivatives() = VectorXd::Zero(5);
  // A(1, 1).derivatives() = VectorXd::Zero(5);
  // A(0, 1).derivatives() = dA01;
  // A(1, 0).derivatives() = dA10;
  Vector2<AutoDiffXd> x = A.ldlt().solve(b);
  Vector2<AutoDiffXd> Ax = A * x;
  MatrixX<AutoDiffXd> dAx = math::ExtractGradient(A * x);
  MatrixX<AutoDiffXd> db = math::ExtractGradient(b);
  EXPECT_TRUE(CompareMatrices(dAx, db, 1e-15));
}

}  // namespace
}  // namespace test
}  // namespace drake
