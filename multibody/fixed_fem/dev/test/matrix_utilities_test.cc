#include "drake/multibody/fixed_fem/dev/matrix_utilities.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/fixed_fem/dev/test/test_utilities.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
namespace {
Matrix3<double> MakeArbitraryMatrix() {
  Matrix3<double> A;
  // clang-format off
  A << 1.2, 2.3, 3.4,
       4.5, 5.6, 6.7,
       7.8, 8.9, 9.0;
  // clang-format on
  return A;
}

Matrix3<AutoDiffXd> MakeAutoDiffMatrix() {
  const Matrix3<double> A = MakeArbitraryMatrix();
  Matrix3<AutoDiffXd> A_ad;
  math::initializeAutoDiff(A, A_ad);
  return A_ad;
}

const double kTol = 1e-14;

/* Calculates the absolute tolerance kTol scaled by the condition number of the
 input matrix A. */
double CalcTolerance(const Matrix3<double>& A) {
  return test::CalcConditionNumber<double>(A) * kTol;
}

GTEST_TEST(MatrixUtilitiesTest, PolarDecompose) {
  const Matrix3<double> A = MakeArbitraryMatrix();
  Matrix3<double> R, S;
  PolarDecompose<double>(A, &R, &S);
  /* Tests reconstruction. */
  EXPECT_TRUE(CompareMatrices(A, R * S, CalcTolerance(A)));
  /* Tests symmetry of S. */
  EXPECT_TRUE(CompareMatrices(S, S.transpose(), kTol));
  /* Tests R is a rotation matrix. */
  EXPECT_TRUE(math::RotationMatrix<double>::IsValid(R, kTol));
}

GTEST_TEST(MatrixUtilitiesTest, AddScaledRotationalDerivative) {
  const Matrix3<AutoDiffXd> F = MakeAutoDiffMatrix();
  Matrix3<AutoDiffXd> R, S;
  PolarDecompose<AutoDiffXd>(F, &R, &S);
  Eigen::Matrix<AutoDiffXd, 9, 9> scaled_dRdF =
      Eigen::Matrix<AutoDiffXd, 9, 9>::Zero();
  AutoDiffXd scale = 1.23;
  AddScaledRotationalDerivative<AutoDiffXd>(R, S, scale, &scaled_dRdF);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Matrix3<double> scaled_dRijdF;
      for (int k = 0; k < kSpaceDimension; ++k) {
        for (int l = 0; l < kSpaceDimension; ++l) {
          scaled_dRijdF(k, l) = scaled_dRdF(3 * j + i, 3 * l + k).value();
        }
      }
      EXPECT_TRUE(
          CompareMatrices(scale * Eigen::Map<const Matrix3<double>>(
                                      R(i, j).derivatives().data(), 3, 3),
                          scaled_dRijdF, CalcTolerance(scaled_dRijdF)));
    }
  }
}

GTEST_TEST(MatrixUtilitiesTest, CalcCofactorMatrix) {
  const Matrix3<double> A = MakeArbitraryMatrix();
  Matrix3<double> C;
  CalcCofactorMatrix<double>(A, &C);
  EXPECT_TRUE(CompareMatrices(
      A.inverse(), 1.0 / A.determinant() * C.transpose(), CalcTolerance(A)));
}

GTEST_TEST(MatrixUtilitiesTest, AddScaledCofactorMatrixDerivative) {
  const Matrix3<AutoDiffXd> A = MakeAutoDiffMatrix();
  Matrix3<AutoDiffXd> C;
  CalcCofactorMatrix<AutoDiffXd>(A, &C);
  Eigen::Matrix<AutoDiffXd, 9, 9> scaled_dCdA =
      Eigen::Matrix<AutoDiffXd, 9, 9>::Zero();
  AutoDiffXd scale = 1.23;
  AddScaledCofactorMatrixDerivative<AutoDiffXd>(A, scale, &scaled_dCdA);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Matrix3<double> scaled_dCijdA;
      for (int k = 0; k < kSpaceDimension; ++k) {
        for (int l = 0; l < kSpaceDimension; ++l) {
          scaled_dCijdA(k, l) = scaled_dCdA(3 * j + i, 3 * l + k).value();
        }
      }
      EXPECT_TRUE(
          CompareMatrices(scale * Eigen::Map<const Matrix3<double>>(
                                      C(i, j).derivatives().data(), 3, 3),
                          scaled_dCijdA, CalcTolerance(scaled_dCijdA)));
    }
  }
}

/* Tests that given a sparse matrix A, if we convert it to a vector of triplets,
 then the vector of triplets can reconstruct a sparse matrix B that is bitwise
 equal to the original matrix A. */
GTEST_TEST(MatrixUtilitiesTest, ConvertEigenSparseMatrixToTriplets) {
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.emplace_back(0, 0, 7);
  triplets.emplace_back(1, 3, 4);
  triplets.emplace_back(2, 2, 8);
  Eigen::SparseMatrix<double> A(4, 4);
  A.setFromTriplets(triplets.begin(), triplets.end());
  const auto expected_triplets = ConvertEigenSparseMatrixToTriplets(A);
  Eigen::SparseMatrix<double> B(4, 4);
  B.setFromTriplets(expected_triplets.begin(), expected_triplets.end());
  // We verify the B is an exact bit by bit copy of A.
  // Eigen does not offer SparseMatrix::operator==() and therefore we compare
  // the results by explicitly comparing the individual components of the CCS
  // format.
  Eigen::Map<VectorX<double>> A_values(A.valuePtr(), A.nonZeros());
  Eigen::Map<VectorX<double>> B_values(B.valuePtr(), B.nonZeros());
  EXPECT_EQ(A_values, B_values);

  Eigen::Map<VectorX<int>> A_inner(A.innerIndexPtr(), A.innerSize());
  Eigen::Map<VectorX<int>> B_inner(B.innerIndexPtr(), B.innerSize());
  EXPECT_EQ(A_inner, B_inner);

  Eigen::Map<VectorX<int>> A_outer(A.outerIndexPtr(), A.outerSize());
  Eigen::Map<VectorX<int>> B_outer(B.outerIndexPtr(), B.outerSize());
  EXPECT_EQ(A_outer, B_outer);
}
}  // namespace
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
