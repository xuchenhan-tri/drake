#include "drake/multibody/contact_solvers/schur_complement.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::unique_ptr;
using std::vector;

constexpr double kEps = 2e-13;
// Size of the matrix in this test.
constexpr int kDofs = 9;

// clang-format off
const Matrix3d M00 =
    (Eigen::Matrix3d() << 11, 2, 3,
                          2, 50, 6,
                          3, 6, 190).finished();
const Matrix3d M10 =
    (Eigen::Matrix3d() << 1.1, 1.0, 0.9,
                          1.2, 1.5, 1.6,
                          1.3, 1.8, 1.9).finished();
const Matrix3d M11 =
    (Eigen::Matrix3d() << 110, 12, 13,
                          12, 150, 16,
                          13, 16, 190).finished();
const Matrix3d M20 =
    (Eigen::Matrix3d() << 11, 12, 13,
                          14, 21, 16,
                          17, 18, 29).finished();
const Matrix3d M22 =
    (Eigen::Matrix3d() << 210, 22, 23,
                          22, 205, 26,
                          23, 26, 290).finished();
// clang-format on

/* Makes a block sparse symmetric matrix with 3x3 dense blocks that looks like
   M =   M00 | M01 | M02
        -----------------
         M10 | M11 |  0
        -----------------
         M20 |  0  | M22
where M02 = M20.transpose(). We choose values so that M is diagonally dominant
and thus SPD. */
Block3x3SparseSymmetricMatrix MakeBlockSparseMatrix() {
  std::vector<std::vector<int>> sparsity_pattern;
  sparsity_pattern.emplace_back(vector<int>{0, 1, 2});
  sparsity_pattern.emplace_back(vector<int>{1});
  sparsity_pattern.emplace_back(vector<int>{2});
  BlockSparsityPattern block_pattern({{3, 3, 3}}, std::move(sparsity_pattern));
  Block3x3SparseSymmetricMatrix M(std::move(block_pattern));
  M.SetBlock(0, 0, M00);
  M.SetBlock(1, 0, M10);
  M.SetBlock(2, 0, M20);
  M.SetBlock(1, 1, M11);
  M.SetBlock(2, 2, M22);
  return M;
}

/* Constructs an arbitrary block sparse matrix (see MakeBlockSparseMatrix())

   M =   M00 | M01 | M02
        -----------------
         M10 | M11 |  0
        -----------------
         M20 |  0  | M22

and returns the Schur complement of the M11 block. */
SchurComplement MakeSchurComplement() {
  Block3x3SparseSymmetricMatrix block_sparse_matrix = MakeBlockSparseMatrix();
  const std::unordered_set<int> eliminated_blocks = {1};
  return SchurComplement(block_sparse_matrix, eliminated_blocks);
}

GTEST_TEST(SchurComplementTest, GetDComplement) {
  const SchurComplement schur_complement = MakeSchurComplement();
  const MatrixXd S = schur_complement.get_D_complement();

  MatrixXd A = MatrixXd(6, 6);
  A.topLeftCorner<3, 3>() = M00;
  A.bottomLeftCorner<3, 3>() = M20;
  A.topRightCorner<3, 3>() = M20.transpose();
  A.bottomRightCorner<3, 3>() = M22;

  MatrixXd D = M11;

  MatrixXd B = MatrixXd::Zero(3, 6);
  B.topLeftCorner<3, 3>() = M10;
  const MatrixXd expected = A - B.transpose() * D.llt().solve(B);
  EXPECT_TRUE(CompareMatrices(S, expected, kEps));
}

GTEST_TEST(SchurComplementTest, SolveForX) {
  const SchurComplement schur_complement = MakeSchurComplement();
  const VectorXd y = VectorXd::LinSpaced(6, 0.0, 12.0);
  const VectorXd x = schur_complement.SolveForX(y);

  MatrixXd D = M11;
  MatrixXd B = MatrixXd::Zero(3, 6);
  B.topLeftCorner<3, 3>() = M10;
  /* The system of equation reads
       Dx  + By = 0
       Bᵀx + Ay = a
     Using equation (1), we get
       x = -D⁻¹By */
  const VectorXd expected_x = D.llt().solve(-B * y);
  EXPECT_TRUE(CompareMatrices(x, expected_x, kEps));
}

GTEST_TEST(SchurComplementTest, Solve) {
  const SchurComplement schur_complement = MakeSchurComplement();
  const VectorXd rhs = VectorXd::LinSpaced(9, 0.0, 12.0);
  const VectorXd z = schur_complement.Solve(rhs);

  const MatrixXd M = MakeBlockSparseMatrix().MakeDenseMatrix();
  const VectorXd expected_z = M.llt().solve(rhs);
  EXPECT_TRUE(CompareMatrices(z, expected_z, kEps));
}

}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
