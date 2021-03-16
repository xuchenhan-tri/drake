#include "drake/multibody/fixed_fem/dev/eigen_conjugate_gradient_solver.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/contact_solvers/sparse_linear_operator.h"
#include "drake/multibody/fixed_fem/dev/eigen_sparse_ldlt_solver.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
namespace {
/* Dimension of the system. */
constexpr int kD = 4;
/* Tolerance for the CG solver. */
const double kTol = 1e-14;

/* Generate an arbitrary kD-by-kD matrix. */
Eigen::Matrix<double, kD, kD> MakeMatrix() {
  Eigen::Matrix<double, kD, kD> B;
  // clang-format off
  B << 31.705908, 62.927869, 93.446683, 65.319711,
       12.200296, 32.078717, 13.422328, 25.142288,
       53.688842, 12.454311, 33.192683, 85.237341,
       14.656643, 42.836689, 13.362649, 85.326259;
  // clang-format on
  return B;
}

/* Generate an arbitrary symmetric positive definite (SPD) matrix. */
Eigen::Matrix<double, kD, kD> MakeSpdMatrix() {
  Eigen::Matrix<double, kD, kD> B = MakeMatrix();
  /* A = Bᵀ * B + ε * I₄ is guaranteed to be symmetric positive definite. */
  return B.transpose() * B + 0.1 * Eigen::Matrix<double, kD, kD>::Identity();
}

/* Generate an arbitrary right hand side vector b. */
Vector<double, kD> MakeRhs() {
  Vector<double, kD> b;
  b << 56.134, 43.548, 23.545, 12.374;
  return b;
}

/* Tests the solution of the linear solve is correct as measure by
 reconstruction errors. */
void TestReconstruction(const LinearSystemSolver<double>& solver) {
  const Vector<double, kD> b = MakeRhs();
  solver.Compute();
  Eigen::Matrix<double, kD, 1> x;
  solver.Solve(b, &x);

  VectorX<double> Ax(kD);
  solver.A().Multiply(x, &Ax);
  EXPECT_LE((Ax - b).norm(), b.norm() * kTol);
}

GTEST_TEST(LinearSystemSolverTest, DoubleDenseSolverTest) {
  /* TODO(xuchenhan-tri) The only reason that we are testing with a
   SparseLinearOperator here is because that is the only linear operator
   available at the moment. Change to a more appropriate dense linear
   operator when something like that lands. */
  const Eigen::SparseMatrix<double> A = MakeSpdMatrix().sparseView();
  contact_solvers::internal::SparseLinearOperator<double> A_op("A", &A);
  EigenConjugateGradientSolver<double> cg(&A_op, kTol);
  TestReconstruction(cg);
  /* EigenSparseLdltSolver doesn't support dense matrices as its name suggests.
   */
}

GTEST_TEST(LinearSystemSolverTest, DoubleSparseSolverTest) {
  Eigen::SparseMatrix<double> A =
      Eigen::Matrix<double, kD, kD>(Vector4<double>(1, 2, 3, 4).asDiagonal())
          .sparseView();
  /* Add a few off-diagonal entries to make A not completely trivial to solve
   while still maintaining the SPD-ness of A. */
  A.coeffRef(0, 1) = 0.1;
  A.coeffRef(1, 0) = 0.1;

  contact_solvers::internal::SparseLinearOperator<double> A_op("A", &A);
  EigenConjugateGradientSolver<double> cg(&A_op, kTol);
  TestReconstruction(cg);
  EigenSparseLdltSolver<double> ldlt(&A_op);
  TestReconstruction(ldlt);
}

/* Tests the derivatives of the solution with respect to the right hand side
 matches the expected values.
 @pre solver.A() and the input matrix `A` are the same linear operator. */
void TestAutoDiffRhs(const Eigen::Matrix<double, kD, kD>& A,
                     const LinearSystemSolver<AutoDiffXd>& solver) {
  /* Build the right hand side b. */
  const Vector<double, kD> b = MakeRhs();
  Vector<AutoDiffXd, kD> b_autodiff;
  math::initializeAutoDiff(b, b_autodiff);
  solver.Compute();
  Eigen::Matrix<AutoDiffXd, kD, 1> x;
  solver.Solve(b_autodiff, &x);
  /* Use LU decomposition to invert the matrix A. */
  const Eigen::FullPivLU<Eigen::Matrix<double, kD, kD>> lu(A);
  for (int i = 0; i < kD; ++i) {
    Vector<double, kD> unit_derivative = Vector<double, kD>::Zero();
    unit_derivative(i) = 1.0;
    /* We are checking the gradient of x = f(b) = A\b w.r.t b, namely, grad =
     df/db. We know that this gradient is such that grad.row(i) = A \ eᵢ, hence
     the check below. */
    const Vector<double, kD> expected_derivatives = lu.solve(unit_derivative);
    EXPECT_TRUE(CompareMatrices(x(i).derivatives(), expected_derivatives,
                                std::numeric_limits<double>::epsilon()));
  }
}

GTEST_TEST(LinearSystemSolver, AutoDiffSolverTestRhs) {
  using T = AutoDiffXd;
  /* Build the operator A. */
  const Eigen::Matrix<double, kD, kD> A = MakeSpdMatrix();
  const Eigen::SparseMatrix<T> A_sparse = A.cast<T>().sparseView();
  contact_solvers::internal::SparseLinearOperator<T> A_op("A", &A_sparse);
  /* Build the solve and solve for x. */
  EigenConjugateGradientSolver<T> cg(&A_op, kTol);
  TestAutoDiffRhs(A, cg);
  EigenSparseLdltSolver<T> ldlt(&A_op);
  TestAutoDiffRhs(A, ldlt);
}

/* Tests the derivatives of the solution with respect to the left hand side
 matches the expected values.
 @pre solver.A() and the input matrix `A` are the same linear operator.
 @pre A = Bᵀ * B + constant. */
void TestAutoDiffLhs(const Eigen::Matrix<AutoDiffXd, kD, kD>& A,
                     const Eigen::Matrix<AutoDiffXd, kD, kD>& B,
                     const LinearSystemSolver<AutoDiffXd>& solver) {
  /* Set up solver and rhs and solve for the solution, x_solver. */
  using T = AutoDiffXd;
  solver.Compute();
  const Vector<double, kD> b = MakeRhs();
  Vector<T, kD> x_solver;
  solver.Solve(b.cast<T>(), &x_solver);

  /* Use LU decomposition to get expected values for A⁻¹, A⁻¹Bᵀ and x. */
  const Eigen::FullPivLU<Eigen::Matrix<T, kD, kD>> lu(A);
  const Eigen::Matrix<T, kD, kD> A_inverse =
      lu.solve(Eigen::Matrix<T, kD, kD>::Identity());
  const Eigen::Matrix<T, kD, kD> A_inverse_BT = lu.solve(B.transpose());
  Vector<T, kD> x = lu.solve(b.cast<T>());
  const Vector<T, kD> Bx = B * x;
  EXPECT_TRUE(CompareMatrices(x, x_solver, kTol));

  for (int i = 0; i < kD; ++i) {
    const Eigen::Matrix<double, kD, kD> xi_derivatives_solver =
        Eigen::Map<const Eigen::Matrix<double, kD, kD>>(
            x_solver(i).derivatives().data(), kD, kD);
    /* Build the expected derivatives of x(i) w.r.t. B:
     x = f(B) = A⁻¹b.
     ∂xᵢ/∂Bⱼₖ = ∂A⁻¹ᵢₗ/∂Bⱼₖbₗ = −A⁻¹ᵢₚ * ∂Aₚₛ/∂Bⱼₖ  * A⁻¹ₛₗbₗ (1).
     Note that A = Bᵀ * B + ε * I₄ and thus  ∂Aₚₛ/∂Bⱼₖ = δₖₚBⱼₛ+δₖₛBⱼₚ.
     Plugging it into (1) and after some algebra we get
     ∂xᵢ/∂Bⱼₖ = -A⁻¹ᵢₖ(Bx)ⱼ - (A⁻¹B)ᵢⱼxₖ. */
    Eigen::Matrix<double, kD, kD> expected_derivatives;
    for (int j = 0; j < kD; ++j) {
      for (int k = 0; k < kD; ++k) {
        expected_derivatives(j, k) =
            -(A_inverse(i, k) * Bx(j) + A_inverse_BT(i, j) * x(k)).value();
      }
    }
    /* The derivatives matches expected value. */
    EXPECT_TRUE(
        CompareMatrices(xi_derivatives_solver, expected_derivatives, kTol));
  }
}

/* Tests the derivatives of the solution with respect to the left hand side
 matches the expected values. */
GTEST_TEST(EigenConjugateGradientSolverTest, AutoDiffSolverTestLhs) {
  /* We take care to make sure the derivatives of A is compatible with the fact
  that A remains SPD. In particular, we cannot initialize the derivatives with
  following code block:
  ```
  const Eigen::Matrix<double, kD, kD> A = MakeSpdMatrix();
  Eigen::Matrix<T, kD, kD> A_autodiff;
  math::initializeAutoDiff(A, A_autodiff);
  ```
  because then dA is not guaranteed to be symmetric. */
  using T = AutoDiffXd;
  const Eigen::Matrix<double, kD, kD> B = MakeMatrix();
  Eigen::Matrix<T, kD, kD> B_autodiff;
  math::initializeAutoDiff(B, B_autodiff);
  /* A = Bᵀ * B + ε * I₄ is guaranteed to be symmetric positive definite. */
  Eigen::Matrix<T, kD, kD> A_autodiff =
      B_autodiff.transpose() * B_autodiff +
      Eigen::Matrix<T, kD, kD>::Identity() * 0.1;
  /* Set up solver and rhs and solve for the solution, x_solver. */
  const Eigen::SparseMatrix<T> A_sparse = A_autodiff.sparseView();
  contact_solvers::internal::SparseLinearOperator<T> A_op("A", &A_sparse);
  EigenConjugateGradientSolver<T> cg(&A_op, kTol);
  TestAutoDiffLhs(A_autodiff, B_autodiff, cg);
  EigenSparseLdltSolver<T> ldlt(&A_op);
  TestAutoDiffLhs(A_autodiff, B_autodiff, ldlt);
}
}  // namespace
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
