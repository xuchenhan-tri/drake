#include <iostream>

#include "drake/common/nice_type_name.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <gtest/gtest.h>
#include <unsupported/Eigen/IterativeSolvers>

#include <iostream>
#define PRINT_VAR(a) std::cout << #a": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a":\n" << a << std::endl;


class MatrixReplacement;
using Eigen::SparseMatrix;

namespace Eigen {
namespace internal {
// MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
template <>
struct traits<MatrixReplacement>
    : public Eigen::internal::traits<Eigen::SparseMatrix<double> > {};
}  // namespace internal
}  // namespace Eigen

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement> {
 public:
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  Index rows() const { return mp_mat->rows(); }
  Index cols() const { return mp_mat->cols(); }

  template <typename Rhs>
  Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct> operator*(
      const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct>(
        *this, x.derived());
  }

  // Custom API:
  MatrixReplacement() : mp_mat(0) {}

  void attachMyMatrix(const SparseMatrix<double>& mat) { mp_mat = &mat; }
  const SparseMatrix<double> my_matrix() const { return *mp_mat; }

 private:
  const SparseMatrix<double>* mp_mat;
};

// Implementation of MatrixReplacement * Eigen::DenseVector though a
// specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {

template <typename Rhs>
struct generic_product_impl<MatrixReplacement, Rhs, SparseShape, DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<MatrixReplacement, Rhs,
                                generic_product_impl<MatrixReplacement, Rhs> > {
  typedef typename Product<MatrixReplacement, Rhs>::Scalar Scalar;

  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const MatrixReplacement& lhs,
                            const Rhs& rhs, const Scalar& alpha) {
    // This method should implement "dst += alpha * lhs * rhs" inplace,
    // however, for iterative solvers, alpha is always equal to 1, so let's not
    // bother about it.
    assert(alpha == Scalar(1) && "scaling is not implemented");
    EIGEN_ONLY_USED_FOR_DEBUG(alpha);

    // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
    // but let's do something fancier (and less efficient):
    for (Index i = 0; i < lhs.cols(); ++i)
      dst += rhs(i) * lhs.my_matrix().col(i);
  }
};

}  // namespace internal
}  // namespace Eigen

GTEST_TEST(EigenTest, MatrixFree) {
  int n = 10;
  Eigen::SparseMatrix<double> S =
      Eigen::MatrixXd::Random(n, n).sparseView(0.5, 1);
  S = S.transpose() * S;

  MatrixReplacement A;
  A.attachMyMatrix(S);

  Eigen::VectorXd b(n), x;
  b.setRandom();

  // Solve Ax = b using various iterative solver with matrix-free version:
  {
    Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper,
                             Eigen::IdentityPreconditioner>
        cg;
    cg.compute(A);
    x = cg.solve(b);
    std::cout << "CG:       #iterations: " << cg.iterations()
              << ", estimated error: " << cg.error() << std::endl;
  }

  {
    Eigen::BiCGSTAB<MatrixReplacement, Eigen::IdentityPreconditioner> bicg;
    bicg.compute(A);
    x = bicg.solve(b);
    std::cout << "BiCGSTAB: #iterations: " << bicg.iterations()
              << ", estimated error: " << bicg.error() << std::endl;
  }

  {
    Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner> gmres;
    gmres.compute(A);
    x = gmres.solve(b);
    std::cout << "GMRES:    #iterations: " << gmres.iterations()
              << ", estimated error: " << gmres.error() << std::endl;
  }

  {
    Eigen::DGMRES<MatrixReplacement, Eigen::IdentityPreconditioner> gmres;
    gmres.compute(A);
    x = gmres.solve(b);
    std::cout << "DGMRES:   #iterations: " << gmres.iterations()
              << ", estimated error: " << gmres.error() << std::endl;
  }

  {
    Eigen::MINRES<MatrixReplacement, Eigen::Lower | Eigen::Upper,
                  Eigen::IdentityPreconditioner>
        minres;
    minres.compute(A);
    x = minres.solve(b);
    std::cout << "MINRES:   #iterations: " << minres.iterations()
              << ", estimated error: " << minres.error() << std::endl;
  }

// This does not even compile because things like isCompressed() are not defined
// for MatrixReplacement.
#if 0
  {
    Eigen::SparseLU<MatrixReplacement> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    x = solver.solve(b);
  }
#endif
}

GTEST_TEST(SparseLU, DenseRhs) {
  int n = 10;
  Eigen::SparseMatrix<double> S =
      Eigen::MatrixXd::Random(n, n).sparseView(0.5, 1);
  S = S.transpose() * S;
  Eigen::VectorXd b(n), x;
  Eigen::SparseVector<double> xs;
  b.setRandom();

  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.analyzePattern(S);
  solver.factorize(S);
  const Eigen::Solve<Eigen::SparseLU<Eigen::SparseMatrix<double>>,
                     Eigen::VectorXd>
      solver_output = solver.solve(b);
  // Result of type Eigen::Solve<SparseLU<...>, Eigen::VectorXd> copies just
  // fine into a VectorXd.
  x = solver_output;
  // xs = x; // This does not compile, there is no such assignment operator.
  // xs = x.sparseView(); // This does copy x into xs.
  // xs = solver_output; // This assignment does not work.

  std::cout << ::drake::NiceTypeName::Get(x) << std::endl;   // simple baseline.
  std::cout << ::drake::NiceTypeName::Get(xs) << std::endl;  // simple baseline.
  std::cout << ::drake::NiceTypeName::Get(solver_output) << std::endl;

  std::cout << "SparseLU error: " << (S * x - b).norm() << std::endl;
}

GTEST_TEST(SparseLU, SparseRhs) {
  int n = 10;
  Eigen::SparseMatrix<double> S =
      Eigen::MatrixXd::Random(n, n).sparseView(0.5, 1);
  S = S.transpose() * S;
  Eigen::VectorXd b(n), x;
  Eigen::SparseVector<double> xs;
  b.setRandom();
  Eigen::SparseVector<double> bs = b.sparseView();

  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.analyzePattern(S);
  solver.factorize(S);
  const Eigen::Solve<Eigen::SparseLU<Eigen::SparseMatrix<double>>,
                     Eigen::SparseVector<double>>
      solver_output = solver.solve(bs);
  // Result of type Eigen::Solve<SparseLU<...>, Eigen::SparseVector> copies just
  // fine into a SparseVector. Also into a dense vector since assignment from
  // sparse to dense works.
  // x = solver_output; // This compiles just fine.
  xs = solver_output;  // Unlike the dense RHS case, this compiles OK.
  // xs = solver_output; // This assignment does not compile.
  // xs = x; // This does not compile, there is no such assignment operator.
  // xs = x.sparseView(); // This does copy x into xs.
  // x = xs;  // Assignment from sparse to dense is allowed.

  std::cout << ::drake::NiceTypeName::Get(x) << std::endl;   // simple baseline.
  std::cout << ::drake::NiceTypeName::Get(xs) << std::endl;  // simple baseline.
  std::cout << ::drake::NiceTypeName::Get(solver_output) << std::endl;
  std::cout << "SparseLU error: " << (S * xs - bs).norm() << std::endl;
}

// Makes 1D Finite differences Laplacian operator.
Eigen::SparseMatrix<double> MakeLaplacianOperator(int n) {
  std::vector<Eigen::Triplet<double>> non_zeros;
  non_zeros.reserve(n);

  // Inside grid excluding boundaries.
  for (int i = 1; i < n-1; ++i) {
    non_zeros.push_back({i, i - 1, -1.0});
    non_zeros.push_back({i, i, 2.0});
    non_zeros.push_back({i, i + 1, -1.0});
  }

  // Dirichlet BCs, so that the operator is invertible.
  non_zeros.push_back({0, 0, 1.0});
  non_zeros.push_back({n-1, n-1, 1.0});

  Eigen::SparseMatrix<double> A(n, n);
  A.setFromTriplets(non_zeros.begin(), non_zeros.end());

  return A;
}

GTEST_TEST(SparseLU, SparseLaplacianOperator) {
  int n = 10;
  Eigen::SparseMatrix<double> A = MakeLaplacianOperator(n);
  Eigen::SparseVector<double> xs(n);
  xs.coeffRef(n / 2) = 1.0;

  Eigen::SparseVector<double> bs = A * xs;
  PRINT_VAR(xs.nonZeros());
  PRINT_VAR(bs.nonZeros());
  PRINT_VAR(Eigen::VectorXd(bs).transpose());
  PRINT_VAR(bs);

  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.analyzePattern(A);
  solver.factorize(A);
  const Eigen::SparseVector<double> solver_output = solver.solve(bs);
  PRINT_VAR(solver_output.nonZeros());
  PRINT_VAR(Eigen::VectorXd(solver_output).transpose());
  PRINT_VAR(solver_output);
}

GTEST_TEST(SparseLU, SparseLaplacianOperatorMultipleRhs) {
  int n = 10;
  Eigen::SparseMatrix<double> A = MakeLaplacianOperator(n);
  Eigen::SparseMatrix<double> xs = Eigen::MatrixXd::Identity(n, n).sparseView();
  Eigen::SparseMatrix<double> bs = A * xs;

  PRINT_VAR(A.nonZeros());
  PRINT_VAR(xs.nonZeros());
  PRINT_VAR(bs.nonZeros());
  PRINT_VARn(Eigen::MatrixXd(bs));
  PRINT_VARn(bs);
  PRINT_VARn(A);

  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.analyzePattern(A);
  solver.factorize(A);
  const Eigen::SparseMatrix<double> solver_output = solver.solve(bs);
  PRINT_VAR(solver_output.nonZeros());
  PRINT_VARn(Eigen::MatrixXd(solver_output));
  PRINT_VARn(solver_output);
}
