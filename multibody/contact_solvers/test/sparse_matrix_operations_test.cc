#include "drake/common/drake_assert.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <gtest/gtest.h>
#include <unsupported/Eigen/IterativeSolvers>

#include <iostream>
#define PRINT_VAR(a) std::cout << #a": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a":\n" << a << std::endl;


using Eigen::SparseMatrix;
using SparseMatrixd = SparseMatrix<double>;
using SparseVectord = Eigen::SparseVector<double>;
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using Triplet = Eigen::Triplet<double>;

SparseMatrixd MakeDiagonalSparse(int n, double value) {
  std::vector<Triplet> triplets;
  triplets.reserve(n);
  for (int i = 0; i < n; ++i) {
    triplets.emplace_back(i, i, value);
  }
  SparseMatrixd S(n, n);
  S.setFromTriplets(triplets.begin(), triplets.end());
  return S;
}

// Make a "fake" Jacobian for a set of paticles such that each particle is the
// contact point.
// nc is the number of contact ponts, which in this case coincides with the
// number of particles.
// nv is the number of velocities.
// particle is a vector of size nc. For each ic-th entry, it contains the index
// of the only particle that contributes to the ic-th velocity.
// The resulting matrix is (3nc)x(3np) in size.
SparseMatrixd MakeParticlesJacobian(int nc, int np,
                                    const std::vector<int>& particle) {
  DRAKE_DEMAND(static_cast<int>(particle.size()) == nc);
  const int num_contributing_particles = particle.size();
  const int nnz = 3 * num_contributing_particles;
  std::vector<Triplet> triplets;
  triplets.reserve(nnz);
  for (int ic = 0; ic < nc; ++ic) {
    const int p = particle[ic];
    DRAKE_DEMAND(p < np);
    triplets.emplace_back(3 * ic, 3 * p, 1.0);
    triplets.emplace_back(3 * ic + 1, 3 * p + 1, 1.0);
    triplets.emplace_back(3 * ic + 2, 3 * p + 2, 1.0);
  }
  SparseMatrixd S(3 * nc, 3 * np);
  S.setFromTriplets(triplets.begin(), triplets.end());
  return S;
}

GTEST_TEST(EigenSparse, MakeSomeBasicMatrices) {
  const int kN = 5;
  SparseMatrixd D = MakeDiagonalSparse(kN, 2.0);
  EXPECT_EQ(D.rows(), kN);
  EXPECT_EQ(D.cols(), kN);
  PRINT_VARn(D);
  PRINT_VAR(D.nonZeros());

  SparseMatrixd J = MakeParticlesJacobian(2, 4, {0, 3});
  PRINT_VARn(J);
  PRINT_VAR(J.nonZeros());
}

GTEST_TEST(EigenSparse, MultiplyByDiagonal) {
  const int nc = 3;
  const int np = 5;
  const int nv = 3 * np;
  SparseMatrixd M = MakeDiagonalSparse(nv, 2.0);
  PRINT_VAR(M.nonZeros());

  SparseMatrixd J = MakeParticlesJacobian(nc, np, {0, 3, 1});
  PRINT_VAR(J.rows());
  PRINT_VAR(J.cols());
  PRINT_VARn(J);  
  PRINT_VAR(J.nonZeros());

  SparseMatrixd MxJ = M * J.transpose();
  PRINT_VARn(MxJ);
  PRINT_VAR(MxJ.nonZeros());

  const VectorXd dnc = 1.5 * VectorXd::Ones(3 * nc);
  SparseMatrixd dnc_x_J = dnc.asDiagonal() * J;
  PRINT_VARn(dnc_x_J);
  PRINT_VAR(dnc_x_J.nonZeros());

  const VectorXd dnv = 1.7 * VectorXd::Ones(nv);
  SparseMatrixd dnv_x_JT = dnv.asDiagonal() * J.transpose();
  PRINT_VAR(dnv_x_JT.rows());
  PRINT_VAR(dnv_x_JT.cols());
  PRINT_VARn(dnv_x_JT);
  PRINT_VAR(dnv_x_JT.nonZeros());

  VectorXd xv(nv);
  xv << 1.2, 1.3;
  SparseVectord xv_sparse = xv.sparseView();
  PRINT_VARn(xv_sparse);
  PRINT_VAR(xv_sparse.nonZeros());
  SparseVectord dnv_x_xv_sparse = dnv.asDiagonal() * xv_sparse;
  PRINT_VARn(dnv_x_xv_sparse);
  PRINT_VAR(dnv_x_xv_sparse.nonZeros());
}

GTEST_TEST(EigenSparse, BasisVectors) { 
  SparseVectord ei(5);
  ei.coeffRef(2) = 1.5;
  PRINT_VARn(ei);
  PRINT_VAR(ei.nonZeros());
  PRINT_VAR(*ei.innerIndexPtr());
  // Apparently SparseVector has no ei.outerIndexPtr(), makes sense.
  EXPECT_EQ(ei.outerIndexPtr(), nullptr);
  EXPECT_EQ(ei.innerNonZeroPtr(), nullptr); //ditto, not used.
  PRINT_VAR(*ei.valuePtr());

  // From those prints above the conclusion is that the fastest way to get a
  // basis is by modifying innerIndexPtr() to be equal to i.
  *ei.innerIndexPtr() = 3;
  *ei.valuePtr() = 1.0;
  PRINT_VAR("Generating basis vectors.")
  for (int i = 0; i<ei.size(); ++i) {
    *ei.innerIndexPtr() = i;
    PRINT_VARn(ei);
    EXPECT_EQ(ei.nonZeros(), 1);
  }

  // This will resize.  
  ei.coeffRef(2) = 0;
  ei.coeffRef(3) = 1.5;
  PRINT_VARn(ei);
  PRINT_VAR(ei.nonZeros());  

}

GTEST_TEST(EigenSparse, ResettingNnz) {
  const int nv = 5;
  VectorXd xv(nv);  
  xv.setZero();
  xv << 1.2, 1.3;
  std::cout << "Initial vector:" << std::endl;
  SparseVectord xv_sparse = xv.sparseView();
  PRINT_VAR(xv_sparse.size());
  PRINT_VAR(xv_sparse.nonZeros());
  PRINT_VAR(xv_sparse.outerSize());
  PRINT_VAR(xv_sparse.innerSize());
  PRINT_VARn(xv_sparse);
  EXPECT_EQ(xv_sparse.nonZeros(), 2);

  std::cout << ".resize()" << std::endl;
  xv.resize(nv);
  PRINT_VAR(xv_sparse.size());
  PRINT_VAR(xv_sparse.nonZeros());
  PRINT_VAR(xv_sparse.outerSize());
  PRINT_VAR(xv_sparse.innerSize());
  PRINT_VARn(xv_sparse);

  // Makes nnz = 0 but does not free memory.
  std::cout << ".setZero()" << std::endl;
  xv_sparse.setZero();
  PRINT_VAR(xv_sparse.size());
  PRINT_VAR(xv_sparse.nonZeros());
  PRINT_VAR(xv_sparse.outerSize());
  PRINT_VAR(xv_sparse.innerSize());

  // To free non-zeros, we need to call squeeze.
  std::cout << ".data().squeeze()" << std::endl;
  xv_sparse.data().squeeze();
  PRINT_VAR(xv_sparse.size());
  PRINT_VAR(xv_sparse.nonZeros());
  PRINT_VAR(xv_sparse.outerSize());
  PRINT_VAR(xv_sparse.innerSize());
  PRINT_VARn(xv_sparse);
}
