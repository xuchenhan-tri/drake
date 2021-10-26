#include "gtest/gtest.h"
#include <Eigen/Dense>

#include "conex/debug_macros.h"
#include "drake/multibody/contact_solvers/geodesic_interior_point_method.h"
#include "drake/multibody/contact_solvers/test/sparse_test_data.h"

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

MatrixXd RandomSym(int n) {
  Eigen::MatrixXd M = MatrixXd::Random(n, n);
  return M + M.transpose();
}
constexpr double eps = 1e-9;

template <typename T>
Eigen::MatrixXd BlockDiagonal(int nr, int nc, const std::vector<T>& X) {
  Eigen::MatrixXd Y(nr, nc);
  Y.setZero();
  int r = 0;
  int c = 0;
  for (const auto& xi : X) {
    nr = xi.rows();
    nc = xi.cols();
    Y.block(r, c, nr, nc) = xi;
    r += nr;
    c += nc;
  }
  return Y;
}

inline Eigen::VectorXd Vect(const SpinFactorProduct& x) {
  Eigen::VectorXd y(x.w0.size() * 3);
  int offset = 0;
  for (int i = 0; i < x.w0.size(); i++) {
    y(offset + 2) = x.w0(i);
    y.segment(offset, 2) = x.w1.col(i);
    offset += 3;
  }
  return y;
}

void DoNewtonTests() {
  // int n = 6;
  // int nc = 5;
  // int m = 3 * nc;

  // MatrixXd J = MatrixXd::Random(m, n);
  drake::TestData test_data = drake::FourStacks();
  MatrixXd J = test_data.J;
  MatrixXd M = test_data.M;
  int m = J.rows();
  int n = J.cols();
  int nc = test_data.num_contacts;

  // auto M = RandomSym(n); M = M * M.transpose();
  VectorXd R = RandomSym(m).diagonal();
  R = R.cwiseProduct(R);
  NewtonDirection dir(test_data.Jtriplets, test_data.blocks_of_M,
                      test_data.num_patches, R, nc);

  VectorXd vstar = VectorXd::Random(n, 1);
  VectorXd v_hat = VectorXd::Random(m, 1);
  VectorXd w0(nc);

  SpinFactorProduct ev(nc);
  ev.w0.setConstant(1);
  ev.w1.setConstant(0);
  VectorXd e = Vect(ev);

  SpinFactorProduct w(nc);
  w.w1 = MatrixXd::Random(2, nc);
  w.w0 = 3 * (w.w1.transpose() * w.w1).diagonal().array().sqrt();

  double k = 300;

  for (int i = 0; i < 2; i++) {
    dir.SetW(w);
    auto Qv = dir.Qwsqrt();
    auto Q = BlockDiagonal(Qv.size() * 3, Qv.size() * 3, Qv);

    MatrixXd temp = MatrixXd::Identity(m, m) + Q * R.asDiagonal() * Q;
    MatrixXd W = temp.inverse();
    MatrixXd temp2 = R.cwiseInverse().asDiagonal();
    temp2 += Q * Q;
    MatrixXd W1 = MatrixXd::Identity(m, m) - (Q * (temp2).eval().inverse()) * Q;

    EXPECT_NEAR((W - W1).norm(), 0, eps);

    temp2 = R.asDiagonal().inverse();
    temp2 += Q * Q;
    EXPECT_NEAR(
        (W - (MatrixXd::Identity(m, m) - (Q * (temp2).eval().inverse()) * Q))
            .norm(),
        0, eps);
    EXPECT_NEAR((Q * W * Q - BlockDiagonal(3 * nc, 3 * nc, dir.QWQ())).norm(),
                0, eps);

    MatrixXd S = J.transpose() * Q * W * Q * J;
    MatrixXd b =
        M * vstar +
        J.transpose() * Q *
            (k * e + W * (k * e - Q * (-v_hat + k * R.asDiagonal() * Q * (e))));

    Eigen::LLT<Eigen::MatrixXd> llt(M + S);
    VectorXd v = llt.solve(b);
    VectorXd d =
        W * (e - 1.0 / k * Q * (J * v - v_hat + k * R.asDiagonal() * Q * (e)));

    VectorXd dcalc;
    VectorXd vcalc;
    dir.CalculateNewtonDirection(1.0 / k, vstar, v_hat, &dcalc, &vcalc);
    EXPECT_NEAR((d - dcalc).norm(), 0, eps);

    VectorXd res1 = M * (v - vstar) - k * J.transpose() * Q * (e + d);
    VectorXd res2 =
        Q * (J * v - v_hat + R.asDiagonal() * k * Q * (e + d)) - k * (e - d);

    EXPECT_NEAR(res1.norm(), 0, eps);
    EXPECT_NEAR(res2.norm(), 0, eps);

    double norminf = NormInf(SpinFactorProduct{d, nc});
    double scale = 2.0 / (norminf * norminf);
    if (scale < 1) {
      d.array() *= scale;
    }

    SpinFactorProduct newton_dir(d, nc);
    TakeStep(w, newton_dir, &w);
  }
}

GTEST_TEST(GeodesicSolver, NewtonTests) {
  std::srand(0);
  for (int i = 0; i < 3; i++) {
    DoNewtonTests();
  }
}
