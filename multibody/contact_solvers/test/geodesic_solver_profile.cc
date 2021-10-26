#define CONEX_ENABLE_TIMER 1

#include "drake/multibody/contact_solvers/geodesic_interior_point_method.h"

#include "drake/multibody/contact_solvers/test/sparse_test_data.h"

namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

MatrixXd RandomSym(int n) {
  Eigen::MatrixXd M = MatrixXd::Random(n, n);
  return M + M.transpose();
}

void DoMain() {
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

  SpinFactorProduct w(nc);
  w.w1 = MatrixXd::Random(2, nc);
  w.w0 = 3 * (w.w1.transpose() * w.w1).diagonal().array().sqrt();

  VectorXd vstar = VectorXd::Random(n, 1);
  VectorXd v_hat = VectorXd::Random(m, 1);
  auto sol = GeodesicSolver(w, test_data.Jtriplets, test_data.blocks_of_M, R,
                            test_data.num_patches, nc, vstar, v_hat);

  DUMP("RESTART");
  auto sol2 = GeodesicSolver(sol, test_data.Jtriplets, test_data.blocks_of_M, R,
                             test_data.num_patches, nc, vstar, v_hat);
}

}  // namespace

int main() {
  for (int i = 0; i < 1; i++) {
    DoMain();
    std::cout << "\n";
  }
  return 0;
}
