#define CONEX_ENABLE_TIMER 1
#include "drake/multibody/contact_solvers/supernodal_solver.h"

namespace conex {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::get;
using std::vector;

using RowData = vector<std::pair<MatrixXd, vector<int>>>;

MatrixXd RandomPSD(int n) {
  MatrixXd M = MatrixXd::Random(n, n);
  return M * M.transpose() + Eigen::MatrixXd::Identity(n, n);
}

void DoMain() {
  int num_blocks = 500;
  int size_blocks = 6;
  int size_row_blocks_J = 3;

  std::vector<Eigen::MatrixXd> blocks_of_W(num_blocks);
  std::vector<Eigen::MatrixXd> blocks_of_M(num_blocks);
  std::vector<BlockMatrixTriplet> Jtriplets(num_blocks);
  for (int i = 0; i < num_blocks; i++) {
    blocks_of_W.at(i) = RandomPSD(size_row_blocks_J);
    blocks_of_M.at(i) = RandomPSD(size_blocks);
    get<0>(Jtriplets.at(i)) = i;
    get<1>(Jtriplets.at(i)) = i;
    get<2>(Jtriplets.at(i)) = MatrixXd::Random(size_row_blocks_J, size_blocks);
  }

#if 0
  RowData diagonal_blocks;
  vector<RowData> row_data(num_blocks);
  int offset = 0;
  for (size_t i = 0; i < num_blocks; i++) {
    vector<int> cols(size_blocks);
    for (int k = 0; k < size_blocks; k++) {
      cols.at(k) = k + offset;
    }
    offset += size_blocks;
    diagonal_blocks.emplace_back(MatrixXd::Random(size_blocks, size_blocks),
                                 cols);
    row_data.at(i).emplace_back(
        MatrixXd::Random(size_row_blocks_J, size_blocks), cols);
  }

  START_TIMER(SupernodalOldSetup)
  SuperNodalSolver solver(diagonal_blocks, row_data);
  END_TIMER
#endif

#if CONEX_ENABLE_TIMER
  START_TIMER(SupernodalSetup)
  SuperNodalSolver solver(num_blocks, Jtriplets, blocks_of_M);
  END_TIMER
#endif

  SuperNodalSolver solver(num_blocks, Jtriplets, blocks_of_M);
  START_TIMER(SupernodalSetWeight)
  solver.SetWeightMatrix(blocks_of_W);
  END_TIMER

  MatrixXd y_ref(num_blocks * size_blocks, 1);
#if 0
  MatrixXd T_ref = solver.FullMatrix();
  MatrixXd x_ref = MatrixXd::Random(T_ref.rows(), 1);
  MatrixXd y_ref = T_ref * x_ref;
  START_TIMER(LLT_Dense)
  Eigen::LLT<MatrixXd> llt(T_ref);
  llt.solve(y_ref);
  END_TIMER
#endif

  START_TIMER(LLT_Supernodal)
  solver.Factor();
  solver.Solve(y_ref);
  END_TIMER
}

}  // namespace conex

int main() {
  for (int i = 0; i < 100; i++) {
    conex::DoMain();
    std::cout << "\n";
  }
  return 0;
}
