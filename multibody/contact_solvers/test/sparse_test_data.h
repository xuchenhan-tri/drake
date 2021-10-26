#include "drake/multibody/contact_solvers/supernodal_solver.h"
#include <Eigen/Dense>

namespace drake {

struct TestData {
  Eigen::MatrixXd M;
  Eigen::MatrixXd J;
  std::vector<Eigen::MatrixXd> blocks_of_M;
  std::vector<conex::BlockMatrixTriplet> Jtriplets;
  int num_contacts;
  int num_patches;
};

inline TestData FourStacks() {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using conex::BlockMatrixTriplet;
  // For this problem each patch has a single contact point.
  // Therefore there'll be num_patches blocks of W.
  const int num_patches = 8;
  const int num_trees = 8;
  // A typical block of J.
  MatrixXd J3x6(3, 6);
  // clang-format off
  J3x6 << 1, 2, 3, 4, 5, 6,
          1, 2, 3, 4, 5, 6,
          1, 2, 3, 4, 5, 6;
  // clang-format on
  const MatrixXd Z3x6 = MatrixXd::Zero(3, 6);
  // Three patches:
  //  - Patch 1: 1 point.
  //  - Patch 2: 2 points.
  //  - Patch 3: 1 point.
  // Total of 12 rows.
  // Three trees of six dofs each = 18.
  // These are the blocks (and they are all of size 3x6):
  // (p,t) = (0,6). 3x6.
  // (p,t) = (0,7). 3x6.
  // (p,t) = (1,4). 3x6.
  // (p,t) = (1,5). 3x6.
  // (p,t) = (2,6). 3x6.
  // (p,t) = (3,0). 3x6.
  // (p,t) = (4,4). 3x6.
  // (p,t) = (5,2). 3x6.
  // (p,t) = (6,0). 3x6.
  // (p,t) = (6,1). 3x6.
  // (p,t) = (7,2). 3x6.
  // (p,t) = (7,3). 3x6.
  Eigen::MatrixXd J(24, 48);
  // clang-format off
  J <<        
       Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, J3x6, J3x6,
       Z3x6, Z3x6, Z3x6, Z3x6, J3x6, J3x6, Z3x6, Z3x6, 
       Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, J3x6, Z3x6, 
       J3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6,
       Z3x6, Z3x6, Z3x6, Z3x6, J3x6, Z3x6, Z3x6, Z3x6,
       Z3x6, Z3x6, J3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6,
       0.6*J3x6, J3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6,
       Z3x6, Z3x6, J3x6, J3x6, Z3x6, Z3x6, Z3x6, Z3x6;
  // clang-format on
  std::vector<BlockMatrixTriplet> Jtriplets;
  // Patch 0:
  Jtriplets.push_back({0, 6, J3x6});
  Jtriplets.push_back({0, 7, J3x6});
  // Patch 1:
  Jtriplets.push_back({1, 4, J3x6});
  Jtriplets.push_back({1, 5, J3x6});
  // Patch 2:
  Jtriplets.push_back({2, 6, J3x6});
  // Patch 3:
  Jtriplets.push_back({3, 0, J3x6});
  // Patch 4:
  Jtriplets.push_back({4, 4, J3x6});
  // Patch 5:
  Jtriplets.push_back({5, 2, J3x6});
  // Patch 6:
  Jtriplets.push_back({6, 0, 0.6 * J3x6});
  Jtriplets.push_back({6, 1, J3x6});
  // Patch 7:
  Jtriplets.push_back({7, 2, J3x6});
  Jtriplets.push_back({7, 3, J3x6});

  const MatrixXd Mt = VectorXd::LinSpaced(6, 1.0, 6.0).asDiagonal();
  Eigen::MatrixXd M = MatrixXd::Zero(6*num_trees, 6*num_trees);
  std::vector<Eigen::MatrixXd> blocks_of_M(num_trees);
  for (int i = 0; i < num_trees; ++i) {
    M.block(6*i, 6*i, 6, 6) = Mt;
    blocks_of_M.at(i) = Mt; 
  }

  TestData data;
  data.blocks_of_M = blocks_of_M;
  data.M = M;
  data.J = J;
  data.Jtriplets = Jtriplets;
  data.num_contacts = 8;
  data.num_patches = num_patches;
  return data;
}


} // namespace drake
