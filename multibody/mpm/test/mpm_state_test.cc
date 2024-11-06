#include "../mpm_state.h"

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;
using multibody::contact_solvers::internal::Block3x3SparseSymmetricMatrix;
using multibody::contact_solvers::internal::BlockSparsityPattern;

/* Adds a particle at position x0 to `particles`. All other data are arbitrary.
 */
void AddParticle(Particles<double>* particles, Vector3d x0) {
  particles->data.x.push_back(x0);
  Matrix3d F0 =
      (Matrix3d() << 1.0, 0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 1.0).finished();
  particles->data.F.push_back(F0);
  particles->data.m.push_back(1.0);
  particles->data.v.push_back(Vector3d(0.0, 0.0, 0.0));
  particles->data.C.push_back(Matrix3d::Zero());
  particles->data.tau_v0.push_back(Matrix3d::Zero());
}

GTEST_TEST(MpmStateTest, MakeTangentMatrix) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  Particles<double> particles;
  /* A single particle produces a fully-connected graph of 27 nodes. */
  const Vector3d x0 = Vector3d(dx, dx, dx);
  AddParticle(&particles, x0);
  const double dt = 0.02;
  {
    MpmState<double> state(dt, &grid, &particles);
    const Block3x3SparseSymmetricMatrix tangent_matrix =
        state.MakeTangentMatrix();
    const BlockSparsityPattern& sparsity_pattern =
        tangent_matrix.sparsity_pattern();
    EXPECT_EQ(sparsity_pattern.block_sizes(), std::vector<int>(27, 3));
    const std::vector<std::vector<int>>& neighbors =
        sparsity_pattern.neighbors();
    ASSERT_EQ(neighbors.size(), 27);
    for (int i = 0; i < 27; ++i) {
      std::vector<int> expected_neighbors;
      for (int j = i; j < 27; ++j) {
        expected_neighbors.push_back(j);
      }
      EXPECT_EQ(neighbors[i], expected_neighbors);
    }
    /* The number of non-zero blocks in a fully-connected graph with n nodes is
     n*(n+1)/2. n = 27 and each block as 3x3 = 9 entries. */
    EXPECT_EQ(sparsity_pattern.CalcNumNonzeros(), 27 * 28 / 2 * 9);
  }

  /* Now we add a new particle to activate another pad that shares a 3x3 face
   with the existing pad.*/
  AddParticle(&particles, x0 + Vector3d(2.0 * dx, 0.0, 0.0));
  {
    MpmState<double> state(dt, &grid, &particles);
    const Block3x3SparseSymmetricMatrix tangent_matrix =
        state.MakeTangentMatrix();
    const BlockSparsityPattern& sparsity_pattern =
        tangent_matrix.sparsity_pattern();
    /* The total number of nodes is 27*2 - 9 (9 shared nodes between the two
     pads).*/
    EXPECT_EQ(sparsity_pattern.block_sizes(), std::vector<int>(45, 3));
    /* It's a bit too complicated to enumerate all the neighbors in this setup.
     Instead, we confirm that the total number of non-zero entries is correct.
     The total number of edges should be doubled except the 3x3 node clique in
     the overlap region is double counted, so we remove those. */
    EXPECT_EQ(sparsity_pattern.CalcNumNonzeros(), (27 * 28 - 9 * 10 / 2) * 9);
  }

  /* Finally we add another particle to activate the pad straddled between the
   two existing pads. */
  AddParticle(&particles, x0 + Vector3d(dx, 0.0, 0.0));
  {
    MpmState<double> state(dt, &grid, &particles);
    const Block3x3SparseSymmetricMatrix tangent_matrix =
        state.MakeTangentMatrix();
    const BlockSparsityPattern& sparsity_pattern =
        tangent_matrix.sparsity_pattern();
    /* The total number of nodes is 27*3 - 18 * 2 (18 shared nodes between two
     consecutive pads).*/
    EXPECT_EQ(sparsity_pattern.block_sizes(),
              std::vector<int>(27 * 3 - 18 * 2, 3));
    /* The total number of edges should be equal to that of 3 fully connected
     pads minus the two double-counted clique in the overlap of two consecutive
     pads (with 18 nodes). */
    EXPECT_EQ(sparsity_pattern.CalcNumNonzeros(),
              (27 * 28 / 2 * 3 - 18 * 19) * 9);
  }
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
