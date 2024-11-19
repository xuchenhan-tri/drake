#include "../mpm_state.h"

#include "../mock_sparse_grid.h"
#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::Vector3i;
using multibody::contact_solvers::internal::Block3x3SparseSymmetricMatrix;
using multibody::contact_solvers::internal::BlockSparsityPattern;

/* Adds a particle at position x0 to `particles`. All other data are arbitrary.
 */
template <typename T>
void AddParticle(Particles<T>* particles,
                 const Eigen::Ref<const Vector3<T>>& x0) {
  particles->data.x.push_back(x0);
  Matrix3<T> F0 =
      (Matrix3<T>() << 1.0, 0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 1.0).finished();
  particles->data.F.push_back(F0);
  particles->data.m.push_back(1.0);
  particles->data.v.push_back(Vector3<T>(0.0, 0.0, 0.0));
  particles->data.C.push_back(Matrix3<T>::Zero());
  particles->data.in_constraint.push_back(false);
  particles->data.volume.push_back(0.01);
  particles->data.tau_v0.push_back(Matrix3<T>::Zero());
}

/* Adds a linear corotated model to all particles.
 @note this function should only be called once in the lifespan of Particles. */
template <typename T>
void AddDefaultMaterial(ParticleData<T>* particle_data) {
  const int num_particles = particle_data->m.size();
  const fem::internal::CorotatedModel<T> model(1e4, 0.45);
  particle_data->constitutive_models.emplace_back(model);
  for (int i = 0; i < num_particles; ++i) {
    particle_data->strain_data.emplace_back(model.MakeDefaultData());
  }
  particle_data->materials.emplace_back(0, num_particles);
}

GTEST_TEST(MpmStateTest, MakeTangentMatrix) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  Particles<double> particles;
  /* A single particle produces a fully-connected graph of 27 nodes. */
  const Vector3d x0 = Vector3d(dx, dx, dx);
  AddParticle<double>(&particles, x0);
  const double dt = 0.02;
  {
    MpmImplicitData<double> data(particles.data);
    MpmState<double> state(dt, &grid, &particles, &data);
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
  AddParticle<double>(&particles, x0 + Vector3d(2.0 * dx, 0.0, 0.0));
  {
    MpmImplicitData<double> data(particles.data);
    MpmState<double> state(dt, &grid, &particles, &data);
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
  AddParticle<double>(&particles, x0 + Vector3d(dx, 0.0, 0.0));
  {
    MpmImplicitData<double> data(particles.data);
    MpmState<double> state(dt, &grid, &particles, &data);
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

GTEST_TEST(MpmStateTest, Residual) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  Particles<double> particles;
  // TODO(xuchenhan-tri): Test that there's actually only a single block.
  /* Add a particle so that all grid nodes touched by this particle are in a
   single grid block. */
  const Vector3d x0 = Vector3d(dx, dx, dx);
  AddParticle<double>(&particles, x0);
  AddDefaultMaterial(&particles.data);
  particles.data.F[0] = Matrix3d::Identity();
  const double dt = 0.02;
  const double kTol = 1e-14;
  MpmImplicitData<double> data(particles.data);

  MpmState<double> state(dt, &grid, &particles, &data);
  /* A single particle activates 27 grid ndoes. */
  const int expected_num_dofs = 27 * 3;
  EXPECT_EQ(state.num_dofs(), expected_num_dofs);
  EXPECT_TRUE(
      CompareMatrices(state.dv(), VectorX<double>::Zero(expected_num_dofs)));
  /* We set b to be an arbitrary value with an arbitrary size to test that the
   function does not crash. */
  VectorX<double> b = VectorX<double>::LinSpaced(42, 0, 1);
  state.CalcResidual(&b);
  EXPECT_TRUE(CompareMatrices(b, VectorX<double>::Zero(expected_num_dofs)));

  /* Give the grid a constant velocity field so that it doesn't induce any
   deformation on the particle. Consequently, the only residual comes from the
   M * dv term. */
  VectorX<double> ddv = VectorX<double>::Ones(expected_num_dofs);
  state.IncrementDv(ddv);
  EXPECT_TRUE(CompareMatrices(state.dv(), ddv));
  state.CalcResidual(&b);
  BsplineWeights<double> weights(x0, dx);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        const double weight = weights.weight(i, j, k);
        /* We make use of the fact that SpGrid follows lexicographical order
         within a block.*/
        const int node_index = i * 9 + j * 3 + k;
        EXPECT_TRUE(CompareMatrices(b.segment<3>(node_index * 3),
                                    weight * ddv.segment<3>(node_index * 3),
                                    kTol));
      }
    }
  }

  /* Reset dv to zero. */
  state.IncrementDv(-ddv);
  EXPECT_TRUE(CompareMatrices(state.dv(),
                              VectorX<double>::Zero(expected_num_dofs), kTol));
  state.CalcResidual(&b);
  EXPECT_TRUE(
      CompareMatrices(b, VectorX<double>::Zero(expected_num_dofs), kTol));

  /* Add a non-constant velocity field and confirm that the residual is no
   longer equal to M * dv, except for the center node; The center node is right
   on top of the particle and because of the xᵢ − xₚ term in computing the
   residual, the contribution to the residual from the particle deformation at
   this node is zero. */
  ddv = VectorX<double>::LinSpaced(expected_num_dofs, 0, 1);
  state.IncrementDv(ddv);
  state.CalcResidual(&b);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        const double weight = weights.weight(i, j, k);
        /* We make use of the fact that SpGrid follows lexicographical order
         within a block.*/
        const int node_index = i * 9 + j * 3 + k;
        if (i == 1 && j == 1 && k == 1) {
          EXPECT_TRUE(CompareMatrices(b.segment<3>(node_index * 3),
                                      weight * ddv.segment<3>(node_index * 3),
                                      kTol));
        } else {
          EXPECT_FALSE(CompareMatrices(b.segment<3>(node_index * 3),
                                       weight * ddv.segment<3>(node_index * 3),
                                       1.0));
        }
      }
    }
  }
}

GTEST_TEST(MpmStateTest, CalcTotalEnergy) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  Particles<double> particles;
  // TODO(xuchenhan-tri): Test that there's actually only a single block.
  /* Add a particle so that all grid nodes touched by this particle are in a
   single grid block. */
  const Vector3d x0 = Vector3d(dx, dx, dx);
  AddParticle<double>(&particles, x0);
  AddDefaultMaterial(&particles.data);
  particles.data.F[0] = Matrix3d::Identity();
  const double dt = 0.02;

  MpmImplicitData<double> data(particles.data);
  MpmState<double> state(dt, &grid, &particles, &data);
  double energy = state.CalcTotalEnergy();
  EXPECT_EQ(energy, 0.0);

  /* A single particle activates 27 grid ndoes. */
  const int expected_num_dofs = 27 * 3;
  EXPECT_EQ(state.num_dofs(), expected_num_dofs);
  /* Arbitrary velocity field. */
  VectorX<double> ddv = VectorX<double>::LinSpaced(expected_num_dofs, 0.0, 1.0);
  state.IncrementDv(ddv);
  const VectorX<double>& dv = state.dv();
  energy = state.CalcTotalEnergy();

  /* This is tested in the ParticleData class. */
  const double potential_energy =
      particles.data.ComputeTotalEnergy(state.data().F);

  double kinetic_energy = 0.0;
  const std::vector<std::pair<Vector3i, GridData<double>>> grid_data =
      grid.GetGridData();
  for (const auto& [node, node_data] : grid_data) {
    const int i = node[0];
    const int j = node[1];
    const int k = node[2];
    /* We make use of the fact that SpGrid follows lexicographical order
     within a block.*/
    const int node_index = i * 9 + j * 3 + k;
    kinetic_energy +=
        0.5 * node_data.m * dv.segment<3>(node_index * 3).squaredNorm();
  }
  EXPECT_DOUBLE_EQ(energy, kinetic_energy + potential_energy);
}

GTEST_TEST(MpmStateTest, ResidualIsDerivativeOfEnergy) {
  const double dx = 0.01;
  MockSparseGrid<AutoDiffXd> grid_ad(dx);
  Particles<AutoDiffXd> particles_ad;
  const Vector3<AutoDiffXd> x0_ad(dx, dx, dx);
  const Vector3<AutoDiffXd> x1_ad(1.1 * dx, 1.2 * dx, 1.3 * dx);
  AddParticle<AutoDiffXd>(&particles_ad, x0_ad);
  AddParticle<AutoDiffXd>(&particles_ad, x1_ad);
  AddDefaultMaterial(&particles_ad.data);
  const AutoDiffXd dt_ad = 0.02;

  SparseGrid<double> grid(dx);
  Particles<double> particles;
  const Vector3<double> x0(dx, dx, dx);
  const Vector3<double> x1(1.1 * dx, 1.2 * dx, 1.3 * dx);
  AddParticle<double>(&particles, x0);
  AddParticle<double>(&particles, x1);
  AddDefaultMaterial(&particles.data);
  const double dt = 0.02;

  const double kTol = 1e-10;

  MpmImplicitData<AutoDiffXd> data_ad(particles_ad.data);
  MpmState<AutoDiffXd, MockSparseGrid> state_ad(dt_ad, &grid_ad, &particles_ad,
                                                &data_ad);
  MpmImplicitData<double> data(particles.data);
  MpmState<double> state(dt, &grid, &particles, &data);
  const int num_dofs = state.num_dofs();
  ASSERT_EQ(num_dofs, state_ad.num_dofs());

  VectorX<double> ddv = VectorX<double>::LinSpaced(num_dofs, 0.0, 1.0);
  VectorX<AutoDiffXd> ddv_ad;
  ddv_ad.resize(num_dofs);
  math::InitializeAutoDiff(ddv, &ddv_ad);

  state_ad.IncrementDv(ddv_ad);
  state.IncrementDv(ddv);

  const AutoDiffXd energy = state_ad.CalcTotalEnergy();
  VectorX<double> residual;
  state.CalcResidual(&residual);
  EXPECT_TRUE(CompareMatrices(energy.derivatives(), residual, kTol));
}

GTEST_TEST(MpmStateTest, HessianIsDerivativeOfResidual) {
  const double dx = 0.01;
  MockSparseGrid<AutoDiffXd> grid_ad(dx);
  Particles<AutoDiffXd> particles_ad;
  const Vector3<AutoDiffXd> x0_ad(dx, dx, dx);
  // const Vector3<AutoDiffXd> x1_ad(1.1 * dx, 1.2 * dx, 1.3 * dx);
  AddParticle<AutoDiffXd>(&particles_ad, x0_ad);
  // AddParticle<AutoDiffXd>(&particles_ad, x1_ad);
  AddDefaultMaterial(&particles_ad.data);
  const AutoDiffXd dt_ad = 0.01;

  SparseGrid<double> grid(dx);
  Particles<double> particles;
  const Vector3<double> x0(dx, dx, dx);
  // const Vector3<double> x1(1.1 * dx, 1.2 * dx, 1.3 * dx);
  AddParticle<double>(&particles, x0);
  // AddParticle<double>(&particles, x1);
  AddDefaultMaterial(&particles.data);
  const double dt = 0.01;

  const double kTol = 1e-10;

  MpmImplicitData<AutoDiffXd> data_ad(particles_ad.data);
  MpmState<AutoDiffXd, MockSparseGrid> state_ad(dt_ad, &grid_ad, &particles_ad,
                                                &data_ad);
  MpmImplicitData<double> data(particles.data);
  MpmState<double> state(dt, &grid, &particles, &data);
  const int num_dofs = state.num_dofs();
  ASSERT_EQ(num_dofs, state_ad.num_dofs());

  VectorX<double> ddv = VectorX<double>::LinSpaced(num_dofs, 0.0, 1.0);
  VectorX<AutoDiffXd> ddv_ad;
  ddv_ad.resize(num_dofs);
  math::InitializeAutoDiff(ddv, &ddv_ad);

  state_ad.IncrementDv(ddv_ad);
  state.IncrementDv(ddv);

  VectorX<AutoDiffXd> residual;
  state_ad.CalcResidual(&residual);

  Block3x3SparseSymmetricMatrix tangent_matrix = state.MakeTangentMatrix();
  state.CalcTangentMatrix(&tangent_matrix);
  const MatrixXd dense_tangent_matrix = tangent_matrix.MakeDenseMatrix();

  EXPECT_TRUE(CompareMatrices(dense_tangent_matrix.col(0),
                              residual(0).derivatives(), kTol));
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
