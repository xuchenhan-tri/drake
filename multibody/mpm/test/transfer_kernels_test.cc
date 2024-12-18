#include "drake/multibody/mpm/transfer_kernels.h"

#include <numeric>

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

using ScalarTypes = ::testing::Types<float, double>;

template <typename T>
class TransferKernelsTest : public ::testing::Test {};

TYPED_TEST_SUITE(TransferKernelsTest, ScalarTypes);

TYPED_TEST(TransferKernelsTest, Constant) {
  using T = TypeParam;
  ParticleData<T> particle_data;
  const T dx = 0.1;

  fem::DeformableBodyConfig<double> config;
  const int kNumParticles = 33;
  const Vector3d x0 = Vector3d(0.6 * dx, 0.6 * dx, 0.6 * dx);
  std::vector<Vector3d> positions(kNumParticles);
  for (int i = 0; i < kNumParticles; ++i) {
    positions[i] = x0;
  }
  const T total_volume = 1.0;
  particle_data.AddParticles(positions, total_volume, config);

  std::vector<int> data_indices(kNumParticles);
  std::iota(data_indices.begin(), data_indices.end(), 0);

  Pad<Vector3<T>> grid_x;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        grid_x[i][j][k] = Vector3<T>(i * dx, j * dx, k * dx);
      }
    }
  }
  Pad<GridData<T>> grid_data;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        grid_data[i][j][k].m = 0.0;
        grid_data[i][j][k].v = Vector3<T>::Zero();
      }
    }
  }
  const T D_inverse_dt = 1.0;
  WorkingSet<T> working_set(dx, D_inverse_dt);
  P2G(particle_data, data_indices, grid_x, &grid_data, &working_set);

  T total_grid_mass = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        total_grid_mass += grid_data[i][j][k].m;
      }
    }
  }
  const T kTolerance = 1024 * std::numeric_limits<T>::epsilon();
  EXPECT_NEAR(total_grid_mass, config.mass_density() * total_volume,
              kTolerance);
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
