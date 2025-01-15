#include "drake/multibody/mpm/transfer.h"

#include <numeric>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

/* Sample N^3 points in the box [0, 1] x [0, 1] x [0, 1]*/
std::vector<Vector3<double>> SamplePoints(int N) {
  std::vector<Vector3<double>> points;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        points.push_back(
            Vector3<double>(i / (N - 1.0), j / (N - 1.0), k / (N - 1.0)));
      }
    }
  }
  return points;
}

using ScalarTypes = ::testing::Types<float, double>;

template <typename T>
class TransferTest : public ::testing::Test {};

TYPED_TEST_SUITE(TransferTest, ScalarTypes);

TYPED_TEST(TransferTest, Constant) {
  using T = TypeParam;
  const double dx = 0.2;
  const T dt = 1e-2;
  const int N = 10;

  SparseGrid<T> grid(dx);
  Particles<T> particles;
  multibody::fem::DeformableBodyConfig<double> config;
  config.set_mass_density(1.0);
  particles.data.AddParticles(SamplePoints(N), /* total volume */ 1.0, config);
  particles.Sort(grid);

  Transfer<T> transfer(dt, &grid, &particles);
  grid.Allocate(particles.sorter);
  transfer.ScalarParticleToGrid(particles, &grid);

  MassAndMomentum<T> grid_mass_and_momentum =
      grid.ComputeTotalMassAndMomentum();
  MassAndMomentum<T> particle_mass_and_momentum =
      particles.data.ComputeTotalMassAndMomentum(dx);
  EXPECT_NEAR(grid_mass_and_momentum.mass, particle_mass_and_momentum.mass,
              128 * std::numeric_limits<T>::epsilon());
  EXPECT_TRUE(CompareMatrices(grid_mass_and_momentum.linear_momentum,
                              particle_mass_and_momentum.linear_momentum,
                              128 * std::numeric_limits<T>::epsilon()));
  EXPECT_TRUE(CompareMatrices(grid_mass_and_momentum.angular_momentum,
                              particle_mass_and_momentum.angular_momentum,
                              128 * std::numeric_limits<T>::epsilon()));

  grid.Allocate(particles.sorter);
  transfer.SimdParticleToGrid(particles, &grid);
  grid_mass_and_momentum = grid.ComputeTotalMassAndMomentum();
  EXPECT_NEAR(grid_mass_and_momentum.mass, particle_mass_and_momentum.mass,
              128 * std::numeric_limits<T>::epsilon());
  EXPECT_TRUE(CompareMatrices(grid_mass_and_momentum.linear_momentum,
                              particle_mass_and_momentum.linear_momentum,
                              128 * std::numeric_limits<T>::epsilon()));
  EXPECT_TRUE(CompareMatrices(grid_mass_and_momentum.angular_momentum,
                              particle_mass_and_momentum.angular_momentum,
                              128 * std::numeric_limits<T>::epsilon()));
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
