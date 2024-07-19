#include "drake/multibody/mpm/particles.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::Matrix3f;
using Eigen::Vector3d;
using Eigen::Vector3f;

GTEST_TEST(Particles, Size) {
  EXPECT_EQ(sizeof(Matrix3d), 9 * sizeof(double));
  EXPECT_EQ(sizeof(Vector3d), 3 * sizeof(double));
  EXPECT_EQ(sizeof(Particle<double>),
            sizeof(double) + 2 * sizeof(Vector3d) + 3 * sizeof(Matrix3d));

  EXPECT_EQ(sizeof(Matrix3f), 9 * sizeof(float));
  EXPECT_EQ(sizeof(Vector3f), 3 * sizeof(float));
  EXPECT_EQ(sizeof(Particle<float>),
            sizeof(float) + 2 * sizeof(Vector3f) + 3 * sizeof(Matrix3f));
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
