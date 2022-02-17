#include "drake/multibody/fem/fem_state.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/expect_throws_message.h"

namespace drake {
namespace multibody {
namespace fem {
namespace {

constexpr int kNumDofs = 12;
using Eigen::VectorXd;

/* Arbitrary values for the state. */
VectorX<double> q() {
  Vector<double, kNumDofs> q;
  q << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2;
  return q;
}
VectorX<double> v() {
  Vector<double, kNumDofs> v;
  v << 1.1, 1.2, 2.3, 2.4, 2.5, 2.6, 2.7, 1.8, 1.9, 2.0, 2.1, 2.2;
  return v;
}
VectorX<double> a() {
  Vector<double, kNumDofs> a;
  a << 2.1, 2.2, 3.3, 3.4, 3.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2;
  return a;
}

/* Verify setters and getters are working properly. */
GTEST_TEST(FemStateTest, GetStates) {
  FemState<double> state(q(), v(), a());
  EXPECT_EQ(state.num_dofs(), kNumDofs);
  const VectorXd& positions = state.GetPositions();
  const VectorXd& velocities = state.GetVelocities();
  const VectorXd& accelerations = state.GetAccelerations();
  EXPECT_EQ(positions, q());
  EXPECT_EQ(velocities, v());
  EXPECT_EQ(accelerations, a());
}

GTEST_TEST(FemStateTest, SetStates) {
  FemState<double> state(q(), v(), a());
  state.SetPositions(-1.23 * q());
  state.SetVelocities(3.14 * v());
  state.SetAccelerations(-1.29 * a());
  EXPECT_EQ(state.GetPositions(), -1.23 * q());
  EXPECT_EQ(state.GetVelocities(), 3.14 * v());
  EXPECT_EQ(state.GetAccelerations(), -1.29 * a());
  /* Setting values with incompatible sizes should throw. */
  EXPECT_THROW(state.SetPositions(VectorXd::Constant(1, 1.0)), std::exception);
  EXPECT_THROW(state.SetVelocities(VectorXd::Constant(1, 1.0)),
               std::exception);
  EXPECT_THROW(state.SetAccelerations(VectorXd::Constant(1, 1.0)),
               std::exception);
}

}  // namespace
}  // namespace fem
}  // namespace multibody
}  // namespace drake
