#include "drake/fem/half_space.h"

#include <limits>
#include <memory>

#include <gtest/gtest.h>

namespace drake {
namespace fem {
namespace {

class HalfSpaceTest :public ::testing::Test {
 public:
  void SetUp() override {
    half_space_ = std::make_unique<HalfSpace<double>>(Vector3<double>(1, 2, 3),
                                                      Vector3<double>(1, 1, 1));
  }

 protected:
  std::unique_ptr<HalfSpace<double>> half_space_;
};

TEST_F(HalfSpaceTest, NormalTest) {
  double n = 1 / std::sqrt(3.0);
  Vector3<double> normal(n, n, n);
  Vector3<double> x = Vector3<double>::Random();
  EXPECT_NEAR((normal - half_space_->Normal(x)).norm(), 0.0,
              std::numeric_limits<double>::epsilon());
}

TEST_F(HalfSpaceTest, SignedDistanceTest) {
  Vector3<double> p0(2, 1, 3);  // p0 is on the surface.
  Vector3<double> p1(2, 3, 4);  // p1 is outside of the half space.
  Vector3<double> p2(1, 0, 2);  // p2 is inside the half space.
  // The scale of round off error.
  double scale = 2.0 * std::sqrt(3.0);
  EXPECT_NEAR(half_space_->SignedDistance(p0), 0,
              std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(half_space_->SignedDistance(p1), std::sqrt(3.0),
              scale * std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(half_space_->SignedDistance(p2), -std::sqrt(3.0),
              scale * std::numeric_limits<double>::epsilon());
}
}  // namespace
}  // namespace fem
}  // namespace drake
