#include "drake/fem/collision_object.h"

#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "drake/fem/half_space.h"

namespace drake {
namespace fem {
namespace {

class CollisionObjectTest : public ::testing::Test {
 public:
  void SetUp() override {
    auto update = [](double time, CollisionObject<double>* cb) {
      double angular_velocity = M_PI / 4.0;
      double translation_velocity = 2.0;

      double c = std::cos(angular_velocity * time / 2.0);
      double s = std::sin(angular_velocity * time / 2.0);
      // Rotates around x-axis;
      Eigen::Quaternion<double> q(c, s, 0.0, 0.0);
      cb->set_rotation(q);
      // Unit speed translation upward.
      Vector3<double> translation(0, 0, time * translation_velocity);
      cb->set_translation(translation);
    };
    auto ls = std::make_unique<HalfSpace<double>>(Vector3<double>(0, 0, 0),
                                                  Vector3<double>(0, 0, 1));
    cb_ = std::make_unique<CollisionObject<double>>(std::move(ls), update);
  }

 protected:
  std::unique_ptr<CollisionObject<double>> cb_;
};

TEST_F(CollisionObjectTest, InitialStateQuery) {
  Vector3<double> p0(0, 0, M_PI / 4.0);
  Vector3<double> p1(0, -1, M_PI / 4.0);
  Vector3<double> p2(0, 1, M_PI / 4.0);

  Vector3<double> analytic_normal(0, 0, 1.0);
  Vector3<double> normal;
  double signed_distance = cb_->Query(p0, &normal);
  EXPECT_NEAR(signed_distance, M_PI / 4.0,
              std::numeric_limits<double>::epsilon());
  EXPECT_EQ(normal, analytic_normal);

  signed_distance = cb_->Query(p1, &normal);
  EXPECT_NEAR(signed_distance, M_PI / 4.0,
              std::numeric_limits<double>::epsilon());
  EXPECT_EQ(normal, analytic_normal);

  signed_distance = cb_->Query(p2, &normal);
  EXPECT_NEAR(signed_distance, M_PI / 4.0,
              std::numeric_limits<double>::epsilon());
  EXPECT_EQ(normal, analytic_normal);
}

TEST_F(CollisionObjectTest, Query) {
  double time = 1.0;
  cb_->Update(time);
  Vector3<double> p0(0, 0, 2);   // p0 is on the surface.
  Vector3<double> p1(0, -1, 2);  // p1 is outside the half space.
  Vector3<double> p2(0, 1, 2);   // p2 is inside of the half space.

  Vector3<double> analytic_normal(0, -1.0 / std::sqrt(2.0),
                                  1.0 / std::sqrt(2.0));
  Vector3<double> normal;
  double signed_distance = cb_->Query(p0, &normal);
  std::cout << "SDF = " << signed_distance << std::endl;
  EXPECT_NEAR(signed_distance, 0.0, std::numeric_limits<double>::epsilon());
  EXPECT_NEAR((normal - analytic_normal).norm(), 0.0,
              std::numeric_limits<double>::epsilon());

  signed_distance = cb_->Query(p1, &normal);
  EXPECT_NEAR(signed_distance, 1.0 / std::sqrt(2.0),
              std::numeric_limits<double>::epsilon());
  EXPECT_NEAR((normal - analytic_normal).norm(), 0.0,
              std::numeric_limits<double>::epsilon());

  signed_distance = cb_->Query(p2, &normal);
  EXPECT_NEAR(signed_distance, -1.0 / std::sqrt(2.0),
              std::numeric_limits<double>::epsilon());
  EXPECT_NEAR((normal - analytic_normal).norm(), 0.0,
              std::numeric_limits<double>::epsilon());
}
}  // namespace
}  // namespace fem
}  // namespace drake
