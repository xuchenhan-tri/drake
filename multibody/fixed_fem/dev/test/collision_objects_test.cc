#include <gtest/gtest.h>

#include "drake/multibody/fixed_fem/dev/collsion_objects.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
using geometry::GeometryId;
using geometry::ProximityProperties;
using geometry::Shape;

class CollisionObjectsTest : public ::testing::Test {
  void SetUp() override {
    // Build some arbitrary proximity properties.
  }

 protected:
  CollisionObjects collision_objects_;
  ProximityProperties proximity_properties_;

  double resolution_hint() const { return collision_objects_.resolution_hint_; }
};

namespace {
TEST_F(CollisionObjectTest, AddSphere) {
  GeometryId id = GeometryId::get_new_id();
  collision_objects_.AddCollisionObject(id, shape, proximity_properties);
}
}  // namespace
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
