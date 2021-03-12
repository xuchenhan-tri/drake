#pragma once

#include "drake/common/unused.h"
#include "drake/multibody/fixed_fem/dev/fem_state.h"

namespace drake {
namespace multibody {
namespace fixed_fem {

template <typename T>
class CollisionObject {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(CollisionObject);

  CollisionObject(
      const geometry::SurfaceMesh<double>& surface_mesh,
      const geometry::ProximityProperties& proximity_properties,
      const std::function<void(const U& time, math::RigidTransform<U>* X_WG,
                               SpatialVelocity<U>* V_WG)>&
          motion_update_callback)
      : surface_mesh_(surface_mesh),
        proximity_properties_(proximity_properties),
        motion_update_callback_(motion_update_callback) {}

  const Vector3<U>& rotational_velocity() const {
    return spatial_velocity_.rotational();
  }

  const Vector3<U>& translational_velocity() const {
    return spatial_velocity_.translational();
  }

  void UpdatePositionAndVelocity(const U& time) {
    motion_update_callback_(time, &pose_, &spatial_velocity_);
  }

  /* Return the pose of the collision object in world space, X_WG. */
  const math::RigidTransform<U>& get_pose() const { return pose_; }

 private:
  const geometry::SurfaceMesh<double> surface_mesh_;
  math::RigidTransform<U> pose_{};
  SpatialVelocity<U> spatial_velocity_{};
  const geometry::ProximityProperties proximity_properties_;
  const std::function<void(const U& time, math::RigidTransform<U>* X_WG,
                           SpatialVelocity<U>* V_WG)>
      motion_update_callback_;
};
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
