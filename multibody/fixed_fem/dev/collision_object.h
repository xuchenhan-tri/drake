#pragma once

#include "drake/geometry/proximity/surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/math/spatial_algebra.h"

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
      const std::function<void(const T& time, math::RigidTransform<T>* X_WG,
                               SpatialVelocity<T>* V_WG)>&
          motion_update_callback)
      : surface_mesh_(surface_mesh),
        proximity_properties_(proximity_properties),
        motion_update_callback_(motion_update_callback) {}

  const Vector3<T>& rotational_velocity() const {
    return spatial_velocity_.rotational();
  }

  const Vector3<T>& translational_velocity() const {
    return spatial_velocity_.translational();
  }

  void UpdatePositionAndVelocity(const T& time) {
    motion_update_callback_(time, &pose_, &spatial_velocity_);
  }

  /* Return the pose of the collision object in world space, X_WG. */
  const math::RigidTransform<T>& pose() const { return pose_; }

  const geometry::SurfaceMesh<double>& mesh() const { return surface_mesh_; }

  const geometry::ProximityProperties& proximity_properties() const {
    return proximity_properties_;
  }

 private:
  const geometry::SurfaceMesh<double> surface_mesh_;
  math::RigidTransform<T> pose_{};
  SpatialVelocity<T> spatial_velocity_{};
  const geometry::ProximityProperties proximity_properties_;
  const std::function<void(const T& time, math::RigidTransform<T>* X_WG,
                           SpatialVelocity<T>* V_WG)>
      motion_update_callback_;
};
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
