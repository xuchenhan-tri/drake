#pragma once

#include "drake/geometry/proximity/surface_mesh.h"
#include "drake/geometry/proximity_properties.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {

class CollisionObject {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(CollisionObject);

  CollisionObject(const geometry::SurfaceMesh<double>& surface_mesh,
                  const geometry::ProximityProperties& proximity_properties)
      : surface_mesh_(surface_mesh),
        proximity_properties_(proximity_properties) {}

  const geometry::SurfaceMesh<double>& mesh() const { return surface_mesh_; }

  const geometry::ProximityProperties& proximity_properties() const {
    return proximity_properties_;
  }

 private:
  const geometry::SurfaceMesh<double> surface_mesh_;
  const geometry::ProximityProperties proximity_properties_;
};
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
