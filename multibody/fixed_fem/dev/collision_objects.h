#pragma once
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/surface_mesh.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
/* Representation of a group of rigid collision objects, consisting of their
 surface meshes and proximity properties.
 @tparam_nonsymbolic_scalar T. */
template <typename T>
class CollisionObjects : public geometry::ShapeReifier {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(CollisionObjects);

  CollisionObjects() = default;

  /* Registers the geometry with the given `id`, `shape` and
   `proximity_properties` in `this` CollisionObjects. */
  void AddCollisionObject(
      geometry::GeometryId id, const geometry::Shape& shape,
      const geometry::ProximityProperties& proximity_properties) {
    DRAKE_DEMAND(rigid_representations_.find(id) ==
                 rigid_representations_.end());
    ReifyData data{id, proximity_properties};
    shape.Reify(this, &data);
    poses_.insert({id, math::RigidTransform<T>()});
  }

  /* Returns the surface mesh of the geometry with GeometryId `id`.
   @pre The geometry with `id` has been registered in `this` CollisionObject
   with AddCollisionObject(). */
  const geometry::SurfaceMesh<double>& mesh(geometry::GeometryId id) const {
    const auto it = rigid_representations_.find(id);
    DRAKE_DEMAND(it != rigid_representations_.end());
    return *(it->second.surface_mesh);
  }

  /* Returns the proximity properties of the geometry with GeometryId `id`.
   @pre The geometry with `id` has been registered in `this` CollisionObject
   with AddCollisionObject(). */
  const geometry::ProximityProperties& proximity_properties(
      geometry::GeometryId id) const {
    const auto it = rigid_representations_.find(id);
    DRAKE_DEMAND(it != rigid_representations_.end());
    return it->second.properties;
  }

  /* Returns the world pose of the geometry with GeometryId `id`.
   @pre The geometry with `id` has been registered in `this` CollisionObject
   with AddCollisionObject(). */
  const math::RigidTransform<T>& pose(geometry::GeometryId id) const {
    auto it = poses_.find(id);
    DRAKE_DEMAND(it != poses_.end());
    return it->second;
  }

  /* Updates the pose of the geometry with GeometryId `id` in world frame to the
   given `pose`.
   @pre The geometry with `id` has been registered in `this` CollisionObject
   with AddCollisionObject(). */
  void UpdatePoseInWorld(geometry::GeometryId id,
                         const math::RigidTransform<T>& pose) {
    auto it = poses_.find(id);
    DRAKE_DEMAND(it != poses_.end());
    it->second = pose;
  }

  const std::vector<geometry::GeometryId>& geometry_ids() const {
    return geometry_ids_;
  }

 private:
  friend class CollisionObjectsTest;
  /* Data to be used during reification. It is passed as the `user_data`
   parameter in the ImplementGeometry API. */
  struct ReifyData {
    geometry::GeometryId id;
    const geometry::ProximityProperties& properties;
  };

  struct RigidRepresentation {
    RigidRepresentation() = default;
    RigidRepresentation(std::unique_ptr<geometry::SurfaceMesh<double>> mesh,
                        const geometry::ProximityProperties& props)
        : surface_mesh(std::move(mesh)), properties(props) {}
    copyable_unique_ptr<geometry::SurfaceMesh<double>> surface_mesh;
    geometry::ProximityProperties properties;
  };

  void ImplementGeometry(const geometry::Sphere& sphere,
                         void* user_data) override;
  void ImplementGeometry(const geometry::Cylinder& cylinder,
                         void* user_data) override;
  void ImplementGeometry(const geometry::HalfSpace&, void* user_data) override;
  void ImplementGeometry(const geometry::Box& box, void* user_data) override;
  void ImplementGeometry(const geometry::Capsule& capsule,
                         void* user_data) override;
  void ImplementGeometry(const geometry::Ellipsoid& ellipsoid,
                         void* user_data) override;
  void ImplementGeometry(const geometry::Mesh&, void*) override;
  void ImplementGeometry(const geometry::Convex& convex,
                         void* user_data) override;

  template <typename ShapeType>
  void MakeRigidRepresentation(const ShapeType& shape, const ReifyData& data);
  std::map<geometry::GeometryId, RigidRepresentation> rigid_representations_;
  std::map<geometry::GeometryId, math::RigidTransform<T>> poses_;
  std::vector<geometry::GeometryId> geometry_ids_;
  double resolution_hint_{0.01};
};
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fixed_fem::internal::CollisionObjects);
