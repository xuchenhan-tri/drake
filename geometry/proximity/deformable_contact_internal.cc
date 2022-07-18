#include "drake/geometry/proximity/deformable_contact_internal.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity/deformable_contact_geometries.h"
#include "drake/geometry/proximity/deformable_mesh_intersection.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"

namespace drake {
namespace geometry {
namespace internal {
namespace deformable {

const DeformableGeometry& Geometries::deformable_geometry(GeometryId id) const {
  if (is_deformable(id)) return deformable_geometries_.at(id);
  throw std::runtime_error(
      fmt::format("There is no deformable geometry with GeometryId {}", id));
}

const RigidGeometry& Geometries::rigid_geometry(GeometryId id) const {
  if (is_rigid(id)) return rigid_geometries_.at(id);
  throw std::runtime_error(
      fmt::format("There is no rigid geometry with GeometryId {}", id));
}

void Geometries::RemoveGeometry(GeometryId id) {
  deformable_geometries_.erase(id);
  rigid_geometries_.erase(id);
}

void Geometries::MaybeAddRigidGeometry(const Shape& shape, GeometryId id,
                                       const ProximityProperties& props) {
  // TODO(xuchenhan-tri): Right now, rigid geometries participating in
  // deformable contact share the property "kRezHint" with hydroelastics. It's
  // reasonable to use the contact mesh with the same resolution for both hydro
  // and deformable contact. Consider reorganizing the proximity properties to
  // make this sharing more explicit. We should also avoid having two copies of
  // the same rigid geometry for both hydro and deformable contact.
  if (props.HasProperty(kHydroGroup, kRezHint)) {
    ReifyData data{id, props};
    shape.Reify(this, &data);
  }
}

void Geometries::UpdateRigidWorldPose(
    GeometryId id, const math::RigidTransform<double>& X_WG) {
  if (is_rigid(id)) {
    rigid_geometries_.at(id).set_pose_in_world(X_WG);
  }
}

void Geometries::AddDeformableGeometry(GeometryId, const VolumeMesh<double>&) {}

void Geometries::UpdateDeformableVertexPositions(
    GeometryId id, const Eigen::Ref<const VectorX<double>>& q_WG) {
  if (is_deformable(id)) {
    deformable_geometries_.at(id).UpdateVertexPositions(q_WG);
  }
}

void Geometries::ComputeDeformableRigidContact(
    std::vector<DeformableRigidContact<double>>* deformable_rigid_contact)
    const {
  DRAKE_DEMAND(deformable_rigid_contact != nullptr);
  deformable_rigid_contact->clear();
  deformable_rigid_contact->reserve(num_deformable_geometries());

  for (const auto& [deformable_id, deformable_geometry] :
       deformable_geometries_) {
    const VolumeMesh<double>& deformable_mesh =
        deformable_geometry.deformable_mesh().mesh();
    DeformableRigidContact<double> contact_data(deformable_id,
                                                deformable_mesh.num_vertices());
    for (const auto& [rigid_id, rigid_geometry] : rigid_geometries_) {
      const math::RigidTransform<double>& X_WR = rigid_geometry.pose_in_world();
      const auto& rigid_bvh = rigid_geometry.rigid_mesh().bvh();
      const auto& rigid_tri_mesh = rigid_geometry.rigid_mesh().mesh();
      AppendDeformableRigidContact(deformable_geometry, rigid_id,
                                   rigid_tri_mesh, rigid_bvh, X_WR,
                                   &contact_data);
    }
    deformable_rigid_contact->emplace_back(std::move(contact_data));
  }
}

void Geometries::ImplementGeometry(const Sphere& sphere, void* user_data) {
  AddRigidGeometry(sphere, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const Cylinder& cylinder, void* user_data) {
  AddRigidGeometry(cylinder, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const Box& box, void* user_data) {
  AddRigidGeometry(box, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const Capsule& capsule, void* user_data) {
  AddRigidGeometry(capsule, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const Ellipsoid& ellipsoid,
                                   void* user_data) {
  AddRigidGeometry(ellipsoid, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const Mesh& mesh, void* user_data) {
  AddRigidGeometry(mesh, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const Convex& convex, void* user_data) {
  AddRigidGeometry(convex, *static_cast<ReifyData*>(user_data));
}

template <typename ShapeType>
void Geometries::AddRigidGeometry(const ShapeType& shape,
                                  const ReifyData& data) {
  /* Forward to hydroelastics to construct the geometry. */
  std::optional<internal::hydroelastic::RigidGeometry> hydro_rigid_geometry =
      internal::hydroelastic::MakeRigidRepresentation(shape, data.properties);
  /* Unsupported geometries will be handle through the
   `ThrowUnsupportedGeometry()` code path. */
  DRAKE_DEMAND(hydro_rigid_geometry.has_value());
  rigid_geometries_.insert(
      {data.id, RigidGeometry(hydro_rigid_geometry->release_mesh())});
}

void Geometries::ThrowUnsupportedGeometry(const std::string& shape_name) {
  static const logging::Warn log_once(
      "Rigid (non-deformable) {} shapes are not currently supported for "
      "deformable contact; registration is allowed, but an error will be "
      "thrown during contact.",
      shape_name);
}

}  // namespace deformable
}  // namespace internal
}  // namespace geometry
}  // namespace drake
