#include "drake/geometry/proximity/deformable_contact_internal.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity/deformable_contact_geometries.h"
#include "drake/geometry/proximity/deformable_mesh_intersection.h"

namespace drake {
namespace geometry {
namespace internal {
namespace deformable {

using std::move;

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
  if (props.HasProperty(kDeformableContactGroup, kRezHint)) {
    const double resolution_hint =
        props.GetProperty<double>(kDeformableContactGroup, kRezHint);
    ReifyData data{id, resolution_hint, std::nullopt};
    shape.Reify(this, &data);
  }
}

void Geometries::UpdateRigidWorldPose(
    GeometryId id, const math::RigidTransform<double>& X_WG) {
  if (is_rigid(id)) {
    rigid_geometries_.at(id).set_pose_in_world(X_WG);
  }
}

void Geometries::MaybeAddDeformableGeometry(const Shape& shape, GeometryId id,
                                            const VolumeMesh<double>& mesh) {
  ReifyData data{id, std::nullopt, mesh};
  shape.Reify(this, &data);
}

void Geometries::UpdateDeformableVertexPositions(
    GeometryId id, const Eigen::Ref<const VectorX<double>>& q_WG) {
  if (is_deformable(id)) {
    deformable_geometries_.at(id).UpdateVertexPositions(q_WG);
  }
}

void Geometries::ComputeAllDeformableContactData(
    std::vector<DeformableContactData<double>>* deformable_contact_data) const {
  DRAKE_DEMAND(deformable_contact_data != nullptr);
  deformable_contact_data->clear();
  deformable_contact_data->reserve(num_deformable_geometries());
  for (const auto& it : deformable_geometries_) {
    DeformableContactData<double> contact_data =
        CalcDeformableContactData(it.first);
    if (contact_data.num_contact_points() > 0) {
      deformable_contact_data->emplace_back(move(contact_data));
    }
  }
}

  void Geometries::ImplementGeometry(const Sphere& sphere, void* user_data) {
    MakeShape(sphere, *static_cast<ReifyData*>(user_data));
  }

  void Geometries::ImplementGeometry(const Cylinder& cylinder,
                                     void* user_data) {
    MakeShape(cylinder, *static_cast<ReifyData*>(user_data));
  }

  void Geometries::ImplementGeometry(const HalfSpace& half_space,
                                     void* user_data) {
    MakeShape(half_space, *static_cast<ReifyData*>(user_data));
  }

  void Geometries::ImplementGeometry(const Box& box, void* user_data) {
    MakeShape(box, *static_cast<ReifyData*>(user_data));
  }

  void Geometries::ImplementGeometry(const Capsule& capsule, void* user_data) {
    MakeShape(capsule, *static_cast<ReifyData*>(user_data));
  }

  void Geometries::ImplementGeometry(const Ellipsoid& ellipsoid,
                                     void* user_data) {
    MakeShape(ellipsoid, *static_cast<ReifyData*>(user_data));
  }

  void Geometries::ImplementGeometry(const Mesh& mesh, void* user_data) {
    MakeShape(mesh, *static_cast<ReifyData*>(user_data));
  }

  void Geometries::ImplementGeometry(const Convex& convex, void* user_data) {
    MakeShape(convex, *static_cast<ReifyData*>(user_data));
  }

  template <typename ShapeType>
  void Geometries::MakeShape(const ShapeType& shape, const ReifyData& data) {
    if (data.mesh) {
      DeformableGeometry geometry(shape, *(data.mesh));
      deformable_geometries_.insert({data.id, move(geometry)});
    } else {
      std::optional<RigidGeometry> geometry =
          MakeRigidRepresentation(shape, *(data.resolution_hint));
      if (geometry) {
        rigid_geometries_.insert({data.id, move(*geometry)});
      } else {
        throw std::logic_error(
            "The provided shape cannot be instantiated as rigid geometry "
            "participating in deformable contact. ");
      }
    }
  }

  DeformableContactData<double> Geometries::CalcDeformableContactData(
      GeometryId deformable_id) const {
    DRAKE_DEMAND(is_deformable(deformable_id));
    std::vector<DeformableRigidContactSurface<double>>
        deformable_rigid_contact_surfaces;
    for (const auto& it : rigid_geometries_) {
      DeformableRigidContactSurface<double> contact_surface =
          CalcDeformableRigidContactSurface(it.first, deformable_id);
      if (contact_surface.num_contact_points() != 0) {
        deformable_rigid_contact_surfaces.emplace_back(move(contact_surface));
      }
    }

    const VolumeMesh<double>& deformable_mesh =
        deformable_geometries_.at(deformable_id).deformable_mesh().mesh();

    return {move(deformable_rigid_contact_surfaces), deformable_mesh};
  }

  DeformableRigidContactSurface<double>
  Geometries::CalcDeformableRigidContactSurface(
      GeometryId rigid_id, GeometryId deformable_id) const {
    DRAKE_DEMAND(is_deformable(deformable_id));
    DRAKE_DEMAND(is_rigid(rigid_id));

    const DeformableGeometry& deformable_geometry =
        deformable_geometries_.at(deformable_id);
    const RigidGeometry& rigid_geometry = rigid_geometries_.at(rigid_id);
    const math::RigidTransform<double>& X_WR = rigid_geometry.pose_in_world();
    const auto& rigid_bvh = rigid_geometry.rigid_mesh().bvh();
    const auto& rigid_tri_mesh = rigid_geometry.rigid_mesh().mesh();

    std::unique_ptr<DeformableRigidContactSurface<double>> contact_surface =
        ComputeContactSurfaceFromDeformableVolumeRigidSurface(
            deformable_id, deformable_geometry, rigid_id, rigid_tri_mesh,
            rigid_bvh, X_WR);
    // TODO(xuchenhan-tri): Remove this copy.
    return *contact_surface;
  }

}  // namespace deformable
}  // namespace internal
}  // namespace geometry
}  // namespace drake
