#pragma once
#include <utility>
#include <vector>

#include "drake/geometry/geometry_ids.h"
#include "drake/math/orthonormal_basis.h"
#include "drake/multibody/fixed_fem/dev/deformable_contact.h"
#include "drake/multibody/fixed_fem/dev/fem_indexes.h"
namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
/* A wrapper around DeformableContactSurface that provides additional
 information about the geometries/bodies involved in the contact and proximity
 properties of the contacts. */
template <typename T>
struct DeformableRigidContactData {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(DeformableRigidContactData)
  DeformableRigidContactData(DeformableContactSurface<T>&& contact_surface_in,
                             geometry::GeometryId rigid_id_in,
                             SoftBodyIndex deformable_id_in, T&& k, T&& d,
                             double mu)
      : contact_surface(std::move(contact_surface_in)),
        rigid_id(rigid_id_in),
        deformable_id(deformable_id_in),
        stiffness(std::move(k)),
        dissipation(std::move(d)),
        friction(mu),
        R_CWs(contact_surface.num_polygons()) {
    for (int ic = 0; ic < contact_surface.num_polygons(); ++ic) {
      R_CWs[ic] = math::ComputeBasisFromAxis(
                      2, contact_surface.polygon_data(ic).unit_normal)
                      .transpose();
    }
  }
  DeformableContactSurface<T> contact_surface;
  geometry::GeometryId rigid_id;  // The id of the rigid geometry in contact.
  SoftBodyIndex deformable_id;    // The id of deformable body in contact.
  T stiffness;                    // Combined stiffness at the contact point.
  T dissipation;                  // Combined dissipation at the contact point.
  double friction;                // Combined friction at the contact point.
  /* The rotation matrix mapping world frame quantities into contact frames at
   each contact point. */
  std::vector<Matrix3<T>> R_CWs;
};
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
