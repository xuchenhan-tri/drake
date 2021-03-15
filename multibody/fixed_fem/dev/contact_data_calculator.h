#pragma once

#include <memory>
#include <vector>

#include "drake/math/cross_product.h"
#include "drake/math/orthonormal_basis.h"
#include "drake/multibody/contact_solvers/point_contact_data.h"
#include "drake/multibody/contact_solvers/sparse_linear_operator.h"
#include "drake/multibody/fixed_fem/dev/collision_object.h"
#include "drake/multibody/fixed_fem/dev/deformable_contact.h"
#include "drake/multibody/plant/coulomb_friction.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {

template <typename T>
class ContactDataCalculator {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ContactDataCalculator);

  ContactDataCalculator() = default;

  // TODO(xuchenhan-tri): Instead of passing in a deformable mesh in world
  // coordinate, pass in a deformable body with its mesh in the deformable
  // frame, pose in world frame and proximity properties.
  const contact_solvers::internal::PointContactData<T> ComputeContactData(
      const geometry::VolumeMesh<T>& deformable_mesh_W,
      const std::vector<CollisionObject<T>>& collision_objects) {
    // Number of generalized velocities.
    const int nv =
        deformable_mesh_W.num_vertices() * 3 + collision_objects.size() * 6;
    // Number of contact points.
    int nc = 0;
    std::vector<DeformableContactSurface<T>> contact_surfaces;
    for (const auto& collision_object : collision_objects) {
      contact_surfaces.emplace_back(ComputeTetMeshTriMeshContact(
          deformable_mesh_W, collision_object.mesh(), collision_object.pose()));
      nc += contact_surfaces.back().num_polygons();
    }

    phi0_.resize(nc);
    stiffness_.resize(nc);
    dissipation_.resize(nc);
    mu_.resize(nc);

    // TODO(xuchenhan-tri): Set these data properly. They are set to zero for
    //  now because PGS does not use them.
    phi0_.setZero();
    stiffness_.setZero();
    dissipation_.setZero();

    /* Calculate the contact Jacobian which maps the generalized velocity to the
     contact velocity. Suppose a deformable body A and a rigid body B are in
     contact and a point Q. We get the contact velocity in world frame by
     substracting the velocity of a point (Bq) coincident with p_WQ on
     the rigid body B from the velocity of a point (Aq) coincident with p_WQ on
     the deformable body A. In monogram notation, this is v_WAq  - v_WBq. We
     then express the contact velocity in the contact frame C and get R_CW *
     (v_WAq - v_WBq), where R_CW is the rotation matrix that maps vectors
     measured in world frame to the contact frame. */
    Jc_matrix_.resize(3 * nc, nv);
    std::vector<Eigen::Triplet<T>> triplets;
    int contact_start = 0;
    for (int i = 0; i < static_cast<int>(contact_surfaces.size()); ++i) {
      const auto& contact_surface = contact_surfaces[i];
      const auto& collision_object = collision_objects[i];
      for (int j = 0; j < contact_surface.num_polygons(); ++j) {
        const ContactPolygonData<T>& polygon_data =
            contact_surface.polygon_data(j);
        /* Get the rotation matrix mapping the world frame quantities to the
         contact frame quantities. */
        Matrix3<T> R_CW =
            math::ComputeBasisFromAxis(2, polygon_data.unit_normal).transpose();
        const int starting_row = 3 * (contact_start + j);
        // Build the contact Jacobian corresponding to the deformable object.
        {
          /* The contribution to the contact velocity from the deformable object
           A is R_CW * v_WAq. Note
             v_WAq = b₀ * v_WVᵢ₀ + b₁ * v_WVᵢ₁ + b₂ * v_WVᵢ₂ + b₃ * v_WVᵢ₃,
           where bₗ is the barycentric weight corresponding to vertex kₗ and
           v_WVₖₗ is the velocity of that vertex. */
          const Vector4<T>& barycentric_weights = polygon_data.b_centroid;
          const geometry::VolumeElement tet_element =
              deformable_mesh_W.element(polygon_data.tet_index);
          for (int l = 0; l < 4; ++l) {
            const int starting_col = 3 * tet_element.vertex(l);
            AddMatrix3ToEigenTriplets(R_CW * barycentric_weights(l),
                                      starting_row, starting_col, &triplets);
          }
        }
        // Build the contact Jacobian corresponding to the collision object.
        {
          /* The rigid objects entries are after the deformable dofs.
           The contribution to the contact velocity from the deformable object
           A is -R_CW * v_WBq. Note that v_WBq is the translational part of the
           spatial velocity V_WBq which can be obtained from the spatial
           velocity V_WB by v_WBq = v_WB + w_WB × p_BoBq_W, where p_BoBq_W is
           the position of Bq measured in the rigid B frame and expressed in the
           world frame. Hence the contact Jacobian with respect to the
           translational velocities dofs are -R_CW and the contact Jacobian
           w.r.t. the rotational velocity dofs are R_CW * [p_BoBq_W×], where
           [p_BoBq_W×] is the skew symmetric matrix representation of crossing
           producting with p_BoBq_W on the left. */
          const int starting_col_rotational =
              3 * deformable_mesh_W.num_vertices() + 6 * i;
          const Vector3<T>& p_Bq_W = polygon_data.centroid;
          const Vector3<T>& p_Bo_W = collision_object.pose().translation();
          const Vector3<T> p_BoBq_W = p_Bq_W - p_Bo_W;
          AddMatrix3ToEigenTriplets(
              R_CW * math::VectorToSkewSymmetric(p_BoBq_W), starting_row,
              starting_col_rotational, &triplets);
          const int starting_col_translational =
              3 * deformable_mesh_W.num_vertices() + 6 * i + 3;
          AddMatrix3ToEigenTriplets(-R_CW, starting_row,
                                    starting_col_translational, &triplets);
        }
      }
      // TODO(xuchenhan-tri): Set friction coeff based on proximity
      // properties of both objects in contact.
      const geometry::ProximityProperties& properties =
          collision_object.proximity_properties();
      const CoulombFriction<double> coulomb_friction =
          properties.GetProperty<CoulombFriction<double>>(
              geometry::internal::kMaterialGroup,
              geometry::internal::kFriction);
      mu_.segment(contact_start, contact_surface.num_polygons()) =
          VectorX<T>::Ones(contact_surface.num_polygons()) *
          coulomb_friction.static_friction();
      contact_start += contact_surface.num_polygons();
    }
    Jc_matrix_.setFromTriplets(triplets.begin(), triplets.end());
    Jc_ = std::make_unique<contact_solvers::internal::SparseLinearOperator<T>>(
        "Jc", &Jc_matrix_);
    return contact_solvers::internal::PointContactData<T>(
        &phi0_, Jc_.get(), &stiffness_, &dissipation_, &mu_);
  }

 private:
  void AddMatrix3ToEigenTriplets(const Eigen::Ref<const Matrix3<T>>& A,
                                 int starting_row, int starting_col,
                                 std::vector<Eigen::Triplet<T>>* triplets) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        triplets->emplace_back(starting_row + i, starting_col + j, A(i, j));
      }
    }
  }
  VectorX<T> phi0_;
  std::unique_ptr<contact_solvers::internal::LinearOperator<T>> Jc_;
  Eigen::SparseMatrix<T> Jc_matrix_;
  VectorX<T> stiffness_;
  VectorX<T> dissipation_;
  VectorX<T> mu_;
};
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
