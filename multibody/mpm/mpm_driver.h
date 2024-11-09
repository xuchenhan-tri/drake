#pragma once

#include "particles.h"
#include "sparse_grid.h"

#include "drake/common/copyable_unique_ptr.h"
#include "drake/common/parallelism.h"
#include "drake/geometry/geometry_instance.h"
#include "drake/geometry/query_object.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* Solves the contact problem for a single particle against a rigid body
 assuming the rigid body has infinite mass and inertia.

 Let phi be the penetration distance (positive when penetration occurs) and vn
 be the relative velocity of the particle with respect to the rigid body in the
normal direction (vn>0 when separting). Then we have phi_dot = -vn.

In the normal direction, the contact force is modeled as a linear elastic system
with Hunt-Crossley dissipation.

  f = k * phi_+ * (1 + d * phi_dot)_+

  where phi_+ = max(0, phi)

The momentum balance in the normal direction becomes

m(vn_next - vn) = k * dt * (phi0 - dt * vn_next)_+ * (1 - d * vn_next)_+

where we used the fact that phi = phi0 - dt * vn_next. This is a quadratic
equation in vn_next, and we solve it to get the next velocity vn_next.

The quadratic equation is ax^2 + bx + c = 0, where

a = k * d * dt^2
b = -m - (k * dt * (dt + d * phi0))
c = k * dt * phi0 + m * vn

After solving for vn_next, we check if the friction force lies in the friction
cone, if not, we project the velocity back into the friction cone. */
template <typename T>
class ContactForceSolver {
 public:
  ContactForceSolver(T dt, T k, T d) : dt_(dt), k_(k), d_(d) {}
  // TODO(xuchenhan-tri): Take in the entire velocity vector and return the
  // next velocity (vector) after treating friction.
  T Solve(T m, T v0, T phi0, T volume) const {
    T v_hat = std::min(phi0 / dt_, 1 / d_);
    if (v0 > v_hat) return v0;
    T effective_k = k_ * volume;
    T a = effective_k * d_ * dt_ * dt_;
    T b = -m - (effective_k * dt_ * (dt_ + d_ * phi0));
    T c = effective_k * dt_ * phi0 + m * v0;
    T discriminant = b * b - 4.0 * a * c;
    T v_next = (-b - std::sqrt(discriminant)) / (2.0 * a);
    return v_next;
  }

 private:
  T dt_;
  T k_;
  T d_;
};

struct ContactPair {
  int particle_index{};
  int contact_particle_index{};
  int constraint_index{};
  int rigid_body_index{};
  double friction_coeffcient{};
  Vector3<double> nhat_W;
  double penetration_depth{};
  Vector3<double> rigid_velocity;
  Vector3<double> rigid_position;
};

template <typename T>
class MpmDriver {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MpmDriver);

  MpmDriver(T dt, T dx, int num_subteps, Parallelism parallelism = false)
      : dt_(dt),
        num_subteps_(num_subteps),
        substep_dt_(dt / num_subteps),
        dx_(dx),
        grid_(std::make_unique<SparseGrid<T>>(dx, parallelism)),
        parallelism_(parallelism) {
    DRAKE_THROW_UNLESS(num_subteps > 0);
    DRAKE_THROW_UNLESS(dt > 0);
    DRAKE_THROW_UNLESS(dx > 0);
  }

  /* Sample particles inside the given geometry.
   @param[in] geometry_instance The geometry instance to sample particles
   inside. Only the shape and pose of the geometry is used; all the geometry
   properties are discarded.
   @param[in] particles_per_cell The targeted number of particle to be sampled
   in each grid cell (of size dx * dx * dx).
   @param[in] config  The physical properties of the material. */
  void SampleParticles(
      std::unique_ptr<geometry::GeometryInstance> geometry_instance,
      int particles_per_cell, const fem::DeformableBodyConfig<T>& config);

  void AdvanceOneTimeStep();

  void AdvanceOneTimeStep(
      const geometry::QueryObject<double>& query_object,
      const std::vector<multibody::SpatialVelocity<double>>& spatial_velocities,
      const std::vector<math::RigidTransform<double>>& poses,
      const std::unordered_map<geometry::GeometryId, multibody::BodyIndex>&
          geometry_id_to_body_index);

  // TODO(xuchenhan-tri): Move these geometry operations to SceneGraph.
  std::vector<ContactPair> CalcContactPairs(
      const geometry::QueryObject<double>& query_object,
      const std::vector<multibody::SpatialVelocity<double>>& spatial_velocities,
      const std::vector<math::RigidTransform<double>>& poses,
      const std::unordered_map<geometry::GeometryId, multibody::BodyIndex>&
          geometry_id_to_body_index) const;

  Particles<T> MakeContactParticles(
      const Particles<T>& all_particles,
      const std::vector<ContactPair>& contact_pairs) const;

  double ApplyImpulse(const std::vector<ContactPair>& contact_pairs,
                      const ContactForceSolver<double>& solver,
                      ParticleData<T>* particle_data,
                      std::vector<Vector3<double>>* impulses) const;

  void SolveContact(const std::vector<ContactPair>& contact_pairs);

  const Particles<T>& particles() const { return particles_; }

  const std::vector<multibody::ExternallyAppliedSpatialForce<double>>&
  rigid_forces() const {
    return rigid_forces_;
  }

 private:
  void UpdateParticleStress();

  T dt_{0.0};
  int num_subteps_{0};
  T substep_dt_{0.0};
  T dx_{0.0};
  Vector3<T> gravity_{0, 0, -9.81};
  copyable_unique_ptr<SparseGrid<T>> grid_;
  Particles<T> particles_;
  Parallelism parallelism_;
  std::vector<multibody::ExternallyAppliedSpatialForce<double>> rigid_forces_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
