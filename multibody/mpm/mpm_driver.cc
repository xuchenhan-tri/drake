#include "mpm_driver.h"

#include <iostream>
#include <variant>

#include "sort_particles.h"
#include "transfer.h"

#include "drake/common/ssize.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/corotated_model.h"
#include "drake/multibody/mpm/poisson_disk.h"
// TODO(xuchenhan-tri): contact_properties is internal to the plant. Computing
// contact points should be done in the contact manager.
#include "drake/multibody/plant/contact_properties.h"
#include "drake/multibody/plant/coulomb_friction.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

using drake::geometry::SignedDistanceToPoint;
using Eigen::Vector3d;
using fem::DeformableBodyConfig;
using geometry::Box;
using geometry::GeometryInstance;
using geometry::Shape;
using geometry::ShapeReifier;
using geometry::Sphere;
using math::RigidTransform;

class BoundingBoxCalculator : public ShapeReifier {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BoundingBoxCalculator);

  BoundingBoxCalculator() = default;

  using BoundingBox = std::array<std::array<double, 3>, 2>;

  BoundingBox Compute(const Shape& shape) {
    BoundingBox result;
    shape.Reify(this, &result);
    return result;
  }

  void ImplementGeometry(const Sphere& sphere, void* data) override {
    DRAKE_ASSERT(data != nullptr);
    BoundingBox& result = *static_cast<BoundingBox*>(data);
    for (int d = 0; d < 3; ++d) {
      result[0][d] = -sphere.radius();
      result[1][d] = sphere.radius();
    }
  }

  void ImplementGeometry(const Box& box, void* data) override {
    DRAKE_ASSERT(data != nullptr);
    BoundingBox& result = *static_cast<BoundingBox*>(data);
    for (int d = 0; d < 3; ++d) {
      result[0][d] = -box.size()(d) / 2.0;
      result[1][d] = box.size()(d) / 2.0;
    }
  }
};

template <typename T>
ConstitutiveModelVariant<T> MakeConstitutiveModel(
    const DeformableBodyConfig<T>& config) {
  switch (config.material_model()) {
    case fem::MaterialModel::kCorotated:
      return fem::internal::CorotatedModel<T>(config.youngs_modulus(),
                                              config.poissons_ratio());
    case fem::MaterialModel::kLinearCorotated:
      return fem::internal::LinearCorotatedModel<T>(config.youngs_modulus(),
                                                    config.poissons_ratio());
    case fem::MaterialModel::kLinear:
      return fem::internal::LinearConstitutiveModel<T>(config.youngs_modulus(),
                                                       config.poissons_ratio());
    case fem::MaterialModel::kStvkHenckyVonMises:
      return fem::internal::StvkHenckyVonMisesModel<T>(config.youngs_modulus(),
                                                       config.poissons_ratio(),
                                                       config.yield_stress());
  }
  DRAKE_UNREACHABLE();
}

template <typename T>
void MpmDriver<T>::SampleParticles(
    std::unique_ptr<GeometryInstance> geometry_instance, int particles_per_cell,
    const DeformableBodyConfig<T>& config) {
  DRAKE_THROW_UNLESS(geometry_instance != nullptr);
  DRAKE_THROW_UNLESS(particles_per_cell > 0);

  BoundingBoxCalculator calculator;
  const std::array<std::array<double, 3>, 2> bounding_box =
      calculator.Compute(geometry_instance->shape());
  const double sampling_radius =
      dx_ / std::cbrt(particles_per_cell * 4.0 / 3.0 * M_PI);
  /* Sample the particles in the geometry's bounding box. */
  const std::vector<Vector3<double>> q_GP_candidates =
      PoissonDiskSampling<double>(sampling_radius, bounding_box[0],
                                  bounding_box[1]);
  /* Reject points that fall outside of the shape. */
  std::vector<Vector3<double>> q_GPs =
      FilterPoints(q_GP_candidates, geometry_instance->shape());
  std::vector<Vector3<T>> q_GPs_T;
  q_GPs_T.reserve(q_GPs.size());
  for (const Vector3<double>& q_GP : q_GPs) {
    q_GPs_T.push_back(q_GP.cast<T>());
  }
  SortParticlePositions(grid_->spgrid(), &q_GPs_T, dx_);
  const int num_particles = ssize(q_GPs);
  const T mass_density = config.mass_density();
  const double total_volume = geometry::CalcVolume(geometry_instance->shape());
  const T volume_per_particle = total_volume / num_particles;
  const RigidTransform<double>& X_WG = geometry_instance->pose();
  auto& particle_data = particles_.data;
  const int num_existing_particles = ssize(particle_data.m);
  ConstitutiveModelVariant<T> constitutive_model =
      MakeConstitutiveModel<T>(config);
  for (int i = 0; i < num_particles; ++i) {
    const Vector3<double> q_WP = X_WG * q_GPs[i];
    particle_data.m.push_back(mass_density * volume_per_particle);
    particle_data.x.push_back(q_WP.cast<T>());
    particle_data.v.push_back({0, 0, 0});
    particle_data.F.push_back(Matrix3<T>::Identity());
    particle_data.tau_v0.push_back(Matrix3<T>::Zero());
    particle_data.C.push_back(Matrix3<T>::Zero());
    particle_data.in_constraint.push_back(false);
    particle_data.volume.push_back(volume_per_particle);
    std::visit(
        [&, this](auto& model) {
          particle_data.strain_data.push_back(model.MakeDefaultData());
        },
        constitutive_model);
  }
  particle_data.constitutive_models.push_back(constitutive_model);
  particle_data.materials.push_back(
      {num_existing_particles, num_existing_particles + num_particles});
}

template <typename T>
void MpmDriver<T>::AdvanceOneTimeStep() {
  for (int i = 0; i < num_subteps_; ++i) {
    particles_.data.UpdateStress(/*apply plasticity*/ true);
    // Particle to grid transfer.
    Transfer<T> transfer(substep_dt_, grid_.get_mutable(), &particles_);
    transfer.ParallelSimdParticleToGrid(parallelism_);
    // Grid velocity update.
    grid_->ExplicitVelocityUpdate(gravity_ * substep_dt_);
    // Grid to particle transfer.
    transfer.ParallelSimdGridToParticle(parallelism_);
  }
}

template <typename T>
void MpmDriver<T>::AdvanceOneTimeStep(
    const geometry::QueryObject<double>& query_object,
    const std::vector<multibody::SpatialVelocity<double>>& spatial_velocities,
    const std::vector<math::RigidTransform<double>>& poses,
    const std::unordered_map<geometry::GeometryId, multibody::BodyIndex>&
        geometry_id_to_body_index) {
  rigid_forces_.resize(poses.size());
  for (int i = 0; i < ssize(rigid_forces_); ++i) {
    /* We use `p_BoBq_B` to temporarily store p_WB. We will replace it with the
     actual value of p_BoBq_B later on. */
    auto& force = rigid_forces_[i];
    force.body_index = BodyIndex(i);
    force.p_BoBq_B = poses[i].translation();
    force.F_Bq_W.SetZero();
  }
  for (int i = 0; i < num_subteps_; ++i) {
    particles_.data.UpdateStress(/*apply plasticity*/ true);
    /* Update particle's contact (and friction) momentum and accumulate the
     opposite momentum in rigid_forces_. */
    for (auto& v : particles_.data.v) {
      v += gravity_.template cast<T>() * substep_dt_;
    }
    const std::vector<ContactPair> contact_pairs = CalcContactPairs(
        query_object, spatial_velocities, poses, geometry_id_to_body_index);
    // Particle to grid transfer.
    Transfer<T> transfer(substep_dt_, grid_.get_mutable(), &particles_);
    transfer.ParallelSimdParticleToGrid(parallelism_);
    SolveContact(contact_pairs);
    // Grid velocity update.
    grid_->ExplicitVelocityUpdate(Vector3<T>::Zero());
    // Grid to particle transfer.
    transfer.ParallelSimdGridToParticle(parallelism_);
  }
  /* Restore p_BoBq_B value and divide by dt to turn impulse into forces. */
  for (int i = 0; i < ssize(rigid_forces_); ++i) {
    rigid_forces_[i].p_BoBq_B = Vector3<double>::Zero();
    rigid_forces_[i].F_Bq_W.rotational() /= dt_;
    rigid_forces_[i].F_Bq_W.translational() /= dt_;
  }
}

template <typename T>
std::vector<ContactPair> MpmDriver<T>::CalcContactPairs(
    const geometry::QueryObject<double>& query_object,
    const std::vector<multibody::SpatialVelocity<double>>& spatial_velocities,
    const std::vector<math::RigidTransform<double>>& poses,
    const std::unordered_map<geometry::GeometryId, multibody::BodyIndex>&
        geometry_id_to_body_index) const {
  std::vector<ContactPair> contact_pairs;
  int contact_particle_index = 0;
  int constraint_index = 0;
  const auto& particle_data = particles_.data;
  for (int p = 0; p < ssize(particle_data.m); ++p) {
    const Vector3<double>& p_WP = particle_data.x[p].template cast<double>();
    // TODO(xuchenhan-tri): Consider building a constraint for particles that
    // are within a margin of the rigid body.
    const std::vector<SignedDistanceToPoint<double>>& signed_distances =
        query_object.ComputeSignedDistanceToPoint(p_WP, 0);
    if (signed_distances.empty()) {
      continue;
    }
    for (const SignedDistanceToPoint<double>& sd : signed_distances) {
      const double& phi = -sd.distance;
      DRAKE_THROW_UNLESS(phi >= 0.0);

      const int body_index = geometry_id_to_body_index.at(sd.id_G);

      const CoulombFriction<double>& coulomb_friction =
          multibody::internal::GetCoulombFriction(sd.id_G,
                                                  query_object.inspector());
      const double mu = coulomb_friction.dynamic_friction();

      const Vector3<double> nhat_W = sd.grad_W.normalized();

      /* World frame position of the origin of the rigid body. */
      const Vector3<double>& p_WR = poses[body_index].translation();
      /* Position of the contact point in the rigid body frame, expressed in the
       world frame. */
      const Vector3<double> p_RP_W = p_WP - p_WR;
      /* World frame velocity of a point affixed to the rigid body that
       coincide with the particle. */
      const Vector3<double> v_WRp =
          spatial_velocities[body_index].Shift(p_RP_W).translational();

      contact_pairs.emplace_back(
          ContactPair{.particle_index = p,
                      .contact_particle_index = contact_particle_index,
                      .constraint_index = constraint_index++,
                      .rigid_body_index = body_index,
                      .friction_coeffcient = mu,
                      .nhat_W = nhat_W,
                      .penetration_depth = phi,
                      .rigid_velocity = v_WRp,
                      .rigid_position = p_RP_W});
    }
    ++contact_particle_index;
  }
  return contact_pairs;
}

template <typename T>
double MpmDriver<T>::ApplyImpulse(
    const std::vector<ContactPair>& contact_pairs,
    const ContactForceSolver<double>& solver, ParticleData<T>* particle_data,
    std::vector<Vector3<double>>* impulses) const {
  DRAKE_DEMAND(particle_data != nullptr);
  DRAKE_DEMAND(impulses != nullptr);
  DRAKE_DEMAND(contact_pairs.size() == impulses->size());
  for (Vector3<T>& f : particle_data->f) {
    f.setZero();
  }
  double impulse_error = 0.0;
  for (const ContactPair& pair : contact_pairs) {
    const int p = pair.contact_particle_index;
    const int c = pair.constraint_index;
    const Vector3<double>& vp = particle_data->v[p].template cast<double>();
    const double volume = particle_data->volume[p];
    const double mp = particle_data->m[p];
    const Vector3<double> vc = vp - pair.rigid_velocity;
    const Vector3<double>& nhat_W = pair.nhat_W;
    const double vn = vc.dot(nhat_W);
    const double vn_next = solver.Solve(mp, vn, pair.penetration_depth, volume);
    const Vector3<double>& old_impulse = (*impulses)[c];
    Vector3<double> new_impulse;
    if (vn_next != vn) {
      const Vector3<double> vt = vc - vn * nhat_W;
      double dvn = vn_next - vn;
      double fn = dvn * mp;
      Vector3<double> ft = -vt * mp;
      if (ft.norm() > fn * pair.friction_coeffcient) {
        ft.normalize();
        ft *= fn * pair.friction_coeffcient;
      }
      new_impulse = old_impulse + fn * nhat_W + ft;
    } else {
      new_impulse = old_impulse;  //  Vector3<double>::Zero();
    }
    const Vector3<double> df = new_impulse - old_impulse;
    impulse_error += df.norm() / (old_impulse.norm() + 1e-9);
    (*impulses)[c] = new_impulse;
    particle_data->f[p] += df.template cast<T>();
  }
  return (impulse_error / contact_pairs.size());
}

template <typename T>
Particles<T> MpmDriver<T>::MakeContactParticles(
    const Particles<T>& all_particles,
    const std::vector<ContactPair>& contact_pairs) const {
  int num_contact_particles = 0;
  for (const ContactPair& pair : contact_pairs) {
    const int c = pair.contact_particle_index;
    num_contact_particles = std::max(num_contact_particles, c + 1);
  }
  Particles<T> contact_particles;
  auto& contact_particle_data = contact_particles.data;
  contact_particle_data.m.resize(num_contact_particles);
  contact_particle_data.x.resize(num_contact_particles);
  contact_particle_data.v.resize(num_contact_particles);
  contact_particle_data.volume.resize(num_contact_particles);
  contact_particle_data.f.resize(num_contact_particles);
  for (const ContactPair& pair : contact_pairs) {
    const int p = pair.particle_index;
    const int c = pair.contact_particle_index;
    contact_particle_data.m[c] = all_particles.data.m[p];
    contact_particle_data.x[c] = all_particles.data.x[p];
    contact_particle_data.v[c] = all_particles.data.v[p];
    contact_particle_data.volume[c] = all_particles.data.volume[p];
  }
  return contact_particles;
}

template <typename T>
void MpmDriver<T>::SolveContact(const std::vector<ContactPair>& contact_pairs) {
  const double kStiffness = 1e7;
  const double kDamping = 1.0;
  const double substep_dt = dt_ / double(num_subteps_);
  ContactForceSolver<double> solver(substep_dt, kStiffness, kDamping);

  Particles<T> contact_particles =
      MakeContactParticles(particles_, contact_pairs);
  std::vector<Vector3<double>> impulses(ssize(contact_pairs),
                                        Vector3<double>::Zero());

  Transfer<T> transfer(substep_dt, grid_.get_mutable(), &contact_particles,
                       false);

  double impulse_error = 1e10;
  const double kTol = 1e-3;
  int count = 0;
  int max_iterations = 1000;
  while (impulse_error > kTol && count < max_iterations) {
    ++count;
    impulse_error =
        ApplyImpulse(contact_pairs, solver, &contact_particles.data, &impulses);
    transfer.ContactP2G2P();
  }
  if (count == max_iterations) {
    std::cout << "Contact solver did not converge." << std::endl;
  } else {
    std::cout << "Contact solver converged in " << count << " iterations."
              << std::endl;
  }

  /* Accumulate the contact impulses on the rigid bodies. */
  for (int i = 0; i < ssize(contact_pairs); ++i) {
    const ContactPair& pair = contact_pairs[i];
    const Vector3<double>& impulse = impulses[i];
    const Vector3<double>& p_RP_W = pair.rigid_position;
    /* The impulse on the rigid is the opposite of that on the particle. */
    const Vector3<double> l_WR_W = -impulse;
    const Vector3<double> h_WPRo_W = p_RP_W.cross(l_WR_W);
    rigid_forces_[pair.rigid_body_index].F_Bq_W +=
        SpatialForce<double>(h_WPRo_W, l_WR_W);
  }
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::internal::MpmDriver<double>;
template class drake::multibody::mpm::internal::MpmDriver<float>;