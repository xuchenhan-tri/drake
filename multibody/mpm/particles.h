#pragma once

#include <variant>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/corotated_model.h"
#include "drake/multibody/fem/linear_constitutive_model.h"
#include "drake/multibody/fem/linear_corotated_model.h"
#include "drake/multibody/fem/stvk_hencky_von_mises_model.h"
#include "drake/multibody/mpm/math.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* Data structure for conveniently accessing/modifying the attributes of a
 single particle. */
template <typename T>
struct Particle {
  Particle(T* m_in, Vector3<T>* x_in, Vector3<T>* v_in, Matrix3<T>* F_in,
           Matrix3<T>* C_in, Matrix3<T>* tau_v0_in)
      : m(*m_in), x(*x_in), v(*v_in), F(*F_in), C(*C_in), tau_v0(*tau_v0_in) {}
  T& m;
  Vector3<T>& x;
  Vector3<T>& v;
  Matrix3<T>& F;
  Matrix3<T>& C;
  Matrix3<T>& tau_v0;
};

template <typename T>
using ConstitutiveModelVariant =
    std::variant<fem::internal::CorotatedModel<T>,
                 fem::internal::LinearCorotatedModel<T>,
                 fem::internal::LinearConstitutiveModel<T>,
                 fem::internal::StvkHenckyVonMisesModel<T>>;

template <typename T>
using DeformationGradientDataVariant =
    std::variant<fem::internal::CorotatedModelData<T>,
                 fem::internal::LinearCorotatedModelData<T>,
                 fem::internal::LinearConstitutiveModelData<T>,
                 fem::internal::StvkHenckyVonMisesModelData<T>>;

/* The collection of all physical attributes we care about for all particles.
 All quantities are measured and expressed in the world frame (when
 applicable).
 @tparam double or float. */
template <typename T>
struct ParticleData {
  Particle<T> particle(int i) {
    return Particle<T>(&m[i], &x[i], &v[i], &F[i], &C[i], &tau_v0[i]);
  }

  std::vector<T> m;           // mass
  std::vector<Vector3<T>> x;  // positions
  std::vector<Vector3<T>> v;  // velocity
  std::vector<Matrix3<T>> F;  // deformation gradient
  std::vector<Matrix3<T>> C;  // affine velocity field
  std::vector<Matrix3<T>>
      tau_v0;             // Kirchhoff stress scaled by reference volume
  std::vector<T> volume;  // reference volume
  std::vector<DeformationGradientDataVariant<T>>
      strain_data;  // Deformation gradient dependent data that is used to
                    // calculate the energy density and its derivatives.

  std::vector<ConstitutiveModelVariant<T>> constitutive_models;
  std::vector<std::pair<int, int>>
      materials;  // Suppose materials[k] = (i, j), then particles with indices
                  // in [i, j) have the same material: constitutive_models[k].
  std::vector<Vector3<T>> f;  // contact impulse.

  // TODO(xuchenhan-tri): Consider decoupling the following data from the
  // particle data structure.

  /* Data used to iterate over all particles in the grid. */
  /* All but last entry store indices of particles marking the boundary of a new
   block. The last entry stores the number of particles. */
  std::vector<int> sentinel_particles;
  /* The order in which the particle data should be accessed when used in tandem
   with a grid. That is, particle_data[particle_indices()[p]] gives the particle
   data for the p-th particle. */
  std::vector<int> data_indices;
  /* Returns the base node offset of the associated grid for each particle. */
  std::vector<uint64_t> base_node_offsets;
  /* Helper data to sort the particles according to their base nodes. */
  std::vector<uint64_t> particle_sorters;
  /* We color SPGrid blocks so that writing to different blocks with the same
  color is guaranteed to be free of write hazards. This function returns the
  block indices for each color associated with this particle data. */
  std::array<std::vector<int>, 8> colored_blocks;
};

template <typename T>
struct ContactParticleData {
  std::vector<double> m;                         // mass
  std::vector<double> volume;                    // volume
  std::vector<Vector3<T>> x;                     // positions
  std::vector<Vector3<T>> v;                     // particle velocity
  std::vector<std::vector<Vector3<T>>> f;        // particle contact momentum
  std::vector<std::vector<Vector3<double>>> vr;  // rigid velocity
  std::vector<std::vector<Vector3<double>>>
      pr;  // position of the contact point in the rigid frame
  std::vector<std::vector<Vector3<double>>>
      nhat_W;                                  // contact normal in world frame
  std::vector<std::vector<double>> phi;        // penetration depth
  std::vector<std::vector<int>> body_indices;  // rigid body indices;
  std::vector<std::vector<double>> mu;         // friction coefficients.
};

template <typename T>
MassAndMomentum<T> ComputeTotalMassAndMomentum(const ParticleData<T>& particles,
                                               const T& dx) {
  MassAndMomentum<T> result;
  const T D = dx * dx * 0.25;
  const int num_particles = particles.m.size();
  for (int i = 0; i < num_particles; ++i) {
    result.mass += particles.m[i];
    result.linear_momentum += particles.m[i] * particles.v[i];
    const Matrix3<T> B = particles.C[i] * D;  // C = B * D^{-1}
    result.angular_momentum +=
        particles.m[i] *
        (particles.x[i].cross(particles.v[i]) + ContractWithLeviCivita(B));
  }
  return result;
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
