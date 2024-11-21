#pragma once

#include <variant>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/parallelism.h"
#include "drake/common/unused.h"
#include "drake/multibody/fem/corotated_model.h"
#include "drake/multibody/fem/linear_constitutive_model.h"
#include "drake/multibody/fem/linear_corotated_model.h"
#include "drake/multibody/fem/stvk_hencky_von_mises_model.h"
#include "drake/multibody/mpm/math.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

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
  using Scalar = T;
  T ComputeTotalEnergy(const std::vector<Matrix3<T>>& deformation_gradient) const {
    T result = 0;
    for (int i = 0; i < ssize(materials); ++i) {
      const auto& constitutive_model = constitutive_models[i];
      for (int p = materials[i].first; p < materials[i].second; ++p) {
        std::visit(
            [&, this](auto& model) {
              const Matrix3<T>& F_p = deformation_gradient[p];
              using StrainDataType =
                  typename std::decay_t<decltype(model)>::Data;
              StrainDataType& strain_data_p =
                  std::get<StrainDataType>(strain_data[p]);
              // TODO(xuchenhan-tri): Use the actual F0.
              strain_data_p.UpdateData(F_p, F_p);
              T Psi;
              model.CalcElasticEnergyDensity(strain_data_p, &Psi);
              result += Psi * volume[p];
            },
            constitutive_model);
      }
    }
    return result;
  }

  void UpdateStress(bool apply_plasticity, Parallelism parallelism = false) {
    UpdateStress(&F, &tau_v0, apply_plasticity, parallelism);
  }

  // TODO(xuchenhan-tri): Right not, we have an inconsistency: we pass in
  // deformation gradient and the stress externally so that the data stored in
  // ParticleData aren't polluted, but we don't do the same for the StrainData.
  /* Use the deformation gradient data to compute the volume-scaled Kirchhoff
   stress for each particle. */
  void UpdateStress(std::vector<Matrix3<T>>* deformation_gradient,
                    std::vector<Matrix3<T>>* volume_scaled_stress,
                    bool apply_plasticity = false,
                    Parallelism parallelism = false) const {
    for (int i = 0; i < ssize(materials); ++i) {
      const auto& constitutive_model = constitutive_models[i];
      [[maybe_unused]] const int num_threads = parallelism.num_threads();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
      // TODO(xuchenhan-tri): This looks wrong... We should use data_indices[p]
      // instead of p itself.
      for (int p = materials[i].first; p < materials[i].second; ++p) {
        std::visit(
            [&, this](auto& model) {
              Matrix3<T>& F_p = (*deformation_gradient)[p];
              using StrainDataType =
                  typename std::decay_t<decltype(model)>::Data;
              StrainDataType& strain_data_p =
                  std::get<StrainDataType>(strain_data[p]);
              // TODO(xuchenhan-tri): Use the actual F0.
              if (apply_plasticity) {
                model.ProjectStrain(&F_p, &strain_data_p);
              } else {
                strain_data_p.UpdateData(F_p, F_p);
              }
              auto& tau_v0_p = (*volume_scaled_stress)[p];
              const Matrix3<T>& particle_F = F[p];
              model.CalcFirstPiolaStress(strain_data_p, &tau_v0_p);
              tau_v0_p *= volume[p] * particle_F.transpose();
            },
            constitutive_model);
      }
    }
  }

  void UpdateStressDerivatives(
      const std::vector<Matrix3<T>>& deformation_gradient,
      std::vector<math::FourthOrderTensor<T>>* volume_scaled_stress_derivatives,
      Parallelism parallelism = false) const {
    for (int i = 0; i < ssize(materials); ++i) {
      const auto& constitutive_model = constitutive_models[i];
      [[maybe_unused]] const int num_threads = parallelism.num_threads();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
      for (int p = materials[i].first; p < materials[i].second; ++p) {
        std::visit(
            [&, this](auto& model) {
              const Matrix3<T>& F_p = deformation_gradient[p];
              using StrainDataType =
                  typename std::decay_t<decltype(model)>::Data;
              StrainDataType& strain_data_p =
                  std::get<StrainDataType>(strain_data[p]);
              // TODO(xuchenhan-tri): The the actual F0.
              strain_data_p.UpdateData(F_p, F_p);
              auto& dPdF = (*volume_scaled_stress_derivatives)[p];
              model.CalcFirstPiolaStressDerivative(strain_data_p, &dPdF);
              dPdF.mutable_data() *= volume[p];
            },
            constitutive_model);
      }
    }
  }

  std::vector<T> m;                 // mass
  std::vector<Vector3<T>> x;        // positions
  std::vector<Vector3<T>> v;        // velocity
  std::vector<Matrix3<T>> F;        // deformation gradient
  std::vector<Matrix3<T>> C;        // affine velocity field
  std::vector<bool> in_constraint;  
  std::vector<Matrix3<T>>
      tau_v0;             // Kirchhoff stress scaled by reference volume
  std::vector<T> volume;  // reference volume
  mutable std::vector<DeformationGradientDataVariant<T>>
      strain_data;  // Deformation gradient dependent data that is used to
                    // calculate the energy density and its derivatives.

  std::vector<ConstitutiveModelVariant<T>> constitutive_models;
  std::vector<std::pair<int, int>>
      materials;  // Suppose materials[k] = (i, j), then particles with indices
                  // in [i, j) have the same material: constitutive_models[k].
  std::vector<Vector3<T>> f;  // contact impulse.
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
        particles.m[i] * (particles.x[i].cross(particles.v[i]) +
                          ContractWithLeviCivita<T>(B.transpose()));
  }
  return result;
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
