#pragma once

#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/math.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
struct Particle {
  Particle(const T& m_in, const Vector3<T>& x_in, const Vector3<T>& v_in,
           const Matrix3<T>& F_in, const Matrix3<T>& C_in,
           const Matrix3<T>& P_in)
      : m(m_in), x(x_in), v(v_in), F(F_in), C(C_in), P(P_in) {}
  T m;
  Vector3<T> x;
  Vector3<T> v;
  Matrix3<T> F;
  Matrix3<T> C;
  Matrix3<T> P;
};

template <typename T>
using Particles = std::vector<Particle<T>>;

template <typename T>
MassAndMomentum<T> ComputeTotalMassAndMomentum(const Particles<T>& particles,
                                               const T& dx) {
  MassAndMomentum<T> result;
  const T D = dx * dx * 0.25;
  const int num_particles = particles.size();
  for (int i = 0; i < num_particles; ++i) {
    result.mass += particles[i].m;
    result.linear_momentum += particles[i].m * particles[i].v;
    const Matrix3<T> B = particles[i].C * D;  // C = B * D^{-1}
    result.angular_momentum +=
        particles[i].m *
        (particles[i].x.cross(particles[i].v) + ContractWithLeviCivita(B));
  }
  return result;
}

template <typename T>
Matrix3<SimdScalar<T>> LoadC(const Particles<T>& particles,
                             const std::vector<int>& indices) {
  Matrix3<T> data[indices.size()];
  for (size_t i = 0; i < indices.size(); ++i) {
    data[i] = particles[indices[i]].C;
  }
  return Load(data, indices.size());
}

template <typename T>
Matrix3<SimdScalar<T>> LoadF(const Particles<T>& particles,
                             const std::vector<int>& indices) {
  Matrix3<T> data[indices.size()];
  for (size_t i = 0; i < indices.size(); ++i) {
    data[i] = particles[indices[i]].F;
  }
  return Load(data, indices.size());
}

template <typename T>
Vector3<SimdScalar<T>> LoadX(const Particles<T>& particles,
                             const std::vector<int>& indices) {
  Vector3<T> data[indices.size()];
  for (size_t i = 0; i < indices.size(); ++i) {
    data[i] = particles[indices[i]].x;
  }
  return Load(data, indices.size());
}

template <typename T>
void StoreC(const Matrix3<SimdScalar<T>>& C, Particles<T>* particles,
            const std::vector<int>& indices) {
  Matrix3<T> data[indices.size()];
  Store(C, data, indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    (*particles)[indices[i]].C = data[i];
  }
}

template <typename T>
void StoreF(const Matrix3<SimdScalar<T>>& F, Particles<T>* particles,
            const std::vector<int>& indices) {
  Matrix3<T> data[indices.size()];
  Store(F, data, indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    (*particles)[indices[i]].F = data[i];
  }
}

template <typename T>
void StoreX(const Vector3<SimdScalar<T>>& x, Particles<T>* particles,
            const std::vector<int>& indices) {
  Vector3<T> data[indices.size()];
  Store(x, data, indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    (*particles)[indices[i]].x = data[i];
  }
}

template <typename T>
void StoreV(const Vector3<SimdScalar<T>>& v, Particles<T>* particles,
            const std::vector<int>& indices) {
  Vector3<T> data[indices.size()];
  Store(v, data, indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    (*particles)[indices[i]].v = data[i];
  }
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
