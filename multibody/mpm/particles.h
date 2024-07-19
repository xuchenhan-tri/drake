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
           const Matrix3<T>& P_in, const BSplineWeights<T>& bspline_in)
      : m(m_in),
        x(x_in),
        v(v_in),
        F(F_in),
        C(C_in),
        P(P_in),
        bspline(bspline_in) {}
  T m;
  Vector3<T> x;
  Vector3<T> v;
  Matrix3<T> F;
  Matrix3<T> C;
  Matrix3<T> P;
  BSplineWeights<T> bspline;
};

// TODO(xuchenhan-tri): Compare with AOS.
template <typename T>
struct ParticleData {
  const Particle<T>& particle(int i) const { return particles[i]; }
  Particle<T>& particle(int i) { return particles[i]; }

  std::vector<Particle<T>> particles;
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
