#pragma once

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
struct SolverState {
  SolverState(int num_dofs, int num_particles)
      : dv(VectorX<T>::Zero(num_dofs)),
        F(num_particles, Matrix3<T>::Identity()),
        tau_v0(num_particles, Matrix3<T>::Zero()),
        volume_scaled_stress_derivatives(num_particles) {
    DRAKE_DEMAND(num_dofs > 0);
    DRAKE_DEMAND(num_particles > 0);
  }

  VectorX<T> dv;
  std::vector<Matrix3<T>> F;
  std::vector<Matrix3<T>> tau_v0;
  std::vector<math::FourthOrderTensor<T>> volume_scaled_stress_derivatives;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake