#pragma once

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
struct SolverState {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SolverState)
  SolverState() = default;

  SolverState(int num_dofs, int num_particles)
      : dv(VectorX<T>::Zero(num_dofs)),
        F(num_particles, Matrix3<T>::Identity()),
        tau_v0(num_particles, Matrix3<T>::Zero()),
        volume_scaled_stress_derivatives(num_particles) {
    DRAKE_DEMAND(num_dofs > 0);
    DRAKE_DEMAND(num_particles > 0);
  }

  void Resize(int num_dofs, int num_particles) {
    DRAKE_DEMAND(num_dofs > 0);
    DRAKE_DEMAND(num_particles > 0);
    dv.resize(num_dofs);
    F.resize(num_particles);
    tau_v0.resize(num_particles);
    volume_scaled_stress_derivatives.resize(num_particles);
  }

  int num_dofs() const { return dv.size(); }
  int num_particles() const { return F.size(); }

  VectorX<T> dv;
  std::vector<Matrix3<T>> F;
  std::vector<Matrix3<T>> tau_v0;
  std::vector<math::FourthOrderTensor<T>> volume_scaled_stress_derivatives;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake