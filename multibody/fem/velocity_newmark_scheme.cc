#include "drake/multibody/fem/velocity_newmark_scheme.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T>
void VelocityNewmarkScheme<T>::DoUpdateStateFromChangeInUnknowns(
    const VectorX<T>& dz, FemData<T>* fem_data) const {
  const VectorX<T>& a = fem_data->GetAccelerations();
  const VectorX<T>& v = fem_data->GetVelocities();
  const VectorX<T>& x = fem_data->GetPositions();
  fem_data->SetAccelerations(a + one_over_dt_gamma_ * dz);
  fem_data->SetVelocities(v + dz);
  fem_data->SetPositions(x + dt() * beta_over_gamma_ * dz);
}

template <typename T>
void VelocityNewmarkScheme<T>::DoAdvanceOneTimeStep(
    const FemData<T>& prev_fem_data, const VectorX<T>& unknown_variable,
    FemData<T>* next_fem_data) const {
  const VectorX<T>& an = prev_fem_data.GetAccelerations();
  const VectorX<T>& vn = prev_fem_data.GetVelocities();
  const VectorX<T>& xn = prev_fem_data.GetPositions();
  const VectorX<T>& v = unknown_variable;
  /* Update x, a, v in that order to ensure we handle the case where
   &prev_fem_data == fem_data. */
  next_fem_data->SetPositions(
      xn + dt() * (beta_over_gamma_ * v + (1.0 - beta_over_gamma_) * vn) +
      dt() * dt() * (0.5 - beta_over_gamma_) * an);
  next_fem_data->SetAccelerations(one_over_dt_gamma_ * (v - vn) -
                                  (1.0 - gamma()) / gamma() * an);
  next_fem_data->SetVelocities(v);
}

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::internal::VelocityNewmarkScheme)
