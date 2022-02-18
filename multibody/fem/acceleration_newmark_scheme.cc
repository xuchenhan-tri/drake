#include "drake/multibody/fem/acceleration_newmark_scheme.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T>
void AccelerationNewmarkScheme<T>::DoUpdateStateFromChangeInUnknowns(
    const VectorX<T>& dz, FemData<T>* fem_data) const {
  const VectorX<T>& a = fem_data->GetAccelerations();
  const VectorX<T>& v = fem_data->GetVelocities();
  const VectorX<T>& x = fem_data->GetPositions();
  fem_data->SetAccelerations(a + dz);
  fem_data->SetVelocities(v + dt() * gamma() * dz);
  fem_data->SetPositions(x + dt() * dt() * beta() * dz);
}

template <typename T>
void AccelerationNewmarkScheme<T>::DoAdvanceOneTimeStep(
    const FemData<T>& prev_fem_data, const VectorX<T>& unknown_variable,
    FemData<T>* next_fem_data) const {
  const VectorX<T>& an = prev_fem_data.GetAccelerations();
  const VectorX<T>& vn = prev_fem_data.GetVelocities();
  const VectorX<T>& xn = prev_fem_data.GetPositions();
  const VectorX<T>& a = unknown_variable;
  /* Update x, v, a in that order to ensure we handle the case where
   &prev_fem_data == next_fem_data. */
  next_fem_data->SetPositions(xn + dt() * vn +
                              dt() * dt() * (beta() * a + (0.5 - beta()) * an));
  next_fem_data->SetVelocities(vn +
                               dt() * (gamma() * a + (1.0 - gamma()) * an));
  next_fem_data->SetAccelerations(a);
}

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::internal::AccelerationNewmarkScheme)
