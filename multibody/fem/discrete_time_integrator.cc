#include "drake/multibody/fem/discrete_time_integrator.h"

// TODO(xuchenhan-tri): Add unit tests for this class.

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T>
Vector3<T> DiscreteTimeIntegrator<T>::weights() const {
  return do_get_weights();
}

template <typename T>
const VectorX<T>& DiscreteTimeIntegrator<T>::GetUnknowns(
    const FemData<T>& fem_data) const {
  return DoGetUnknowns(fem_data);
}

template <typename T>
void DiscreteTimeIntegrator<T>::UpdateStateFromChangeInUnknowns(
    const VectorX<T>& dz, FemData<T>* fem_data) const {
  DRAKE_DEMAND(fem_data != nullptr);
  DRAKE_DEMAND(dz.size() == fem_data->num_dofs());
  DoUpdateStateFromChangeInUnknowns(dz, fem_data);
}

template <typename T>
void DiscreteTimeIntegrator<T>::AdvanceOneTimeStep(
    const FemData<T>& prev_fem_data, const VectorX<T>& unknown_variable,
    FemData<T>* next_fem_data) const {
  DRAKE_DEMAND(next_fem_data != nullptr);
  DRAKE_DEMAND(prev_fem_data.num_dofs() == next_fem_data->num_dofs());
  DRAKE_DEMAND(prev_fem_data.num_dofs() == unknown_variable.size());
  DoAdvanceOneTimeStep(prev_fem_data, unknown_variable, next_fem_data);
}

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::internal::DiscreteTimeIntegrator)
