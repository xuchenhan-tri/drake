#include "drake/multibody/fem/fem_data.h"

namespace drake {
namespace multibody {
namespace fem {

template <typename T>
FemData<T>::FemData(const FemDataInfo<T>& data_info) : info_(data_info) {
  context_ = info_.system.CreateDefaultContext();
}

/* Sugar to get/set a part of the FEM state (q, v, or a). */
template <typename T>
const VectorX<T>& FemData<T>::GetPositions() const {
  context_->get_discrete_state(info_.fem_position_index).value();
}

template <typename T>
const VectorX<T>& FemData<T>::GetVelocities() const {
  context_->get_discrete_state(info_.fem_velocity_index).value();
}

template <typename T>
const VectorX<T>& FemData<T>::GetAccelerations() const {
  context_->get_discrete_state(info_.fem_acceleration_index).value();
}

template <typename T>
void FemData<T>::SetPositions(const VectorX<T>& q) {
  context_->SetDiscreteState(info_.fem_position_index, q);
}

template <typename T>
void FemData<T>::SetVelocities(const VectorX<T>& v) {
  context_->SetDiscreteState(info_.fem_velocity_index, v);
}

template <typename T>
void FemData<T>::SetAccelerations(const VectorX<T>& a) {
  context_->SetDiscreteState(info_.fem_acceleration_index, a);
}

}  // namespace fem
}  // namespace multibody
}  // namespace drake
