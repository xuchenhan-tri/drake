#include "drake/multibody/fem/fem_state.h"

namespace drake {
namespace multibody {
namespace fem {

template <typename T>
FemState<T>::FemState(const Eigen::Ref<const VectorX<T>>& q,
                      const Eigen::Ref<const VectorX<T>>& v,
                      const Eigen::Ref<const VectorX<T>>& a)
    : systems::BasicVector<T>(3 * q.size()),
      num_dofs_(q.size()),
      q_(this->value().segment(num_dofs_, num_dofs_)),
      v_(this->value().segment(num_dofs_, num_dofs_)),
      a_(this->value().segment(num_dofs_, num_dofs_)) {
  SetPositions(q);
  SetVelocities(v);
  SetAccelerations(a);
  DRAKE_DEMAND(q.size() == v.size());
  DRAKE_DEMAND(q.size() == a.size());
}

template <typename T>
const VectorX<T>& FemState<T>::GetPositions() const {
  return q_;
}

template <typename T>
const VectorX<T>& FemState<T>::GetVelocities() const {
  return v_;
}

template <typename T>
const VectorX<T>& FemState<T>::GetAccelerations() const {
  return a_;
}

template <typename T>
void FemState<T>::SetPositions(const Eigen::Ref<const VectorX<T>>& q) {
  DRAKE_THROW_UNLESS(q.size() == num_dofs());
  this->get_mutable_value().head(num_dofs_) = q;
}

template <typename T>
void FemState<T>::SetVelocities(const Eigen::Ref<const VectorX<T>>& v) {
  DRAKE_THROW_UNLESS(v.size() == num_dofs());
  this->get_mutable_value().segment(num_dofs_, num_dofs_) = v;
}

template <typename T>
void FemState<T>::SetAccelerations(const Eigen::Ref<const VectorX<T>>& a) {
  DRAKE_THROW_UNLESS(a.size() == num_dofs());
  this->get_mutable_value().tail(num_dofs_) = a;
}

}  // namespace fem
}  // namespace multibody
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::FemState);
