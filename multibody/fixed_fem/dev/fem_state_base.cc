#include "drake/multibody/fixed_fem/dev/fem_state_base.h"

namespace drake {
namespace multibody {
namespace fem {

template <typename T>
void FemStateBase<T>::SetPositions(const Eigen::Ref<const VectorX<T>>& q) {
  DRAKE_THROW_UNLESS(q.size() == q_.size());
  InvalidateAllCacheEntries();
  q_ = q;
}

template <typename T>
void FemStateBase<T>::SetVelocities(const Eigen::Ref<const VectorX<T>>& v) {
  DRAKE_THROW_UNLESS(v.size() == v_.size());
  InvalidateAllCacheEntries();
  v_ = v;
}

template <typename T>
void FemStateBase<T>::SetAccelerations(const Eigen::Ref<const VectorX<T>>& a) {
  DRAKE_THROW_UNLESS(a.size() == a_.size());
  InvalidateAllCacheEntries();
  a_ = a;
}

template <typename T>
void FemStateBase<T>::ApplyBoundaryCondition(
    const DirichletBoundaryCondition<T>& bc) {
  const auto& bcs = bc.get_bcs();
  if (bcs.size() == 0) {
    return;
  }
  bc.VerifyBcIndexes(this->num_dofs());
  /* Write the BC to the mutable state. */
  for (const auto& [dof_index, boundary_state] : bcs) {
    q_(int{dof_index}) = boundary_state(0);
    v_(int{dof_index}) = boundary_state(1);
    a_(int{dof_index}) = boundary_state(2);
  }
}

}  // namespace fem
}  // namespace multibody
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::FemStateBase);
