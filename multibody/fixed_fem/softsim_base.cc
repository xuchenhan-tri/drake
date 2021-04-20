#include "drake/multibody/fixed_fem/softsim_base.h"

#include "drake/multibody/plant/multibody_plant_softsim_attorney.h"
namespace drake {
namespace multibody {
namespace fixed_fem {
template <typename T>
void SoftsimBase<T>::CalcJacobianTranslationVelocity(
    const systems::Context<T>& context, const Vector3<T>& p_WP,
    geometry::GeometryId geometry_id, EigenPtr<Matrix3X<T>> Jv_v_WAp) const {
  drake::multibody::internal::MultibodyPlantSoftsimAttorney::
      CalcJacobianTranslationVelocity(multibody_plant(), context, p_WP,
                                      geometry_id, Jv_v_WAp);
}

template <typename T>
const T& SoftsimBase<T>::default_contact_stiffness() const {
  return drake::multibody::internal::MultibodyPlantSoftsimAttorney::
      contact_stiffness(multibody_plant());
}

template <typename T>
const T& SoftsimBase<T>::default_contact_dissipation() const {
  return drake::multibody::internal::MultibodyPlantSoftsimAttorney::
      contact_dissipation(multibody_plant());
}
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fixed_fem::SoftsimBase);
