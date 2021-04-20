#include "drake/multibody/plant/multibody_plant_softsim_attorney.h"

#include "drake/multibody/plant/multibody_plant.h"
namespace drake {
namespace multibody {
namespace internal {

template <typename T>
void MultibodyPlantSoftsimAttorney::CalcJacobianTranslationVelocity(
    const MultibodyPlant<T>& mbp, const systems::Context<T>& context,
    const Vector3<T>& p_WP, geometry::GeometryId geometry_id,
    EigenPtr<Matrix3X<T>> Jv_v_WAp) {
  const BodyIndex bodyA_index = mbp.geometry_id_to_body_index_.at(geometry_id);
  const Body<T>& bodyA = mbp.get_body(bodyA_index);
  const Frame<T>& frame_W = mbp.world_frame();
  mbp.internal_tree().CalcJacobianTranslationalVelocity(
      context, JacobianWrtVariable::kV, bodyA.body_frame(), frame_W, p_WP,
      frame_W, frame_W, Jv_v_WAp);
}

template <typename T>
const T& MultibodyPlantSoftsimAttorney::contact_stiffness(
    const MultibodyPlant<T>& mbp) {
  return mbp.penalty_method_contact_parameters_.geometry_stiffness;
}

template <typename T>
const T& MultibodyPlantSoftsimAttorney::contact_dissipation(
    const MultibodyPlant<T>& mbp) {
  return mbp.penalty_method_contact_parameters_.dissipation;
}

template const double& MultibodyPlantSoftsimAttorney::contact_stiffness(
    const MultibodyPlant<double>&);
template const double& MultibodyPlantSoftsimAttorney::contact_dissipation(
    const MultibodyPlant<double>&);
template void MultibodyPlantSoftsimAttorney::CalcJacobianTranslationVelocity(
    const MultibodyPlant<double>&, const systems::Context<double>&,
    const Vector3<double>&, geometry::GeometryId, EigenPtr<Matrix3X<double>>);
}  // namespace internal
}  // namespace multibody
}  // namespace drake
