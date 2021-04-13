#include "drake/multibody/plant/deformable_solver_base.h"

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_deformable_solver_attorney.h"

namespace drake {
namespace multibody {
namespace internal {
template <typename T>
const Vector3<double>& DeformableSolverBase<T>::gravity() const {
  return mbp_->gravity_field().gravity_vector();
}

template <typename T>
double DeformableSolverBase<T>::dt() const {
  return mbp_->time_step();
}

template <typename T>
double DeformableSolverBase<T>::default_contact_stiffness() const {
  return MultibodyPlantDeformableSolverAttorney<T>::contact_stiffness(
      multibody_plant());
}

template <typename T>
double DeformableSolverBase<T>::default_contact_dissipation() const {
  return MultibodyPlantDeformableSolverAttorney<T>::contact_dissipation(
      multibody_plant());
}

template <typename T>
void DeformableSolverBase<T>::DeclareDeformableState(const VectorX<T>& q,
                                                     const VectorX<T>& qdot,
                                                     const VectorX<T>& qddot) {
  MultibodyPlantDeformableSolverAttorney<T>::DeclareDeformableState(
      q, qdot, qddot, mbp_);
}

template <typename T>
void DeformableSolverBase<T>::CalcJacobianTranslationVelocity(
    const systems::Context<T>& context, const Vector3<T>& p_WP,
    geometry::GeometryId geometry_id, EigenPtr<Matrix3X<T>> Jv_v_WAp) const {
  MultibodyPlantDeformableSolverAttorney<T>::CalcJacobianTranslationVelocity(
      multibody_plant(), context, p_WP, geometry_id, Jv_v_WAp);
}
}  // namespace internal
}  // namespace multibody
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::internal::DeformableSolverBase);
