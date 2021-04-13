#include "drake/multibody/plant/multibody_plant_deformable_solver_attorney.h"

#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace multibody {
namespace internal {

template <typename T>
void MultibodyPlantDeformableSolverAttorney<T>::DeclareDeformableState(
    const VectorX<T>& q, const VectorX<T>& qdot, const VectorX<T>& qddot,
    MultibodyPlant<T>* mbp) {
  mbp->DeclareDeformableState(q, qdot, qddot);
}

template <typename T>
void MultibodyPlantDeformableSolverAttorney<T>::CalcJacobianTranslationVelocity(
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
}  // namespace internal
}  // namespace multibody
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::internal::MultibodyPlantDeformableSolverAttorney);
