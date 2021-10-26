#include "drake/multibody/tree/one_dimensional_spring_damper.h"

#include <limits>
#include <utility>
#include <vector>

#include "drake/multibody/tree/body.h"
#include "drake/multibody/tree/multibody_tree.h"

namespace drake {
namespace multibody {

template <typename T>
OneDimensionalSpringDamper<T>::OneDimensionalSpringDamper(const Body<T>& bodyA,
                                                          const Body<T>& bodyB,
                                                          double stiffness,
                                                          double damping)
    : ForceElement<T>(bodyA.model_instance()),
      bodyA_(bodyA),
      bodyB_(bodyB),
      stiffness_(stiffness),
      damping_(damping) {
  DRAKE_THROW_UNLESS(stiffness >= 0);
  DRAKE_THROW_UNLESS(damping >= 0);
}

template <typename T>
void OneDimensionalSpringDamper<T>::DoCalcAndAddForceContribution(
    const systems::Context<T>&,
    const internal::PositionKinematicsCache<T>& pc,
    const internal::VelocityKinematicsCache<T>& vc,
    MultibodyForces<T>* forces) const {
  using std::sqrt;

  const math::RigidTransform<T>& X_WA = pc.get_X_WB(bodyA().node_index());
  const math::RigidTransform<T>& X_WB = pc.get_X_WB(bodyB().node_index());
  const SpatialVelocity<T>& V_WA = vc.get_V_WB(bodyA().node_index());
  const SpatialVelocity<T>& V_WB = vc.get_V_WB(bodyB().node_index());

  const T x = X_WB.translation().y() - X_WA.translation().y();
  const T xdot = V_WB.translational().y() - V_WA.translational().y();

  const T fx_Ao_W = stiffness() * x + damping() * xdot;

  // Force on A, applied at P, expressed in the world frame.
  Vector3<T> f_AP_W = Vector3<T>(0.0, fx_Ao_W, 0.0);

  // Alias to the array of applied body forces:
  std::vector<SpatialForce<T>>& F_Bo_W_array = forces->mutable_body_forces();

  F_Bo_W_array[bodyA().node_index()] +=
      SpatialForce<T>(Vector3<T>::Zero(), f_AP_W);
  F_Bo_W_array[bodyB().node_index()] +=
      SpatialForce<T>(Vector3<T>::Zero(), -f_AP_W);
}

template <typename T>
T OneDimensionalSpringDamper<T>::CalcPotentialEnergy(
    const systems::Context<T>&,
    const internal::PositionKinematicsCache<T>& pc) const {
  const math::RigidTransform<T>& X_WA = pc.get_X_WB(bodyA().node_index());
  const math::RigidTransform<T>& X_WB = pc.get_X_WB(bodyB().node_index());
  const T x = X_WB.translation().y() - X_WA.translation().y();
  return 0.5 * stiffness() * x * x;
}

template <typename T>
T OneDimensionalSpringDamper<T>::CalcConservativePower(
    const systems::Context<T>&,
    const internal::PositionKinematicsCache<T>& pc,
    const internal::VelocityKinematicsCache<T>& vc) const {
  // Since the potential energy is:
  //  V = 1/2⋅k⋅(ℓ-ℓ₀)²
  // The conservative power is defined as:
  //  Pc = -d(V)/dt
  // being positive when the potential energy decreases.

  const math::RigidTransform<T>& X_WA = pc.get_X_WB(bodyA().node_index());
  const math::RigidTransform<T>& X_WB = pc.get_X_WB(bodyB().node_index());
  const SpatialVelocity<T>& V_WA = vc.get_V_WB(bodyA().node_index());
  const SpatialVelocity<T>& V_WB = vc.get_V_WB(bodyB().node_index());

  const T x = X_WB.translation().y() - X_WA.translation().y();
  const T xdot = V_WB.translational().y() - V_WA.translational().y();

  // Since V = 1/2⋅k⋅x² we have that, from its definition:
  // Pc = -d(V)/dt = -k⋅x⋅dx/dt
  const T Pc = -stiffness() * x * xdot;
  return Pc;
}

template <typename T>
T OneDimensionalSpringDamper<T>::CalcNonConservativePower(
    const systems::Context<T>&,
    const internal::PositionKinematicsCache<T>&,
    const internal::VelocityKinematicsCache<T>& vc) const {
  const SpatialVelocity<T>& V_WA = vc.get_V_WB(bodyA().node_index());
  const SpatialVelocity<T>& V_WB = vc.get_V_WB(bodyB().node_index());
  const T xdot = V_WB.translational().y() - V_WA.translational().y();
  // Energy is dissipated at rate Pnc = -d⋅(dℓ/dt)²:
  return -damping() * xdot * xdot;
}

template <typename T>
template <typename ToScalar>
std::unique_ptr<ForceElement<ToScalar>>
OneDimensionalSpringDamper<T>::TemplatedDoCloneToScalar(
    const internal::MultibodyTree<ToScalar>& tree_clone) const {
  const Body<ToScalar>& bodyA_clone =
      tree_clone.get_body(bodyA().index());
  const Body<ToScalar>& bodyB_clone =
      tree_clone.get_body(bodyB().index());

  // Make the OneDimensionalSpringDamper<T> clone.
  auto spring_damper_clone = std::make_unique<OneDimensionalSpringDamper<ToScalar>>(
      bodyA_clone, bodyB_clone, stiffness(), damping());

  return spring_damper_clone;
}

template <typename T>
std::unique_ptr<ForceElement<double>>
OneDimensionalSpringDamper<T>::DoCloneToScalar(
    const internal::MultibodyTree<double>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<ForceElement<AutoDiffXd>>
OneDimensionalSpringDamper<T>::DoCloneToScalar(
    const internal::MultibodyTree<AutoDiffXd>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

template <typename T>
std::unique_ptr<ForceElement<symbolic::Expression>>
OneDimensionalSpringDamper<T>::DoCloneToScalar(
    const internal::MultibodyTree<symbolic::Expression>& tree_clone) const {
  return TemplatedDoCloneToScalar(tree_clone);
}

}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::OneDimensionalSpringDamper)
