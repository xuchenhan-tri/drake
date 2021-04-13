#pragma once
#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/systems/framework/context.h"
namespace drake {
namespace multibody {
template <typename T>
class MultibodyPlant;

namespace internal {
template <typename T>
class DeformableSolverBase;

/* This is an attorney-client pattern providing DeformableSolverBase access to
 the private data/methods of MultibodyPlant in order to be able construct
 contact solver data. This class is meant to be a short-term solution to
 quickly facilitate integration of deformable simulation with MultibodyPlant
 without moving the deformable functionalities out of the dev directory. When
 the deformable functionalities graduates from the dev direction, this class
 should be retired along with DeformableSolverBase. */
template <typename T>
class MultibodyPlantDeformableSolverAttorney {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MultibodyPlantDeformableSolverAttorney);
  MultibodyPlantDeformableSolverAttorney() = delete;

 private:
  friend class DeformableSolverBase<T>;

  /* Declares a deformable state with the given position, velocity and
   acceleration in the given `mbp`. */
  static void DeclareDeformableState(const VectorX<T>& q,
                                     const VectorX<T>& qdot,
                                     const VectorX<T>& qddot,
                                     MultibodyPlant<T>* mbp);
  static double contact_stiffness(const MultibodyPlant<T>& plant) {
    return plant.penalty_method_contact_parameters_.geometry_stiffness;
  }

  static double contact_dissipation(const MultibodyPlant<T>& plant) {
    return plant.penalty_method_contact_parameters_.dissipation;
  }

  // Return a point P's translational velocity (measured and expressed in
  // world frame) Jacobian in the world frame with respect to the generalized
  // velocities in the `mbp`.
  // @param[in] mbp         The plant in which the generalized velocities live.
  // @param[in] context     The state of the multibody system.
  // @param[in] p_WP        The position of the point P in world frame.
  // @param[in] geometry_id The geometry id of the body A to which the point P
  //                        is fixed.
  // @param[out] Jv_v_WAp   Point Ap's velocity Jacobian in the world frame with
  //                        respects to the generalized velocities where Ap is
  //                        the origin of the frame A shifted to P.
  static void CalcJacobianTranslationVelocity(
      const MultibodyPlant<T>& mbp, const systems::Context<T>& context,
      const Vector3<T>& p_WP, geometry::GeometryId geometry_id,
      EigenPtr<Matrix3X<T>> Jv_v_WAp);
};
}  // namespace internal
}  // namespace multibody
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::internal::MultibodyPlantDeformableSolverAttorney);
