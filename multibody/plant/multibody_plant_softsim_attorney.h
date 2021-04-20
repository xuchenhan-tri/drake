#pragma once
#include "drake/common/drake_copyable.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/systems/framework/context.h"
namespace drake {
namespace multibody {
template <typename T>
class MultibodyPlant;
namespace fixed_fem {
template <typename T>
class SoftsimBase;
}
namespace internal {
// This is an attorney-client pattern providing SoftsimBase access to the
// private data/methods of MultibodyPlant in order to be able construct contact
// solver data. This class is meant to be a short-term solution to quickly
// facilitate integration of softsim with MultibodyPlant without SceneGraph's
// support for deformable geometries.
class MultibodyPlantSoftsimAttorney {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MultibodyPlantSoftsimAttorney);
  MultibodyPlantSoftsimAttorney() = delete;

 private:
  template <typename T>
  friend class multibody::fixed_fem::SoftsimBase;
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
  template <typename T>
  static void CalcJacobianTranslationVelocity(
      const MultibodyPlant<T>& mbp, const systems::Context<T>& context,
      const Vector3<T>& p_WP, geometry::GeometryId geometry_id,
      EigenPtr<Matrix3X<T>> Jv_v_WAp);

  template <typename T>
  static const T& contact_stiffness(const MultibodyPlant<T>& mbp);

  template <typename T>
  static const T& contact_dissipation(const MultibodyPlant<T>& mbp);
};
}  // namespace internal
}  // namespace multibody
}  // namespace drake
