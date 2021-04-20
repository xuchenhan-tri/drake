#pragma once
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/geometry_roles.h"
#include "drake/geometry/shape_specification.h"
#include "drake/multibody/contact_solvers/contact_solver.h"
#include "drake/multibody/contact_solvers/contact_solver_results.h"
#include "drake/systems/framework/context.h"
namespace drake {
namespace multibody {

template <typename T>
class MultibodyPlant;

namespace fixed_fem {
/** A pure virtual softsim utility class. */
template <typename T>
class SoftsimBase {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SoftsimBase)

  explicit SoftsimBase(multibody::MultibodyPlant<T>* mbp) : mbp_(mbp) {
    DRAKE_DEMAND(mbp_ != nullptr);
  }

  virtual ~SoftsimBase() = default;

  /* Registers a collision object used for computing rigid-deformable contact
   information given a collision geometry in the MultibodyPlant associated with
   this SoftsimBase.
   @param geometry_id   The GeometryId of the collision geometry.
   @param shape         The shape of the collision geometry.
   @param properties    The proximity properties of the collision geometry.
   @throws std::exception if `geometry_id` is not registered in the associated
   Multibodyplant or if `geometry_id` is already registered in this SoftsimBase.
  */
  virtual void RegisterCollisionObject(
      geometry::GeometryId geometry_id, const geometry::Shape& shape,
      const geometry::ProximityProperties& properties) = 0;

  // TODO(xuchenhan-tri): Consider returns the contact solver data instead of
  //  keeping it in the SoftsimBase.
  /* Assembles the contact data for both rigid-rigid contact and
   rigid-deformable contact.
   @param context0           The context before the contact solve.
   @param v0                 The generalized velocities for rigid dofs.
   @param M0                 The mass matrix for rigid dofs.
   @param minus_tau          The negative non-contact forces on the rigid dofs.
   @param contact_jacobians  The contact Jacobians for the rigid-rigid contacts
                             w.r.t. the rigid dofs.
   @param stiffness          The contact stiffness for rigid-rigid contacts.
   @param damping            The contact dissipation for rigid-rigid contacts.
   @param mu                 The friction coefficients for rigid-rigid contacts.
  */
  virtual void AssembleContactSolverData(
      const systems::Context<T>& context0, const VectorX<T>& v0,
      const MatrixX<T>& M0, VectorX<T>&& minus_tau, VectorX<T>&& phi0,
      const MatrixX<T>& contact_jacobians, VectorX<T>&& stiffness,
      VectorX<T>&& damping, VectorX<T>&& mu) = 0;

  /* Solves the contact problem with the given contact solver and writes out the
   contact results. */
  virtual void SolveContactProblem(
      const contact_solvers::internal::ContactSolver<T>& contact_solver,
      contact_solvers::internal::ContactSolverResults<T>* results) const = 0;

 protected:
  const MultibodyPlant<T>& multibody_plant() const {
    DRAKE_DEMAND(mbp_ != nullptr);
    return *mbp_;
  }

  /* Return a point P's translational velocity (measured and expressed in
   world frame) Jacobian in the world frame with respect to the generalized
   velocities in the `mbp_` owned by this SoftsimBase.
   @param[in] context     The state of the multibody system.
   @param[in] p_WP        The position of the point P in world frame.
   @param[in] geometry_id The geometry id of the body A to which the point P
                          is fixed.
   @param[out] Jv_v_WAp   Point Ap's velocity Jacobian in the world frame with
                          respects to the generalized velocities where Ap is
                          the origin of the frame A shifted to P. */
  void CalcJacobianTranslationVelocity(const systems::Context<T>& context,
                                       const Vector3<T>& p_WP,
                                       geometry::GeometryId geometry_id,
                                       EigenPtr<Matrix3X<T>> Jv_v_WAp) const;

  const T& default_contact_stiffness() const;

  const T& default_contact_dissipation() const;

 private:
  MultibodyPlant<T>* mbp_;
};
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fixed_fem::SoftsimBase);
