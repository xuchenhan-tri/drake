#pragma once
#include "drake/geometry/geometry_ids.h"
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

 private:
  MultibodyPlant<T>* mbp_;
};
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
