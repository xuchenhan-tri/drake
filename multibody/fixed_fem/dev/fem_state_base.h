#pragma once

#include <memory>

#include "drake/common/default_scalars.h"
#include "drake/multibody/fixed_fem/dev/dirichlet_boundary_condition.h"

namespace drake {
namespace multibody {
namespace fem {

template <typename T>
class DirichletBoundaryCondition;

/** An abstract state class that stores the fem states. The states include the
 generalized positions, velocities, and accelerations associated with each node.
 @tparam_nonsymbolic_scalar */
template <typename T>
class FemStateBase {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FemStateBase);

  virtual ~FemStateBase() = default;

  /** @name State getters. Throw an exception if the state doesn't exist.
   @{ */
  const VectorX<T>& GetPositions() const { return q_; }

  const VectorX<T>& GetVelocities() const { return v_; }

  const VectorX<T>& GetAccelerations() const { return a_; }
  /** @} */

  /** @name State setters.
   The size of the values provided must match the current size of the states.
   Throw an exception otherwise.
   @{ */
  void SetPositions(const Eigen::Ref<const VectorX<T>>& q);

  void SetVelocities(const Eigen::Ref<const VectorX<T>>& v);

  void SetAccelerations(const Eigen::Ref<const VectorX<T>>& a);
  /** @} */

  /* Returns the number of generalized positions in the state. */
  int num_dofs() const { return q_.size(); }

  // TODO(xuchenhan-tri): Move this method to DirichletBoundaryCondition
  /** Modifies `this` FEM state so that it complies with the given boundary
   conditions.
   @throw std::exception if the any of the indexes of the dofs under the
   boundary condition specified by the given DirichletBoundaryCondition does
   not exist in `this` FEM state`. */
  void ApplyBoundaryCondition(const DirichletBoundaryCondition<T>& bc);

 protected:
  /** Constructs an %FemStateBase with prescribed generalized positions,
   velocities, and accelerations.
   @param[in] q  The prescribed generalized positions.
   @param[in] v  The prescribed generalized velocities.
   @param[in] a  The prescribed generalized accelerations.
   @pre q.size() == v.size().
   @pre q.size() == a.size(). */
  FemStateBase(const Eigen::Ref<const VectorX<T>>& q,
               const Eigen::Ref<const VectorX<T>>& v,
               const Eigen::Ref<const VectorX<T>>& a)
      : q_(q), v_(v), a_(a) {
    DRAKE_DEMAND(q_.size() == v_.size());
    DRAKE_DEMAND(q_.size() == a_.size());
  }

 private:
  /* Invalidate state-dependent quantities. Should be called on state changes.
   */
  virtual void InvalidateAllCacheEntries() = 0;

  /* Generalized positions. */
  VectorX<T> q_{};
  /* Generalized velocities. */
  VectorX<T> v_{};
  /* Generalized accelerations. */
  VectorX<T> a_{};
};

}  // namespace fem
}  // namespace multibody
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::FemStateBase);
