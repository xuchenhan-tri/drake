#pragma once

#include "drake/common/unused.h"
#include "drake/multibody/fixed_fem/dev/fem_state.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
// TODO(xuchenhan-tri): Try to come up with a better name for this class. It
//  does more than updating the states. It also defines the discretized equation
//  that FEM solver solves.
/* StateUpdater provides the interface to update FemStateBase in FemModelBase.
 In each Newton-Raphson iteration of the FEM solver, we are solving for an
 equation in the form of

        G(q, q̇, q̈) = 0.

 For different models, G takes different forms. For dynamic elasticity,

        G(q, q̇, q̈) = Mq̈ - fₑ(q) - fᵥ(q, q̇) - fₑₓₜ,

 where M is the mass matrix, fₑ(q) is the elastic force, fᵥ(q, q̇) is the
 damping force, and fₑₓₜ are the external forces. For static elasticity,

        G(q, q̇, q̈) = G(q) = -fₑ(q) + fₑₓₜ.

 For the Poisson equation,

        G(q, q̇, q̈) = G(q) = Kq − f.

 With time discretization, one can express `q`, `q̇,` `q̈` in terms of a
 single variable, which we dub the name "unknown" and denote it with `z`.
 For example, for dynamic elasticity with Newmark scheme,

        q̈ = z;
        q̇ = q̇ₙ + dt ⋅ (γ ⋅ z + (1−γ) ⋅ q̈ₙ)
        q = qₙ + dt ⋅ q̇ₙ + dt² ⋅ [β ⋅ z + (0.5−β) ⋅ q̈ₙ].

 Hence, we can write the system of equations as

        G(z) = 0,

 and in each Newton-Raphson iteration, we solve for a linear system of equation
 of the form

        ∇G(z) ⋅ dz = -G(z),

 where ∇G = ∂G/∂z. StateUpdater is responsible for updating the FEM state given
 the solution of the linear solve, `dz`, which is also the change in the unknown
 variables. In addition, StateUpdater also provides the derivatives of the
 states with respect to the unknown variable `z`, namely, `∂q/∂z`,  `∂q̇/∂z`, and
 `∂q̈/∂z`, that are needed for building ∇G.
 @tparam_non_symbolic. */
template <typename T>
class StateUpdater {
 public:
  virtual ~StateUpdater() = default;

  /* For a representative degree of freedom i, returns the derivative of the
   state with respect to the unknown variable `zᵢ`, [∂qᵢ/∂zᵢ, ∂q̇ᵢ/∂zᵢ, ∂q̈ᵢ/∂zᵢ].
   The choice of i is arbitrary as these derivatives are the same for all
   degrees of freedom in the same model. These derivatives can be used as
   weights to combine stiffness, damping and mass matrices (see FemElement) to
   form the tangent matrix (see FemModelBase). If q̇ or q̈ are not a part of the
   state, their derivatives are set to 0. */
  Vector3<T> weights() const { return do_get_weights(); }

  /* Extracts the unknown variable from the given FEM `state`.
   @throw std::exception if the type of concrete FemState for `state` is not
   compatible with the concrete FemModel for `this` model. */
  const VectorX<T>& GetUnknowns(const FemStateBase<T>& state) const {
    return DoGetUnknowns(state);
  }

  /* Updates the FemStateBase `state` given the change in the unknown variables.
   @pre state != nullptr.
   @pre dz.size() == state->num_generalized_positions(). */
  void UpdateStateFromChangeInUnknowns(const VectorX<T>& dz,
                                       FemStateBase<T>* state) const {
    DRAKE_DEMAND(state != nullptr);
    DRAKE_DEMAND(dz.size() == state->num_generalized_positions());
    DoUpdateStateFromChangeInUnknowns(dz, state);
  }

  /* Advances the given `prev_state` by one time step to the `next_state` if the
   states have nonzero ODE order. No-op otherwise.
   @param[in]  prev_state        The state at the previous time step.
   @param[in]  unknown_variable  The unknown variable z.
   @param[out] next_state        The state at the new time step.
   @pre next_state != nullptr.
   @pre The sizes of `prev_state`, `unknown_variable`, and `next_state` are
   compatible. */
  void AdvanceOneTimeStep(const FemStateBase<T>& prev_state,
                          const VectorX<T>& unknown_variable,
                          FemStateBase<T>* next_state) const {
    DRAKE_DEMAND(next_state != nullptr);
    DRAKE_DEMAND(prev_state.num_generalized_positions() ==
                 next_state->num_generalized_positions());
    DRAKE_DEMAND(prev_state.num_generalized_positions() ==
                 unknown_variable.size());
    DoAdvanceOneTimeStep(prev_state, unknown_variable, next_state);
  }

 protected:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(StateUpdater);
  StateUpdater() = default;

  /* Derived classes must override this method to provide an implementation. */
  virtual Vector3<T> do_get_weights() const = 0;

  /* Derived classes must override this method to extract the unknown variable
   from the given FemStateBase. */
  virtual const VectorX<T>& DoGetUnknowns(
      const FemStateBase<T>& state) const = 0;

  /* Derived classes must override this method to udpate the FemStateBase
   `state` given the change in the unknown variables `dz` based on the
   time-stepping scheme of the derived class. The `dz` provided here has
   compatible size with the number of generalized positions in `state` and does
   not need to be checked again.  */
  virtual void DoUpdateStateFromChangeInUnknowns(
      const VectorX<T>& dz, FemStateBase<T>* state) const = 0;

  /* Derived StateUpdaters associated with states that have ODE order greater
   than 0 must override this method to advance a single time step according to
   the specific time stepping scheme. The sizes of `prev_state`, `unknowns`, and
   `next_state` are compatible and do not need to be checked again. */
  virtual void DoAdvanceOneTimeStep(const FemStateBase<T>& prev_state,
                                    const VectorX<T>& unknowns,
                                    FemStateBase<T>* next_state) const {
    unused(prev_state, unknowns, next_state);
  }
};
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
