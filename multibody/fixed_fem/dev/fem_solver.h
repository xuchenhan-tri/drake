#pragma once

#include <algorithm>
#include <memory>
#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fixed_fem/dev/discrete_time_integrator.h"
#include "drake/multibody/fixed_fem/dev/fem_model_base.h"
#include "drake/multibody/fixed_fem/dev/fem_state_base.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

// TODO(xuchenhan-tri): Update the class doc.
/* FemSolver solves for the state of a given FemModel at which the residual of
 the model is sufficiently close to zero. %FemSolver uses a simple
 Newton-Raphson solver to solve for the zero residual state. A common workflow
 for solving a static FEM model looks like:
 ```
 // Creates a solver for the given FemModel.
 FemSolver<double> solver(&model));
 // Optionally, sets the tolerances under which we deem the residual is
 // effectively zero.
 solver.set_absolute_tolerance(kAbsoluteTolereance);
 solver.set_relative_tolerance(kRelativeTolereance);
 // Finally, provide an initial guess and solve for the zero residual state.
 solver.SolveStaticModelWithInitialGuess(&state);
 ```
 The workflow for solving dynamics FEM model is similar. AdvanceOneTimeStep()
 should be called in the place of SolveStaticModelWithInitialGuess().
 @tparam_nonsymbolic_scalar T. */
template <typename T>
class FemSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemSolver);

  /* Constructs a new FemSolver that solves the given `model` with the
   `integrator` provided to advance time.
   @note The `model` and `integrator` pointers persist in `this` FemSolver and
   thus the model and the integrator must outlive this solver.
   @pre model != nullptr.
   @pre integrator != nullptr.*/
  FemSolver(const FemModelBase<T>* model,
            const DiscreteTimeIntegrator<T>* integrator);

  /* Advances the state of the FEM model by one time step with the integrator
   prescribed at construction.
   @param[in] prev_state   The state of the FEM model evaluated at the previous
                           time step.
   @param[out] next_state  The state of the FEM model evaluated at the next time
                           step.
   @pre next_state != nullptr.
   @pre prev_state.num_dofs() == next_state->dofs().
   @throw std::exception if the input `prev_state` or `next_state` is
   incompatible with the FEM model solved by this solver. */
  void AdvanceOneTimeStep(const FemStateBase<T>& prev_state,
                          FemStateBase<T>* next_state) const;

  /* Returns the FEM model that this solver solves for. */
  const FemModelBase<T>& model() const { return *model_; }

  /* Returns the discrete time integrator that this solver uses. */
  const DiscreteTimeIntegrator<T>& integrator() const { return *integrator_; }

  /* Sets the relative tolerance, unitless. The Newton-Raphson iterations are
   considered as converged if ‖dz‖ < `tolerance`⋅‖z‖ where z is the unknown
   variable _or_ if the absolute tolerance criterion is satisfied (See
   set_absolute_tolerance()). The default value is 1e-6. */
  void set_relative_tolerance(const T& tolerance) {
    relative_tolerance_ = tolerance;
  }

  /* Sets the absolute tolerance which has the same unit as the unknown
   variable z. The Newton-Raphson iterations are considered as converged if the
   change in the state is smaller than the absolute tolerance _or_ if the
   relative tolerance criterion is satisfied (See set_relative_tolerance()). The
   default value is 1e-3. */
  void set_absolute_tolerance(const T& tolerance) {
    absolute_tolerance_ = tolerance;
  }

  /* Sets the relative tolerance for the linear solver used in the
   Newton-Raphson iterations if the linear solver is iterative. The default
   (unitless) tolerance is 1e-4. No-op if the linear solver is direct. */
  void set_linear_solve_tolerance(const T& tolerance);

 private:
  /* Uses a Newton-Raphson solver to solve for the equilibrium state that
   satisfies the tolerances. See set_relative_tolerance() and
   set_absolute_tolerance() for convergence criteria. The input FEM state is
   non-null and is guaranteed to be compatible with the FEM model.
   @param[in, out] state  As input, `state` provides an initial guess of
   the solution. As output, `state` reports the equilibrium state. */
  int SolveWithInitialGuess(FemStateBase<T>* state) const;

  /* Reset the scratch data in this class (tangent matrix, residual, and dz) if
   necessary. */
  void ResetScratchDataIfNecessary() const;

  /* The FEM model being solved by `this` solver. */
  const FemModelBase<T>* model_;
  /* The discrete time integrator the solver uses. */
  const DiscreteTimeIntegrator<T>* integrator_;
  /* A scratch sparse matrix to store the tangent matrix of the model. We use
   PETSc matrix for T=double and an Eigen::SparseMatrix otherwise. */
  mutable Eigen::SparseMatrix<T> tangent_matrix_eigen_;
  mutable std::unique_ptr<internal::PetscSymmetricBlockSparseMatrix>
      tangent_matrix_petsc_;
  /* Solver for the tangent matrix when T!=double.  */
  mutable Eigen::ConjugateGradient<Eigen::SparseMatrix<T>,
                                   Eigen::Lower | Eigen::Upper>
      eigen_tangent_matrix_solver_;
  /* A scratch vector to store the residual of the model. */
  mutable VectorX<T> b_;
  /* A scratch vector to store the solution to A * dz = -b, where A is the
   tangent matrix. */
  mutable VectorX<T> dz_;
  /* The relative tolerance for determining the convergence of the Newton
   solver, unitless. */
  T relative_tolerance_{1e-6};
  /* The absolute tolerance for determining the convergence of the Newton
   solver. It has the same unit as the unknown variable z. */
  T absolute_tolerance_{1e-3};
  /* The relative tolerance when solving for A * dz = -b, where A is the tangent
   matrix. */
  T linear_solve_tolerance_{1e-4};
  /* Max number of Newton-Raphson iterations the solver takes before it gives
   up. */
  static constexpr int kMaxIterations_ = 100;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::internal::FemSolver);