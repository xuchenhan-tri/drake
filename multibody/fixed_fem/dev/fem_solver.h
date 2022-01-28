#pragma once

#include <algorithm>
#include <memory>
#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/sparse_linear_operator.h"
#include "drake/multibody/fixed_fem/dev/eigen_conjugate_gradient_solver.h"
#include "drake/multibody/fixed_fem/dev/fem_model_base.h"
#include "drake/multibody/fixed_fem/dev/fem_state_base.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

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
            const DiscreteTimeIntegrator<T>* integrator)
      : model_(model), integrator_(integrator) {
    DRAKE_DEMAND(model_ != nullptr);
    DRAKE_DEMAND(integrator_ != nullptr);
    ResetScratchDataIfNecessary();
  }

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
                          FemStateBase<T>* next_state) const {
    DRAKE_DEMAND(next_state != nullptr);
    model_->ThrowIfModelStateIncompatible(__func__, prev_state);
    model_->ThrowIfModelStateIncompatible(__func__, *next_state);
    /* Make initial guess of the unknown variable that it stays the same. */
    const VectorX<T>& unknown_variable = model_->GetUnknowns(prev_state);
    integrator_->AdvanceOneTimeStep(prev_state, unknown_variable, next_state);
    /* Run Newton-Raphson iterations. */
    SolveWithInitialGuess(next_state);
  }

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
  void set_linear_solve_tolerance(const T& tolerance) {
    linear_solve_tolerance_ = tolerance;
    if constexpr (std::is_same_v<T, double>) {
      if (tangent_matrix_petsc_ != nullptr) {
        tangent_matrix_petsc_->set_relative_tolerance(tolerance);
      }
    } else {
      eigen_tangent_marix_solver_.set_tolerance(tolerance);
    }
  }

 private:
  /* Uses a Newton-Raphson solver to solve for the equilibrium state that
   satisfies the tolerances. See set_relative_tolerance() and
   set_absolute_tolerance() for convergence criteria. The input FEM state is
   non-null and is guaranteed to be compatible with the FEM model.
   @param[in, out] state  As input, `state` provides an initial guess of
   the solution. As output, `state` reports the equilibrium state. */
  int SolveWithInitialGuess(FemStateBase<T>* state) const {
    /* Make sure the scratch quantities are of the correct sizes. */
    ResetScratchDataIfNecessary();
    model_->ApplyBoundaryCondition(state);
    model_->CalcResidual(*state, &b_);
    int iter = 0;
    /* Newton-Raphson iterations. We iterate until any of the following is true:
     1. The max number of allowed iterations is reached;
     2. The norm of the change in the state in a single iteration is smaller
        than the absolute tolerance.
     3. The relative error (the norm of the change in the state divided by the
        norm of the state) is smaller than the unitless relative tolerance. */
    do {
      /* Use PETSc matrix when scalar type is double. Otherwise, use Eigen
       matrix. */
      if constexpr (std::is_same_v<T, double>) {
        model_->CalcTangentMatrix(*state, integrator_->weights(),
                                  tangent_matrix_petsc_.get());
        tangent_matrix_petsc_->AssembleIfNecessary();
        /* Solve for A * dz = -b, where A is the tangent matrix. */
        dz_ = tangent_matrix_petsc_->Solve(
            internal::PetscSymmetricBlockSparseMatrix::SolverType::
                kConjugateGradient,
            internal::PetscSymmetricBlockSparseMatrix::PreconditionerType::
                kIncompleteCholesky,
            -b_);
      } else {
        model_->CalcTangentMatrix(*state, integrator_->weights(),
                                  &tangent_matrix_eigen_);
        /* Solve for A * dz = -b, where A is the tangent matrix. */
        eigen_tangent_matrix_solver_.compute(tangent_matrix_eigen_);
        dz_ = eigen_tangent_matrix_solver_.solve(-b_);
      }
      integrator_->UpdateStateFromChangeInUnknowns(dz_, state);
      model_->CalcResidual(*state, &b_);
      ++iter;
    } while (dz_.norm() > std::max(relative_tolerance_ *
                                       integrator_->GetUnknowns(*state).norm(),
                                   absolute_tolerance_) &&
             iter < kMaxIterations_);
    if (iter == kMaxIterations_) {
      throw std::runtime_error(fmt::format(
          "The solver did not converge in {} iterations. Please provide a "
          "better initial guess. Consider taking a smaller time step, "
          "especially if the constitutive model you are using is nonlinear "
          "(e.g. CorotatedModel).",
          kMaxIterations_));
    }
    return iter;
  }

  /* Reset the scratch data in this class (tangent matrix, residual, and dz) if
   necessary. */
  void ResetScratchDataIfNecessary() const {
    if (b_.size() != model_->num_dofs()) {
      b_.resize(model_->num_dofs());
      dz_.resize(model_->num_dofs());
      if constexpr (std::is_same_v<T, double>) {
        tangent_matrix_petsc_ =
            model_->MakePetscSymmetricBlockSparseTangentMatrix();
      } else {
        tangent_matrix_eigen_ = model_->MakeEigenSparseTangentMatrix();
      }
      set_linear_solve_tolerance(linear_solve_tolerance_);
    }
  }

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
