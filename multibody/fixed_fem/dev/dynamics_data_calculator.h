#pragma once

#include <vector>

#include <Eigen/SparseCore>

#include "drake/multibody/contact_solvers/sparse_linear_operator.h"
#include "drake/multibody/contact_solvers/system_dynamics_data.h"
#include "drake/multibody/fixed_fem/dev/block_diagonal_operator.h"
#include "drake/multibody/fixed_fem/dev/collision_object.h"
#include "drake/multibody/fixed_fem/dev/eigen_sparse_ldlt_solver.h"
#include "drake/multibody/fixed_fem/dev/fem_model_base.h"
#include "drake/multibody/fixed_fem/dev/fem_state_base.h"
#include "drake/multibody/fixed_fem/dev/inverse_operator.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {

template <typename T>
class DynamicsDataCalculator {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DynamicsDataCalculator);

  DynamicsDataCalculator(
      const FemModelBase<T>* fem_model,
      const std::vector<CollisionObject<T>>* collision_objects)
      : fem_model_(fem_model), collision_objects_(collision_objects) {
    // TODO(xuchenhan): Should verify here that the model fed in is of type
    //  DynamicElasticityModel.
    DRAKE_DEMAND(fem_model_ != nullptr);
    DRAKE_DEMAND(collision_objects_ != nullptr);
    Resize();
  }

  /* Returns the SystemDynamicData consumed by the contact solver given the
   state of the deformable object. */
  const contact_solvers::internal::SystemDynamicsData<T> ComputeDynamicsData(
      const FemStateBase<T>& fem_state) {
    /* Ensure that the matrices are of the correct sizes. */
    Resize();
    /* Update the deformable tangent matrix. The rigid mass matrix stays zero
     because we assume the rigid objects have infinite mass and inertia. */
    fem_model_->CalcTangentMatrix(fem_state, &deformable_tangent_matrix_);
    /* Factorize the new tangent matrix. */
    ldlt_.Compute();
    /* Fill out the contact-free velocities, with the deformable dofs in the
     front and rigid dofs trailing. */
    const int num_collision_objects = collision_objects_->size();
    const int num_deformable_dofs = fem_model_->num_dofs();
    v_star_.head(num_deformable_dofs) = fem_state.qdot();
    for (int i = 0; i < num_collision_objects; ++i) {
      v_star_.template segment<3>(num_deformable_dofs + 6 * i) =
          (*collision_objects_)[i].rotational_velocity();
      v_star_.template segment<3>(num_deformable_dofs + 6 * i + 3) =
          (*collision_objects_)[i].translational_velocity();
    }
    return contact_solvers::internal::SystemDynamicsData(&A_inv_, &v_star_);
  }

 private:
  /* Resize `deformable_tangent_matrix_` and `rigid_mass_matrix_inv_` to the
   correct sizes and set sparsity patterns for these two matrices. Resize
   `v_star_` to the correct size. No-op if these quantities are already of the
   correct sizes.
  */
  void Resize() {
    const int num_collision_objects = collision_objects_->size();
    /* Check if the size actually changes because `resize` for
     Eigen::SparseMatrix is destructive. */
    if (num_collision_objects != rigid_mass_matrix_inv_.cols()) {
      rigid_mass_matrix_inv_.resize(6 * num_collision_objects,
                                    6 * num_collision_objects);
    }
    const int num_deformable_dofs = fem_model_->num_dofs();
    if (num_deformable_dofs != deformable_tangent_matrix_.cols()) {
      deformable_tangent_matrix_.resize(num_deformable_dofs,
                                        num_deformable_dofs);
      fem_model_->SetTangentMatrixSparsityPattern(&deformable_tangent_matrix_);
    }
    const int total_num_dofs = 6 * num_collision_objects + num_deformable_dofs;
    v_star_.resize(total_num_dofs);
  }

  const FemModelBase<T>* fem_model_;
  const std::vector<CollisionObject<T>>* collision_objects_;
  Eigen::SparseMatrix<T> deformable_tangent_matrix_;
  const contact_solvers::internal::SparseLinearOperator<T> deformable_tangent_{
      "deformable_tangent", &deformable_tangent_matrix_};
  Eigen::SparseMatrix<T> rigid_mass_matrix_inv_;
  EigenSparseLdltSolver<T> ldlt_{&deformable_tangent_};
  contact_solvers::internal::InverseOperator<T> deformable_tangent_inv_{
      "deformable_tangent_inverse", &ldlt_};
  contact_solvers::internal::SparseLinearOperator<T> rigid_mass_inv_{
      "rigid_mass_inverse", &rigid_mass_matrix_inv_};
  contact_solvers::internal::BlockDiagonalOperator<T> A_inv_{
      "A_inv", {&deformable_tangent_inv_, &rigid_mass_inv_}};
  VectorX<T> v_star_;
};
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
