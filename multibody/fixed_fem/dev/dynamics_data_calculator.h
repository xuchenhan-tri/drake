#pragma once

#include <memory>
#include <vector>

#include "drake/math/cross_product.h"
#include "drake/math/orthonormal_basis.h"
#include "drake/multibody/contact_solvers/point_contact_data.h"
#include "drake/multibody/contact_solvers/sparse_linear_operator.h"
#include "drake/multibody/fixed_fem/dev/collision_object.h"
#include "drake/multibody/fixed_fem/dev/deformable_contact.h"
#include "drake/multibody/plant/coulomb_friction.h"

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
    fem_model_->CalcTangentMatrix(fem_state, &deformable_tangent_matrix);
    /* Fill out the contact-free velocities, with the deformable dofs in the
     front and rigid dofs trailing. */
    const int num_collision_objects = collision_objects_->size();
    const int num_deformable_dofs = fem_model_->num_dofs();
    v_star_.head(num_deformable_dofs) = fem_state.qdot();
    for (int i = 0; i < num_collision_objects; ++i) {
      v_star_.template segment<3>(num_deformable_dofs + 6 * i) =
          (*collsion_objects)[i].rotational_velocity();
      v_star_.template segment<3>(num_deformable_dofs + 6 * i + 3) =
          (*collsion_objects)[i].translational_velocity();
    }
    return contact_solvers::internal::SystemDynamicsData(&A_inv_, &v_star_)
  }

 private:
  /* Resize `deformable_tangent_matrix_` and `rigid_mass_matrix_` to the correct
   sizes and set sparsity patterns for these two matrices. Resize `v_star_` to
   the correct size. No-op if these quantities are already of the correct sizes.
  */
  void Resize() {
    const int num_collision_objects = collision_objects_->size();
    /* Check if the size actually changes because `resize` for
     Eigen::SparseMatrix is destructive. */
    if (num_collision_objects != rigid_mass_matrix_.cols()) {
      rigid_mass_matrix_.resize(num_collision_objects, num_collision_objects);
    }
    const int num_deformable_dofs = fem_model_->num_dofs();
    if (num_deformable_dofs != deformable_tangent_matrix_.cols()) {
      deformable_tangent_matrix.reisze(num_deformable_dofs,
                                       num_deformable_dofs);
      fem_model_->SetTangentMatrixSparsityPattern(&deformable_tangent_matrix);
    }
    const int total_num_dofs = 6 * num_collsiion_objects + num_deformable_dofs;
    v_star_.resize(total_num_dofs);
  }

  /* Adds the given 3-by-3 matrix `A` to the 3-by-3 block starting at `row` and
   `col` of the sparse matrix to be constructed by the given `triplets`. */
  void AddMatrix3ToEigenTriplets(const Eigen::Ref<const Matrix3<T>>& A, int row,
                                 int col,
                                 std::vector<Eigen::Triplet<T>>* triplets) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        triplets->emplace_back(row + i, col + j, A(i, j));
      }
    }
  }

  const FemModelBase<T>* fem_model_;
  const std::vector<CollisionObject<T>>* collision_objects_;
  Eigen::SparseMatrix<T> deformable_tangent_matrix_;
  Eigen::SparseMatrix<T> rigid_mass_inv_matrix_;
  EigenLdltSolver<T> ldlt_(&deformable_tangent_matrix_);
  contact_solvers::internal::InverseOperator<T> deformable_tangent_inv_(
      "deformable_tangent_inverse", &ldlt_);
  contact_solvers::internal::SparseOperator<T> rigid_mass_inv_(
      "rigid_mass_inverse" & rigid_mass_matrix_);
  contact_solvers::internal::BlockDiagonalOperator<T> A_inv_("A_inv" {
      &deformable_tangent_inv_, &rigid_mass_inv_});
  VectorX<T> v_star_;
};
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
