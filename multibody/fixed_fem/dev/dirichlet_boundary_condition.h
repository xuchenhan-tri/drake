#pragma once

#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Sparse>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fixed_fem/dev/fem_indexes.h"
#include "drake/multibody/fixed_fem/dev/petsc_symmetric_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace fem {

/** %DirichletBoundaryCondition provides functionalities related to Dirichlet
 boundary conditions (BC) to the FEM solver. In particular, it provides the
 following functionalities:
 1. storing the information necessary to apply the BC;
 2. modifying a given state to comply with the stored BC;
 3. modifying a given tangent matrix/residual that arises from the FEM system
 without BC and transform it into the tangent matrix/residual for the same
 system under the stored BC.
 @tparam_nonsymbolic_scalar */
template <class T>
class DirichletBoundaryCondition {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DirichletBoundaryCondition);

  /** Constructs an empty boundary condition. */
  DirichletBoundaryCondition() {}

  /** Sets the dof with index `dof_index` to be subject to the prescribed
   `boundary_state`.
   @param[in] dof_index      The index of the degree of freedom to which the
                             boundary condition is applied.
   @param[in] boundary_state The prescibed position, velocity, and acceleration
                             at `dof_index`. */
  void AddBoundaryCondition(
      DofIndex dof_index, const Eigen::Ref<const Vector3<T>>& boundary_state) {
    bcs_[dof_index] = boundary_state;
  }

  /** Returns all boundary conditions stored in `this`
   %DirichletBoundaryCondition as an `std::map` with the index of the dof as
   key and the prescribed boundary values as value. */
  const std::map<DofIndex, Vector3<T>>& get_bcs() const { return bcs_; }

  /** Modifies the given tangent matrix that arises from an FEM system without
   BC into the tangent matrix for the same system subject to `this` BC. More
   specifically, the rows and columns corresponding to dofs under the BC will be
   zeroed out with the exception of the diagonal entries for those dofs which
   will be set to 1.
   @pre tangent_matrix != nullptr.
   @pre tangent_matrix->rows() == tangent_matrix->cols().
   @throw std::exception if the any of the indexes of the dofs under the
   boundary condition specified by `this` %DirichletBoundaryCondition is
   greater than or equal to the `tangent_matrix->cols()`. */
  void ApplyBoundaryConditionToTangentMatrix(
      Eigen::SparseMatrix<T>* tangent_matrix) const {
    DRAKE_DEMAND(tangent_matrix != nullptr);
    DRAKE_DEMAND(tangent_matrix->rows() == tangent_matrix->cols());
    if (bcs_.size() == 0) {
      return;
    }
    /* Check validity of the dof indices stored. */
    VerifyBcIndexes(tangent_matrix->cols());

    /* Zero out all rows and columns of the tangent matrix corresponding to dofs
     under the BC (except the diagonal entry which is set to 1). */
    for (const auto& it : bcs_) {
      const DofIndex dof_index = it.first;
      tangent_matrix->row(dof_index) *= T(0);
      tangent_matrix->col(dof_index) *= T(0);
      tangent_matrix->coeffRef(dof_index, dof_index) = T(1);
    }
  }

  void ApplyBoundaryConditionToTangentMatrix(
      internal::PetscSymmetricBlockSparseMatrix* tangent_matrix) const {
    DRAKE_DEMAND(tangent_matrix != nullptr);
    DRAKE_DEMAND(tangent_matrix->rows() == tangent_matrix->cols());
    if (bcs_.size() == 0) {
      return;
    }
    /* Check validity of the dof indices stored. */
    VerifyBcIndexes(tangent_matrix->cols());

    /* Zero out all rows and columns of the tangent matrix corresponding to dofs
     under the BC (except the diagonal entry which is set to 1). */
    std::vector<int> indexes(bcs_.size());
    int i = 0;
    for (const auto& it : bcs_) {
      indexes[i++] = it.first;
    }
    tangent_matrix->ZeroRowsAndColumns(indexes, /* diagonal entry */ 1.0);
  }

  /** Modifies the given residual that arises from an FEM system without BC into
   the residual for the same system subject to `this` BC. More specifically, the
   entries corresponding to dofs under the BC will be zeroed out.
   @pre residual != nullptr.
   @throw std::exception if any of the indexes of the dofs under the boundary
   condition specified by `this` %DirichletBoundaryCondition  is greater than
   or equal to the `residual->size()`. */
  void ApplyBoundaryConditionToResidual(EigenPtr<VectorX<T>> residual) const {
    DRAKE_DEMAND(residual != nullptr);
    if (bcs_.size() == 0) {
      return;
    }
    /* Check validity of the dof indices stored. */
    VerifyBcIndexes(residual->size());

    /* Zero out all entries of the residual corresponding to dofs under the BC.
     */
    for (const auto& it : bcs_) {
      const DofIndex dof_index = it.first;
      (*residual)(int{dof_index}) = 0;
    }
  }

  /** Verifies that the largest index for the dofs under BC is smaller than the
   given `size`. Otherwise, throw an exception. */
  void VerifyBcIndexes(int size) const {
    const auto& last_bc = bcs_.crbegin();
    if (last_bc->first >= size) {
      throw std::runtime_error(
          "An index of the dirichlet boundary condition is out of the range.");
    }
  }

 private:
  /* We sort the boundary conditions according to dof indices for better
   cache consistency when applying the BC. The value of the map stores the
   value of q, qdot and qddot (in that order and when applicable) of the dof
   with index of the key. */
  std::map<DofIndex, Vector3<T>> bcs_{};
};
}  // namespace fem
}  // namespace multibody
}  // namespace drake
