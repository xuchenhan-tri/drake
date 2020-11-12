#include "drake/fem/backward_euler_objective.h"

#include <vector>

#include "drake/fem/fem_data.h"

namespace drake {
namespace fem {
template <typename T>
void BackwardEulerObjective<T>::CalcResidual(
    const FemState<T>& state, EigenPtr<VectorX<T>> residual) const {
  Eigen::Map<Matrix3X<T>> impulse(residual->data(), 3, residual->size() / 3);
  impulse.setZero();

  const auto& dv = state.get_dv();
  const VectorX<T>& mass = fem_data_.get_mass();
  const T& dt = fem_data_.get_dt();
  const Vector3<T>& gravity = fem_data_.get_gravity();

  // Add -M*x + gravity * dt
  for (int i = 0; i < mass.size(); ++i) {
    impulse.col(i) -= mass(i) * dv.col(i);
    impulse.col(i) += mass(i) * dt * gravity;
  }
  // Add fe * dt.
  force_.AccumulateScaledElasticForce(state, dt, &impulse);
  // Add fd * dt.
  force_.AccumulateScaledDampingForce(state, dt, &impulse);
  // Apply boundary condition.
  Project(&impulse);
  *residual = Eigen::Map<VectorX<T>>(impulse.data(), impulse.size());
}

template <typename T>
void BackwardEulerObjective<T>::Multiply(const FemState<T>& state,
                                         const Eigen::Ref<const Matrix3X<T>>& x,
                                         EigenPtr<Matrix3X<T>> prod) const {
  const VectorX<T>& mass = fem_data_.get_mass();
  DRAKE_DEMAND(prod->cols() == mass.size());
  DRAKE_DEMAND(x.cols() == mass.size());
  const T& dt = fem_data_.get_dt();
  // Get M*x.
  for (int i = 0; i < prod->cols(); ++i) {
    prod->col(i) = mass(i) * x.col(i);
  }
  // Get dt * (alpha * M + beta * K) * x.
  force_.AccumulateScaledDampingForceDifferential(state, -dt, x, prod);
  // Get  dt² * K * x.
  force_.AccumulateScaledElasticForceDifferential(state, -dt * dt, x, prod);
  // Apply boundary condition.
  Project(prod);
}

template <typename T>
void BackwardEulerObjective<T>::SetSparsityPattern(
    Eigen::SparseMatrix<T>* A) const {
  std::vector<Eigen::Triplet<T>> non_zero_entries(get_num_dofs());
  // Diagonal entries contains mass and are non-zero.
  for (int i = 0; i < get_num_dofs(); ++i) {
    non_zero_entries[i] = Eigen::Triplet<T>(i, i, 0);
  }
  // Add in the non-zero entries from the stiffness and damping matrices.
  force_.SetSparsityPattern(&non_zero_entries);
  A->setFromTriplets(non_zero_entries.begin(), non_zero_entries.end());
  A->makeCompressed();
}

template <typename T>
T BackwardEulerObjective<T>::norm(const Eigen::Ref<const VectorX<T>>& x) const {
  // Input has unit of impulse. Convert to the unit of velocity by dividing by
  // mass.
  const auto& tmp_x = Eigen::Map<const Matrix3X<T>>(x.data(), 3, x.size() / 3);
  DRAKE_DEMAND(tmp_x.cols() == fem_data_.get_mass().size());
  const auto& tmp_mass = fem_data_.get_mass().transpose().array();
  return (tmp_x.array().rowwise() / tmp_mass).abs().maxCoeff();
}

template <typename T>
std::unique_ptr<multibody::contact_solvers::internal::LinearOperator<T>>
BackwardEulerObjective<T>::GetA(const FemState<T>& state) const {
  if (matrix_free_) {
    auto multiply = [this, &state](const Eigen::Ref<const VectorX<T>>& x,
                                   EigenPtr<VectorX<T>> y) {
      const Matrix3X<T>& tmp_x =
          Eigen::Map<const Matrix3X<T>>(x.data(), 3, x.size() / 3);
      Matrix3X<T> tmp_y;
      tmp_y.resize(3, tmp_x.cols());
      this->Multiply(state, tmp_x, &tmp_y);
      *y = Eigen::Map<VectorX<T>>(tmp_y.data(), tmp_y.size());
    };
    return std::make_unique<
        multibody::contact_solvers::internal::SparseLinearOperator<T>>(
        "A", multiply, get_num_dofs(), get_num_dofs());
  }
  int num_dofs = get_num_dofs();
  auto& A = state.get_mutable_cache().get_mutable_A();
  if (A.cols() != num_dofs) {
    A.resize(num_dofs, num_dofs);
    SetSparsityPattern(&A);
  }
  BuildA(state, &A);
  return std::make_unique<
      multibody::contact_solvers::internal::SparseLinearOperator<T>>("A", &A);
}

template <typename T>
void BackwardEulerObjective<T>::BuildA(const FemState<T>& state,
                                       Eigen::SparseMatrix<T>* A) const {
  const auto& mass = get_mass();
  const T& dt = fem_data_.get_dt();
  // The dimension of the matrix should be properly set by the caller and
  // should be the same as the number of dofs in the system.
  DRAKE_DEMAND(A->cols() == get_num_dofs());
  // Clear out old data.
  for (int k = 0; k < A->outerSize(); ++k)
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(*A, k); it; ++it) {
      it.valueRef() = 0.0;
    }
  // Add mass to the diagonal entries.
  for (int i = 0; i < mass.size(); ++i) {
    for (int d = 0; d < 3; ++d) {
      A->coeffRef(3 * i + d, 3 * i + d) += mass(i);
    }
  }
  // Add Stiffness and damping matrix to the Jacobian.
  force_.AccumulateScaledStiffnessMatrix(state, dt * dt, A);
  force_.AccumulateScaledDampingMatrix(state, dt, A);
  Project(A);
}

template <typename T>
void BackwardEulerObjective<T>::Project(EigenPtr<Matrix3X<T>> impulse) const {
  const auto& bc = fem_data_.get_v_bc();
  const auto& vertex_indices = fem_data_.get_vertex_indices();
  const Matrix3X<T>& initial_position = fem_data_.get_Q();

  for (const auto& boundary_condition : bc) {
    const auto& vertex_range = vertex_indices[boundary_condition.object_id];
    for (int j = 0; j < static_cast<int>(vertex_range.size()); ++j) {
      if (boundary_condition.bc(vertex_range[j], initial_position)) {
        impulse->col(vertex_range[j]).setZero();
      }
    }
  }
}

template <typename T>
void BackwardEulerObjective<T>::Project(Eigen::SparseMatrix<T>* A) const {
  const auto& bc = fem_data_.get_v_bc();
  const auto& vertex_indices = fem_data_.get_vertex_indices();
  const Matrix3X<T>& initial_position = fem_data_.get_Q();
  for (const auto& boundary_condition : bc) {
    const auto& vertex_range = vertex_indices[boundary_condition.object_id];
    for (int j = 0; j < static_cast<int>(vertex_range.size()); ++j) {
      if (boundary_condition.bc(vertex_range[j], initial_position)) {
        for (int col = 3 * vertex_range[j]; col < 3 * (vertex_range[j] + 1);
             ++col) {
          // Set everything in the row corresponding to Dirichlet entry to 0.
          (*A).row(col) *= 0;
          // Set everything in the column corresponding to Dirichlet entry to 0,
          // and set the diagonal entry to 1.
          for (typename Eigen::SparseMatrix<T>::InnerIterator it(*A, col); it;
               ++it) {
            if (it.index() == col)
              it.valueRef() = 1.0;
            else
              it.valueRef() = 0.0;
          }
        }
      }
    }
  }
}
}  // namespace fem
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::BackwardEulerObjective)
