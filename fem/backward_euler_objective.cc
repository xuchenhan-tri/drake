#include "drake/fem/backward_euler_objective.h"

#include <vector>

#include "drake/fem/fem_data.h"

namespace drake {
namespace fem {

template <typename T>
void BackwardEulerObjective<T>::Update(const Eigen::Ref<const VectorX<T>>& dv) {
  // Move positions states to tmp_q = qₙ + dt * (vₙ + dv).
  const Matrix3X<T>& tmp_q =
      Eigen::Map<const Matrix3X<T>>(dv.data(), 3, dv.size() / 3) *
          fem_data_.get_dt() +
      fem_data_.get_q_hat();
  auto& elements = fem_data_.get_mutable_elements();
  // Update deformation gradient on all elements.
  for (auto& e : elements) {
    e.UpdateF(tmp_q);
  }
}

template <typename T>
void BackwardEulerObjective<T>::CalcResidual(EigenPtr<VectorX<T>> residual) {
  Eigen::Map<Matrix3X<T>> impulse(residual->data(), 3, residual->size() / 3);
  impulse.setZero();
  const VectorX<T>& mass = fem_data_.get_mass();
  const Matrix3X<T>& dv = fem_data_.get_dv();
  const T& dt = fem_data_.get_dt();
  const Vector3<T>& gravity = fem_data_.get_gravity();
  const auto& v_hat = fem_data_.get_v() + fem_data_.get_dv();

  // Add -M*x + gravity * dt
  for (int i = 0; i < mass.size(); ++i) {
    impulse.col(i) -= mass(i) * dv.col(i);
    impulse.col(i) += mass(i) * dt * gravity;
  }
  // Add fe * dt.
  force_.AccumulateScaledElasticForce(dt, &impulse);
  // Add fd * dt.
  force_.AccumulateScaledDampingForce(dt, v_hat, &impulse);
  // Apply boundary condition.
  Project(&impulse);
  *residual = Eigen::Map<VectorX<T>>(impulse.data(), impulse.size());
}

template <typename T>
void BackwardEulerObjective<T>::Multiply(const Eigen::Ref<const Matrix3X<T>>& x,
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
  force_.AccumulateScaledDampingForceDifferential(-dt, x, prod);
  // Get  dt² * K * x.
  force_.AccumulateScaledElasticForceDifferential(-dt * dt, x, prod);
  // Apply boundary condition.
  Project(prod);
}

template <typename T>
void BackwardEulerObjective<T>::SetSparsityPattern(
    Eigen::SparseMatrix<T>* jacobian) const {
  std::vector<Eigen::Triplet<T>> non_zero_entries(get_num_dofs());
  // Diagonal entries contains mass and are non-zero.
  for (int i = 0; i < get_num_dofs(); ++i) {
    non_zero_entries[i] = Eigen::Triplet<T>(i, i, 0);
  }
  // Add in the non-zero entries from the stiffness and damping matrices.
  force_.SetSparsityPattern(&non_zero_entries);
  jacobian->setFromTriplets(non_zero_entries.begin(), non_zero_entries.end());
  jacobian->makeCompressed();
}

template <typename T>
void BackwardEulerObjective<T>::BuildJacobian(
    Eigen::SparseMatrix<T>* jacobian) const {
  const auto& mass = get_mass();
  const T& dt = fem_data_.get_dt();
  // The dimension of the matrix should be properly set by the caller and
  // should be the same as the number of dofs in the system.
  DRAKE_DEMAND(jacobian->cols() == get_num_dofs());
  // Clear out old data.
  for (int k = 0; k < jacobian->outerSize(); ++k)
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(*jacobian, k); it;
         ++it) {
      it.valueRef() = 0.0;
    }
  // Add mass to the diagonal entries.
  for (int i = 0; i < mass.size(); ++i) {
    for (int d = 0; d < 3; ++d) {
      jacobian->coeffRef(3 * i + d, 3 * i + d) += mass(i);
    }
  }
  // Add Stiffness and damping matrix to the Jacobian.
  force_.AccumulateScaledStiffnessMatrix(dt * dt, jacobian);
  force_.AccumulateScaledDampingMatrix(dt, jacobian);
}

template <typename T>
void BackwardEulerObjective<T>::Project(EigenPtr<Matrix3X<T>> impulse) const {
  const auto& bc = fem_data_.get_v_bc();
  const auto& vertex_indices = fem_data_.get_vertex_indices();
  const Matrix3X<T>& initial_position = fem_data_.get_Q();

  for (const auto& boundary_condition : bc) {
    const auto& vertex_range = vertex_indices[boundary_condition.object_id];
    for (int j = 0; j < static_cast<int>(vertex_range.size()); ++j) {
      boundary_condition.bc(vertex_range[j], initial_position, impulse);
    }
  }
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

}  // namespace fem
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::BackwardEulerObjective)
