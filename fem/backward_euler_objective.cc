#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/fem_solver.h"

namespace drake {
namespace fem {

template <typename T>
void BackwardEulerObjective<T>::Update(const VectorX<T>& dv) {
  const Matrix3X<T>& tmp_x =
      Eigen::Map<const Matrix3X<T>>(dv.data(), 3, dv.size() / 3) *
          fem_solver_.get_dt() +
      fem_solver_.get_q_hat();
  auto& elements = fem_solver_.get_mutable_elements();
  for (auto& e : elements) {
    e.UpdateF(tmp_x);
  }
}

template <typename T>
void BackwardEulerObjective<T>::CalcResidual(VectorX<T>* residual) {
  Eigen::Map<Matrix3X<T>> impulse(residual->data(), 3, residual->size() / 3);
  impulse.setZero();
  const VectorX<T>& mass = fem_solver_.get_mass();
  const Matrix3X<T>& dv = fem_solver_.get_dv();
  const T& dt = fem_solver_.get_dt();
  const Vector3<T>& gravity = fem_solver_.get_gravity();
  const auto& v_hat = fem_solver_.get_v() + fem_solver_.get_dv();

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
  DRAKE_DEMAND(prod->cols() == fem_solver_.get_mass().size());
  DRAKE_DEMAND(x.cols() == fem_solver_.get_mass().size());
  const VectorX<T>& mass = fem_solver_.get_mass();
  const T& dt = fem_solver_.get_dt();
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
void BackwardEulerObjective<T>::Project(EigenPtr<Matrix3X<T>> impulse) const {
  const auto& bc = fem_solver_.get_v_bc();
  const auto& vertex_indices = fem_solver_.get_vertex_indices();
  const T& time = fem_solver_.get_time();
  const Matrix3X<T>& initial_position = fem_solver_.get_Q();

  for (const auto& boundary_condition : bc) {
    const auto& vertex_range = vertex_indices[boundary_condition.object_id];
    for (int j = 0; j < static_cast<int>(vertex_range.size()); ++j) {
      boundary_condition.bc(vertex_range[j], time, initial_position, impulse);
    }
  }
}

template <typename T>
int BackwardEulerObjective<T>::get_num_dofs() const { return fem_solver_.get_q().size(); }

template class BackwardEulerObjective<double>;
}  // namespace fem
}  // namespace drake
