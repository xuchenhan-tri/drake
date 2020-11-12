#include "drake/fem/fem_solver.h"
namespace drake {
namespace fem {
template <typename T>
void FemSolver<T>::SetInitialStates(
    const int object_id,
    std::function<void(int, EigenPtr<Matrix3X<T>>)> set_position,
    std::function<void(int, EigenPtr<Matrix3X<T>>)> set_velocity) {
  const int num_objects = data_.get_num_objects();
  const auto& vertex_indices = data_.get_vertex_indices();
  auto& Q = data_.get_mutable_Q();
  auto& q = state_.get_mutable_q();
  auto& v = state_.get_mutable_v();
  q.conservativeResize(3, Q.cols());
  v.conservativeResize(3, Q.cols());

  DRAKE_DEMAND(object_id < num_objects);
  const auto& vertex_range = vertex_indices[object_id];
  Matrix3X<T> init_q(3, vertex_range.size());
  Matrix3X<T> init_v = Matrix3X<T>::Zero(3, vertex_range.size());
  for (int i = 0; i < static_cast<int>(vertex_range.size()); ++i) {
    init_q.col(i) = Q.col(vertex_range[i]);
  }
  for (int i = 0; i < static_cast<int>(vertex_range.size()); ++i) {
    if (set_position != nullptr) {
      set_position(i, &init_q);
    }
    if (set_velocity != nullptr) {
      set_velocity(i, &init_v);
    }
  }
  for (int i = 0; i < static_cast<int>(vertex_range.size()); ++i) {
    q.col(vertex_range[i]) = init_q.col(i);
    Q.col(vertex_range[i]) = init_q.col(i);
    v.col(vertex_range[i]) = init_v.col(i);
  }
}

template <typename T>
void FemSolver<T>::SetBoundaryCondition(
    const int object_id, std::function<bool(int, const Matrix3X<T>&)> bc) {
  const int num_objects = data_.get_num_objects();
  auto& v_bc = data_.get_mutable_v_bc();
  DRAKE_DEMAND(object_id < num_objects);
  v_bc.emplace_back(object_id, bc);
}

template <typename T>
void FemSolver<T>::SolveFreeMotion() const {
  auto& v0 = state_.get_mutable_v0();
  v0 = state_.get_v();
  auto& q0 = state_.get_mutable_q0();
  q0 = state_.get_q();

  auto& dv = state_.get_mutable_dv();
  dv.setZero();
  Eigen::Map<VectorX<T>> z(dv.data(), dv.size());
  newton_solver_.Solve(&z);
}

template <typename T>
void FemSolver<T>::SolveContact() const {
  // Update the position of collision objects.
  auto& collision_objects = data_.get_mutable_collision_objects();
  T time = state_.get_time();
  for (auto& cb : collision_objects) {
    cb->Update(time);
  }
  // Fill SystemDynamicsData.
  VectorX<T> penetration_depth;
  auto A = objective_.GetA(state_);
  drake::multibody::contact_solvers::internal::InverseOperator<T> Ainv(
      "Ainv", &newton_solver_.get_linear_solver(), *A);
  const Matrix3X<T>& v_star = state_.get_v0() + state_.get_dv();
  const VectorX<T>& v_star_tmp =
      Eigen::Map<const VectorX<T>>(v_star.data(), v_star.size());
  drake::multibody::contact_solvers::internal::SystemDynamicsData<T>
      dynamics_data(&Ainv, &v_star_tmp);
  // Fill PointContactData.
  auto& Jc = state_.get_mutable_cache().get_mutable_Jc();
  const auto& dt = data_.get_dt();
  // Perform contact query at the temporary position q0 + dt * v*.
  Matrix3X<T> tmp_q = state_.get_q0() + dt * v_star;
  QueryContact(tmp_q, collision_objects, &Jc, &penetration_depth);
  drake::multibody::contact_solvers::internal::SparseLinearOperator<T> Jc_lop(
      "Jc", &Jc);
  // TODO(xuchenhan-tri) Properly set these coefficients.
  VectorX<T> stiffness = VectorX<T>::Zero(penetration_depth.size());
  VectorX<T> dissipation = VectorX<T>::Zero(penetration_depth.size());
  const T friction_coeff = 0.1;
  VectorX<T> mu = friction_coeff * VectorX<T>::Ones(penetration_depth.size());
  drake::multibody::contact_solvers::internal::PointContactData<T> point_data(
      &penetration_depth, &Jc_lop, &stiffness, &dissipation, &mu);
  // Solve contact constraints with PGS.
  drake::multibody::contact_solvers::internal::PgsSolver<T> pgs;
  drake::multibody::contact_solvers::internal::ContactSolverResults<T> result;
  pgs.SolveWithGuess(dt, dynamics_data, point_data, v_star_tmp, &result);
  // Update positions and velocities.
  const auto& v_new_tmp = result.v_next;
  auto& v = state_.get_mutable_v();
  v = Eigen::Map<const Matrix3X<T>>(v_new_tmp.data(), 3, v_new_tmp.size() / 3);
  auto& q = state_.get_mutable_q();
  q = state_.get_q0() + dt * v;
}
}  // namespace fem
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::FemSolver)
