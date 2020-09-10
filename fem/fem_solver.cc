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
  auto& q = data_.get_mutable_q();
  auto& v = data_.get_mutable_v();

  DRAKE_DEMAND(object_id < num_objects);
  const auto& vertex_range = vertex_indices[object_id];
  Matrix3X<T> init_q(3, vertex_range.size());
  Matrix3X<T> init_v(3, vertex_range.size());
  for (int i = 0; i < static_cast<int>(vertex_range.size()); ++i) {
    init_q.col(i) = Q.col(vertex_range[i]);
    init_v.col(i) = v.col(vertex_range[i]);
  }
  for (int i = 0; i < static_cast<int>(vertex_range.size()); ++i) {
    if (set_position != nullptr) {
      set_position(i, &init_q);
    }
    if (set_velocity != nullptr) {
      set_velocity(i, &init_v);
    } else {
      init_v.col(i).setZero();
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
    const int object_id,
    std::function<bool(int, const Matrix3X<T>&)> bc) {
  const int num_objects = data_.get_num_objects();
  auto& v_bc = data_.get_mutable_v_bc();
  DRAKE_DEMAND(object_id < num_objects);
  v_bc.emplace_back(object_id, bc);
}

template <typename T>
void FemSolver<T>::SolveFreeMotion() {
  auto& q = data_.get_mutable_q();
  const auto& q_n = data_.get_q();
  auto& q_hat = data_.get_mutable_q_star();
  auto& v = data_.get_mutable_v();
  auto& dv = data_.get_mutable_dv();
  auto& elements = data_.get_mutable_elements();
  const auto& dt = data_.get_dt();

  for (auto& e : elements) {
    e.UpdateTimeNPositionBasedState(q_n);
  }
  dv.setZero();
  q_hat = q_n + dt * v;
  Eigen::Map<VectorX<T>> x(dv.data(), dv.size());
  newton_solver_.Solve(&x);
  dv = Eigen::Map<Matrix3X<T>>(x.data(), dv.rows(), dv.cols());
  v += dv;
  q = q_hat + dt * dv;
}

template <typename T>
void FemSolver<T>::SolveContact() {
  T time = data_.get_time();
  auto& v = data_.get_mutable_v();
  auto& q = data_.get_mutable_q();
  const auto& dt = data_.get_dt();

  // Update the position of collision objects.
  auto& collision_objects = data_.get_mutable_collision_objects();
  for (auto& cb : collision_objects) {
    cb->Update(time);
  }
  // Fill SystemDynamicsData and PointContact Data.
  Eigen::SparseMatrix<T> jacobian;
  VectorX<T> penetration_depth;
  const VectorX<T>& v_free = Eigen::Map<VectorX<T>>(v.data(), v.size());
  VectorX<T> tau = VectorX<T>::Zero(v.size());
  contact_jacobian_.QueryContact(&jacobian, &penetration_depth);
  auto J = objective_.GetJacobian();
  drake::multibody::solvers::InverseOperator<T> Ainv(
      "Inverse stiffness matrix", &newton_solver_.get_linear_solver(), *J);
  drake::multibody::solvers::SystemDynamicsData<T> dynamics_data(&Ainv, &v_free,
                                                                 &tau);
  drake::multibody::solvers::SparseLinearOperator<T> Jc("Jc", &jacobian);
  VectorX<T> stiffness = VectorX<T>::Zero(penetration_depth.size());
  VectorX<T> dissipation = VectorX<T>::Zero(penetration_depth.size());
  // TODO(xuchenhan-tri) Properly set friction coefficients.
  const T friction_coeff = 0.1;
  VectorX<T> mu = friction_coeff * VectorX<T>::Ones(penetration_depth.size());
  drake::multibody::solvers::PointContactData<T> point_data(
      &penetration_depth, &Jc, &stiffness, &dissipation, &mu);
  // Solve contact constraints with PGS.
  drake::multibody::solvers::PgsSolver<T> pgs;
  pgs.SetSystemDynamicsData(&dynamics_data);
  pgs.SetPointContactData(&point_data);
  pgs.SolveWithGuess(dt, v_free);
  // Update positions and velocities.
  const auto& v_new_tmp = pgs.GetVelocities();
  Matrix3X<T> v_new =
      Eigen::Map<const Matrix3X<T>>(v_new_tmp.data(), 3, v_new_tmp.size() / 3);
  auto dv = v_new - v;
  q += dt * dv;
  v = v_new;
}
}  // namespace fem
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::drake::fem::FemSolver)
