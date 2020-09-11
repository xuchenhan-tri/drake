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
  EvalF0();
  Evalqstar();

  // TODO(xuchenhan-tri) consider putting dv in a scratch space to avoid
  // allocation every time step.
  Matrix3X<T> dv = Matrix3X<T>::Zero(v0.rows(), v0.cols());
  Eigen::Map<VectorX<T>> z(dv.data(), dv.size());
  newton_solver_.Solve(&z);
  dv = Eigen::Map<Matrix3X<T>>(z.data(), dv.rows(), dv.cols());

  auto& v_star = state_.get_mutable_v_star();
  v_star = v0 + dv;
}

template <typename T>
const std::vector<Matrix3<T>>& FemSolver<T>::EvalF0() const {
  if (!state_.F0_out_of_date()) {
    return state_.get_F0();
  }
  const auto& elements = data_.get_elements();
  const auto& q0 = state_.get_q0();
  auto& F0 = state_.get_mutable_F0();
  int quadrature_offset = 0;
  for (const auto& e : elements) {
    F0[quadrature_offset++] = e.CalcF(q0);
  }
  state_.set_F0_out_of_date(false);
  return state_.get_F0();
}

template <typename T>
const Matrix3X<T>& FemSolver<T>::Evalqstar() const {
  if (!state_.q_star_out_of_date()) {
    return state_.get_q_star();
  }
  auto& q_star = state_.get_mutable_q_star();
  const auto& dt = data_.get_dt();
  const auto& v0 = state_.get_v0();
  const auto& q0 = state_.get_q0();
  q_star = q0 + dt * v0;
  state_.set_q_star_out_of_date(false);
  return state_.get_q_star();
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
  auto A = objective_.GetA();
  drake::multibody::solvers::InverseOperator<T> Ainv(
      "Ainv", &newton_solver_.get_linear_solver(), *A);
  const auto& v_star = state_.get_v_star();
  const VectorX<T>& v_star_tmp = Eigen::Map<const VectorX<T>>(v_star.data(), v_star.size());
  VectorX<T> tau = VectorX<T>::Zero(get_num_position_dofs());
  drake::multibody::solvers::SystemDynamicsData<T> dynamics_data(&Ainv, &v_star_tmp,
                                                                 &tau);
  // Fill PointContactData.
  auto& Jc = state_.get_mutable_Jc();
  const auto& dt = data_.get_dt();
  // Perform contact query at the temporary position q0 + dt * v*.
  Matrix3X<T> tmp_q = state_.get_q0() + dt*v_star;
  QueryContact(tmp_q, collision_objects, &Jc, &penetration_depth);
  drake::multibody::solvers::SparseLinearOperator<T> Jc_lop("Jc", &Jc);
  // TODO(xuchenhan-tri) Properly set these coefficients.
  VectorX<T> stiffness = VectorX<T>::Zero(penetration_depth.size());
  VectorX<T> dissipation = VectorX<T>::Zero(penetration_depth.size());
  const T friction_coeff = 0.1;
  VectorX<T> mu = friction_coeff * VectorX<T>::Ones(penetration_depth.size());
  drake::multibody::solvers::PointContactData<T> point_data(
      &penetration_depth, &Jc_lop, &stiffness, &dissipation, &mu);
  // Solve contact constraints with PGS.
  drake::multibody::solvers::PgsSolver<T> pgs;
  pgs.SetSystemDynamicsData(&dynamics_data);
  pgs.SetPointContactData(&point_data);
  pgs.SolveWithGuess(dt, v_star_tmp);
  // Update positions and velocities.
  const auto& v_new_tmp = pgs.GetVelocities();
  auto& v = state_.get_mutable_v();
  v = Eigen::Map<const Matrix3X<T>>(v_new_tmp.data(), 3, v_new_tmp.size() / 3);
  auto& q = state_.get_mutable_q();
  q = state_.get_q0() + dt * v;
}
}  // namespace fem
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::FemSolver)
