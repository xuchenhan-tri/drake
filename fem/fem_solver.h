#pragma once

#include <memory>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/fem_config.h"
#include "drake/fem/fem_data.h"
#include "drake/fem/fem_element.h"
#include "drake/fem/fem_force.h"
#include "drake/fem/fem_system.h"
#include "drake/fem/newton_solver.h"
#include "drake/fem/contact_jacobian.h"

#include "drake/multibody/solvers/point_contact_data.h"
#include "drake/multibody/solvers/system_dynamics_data.h"
#include "drake/multibody/solvers/pgs_solver.h"
#include "drake/multibody/solvers/sparse_linear_operator.h"
#include "drake/multibody/solvers/inverse_operator.h"

namespace drake {
namespace fem {

template <typename T>
class FemSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemSolver);

  explicit FemSolver(T dt)
      : data_(dt),
        force_(data_.get_elements()),
        objective_(&data_, &force_),
        newton_solver_(&objective_),
        contact_jacobian_(data_.get_q(), data_.get_collision_objects()){}
  /**
   The internal main loop for the FEM simulation that calls NewtonSolver to
   calculate the discrete velocity change. Update the position and velocity
   states from time n to time n+1.
   */
  void AdvanceOneTimeStep(const Eigen::Ref<const Matrix3X<T>>& q_n) {
    auto& q = data_.get_mutable_q();
    auto& q_hat = data_.get_mutable_q_hat();
    auto& v = data_.get_mutable_v();
    auto& dv = data_.get_mutable_dv();
    auto& elements = data_.get_mutable_elements();
    const auto& dt = data_.get_dt();
    T time = data_.get_time();

    for (auto& e : elements){
        e.UpdateTimeNPositionBasedState(q_n);
    }
    dv.setZero();
    q_hat = q_n + dt * v;
    Eigen::Map<VectorX<T>> x(dv.data(), dv.size());
    newton_solver_.Solve(&x);
    dv = Eigen::Map<Matrix3X<T>>(x.data(), dv.rows(), dv.cols());
    v += dv;
    q = q_hat + dt * dv;

    SolveContact();
    data_.set_time(time + dt);
  }

  void SolveContact() {

      T time = data_.get_time();
      auto &collision_objects = data_.get_mutable_collision_objects();
      for (auto& cb : collision_objects){
          cb->Update(time);
      }
      std::cout << "Time = " << time << std::endl;

      T friction_coeff = 1.0;
      auto& v = data_.get_mutable_v();
      auto& q = data_.get_mutable_q();
      const auto& dt = data_.get_dt();

      Eigen::SparseMatrix<T> jacobian;
      VectorX<T> penetration_depth;
      const VectorX<T>& v_free = Eigen::Map<VectorX<T>>(v.data(), v.size());
      VectorX<T> tau = VectorX<T>::Zero(v.size());
      contact_jacobian_.QueryContact(&jacobian, &penetration_depth);
//      if (contact_jacobian_.get_normals().size() > 0)
//          std::cout << contact_jacobian_.get_normals()<< std::endl;
      drake::multibody::solvers::InverseOperator<T> Ainv("Inverse stiffness matrix", &newton_solver_.get_linear_solver());
      drake::multibody::solvers::SystemDynamicsData<T> dynamics_data(&Ainv, &v_free, &tau);
      drake::multibody::solvers::SparseLinearOperator<T> Jc("Jc", &jacobian);
      VectorX<T> stiffness = VectorX<T>::Zero(penetration_depth.size());
      VectorX<T> dissipation = VectorX<T>::Zero(penetration_depth.size());
      VectorX<T> mu = friction_coeff * VectorX<T>::Ones(penetration_depth.size());
      drake::multibody::solvers::PointContactData<T> point_data(&penetration_depth, &Jc, &stiffness, &dissipation, &mu);

      drake::multibody::solvers::PgsSolver<T> pgs;
      pgs.SetSystemDynamicsData(&dynamics_data);
      pgs.SetPointContactData(&point_data);
      pgs.SolveWithGuess(dt, v_free);
      const auto& v_new_tmp = pgs.GetVelocities();
      Matrix3X<T> v_new = Eigen::Map<const Matrix3X<T>>(v_new_tmp.data(), 3, v_new_tmp.size()/3);
      auto dv = v_new - v;
//      std::cout << " dv = \n" << dv << std::endl;
      q += dt * dv;
      v = v_new;
  }
  /**
   Add an object represented by a list of vertices connected by a simplex mesh
  to the simulation.
  @param[in] indices    The list of indices describing the connectivity of the
  mesh. @p indices[i] contains the indices of the 4 vertices in the i-th
  element.
  @param[in] positions  The list of positions of the vertices in the undeformed
  configuration.
  @return The object_id of the newly added_object.
  */
  int AddUndeformedObject(const std::vector<Vector4<int>>& indices,
                          const Matrix3X<T>& positions,
                          const FemConfig& config) {
    return data_.AddUndeformedObject(indices, positions, config);
  }

  /**
      Set the initial positions and velocities of a given object.
      @param[in] object_id     The id the object whose initial conditions are
     being set.
      @param[in] set_position  The function that takes an index i that modifies
     the initial position of the i-th vertex in the chosen object.
      @param[in] set_velocity  The function that takes an index i that modifies
     the initial velocity of the i-th vertex in the chosen object.

      @pre @p object_id < number of existing objects.
   */
  void SetInitialStates(
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
      set_position(i, &init_q);
      set_velocity(i, &init_v);
    }
    for (int i = 0; i < static_cast<int>(vertex_range.size()); ++i) {
      q.col(vertex_range[i]) = init_q.col(i);
      Q.col(vertex_range[i]) = init_q.col(i);
      v.col(vertex_range[i]) = init_v.col(i);
    }
  }

  /**
      Set the boundary condition of a given object. If the current boundary
     condition being set is incompatible with a previous boundary condition, the
     new one will be applied.
      @param[in] object_id   The id the object whose initial conditions are
     being set.
      @param[in] bc          The function that takes an index i and the time t
     that modifies the position and the velocity of the i-th vertex in the
     chosen object at time t.

      @pre @p object_id < number of existing objects.
   */
  void SetBoundaryCondition(
      const int object_id,
      std::function<void(int, const Matrix3X<T>&, EigenPtr<Matrix3X<T>>)> bc) {
    const int num_objects = data_.get_num_objects();
    auto& v_bc = data_.get_mutable_v_bc();
    DRAKE_DEMAND(object_id < num_objects);
    v_bc.emplace_back(object_id, bc);
  }

  void AddCollisionObject(std::unique_ptr<CollisionObject<T>> object) {
      data_.add_collision_object(std::move(object));
  }

  int get_num_position_dofs() const { return data_.get_num_position_dofs(); }

  const std::vector<Vector4<int>>& get_mesh() const { return data_.get_mesh(); }

  const Matrix3X<T>& get_q() const { return data_.get_q(); }

 private:
  FemData<T> data_;
  FemForce<T> force_;
  BackwardEulerObjective<T> objective_;
  NewtonSolver<T> newton_solver_;
  ContactJacobian<T> contact_jacobian_;
};

}  // namespace fem
}  // namespace drake
