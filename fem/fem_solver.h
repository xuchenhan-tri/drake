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
        newton_solver_(&objective_) {}
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
    data_.set_time(time + dt);
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

  int get_num_position_dofs() const { return data_.get_num_position_dofs(); }

  const std::vector<Vector4<int>>& get_mesh() const { return data_.get_mesh(); }

  const Matrix3X<T>& get_q() const { return data_.get_q(); }

 private:
  FemData<T> data_;
  FemForce<T> force_;
  BackwardEulerObjective<T> objective_;
  NewtonSolver<T> newton_solver_;
};

}  // namespace fem
}  // namespace drake
