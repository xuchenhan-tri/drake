#pragma once

#include <memory>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/constitutive_model.h"
#include "drake/fem/corotated_linear_model.h"
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
    auto v = data_.get_mutable_v();
    auto& dv = data_.get_mutable_dv();
    const auto& dt = data_.get_dt();
    T time = data_.get_time();

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
    int num_elements = data_.get_num_elements();
    int num_vertices = data_.get_num_vertices();
    int num_objects = data_.get_num_objects();
    auto& element_indices = data_.get_mutable_element_indices();
    auto& vertex_indices = data_.get_mutable_vertex_indices();
    auto& elements = data_.get_mutable_elements();
    auto& Q = data_.get_mutable_Q();
    auto& q = data_.get_mutable_q();
    auto& v = data_.get_mutable_v();
    auto& dv = data_.get_mutable_dv();
    auto& mass = data_.get_mutable_mass();
    auto& mesh = data_.get_mutable_mesh();

    // Add new elements and record the element indices for this object.
    std::vector<int> local_element_indices(indices.size());
    Vector4<int> particle_offset{num_vertices, num_vertices, num_vertices,
                                 num_vertices};
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      Matrix3X<T> local_positions(3, 4);
      for (int j = 0; j < 4; ++j) {
        local_positions.col(j) = positions.col(indices[i][j]);
      }
      mesh.push_back(indices[i] + particle_offset);
      elements.emplace_back(
          indices[i] + particle_offset, positions,
          std::make_unique<CorotatedLinearElasticity<T>>(
              config.youngs_modulus, config.poisson_ratio, config.mass_damping,
              config.stiffness_damping, local_positions),
          config.density);
      local_element_indices[i] = num_elements++;
    }
    element_indices.push_back(local_element_indices);

    // Record the vertex indices for this object.
    std::vector<int> local_vertex_indices(positions.cols());
    for (int i = 0; i < positions.cols(); ++i)
      local_vertex_indices[i] = num_vertices++;
    vertex_indices.push_back(local_vertex_indices);

    // Allocate for positions and velocities.
    Q.conservativeResize(3, q.cols() + positions.cols());
    Q.rightCols(positions.cols()) = positions;
    q.conservativeResize(3, q.cols() + positions.cols());
    q.rightCols(positions.cols()) = positions;
    v.conservativeResize(3, v.cols() + positions.cols());
    v.rightCols(positions.cols()).setZero();
    dv.resize(3, v.cols());

    // Set mass.
    mass.conservativeResize(mass.size() + positions.cols());
    const int object_id = num_objects;
    SetMassFromDensity(object_id, config.density);
    return object_id;
  }
  /**
      Set the initial positions and velocities of a given object.
      @param[in] object_id     The id the object whose mass is being set.
      @param[in] density     Mass density of the object.
      */

  void SetMassFromDensity(const int object_id, const T density) {
    const auto& vertex_indices = data_.get_vertex_indices();
    const auto& element_indices = data_.get_element_indices();
    const auto& elements = data_.get_elements();
    auto& mass = data_.get_mutable_mass();
    const auto& vertex_range = vertex_indices[object_id];
    // Clear old mass values.
    for (int i = 0; i < static_cast<int>(vertex_range.size()); ++i) {
      mass[i] = 0;
    }
    // Add the mass contribution of each element.
    const auto& element_range = element_indices[object_id];
    for (int i = 0; i < static_cast<int>(element_range.size()); ++i) {
      const auto& element = elements[i];
      const Vector4<int>& local_indices = element.get_indices();
      for (int j = 0; j < static_cast<int>(local_indices.size()); ++j) {
        mass[local_indices[j]] += density * element.get_element_measure();
      }
    }
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
