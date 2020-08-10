#pragma once

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/constitutive_model.h"
#include "drake/fem/fem_element.h"
#include "drake/fem/fem_force.h"
#include "drake/fem/newton_solver.h"

namespace drake {
namespace fem {

template <typename T>
class FemSolver {
 public:
  struct BoundaryCondition {
    int object_id;
    std::function<void(int, T, Matrix3X<T>*)> bc;
  };
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemSolver);

  FemSolver(T dt)
      : dt_(dt),
        elements_({}),
        force_(elements_),
        objective_(*this, force_),
        newton_solver_(objective_) {}
  /**
   Calls NewtonSolver to calculate the discrete velocity change.
   Update the position and velocity states from time n to time n+1.
   */
  void UpdateDiscreteState(const VectorX<T>& q_n, const VectorX<T>& v_n,
                           VectorX<T>* next_q, VectorX<T>* next_v) const;

  /** The internal main loop for the FEM simulation. */
  void AdvanceOneTimeStep() {
    dv_.setZero();
    q_hat_ = q_ + dt_ * v_;
    Eigen::Map<VectorX<T>> x(dv_.data(), dv_.size());
    newton_solver_.Solve(&x);
    dv_ = Eigen::Map<Matrix3X<T>>(x.data(), dv_.rows(),  dv_.cols());
    v_ += dv_;
    ApplyVelocityBoundaryConditions();
    q_ = q_hat_ + dt_ * dv_;
    ApplyPositionBoundaryConditions();
    time_ += dt_;
  }
  /**
   Add an object represented by a list of vertices connected by a simplex mesh
  to the simulation.
  @param[in] indices    The list of indices describing the connectivity of the
  mesh. @p indices[i] contains the indices of the 4 vertices in the i-th
  element.
  @param[in] positions  The list of positions of the vertices in the undeformed
  configuration.
  @param[in] model      The constitutive model that the object is endowed with.
  @return The object_id of the newly added_object.
  */
  template <class ModelType>
  int AddUndeformedObject(const std::vector<Vector4<int>>& indices,
                          const Matrix3X<T>& positions, const ModelType& model,
                          const T density) {
    DRAKE_DEMAND(indices.size() == positions.cols());
    // Add new elements and record the element indices for this object.
    std::vector<int> local_element_indices(indices.size());
    Vector4<int> particle_offset{num_vertices_, num_vertices_, num_vertices_,
                                 num_vertices_};
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      elements_.emplace_back(indices[i] + particle_offset, positions,
                             std::make_unique<ModelType>(model));
      local_element_indices[i] = num_elements_++;
    }
    element_indices_.push_back(local_element_indices);

    // Record the vertex indices for this object.
    std::vector<int> local_vertex_indices(positions.cols());
    for (int i = 0; i < positions.cols().size(); ++i)
      local_vertex_indices[i] = num_vertices_++;
    vertex_indices_.push_back(local_vertex_indices);

    // Allocate for positions and velocities.
    q_.conservativeResize(q_.cols() + positions.cols());
    q_.rightCols(positions.cols()) = positions;
    v_.conservativeResize(v_.cols() + positions.cols());
    v_.rightCols(positions.cols()).setZero();

    // Set mass.
    const int object_id = num_objects_++;
    mass_.conservativeResize(mass_.size() + positions.cols());
    SetMassFromDensity(object_id, density);
    return object_id;
  }
  /**
      Set the initial positions and velocities of a given object.
      @param[in] object_id     The id the object whose mass is being set.
      @param[in] density     Mass density of the object.
      */

  void SetMassFromDensity(const int object_id, const T density) {
    const auto& vertex_range = vertex_indices_[object_id];
    // Clear old mass values.
    for (int i = 0; i < static_cast<int>(vertex_range.size()); ++i) {
      mass_[i] = 0;
    }
    // Add the mass contribution of each element.
    const auto& element_range = element_indices_[object_id];
    for (int i = 0; i < static_cast<int>(element_range.size()); ++i) {
      const auto& element = elements_[i];
      const Vector4<int>& local_indices = element.indices();
      for (int j = 0; j < static_cast<int>(local_indices.size()); ++j) {
        mass_[local_indices[j]] += density * element.element_measure();
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
  void SetInitialStates(const int object_id,
                        std::function<void(int, Matrix3X<T>*)> set_position,
                        std::function<void(int, Matrix3X<T>*)> set_velocity) {
    DRAKE_DEMAND(object_id < num_objects_);
    const auto& vertex_range = vertex_indices_[object_id];
    Matrix3X<T> init_q(vertex_range.size());
    Matrix3X<T> init_v(vertex_range.size());
    for (int i = 0; i < static_cast<int>(vertex_range.size()); ++i) {
      set_position(i, init_q);
      set_velocity(i, init_v);
    }
    for (int i = 0; i < static_cast<int>(vertex_range.size()); ++i) {
      q_.col(vertex_range[i]) = init_q[i];
      v_.col(vertex_range[i]) = init_v[i];
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
  void SetVelocityBoundaryCondition(
      const int object_id, std::function<void(int, T, Matrix3X<T>*)> bc) {
    DRAKE_DEMAND(object_id < num_objects_);
    v_bc_.emplace_back(object_id, bc);
  }
  void SetPositionBoundaryCondition(
      const int object_id, std::function<void(int, T, Matrix3X<T>*)> bc) {
    DRAKE_DEMAND(object_id < num_objects_);
    q_bc_.emplace_back(object_id, bc);
  }

  const VectorX<T>& get_mass() const { return mass_; }

  const Matrix3X<T>& get_dv() const { return dv_; }

  const std::vector<FemElement<T>>& get_elements() const { return elements_; }

  std::vector<FemElement<T>>& get_mutable_elements() { return elements_; }

  const FemForce<T>& get_force() const { return force_; }

  FemForce<T>& get_mutable_force() { return force_; }

    const BackwardEulerObjective<T>& get_objective() const { return objective_; }

    BackwardEulerObjective<T>& get_mutable_objective() { return objective_; }

  T get_dt() const { return dt_; }

  void set_dt(T dt) { dt_ = dt; }

  Vector3<T> gravity() { return gravity_; }

  void set_gravity(Vector3<T>& gravity) { gravity_ = gravity; }

 private:
  void ApplyVelocityBoundaryConditions() {
    for (const auto& boundary_condition : v_bc_) {
      const auto& vertex_range = vertex_indices_[boundary_condition.object_id];
      for (int j = 0; j < static_cast<int>(vertex_range.size()); ++j) {
        boundary_condition.bc(vertex_range[j], time_, &v_);
      }
    }
  }

  void ApplyPositionBoundaryConditions() {
    for (const auto& boundary_condition : q_bc_) {
      const auto& vertex_range = vertex_indices_[boundary_condition.object_id];
      for (int j = 0; j < static_cast<int>(vertex_range.size()); ++j) {
        boundary_condition.bc(vertex_range[j], time_, &q_);
      }
    }
  }

  T dt_;
  std::vector<FemElement<T>> elements_;
  FemForce<T> force_;
  BackwardEulerObjective<T> objective_;
  NewtonSolver<T> newton_solver_;
  // vertex_indices_[i] gives the vertex indices corresponding to object i.
  std::vector<std::vector<int>> vertex_indices_;
  // element_indices_[i] gives the element indices corresponding to object i.
  std::vector<std::vector<int>> element_indices_;
  Matrix3X<T> q_;
  Matrix3X<T> q_hat_;
  Matrix3X<T> v_;
  Matrix3X<T> dv_;
  VectorX<T> mass_;
  Vector3<T> gravity_;
  std::vector<BoundaryCondition> v_bc_;
  std::vector<BoundaryCondition> q_bc_;
  int num_objects_{0};
  int num_vertices_{0};
  int num_elements_{0};
  double time_{0.0};
};

}  // namespace fem
}  // namespace drake
