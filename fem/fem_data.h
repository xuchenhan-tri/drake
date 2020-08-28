#pragma once

#include <vector>
#include <memory>

#include "drake/common/eigen_types.h"
#include "drake/fem/collision_object.h"
#include "drake/fem/constitutive_model.h"
#include "drake/fem/corotated_linear_model.h"
#include "drake/fem/fem_config.h"
#include "drake/fem/fem_element.h"

namespace drake {
namespace fem {
template <typename T>
struct BoundaryCondition {
  BoundaryCondition(int object_id_in,
                    const std::function<void(int, const Matrix3X<T>&,
                                             EigenPtr<Matrix3X<T>>)>& bc_in)
      : object_id(object_id_in), bc(bc_in) {}
  int object_id;
  std::function<void(int, const Matrix3X<T>&, EigenPtr<Matrix3X<T>>)> bc;
};

template <typename T>
class FemData {
 public:
  FemData(T dt) : dt_(dt) {}

  int AddUndeformedObject(const std::vector<Vector4<int>>& indices,
                          const Matrix3X<T>& positions,
                          const FemConfig& config) {
    // Add new elements and record the element indices for this object.
    std::vector<int> local_element_indices(indices.size());
    Vector4<int> particle_offset{num_vertices_, num_vertices_, num_vertices_,
                                 num_vertices_};
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      Matrix3X<T> local_positions(3, 4);
      for (int j = 0; j < 4; ++j) {
        local_positions.col(j) = positions.col(indices[i][j]);
      }
      mesh_.push_back(indices[i] + particle_offset);
      // TODO(xuchenhan-tri): Support customized constitutive models.
      elements_.emplace_back(
          indices[i] + particle_offset, positions,
          std::make_unique<CorotatedLinearElasticity<T>>(
              config.youngs_modulus, config.poisson_ratio, config.mass_damping,
              config.stiffness_damping, local_positions),
          config.density);
      local_element_indices[i] = num_elements_++;
    }
    element_indices_.push_back(local_element_indices);

    // Record the vertex indices for this object.
    std::vector<int> local_vertex_indices(positions.cols());
    for (int i = 0; i < positions.cols(); ++i)
      local_vertex_indices[i] = num_vertices_++;
    vertex_indices_.push_back(local_vertex_indices);

    // Allocate for positions and velocities.
    Q_.conservativeResize(3, q_.cols() + positions.cols());
    Q_.rightCols(positions.cols()) = positions;
    q_.conservativeResize(3, q_.cols() + positions.cols());
    q_.rightCols(positions.cols()) = positions;
    v_.conservativeResize(3, v_.cols() + positions.cols());
    v_.rightCols(positions.cols()).setZero();
    dv_.resize(3, v_.cols());

    // Set mass.
    mass_.conservativeResize(mass_.size() + positions.cols());
    const int object_id = num_objects_;
    SetMassFromDensity(object_id, config.density);
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
      mass_[vertex_range[i]] = 0;
    }
    // Add the mass contribution of each element.
    const auto& element_range = element_indices_[object_id];
    for (int i = 0; i < static_cast<int>(element_range.size()); ++i) {
      const auto& element = elements_[i];
      const Vector4<int>& local_indices = element.get_indices();
      const T fraction = 1.0 / static_cast<T>(local_indices.size());
      for (int j = 0; j < static_cast<int>(local_indices.size()); ++j) {
        mass_[local_indices[j]] +=
            density * element.get_element_measure() * fraction;
      }
    }
  }

  const VectorX<T>& get_mass() const { return mass_; }
  VectorX<T>& get_mutable_mass() { return mass_; }

  const Matrix3X<T>& get_dv() const { return dv_; }
  Matrix3X<T>& get_mutable_dv() { return dv_; }

  const Matrix3X<T>& get_v() const { return v_; }
  Matrix3X<T>& get_mutable_v() { return v_; }

  const Matrix3X<T>& get_q() const { return q_; }
  Matrix3X<T>& get_mutable_q() { return q_; }

  const Matrix3X<T>& get_Q() const { return Q_; }
  Matrix3X<T>& get_mutable_Q() { return Q_; }

  const Matrix3X<T>& get_q_hat() const { return q_hat_; }
  Matrix3X<T>& get_mutable_q_hat() { return q_hat_; }

  void set_q(const Matrix3X<T>& q) { q_ = q; }
  void set_Q(const Matrix3X<T>& Q) { Q_ = Q; }

  const std::vector<FemElement<T>>& get_elements() const { return elements_; }
  std::vector<FemElement<T>>& get_mutable_elements() { return elements_; }

  T get_dt() const { return dt_; }
  void set_dt(T dt) { dt_ = dt; }

  T get_time() const { return time_; }
  void set_time(T time) { time_ = time; }

  const Vector3<T>& get_gravity() const { return gravity_; }

  void set_gravity(Vector3<T>& gravity) { gravity_ = gravity; }

  const std::vector<BoundaryCondition<T>>& get_v_bc() const { return v_bc_; }
  std::vector<BoundaryCondition<T>>& get_mutable_v_bc() { return v_bc_; }

  const std::vector<std::vector<int>>& get_vertex_indices() const {
    return vertex_indices_;
  }
  std::vector<std::vector<int>>& get_mutable_vertex_indices() {
    return vertex_indices_;
  }
  const std::vector<std::vector<int>>& get_element_indices() const {
    return element_indices_;
  }
  std::vector<std::vector<int>>& get_mutable_element_indices() {
    return element_indices_;
  }

  const std::vector<Vector4<int>>& get_mesh() const { return mesh_; }
  std::vector<Vector4<int>>& get_mutable_mesh() { return mesh_; }

  int get_num_objects() const { return vertex_indices_.size(); }
  int get_num_vertices() const { return q_.cols(); }
  int get_num_elements() const { return elements_.size(); }
  int get_num_position_dofs() const { return q_.size(); }

  void add_collision_object(std::unique_ptr<CollisionObject<T>> object) {
    collision_objects_.push_back(std::move(object));
  }
  const std::vector<std::unique_ptr<CollisionObject<T>>>&
  get_collision_objects() const {
    return collision_objects_;
  }
  std::vector<std::unique_ptr<CollisionObject<T>>>&
  get_mutable_collision_objects() {
    return collision_objects_;
  }

 private:
  T dt_;
  std::vector<FemElement<T>> elements_;
  // vertex_indices_[i] gives the global vertex indices corresponding to object
  // i.
  std::vector<std::vector<int>> vertex_indices_;
  // element_indices_[i] gives the global element indices corresponding to
  // object i.
  std::vector<std::vector<int>> element_indices_;
  // mesh_[i] contains the global indices of the 4 vertices in the i-th
  // tetrahedron.
  std::vector<Vector4<int>> mesh_;
  // Initial position.
  Matrix3X<T> Q_;
  // Time n position.
  Matrix3X<T> q_;
  // Time n position + dt * time n velocity.
  Matrix3X<T> q_hat_;
  // Time n velocity.
  Matrix3X<T> v_;
  // Time n+1 velocity - Time n velocity.
  Matrix3X<T> dv_;
  VectorX<T> mass_;
//  Vector3<T> gravity_{0, 0, -9.81};
    Vector3<T> gravity_{0, -9.81, 0};
  // Velocity boundary conditions.
  std::vector<BoundaryCondition<T>> v_bc_;
  int num_objects_{0};
  int num_vertices_{0};
  int num_elements_{0};
  std::vector<std::unique_ptr<CollisionObject<T>>> collision_objects_;
  double time_{0.0};
};

}  // namespace fem
}  // namespace drake
