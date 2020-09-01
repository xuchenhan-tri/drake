#pragma once

#include <memory>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/collision_object.h"
#include "drake/fem/corotated_linear_model.h"
#include "drake/fem/fem_config.h"
#include "drake/fem/fem_element.h"
#include "drake/fem/hyperelastic_constitutive_model.h"

namespace drake {
namespace fem {
    // TODO(xuchenhan-tri): We currently only support zero Dirichelet BC. Make it more general.
template <typename T>
struct BoundaryCondition {
  BoundaryCondition(int object_id_in,
                    const std::function<void(int, const Matrix3X<T>&,
                                             EigenPtr<Matrix3X<T>>)>& bc_in)
      : object_id(object_id_in), bc(bc_in) {}
  int object_id;
  std::function<void(int, const Matrix3X<T>&, EigenPtr<Matrix3X<T>>)> bc;
};

/** A data class that holds FEM vertex states, elements and constants. */
template <typename T>
class FemData {
 public:
  FemData(double dt) : dt_(dt) {}

  int AddUndeformedObject(const std::vector<Vector4<int>>& indices,
                          const Matrix3X<T>& positions,
                          const FemConfig& config);

  /**
      Set the initial positions and velocities of a given object.
      @param[in] object_id     The id the object whose mass is being set.
      @param[in] density     Mass density of the object.
  */
  void SetMassFromDensity(const int object_id, const T density);

  // Setters and getters.

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

  double get_dt() const { return dt_; }
  void set_dt(double dt) { dt_ = dt; }

  double get_time() const { return time_; }
  void set_time(double time) { time_ = time; }

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
  double dt_;
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
  Vector3<T> gravity_{0, 0, -9.81};
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
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::drake::fem::FemData)
