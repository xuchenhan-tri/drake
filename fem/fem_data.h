#pragma once

#include "drake/common/eigen_types.h"
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
  Vector3<T> gravity_{0, 0, -9.81};
  // Velocity boundary conditions.
  std::vector<BoundaryCondition<T>> v_bc_;
  int num_objects_{0};
  int num_vertices_{0};
  int num_elements_{0};
  double time_{0.0};
};

}  // namespace fem
}  // namespace drake
