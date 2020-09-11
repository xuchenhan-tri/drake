#pragma once

#include <memory>
#include <utility>
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
// TODO(xuchenhan-tri): We currently only support zero Dirichelet BC. Make it
// more general.
template <typename T>
struct BoundaryCondition {
  BoundaryCondition(int object_id_in,
                    const std::function<int(int, const Matrix3X<T>&)>& bc_in)
      : object_id(object_id_in), bc(bc_in) {}
  int object_id;
  std::function<bool(int, const Matrix3X<T>&)> bc;
};

/** A data class that holds FEM vertex states, elements and constants. */
template <typename T>
class FemData {
 public:
  explicit FemData(double dt) : dt_(dt) {}

  /**
   Add an object represented by a list of vertices connected by a simplex mesh
  to the simulation. Multiple calls to this method is allowed and the resulting
  vertices and elements will be properly indexed.
  @param[in] indices    The list of indices describing the connectivity of the
  mesh. @p indices[i] contains the indices of the 4 vertices in the i-th
  element.
  @param[in] positions  The list of positions of the vertices in the world frame
  in the reference configuration.
  @return The object_id of the newly added_object.
  */
  int AddUndeformedObject(const std::vector<Vector4<int>>& indices,
                          const Matrix3X<T>& positions,
                          const MaterialConfig& config);

  /**
      Set the initial positions and velocities of a given object.
      @param[in] object_id     The id the object whose mass is being set.
      @param[in] density     Mass density of the object.
  */
  void SetMassFromDensity(const int object_id, const T density);

  void AddHyperelasticCache(const std::vector<std::unique_ptr<HyperelasticConstitutiveModel<T>>>& model, std::vector<std::unique_ptr<HyperelasticCache<T>>>* cache)
  {
      for (int i = cache->size(); i < static_cast<int>(model.size()); ++i){
          cache->emplace_back(model[i]->CreateCache());
      }
  }

  // Setters and getters.
  // ---------- Vertex quntities ----------
  const VectorX<T>& get_mass() const { return mass_; }
  VectorX<T>& get_mutable_mass() { return mass_; }

  const Matrix3X<T>& get_Q() const { return Q_; }
  Matrix3X<T>& get_mutable_Q() { return Q_; }

  const std::vector<std::vector<int>>& get_vertex_indices() const {
    return vertex_indices_;
  }
  std::vector<std::vector<int>>& get_mutable_vertex_indices() {
    return vertex_indices_;
  }

  int get_num_vertices() const { return Q_.cols(); }
  int get_num_position_dofs() const { return Q_.size(); }

  // ---------- Element quantities ----------
  const std::vector<FemElement<T>>& get_elements() const { return elements_; }
  std::vector<FemElement<T>>& get_mutable_elements() { return elements_; }

  const std::vector<std::vector<int>>& get_element_indices() const {
    return element_indices_;
  }
  std::vector<std::vector<int>>& get_mutable_element_indices() {
    return element_indices_;
  }

  const std::vector<Vector4<int>>& get_mesh() const { return mesh_; }
  std::vector<Vector4<int>>& get_mutable_mesh() { return mesh_; }

  int get_num_elements() const { return elements_.size(); }

  // ---------- Constants and external quantities ----------
  double get_dt() const { return dt_; }
  void set_dt(double dt) { dt_ = dt; }

  const Vector3<T>& get_gravity() const { return gravity_; }
  void set_gravity(const Vector3<T>& gravity) { gravity_ = gravity; }

  const std::vector<BoundaryCondition<T>>& get_v_bc() const { return v_bc_; }
  std::vector<BoundaryCondition<T>>& get_mutable_v_bc() { return v_bc_; }

  int get_num_objects() const { return vertex_indices_.size(); }

  void add_collision_object(std::unique_ptr<CollisionObject<T>> object) {
    collision_objects_.push_back(std::move(object));
  }
  const std::vector<std::unique_ptr<CollisionObject<T>>>&
  get_collision_objects() const {
    return collision_objects_;
  }
  std::vector<std::unique_ptr<CollisionObject<T>>>&
  get_mutable_collision_objects() const {
    return collision_objects_;
  }

 private:
  double dt_;
  std::vector<FemElement<T>> elements_;
  /* There are two sets of indices that we employ for most quantities, local and
     global. Local indices start from 0 and only index quantities added in a
    single call to `AddUndeformedObject`. Global indices also start from 0 but
    index over all the quantities in the FemSolver. The local indices and the
    global indices only differ if there is more than one object added through
    `AddUndeformedObject`. The concept of local indices and global indices are
    illustrated in the following drawing. The local indices are the index
    outside the parentheses and the global indices are the ones inside the
    parentheses. There are two objects in this example and we index both the
    triangles and the vertices.

    _________________________________________________________________________
    |                                                              object 0 |
    |                    2(2)           4(4)                     7(7)       |
    |    0(0) X-----------X-----------X          5(5) X-----------X         |
    |         |          /|          /                |          /|         |
    |         |         / |         /                 |         / |         |
    |         |  0(0)  /  |  2(2)  /                  |  3(3)  /  |         |
    |         |       /   |       /                   |       /   |         |
    |         |      /    |      /                    |      /    |         |
    |         |     /     |     /                     |     /     |         |
    |         |    /      |    /                      |    /      |         |
    |         |   /       |   /                       |   /       |         |
    |         |  /        |  /                        |  /        |         |
    |         | /   1(1)  | /                         | /   4(4)  |         |
    |         |/          |/                          |/          |         |
    |    1(1) X-----------X  3(3)                6(6) X-----------X 8(8)    |
    |_______________________________________________________________________|


    _________________________________________________________________________
    |                                                              object 1 |
    |                       2(11)           3(12)                           |
    |        0(9) X-----------X-----------X                                 |
    |             \           |          /                                  |
    |              \          |         /                                   |
    |               \   0(5)  |  1(6)  /                                    |
    |                \        |       /                                     |
    |                 \       |      /                                      |
    |                  \      |     /                                       |
    |                   \     |    /                                        |
    |                    \    |   /                                         |
    |                     \   |  /                                          |
    |                      \  | /                                           |
    |                       \ |/                                            |
    |                   1(10) X                                             |
    |_______________________________________________________________________|

     The input data are indexed locally and FemData data converts them to global
     indices when the data is read in.
   */

  // vertex_indices_[i] gives the global vertex indices corresponding to object
  // i. vertex_indices_[i] is usually a vector of consecutive indices.
  std::vector<std::vector<int>> vertex_indices_;
  // element_indices_[i] gives the global element indices corresponding to
  // object i. element_indices_[i] is usually a vector of consecutive indices.
  std::vector<std::vector<int>> element_indices_;
  // mesh_[i] contains the global indices of the 4 vertices in the i-th
  // tetrahedron.
  std::vector<Vector4<int>> mesh_;
  // Reference vertex positions.
  Matrix3X<T> Q_;
  VectorX<T> mass_;
  Vector3<T> gravity_{0, 0, -9.81};
  // Velocity boundary conditions.
  std::vector<BoundaryCondition<T>> v_bc_;
  int num_objects_{0};
  int num_vertices_{0};
  int num_elements_{0};
  mutable std::vector<std::unique_ptr<CollisionObject<T>>> collision_objects_;
  double time_{0.0};
};
}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::FemData)
