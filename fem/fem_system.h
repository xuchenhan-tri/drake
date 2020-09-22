#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/fem_solver.h"
#include "drake/fem/mesh_utility.h"
#include "drake/fem/parse_vtk.h"
#include "drake/fem/fem_tetmesh.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace fem {
/** A wrapper class around FemSolver that provides the minimal set of APIs for
 * the pancake demo. */
template <typename T>
class FemSystem final : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemSystem)
  explicit FemSystem(double dt) : solver_(dt), dt_(dt) {}

  void CopyPositionStateOut(const systems::Context<T>& context,
                            systems::BasicVector<T>* output) const {
      output->SetFromVector(context.get_discrete_state().get_vector().get_value());
  }

  void AddObjectFromVtkFile(
      const std::string& vtk, const MaterialConfig& config,
      std::function<void(int, EigenPtr<Matrix3X<T>>)> position_transform =
          nullptr,
      std::function<void(int, EigenPtr<Matrix3X<T>>)> velocity_transform =
          nullptr,
      std::function<bool(int, const Matrix3X<T>&)>
          boundary_condition = nullptr);

  void AddRectangularBlock(
      const int nx, const int ny, const int nz, const T h,
      const MaterialConfig& config,
      std::function<void(int, EigenPtr<Matrix3X<T>>)> position_transform =
          nullptr,
      std::function<void(int, EigenPtr<Matrix3X<T>>)> velocity_transform =
          nullptr,
      std::function<bool(int, const Matrix3X<T>&)>
          boundary_condition = nullptr);

  void AddCollisionObject(std::unique_ptr<CollisionObject<T>> object) {
    solver_.AddCollisionObject(std::move(object));
  }

  void AdvanceOneTimeStep(const systems::Context<T>&,
                          systems::DiscreteValues<T>* next_states) const;

  const std::vector<Vector4<int>>& get_indices() const {
    return solver_.get_mesh();
  }

  int get_num_position_dofs() const { return solver_.get_num_position_dofs(); }

  /** `Finalize` must be called after all deformable objects have been added. It declares output ports for this system and set the discrete update function. */
  void Finalize();

  bool is_finalized() const { return finalized_; }

  double get_dt() const { return dt_; }

  const std::vector<std::unique_ptr<FemTetMeshBase>>& get_meshes() const {return meshes_;}

 private:
  mutable FemSolver<T> solver_;
  double dt_{};
  bool finalized_{false};
  std::vector<std::unique_ptr<FemTetMeshBase>> meshes_;
};
}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::FemSystem)
