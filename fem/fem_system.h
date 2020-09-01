#pragma once

#include <functional>
#include <string>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/fem_solver.h"
#include "drake/fem/mesh_utility.h"
#include "drake/fem/vtk_parser.h"
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
    output->SetFromVector(
        context.get_discrete_state().get_vector().get_value());
  }

  void AddObjectFromVtkFile(
      const std::string& vtk, const FemConfig& config,
      std::function<void(int, EigenPtr<Matrix3X<T>>)> position_transform =
          nullptr,
      std::function<void(int, EigenPtr<Matrix3X<T>>)> velocity_transform =
          nullptr,
      std::function<void(int, const Matrix3X<T>&, EigenPtr<Matrix3X<T>>)>
          boundary_condition = nullptr);

  void AddRectangularBlock(
      const int nx, const int ny, const int nz, const T h,
      const FemConfig& config,
      std::function<void(int, EigenPtr<Matrix3X<T>>)> position_transform =
          nullptr,
      std::function<void(int, EigenPtr<Matrix3X<T>>)> velocity_transform =
          nullptr,
      std::function<void(int, const Matrix3X<T>&, EigenPtr<Matrix3X<T>>)>
          boundary_condition = nullptr);

  void AddCollisionObject(std::unique_ptr<CollisionObject<T>> object) {
    solver_.AddCollisionObject(std::move(object));
  }

  void AdvanceOneTimeStep(const systems::Context<T>&,
                          systems::DiscreteValues<T>* next_states) const;

  T get_dt() const { return dt_; }

  const std::vector<Vector4<int>>& get_indices() const {
    return solver_.get_mesh();
  }

  int get_num_position_dofs() const { return solver_.get_num_position_dofs(); }

 private:
  void DeclareStatePortUpdate(
      const Eigen::Ref<const Matrix3X<T>>& initial_position);

  mutable FemSolver<T> solver_;
  VtkParser<T> parser_;
  double dt_{0.01};
  // This flag is turned on when an object has been added to FemSystem.
  bool object_added_{false};
};
}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::drake::fem::FemSystem)
