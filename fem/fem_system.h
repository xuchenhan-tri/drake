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

template <typename T>
class FemSystem final : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemSystem)
  explicit FemSystem(T dt) : solver_(dt), dt_(dt) {}

  void CopyPositionStateOut(const systems::Context<T>& context,
                            systems::BasicVector<T>* output) const {
    output->SetFromVector(
        context.get_discrete_state().get_vector().get_value());
  }

  void AddObjectFromVtkFile(
      const std::string& vtk, const FemConfig& config,
      const std::function<void(int, EigenPtr<Matrix3X<T>>)> position_transform,
      const std::function<void(int, EigenPtr<Matrix3X<T>>)> velocity_transform,
      std::function<void(int, const Matrix3X<T>&, EigenPtr<Matrix3X<T>>)>
          boundary_condition) {
    // We only allow one object in FEM sim so far.
    DRAKE_DEMAND(object_added_ == false);

    std::vector<Vector4<int>> vertex_indices;
    Matrix3X<T> initial_positions = parser_.Parse(vtk, &vertex_indices);
    int object_id =
        solver_.AddUndeformedObject(vertex_indices, initial_positions, config);
    solver_.SetInitialStates(object_id, position_transform, velocity_transform);
    solver_.SetBoundaryCondition(object_id, boundary_condition);
    DeclareStatePortUpdate(initial_positions);
    object_added_ = true;
  }

  void AddRectangularBlock(
      const int nx, const int ny, const int nz, const T h,
      const FemConfig& config,
      const std::function<void(int, EigenPtr<Matrix3X<T>>)> position_transform,
      const std::function<void(int, EigenPtr<Matrix3X<T>>)> velocity_transform,
      std::function<void(int, const Matrix3X<T>&, EigenPtr<Matrix3X<T>>)>
          boundary_condition) {
    // We only allow one object in FEM sim so far.
    DRAKE_DEMAND(object_added_ == false);
    // Build vertex positions.
    Matrix3X<T> initial_positions =
        MeshUtility<T>::AddRectangularBlockVertices(nx, ny, nz, h);
    // Build Mesh connectivity.
    std::vector<Vector4<int>> vertex_indices =
        MeshUtility<T>::AddRectangularBlockMesh(nx, ny, nz);
    // Create the object in the underlying solver.
    int object_id =
        solver_.AddUndeformedObject(vertex_indices, initial_positions, config);
    solver_.SetInitialStates(object_id, position_transform, velocity_transform);
    solver_.SetBoundaryCondition(object_id, boundary_condition);
    DeclareStatePortUpdate(initial_positions);
    object_added_ = true;
  }

  void AddCollisionObject(std::unique_ptr<CollisionObject<T>> object) {
      solver_.AddCollisionObject(std::move(object));
  }

  void AdvanceOneTimeStep(const systems::Context<T>& context,
                          systems::DiscreteValues<T>* next_states) const {
    const VectorX<T>& positions =
        context.get_discrete_state().get_vector().get_value();
    const Matrix3X<T>& q = Eigen::Map<const Matrix3X<T>>(positions.data(), 3,
                                                         positions.size() / 3);
    solver_.AdvanceOneTimeStep(q);
    const auto& new_q = solver_.get_q();
    const VectorX<T>& q_vec =
        Eigen::Map<const VectorX<T>>(new_q.data(), new_q.rows() * new_q.cols());
    next_states->get_mutable_vector().SetFromVector(q_vec);
  }

  T get_dt() const { return dt_; }

  const std::vector<Vector4<int>>& get_indices() const {
    return solver_.get_mesh();
  }

  int get_num_position_dofs() const { return solver_.get_num_position_dofs(); }

 private:
  void DeclareStatePortUpdate(
      const Eigen::Ref<const Matrix3X<T>>& initial_position) {
    const VectorX<T>& tmp = Eigen::Map<const VectorX<T>>(
        initial_position.data(),
        initial_position.rows() * initial_position.cols());
    systems::BasicVector<T> initial_state(tmp);
    this->DeclareDiscreteState(initial_state);
    this->DeclareVectorOutputPort(
        "vertex_positions",
        systems::BasicVector<T>(initial_position.rows() *
                                initial_position.cols()),
        &FemSystem::CopyPositionStateOut);
    this->DeclarePeriodicDiscreteUpdateEvent(dt_, 0.,
                                             &FemSystem::AdvanceOneTimeStep);
  }

  mutable FemSolver<T> solver_;
  VtkParser<T> parser_;
  T dt_{0.01};
  // This flag is turned on when an object has been added to FemSystem.
  bool object_added_{false};
};
}  // namespace fem
}  // namespace drake
