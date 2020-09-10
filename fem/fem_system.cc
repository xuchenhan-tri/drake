#include "drake/fem/fem_system.h"

namespace drake {
namespace fem {

template <typename T>
void FemSystem<T>::AddObjectFromVtkFile(
    const std::string& vtk, const FemConfig& config,
    std::function<void(int, EigenPtr<Matrix3X<T>>)> position_transform,
    std::function<void(int, EigenPtr<Matrix3X<T>>)> velocity_transform,
    std::function<bool(int, const Matrix3X<T>&)>
        boundary_condition) {
  // We only allow one object in FEM sim so far.
  DRAKE_DEMAND(object_added_ == false);

  std::vector<Vector4<int>> vertex_indices;
  Matrix3X<T> initial_positions = ParseVtk<T>(vtk, &vertex_indices);
  int object_id =
      solver_.AddUndeformedObject(vertex_indices, initial_positions, config);
  solver_.SetInitialStates(object_id, position_transform, velocity_transform);
  if (boundary_condition != nullptr) {
    solver_.SetBoundaryCondition(object_id, boundary_condition);
  }
  DeclareStatePortUpdate(solver_.get_q());
  object_added_ = true;
}

template <typename T>
void FemSystem<T>::AddRectangularBlock(
    const int nx, const int ny, const int nz, const T h,
    const FemConfig& config,
    std::function<void(int, EigenPtr<Matrix3X<T>>)> position_transform,
    std::function<void(int, EigenPtr<Matrix3X<T>>)> velocity_transform,
    std::function<bool(int, const Matrix3X<T>&)>
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
  if (boundary_condition != nullptr) {
    solver_.SetBoundaryCondition(object_id, boundary_condition);
  }
  DeclareStatePortUpdate(initial_positions);
  object_added_ = true;
}

template <typename T>
void FemSystem<T>::AdvanceOneTimeStep(
    const systems::Context<T>&, systems::DiscreteValues<T>* next_states) const {
  solver_.AdvanceOneTimeStep();
  const auto& new_q = solver_.get_q();
  const VectorX<T>& q_vec =
      Eigen::Map<const VectorX<T>>(new_q.data(), new_q.rows() * new_q.cols());
  next_states->get_mutable_vector().SetFromVector(q_vec);
}

template <typename T>
void FemSystem<T>::DeclareStatePortUpdate(
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
}  // namespace fem
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::FemSystem)
