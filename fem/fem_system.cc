#include "drake/fem/fem_system.h"

namespace drake {
namespace fem {

template <typename T>
void FemSystem<T>::AddObjectFromVtkFile(
    const std::string& vtk, const MaterialConfig& config,
    std::function<void(int, EigenPtr<Matrix3X<T>>)> position_transform,
    std::function<void(int, EigenPtr<Matrix3X<T>>)> velocity_transform,
    std::function<bool(int, const Matrix3X<T>&)> boundary_condition) {
  DRAKE_DEMAND(!finalized_);
  std::vector<Vector4<int>> vertex_indices;
  Matrix3X<T> initial_positions = ParseVtk<T>(vtk, &vertex_indices);
  meshes_.emplace_back(std::make_unique<FemTetMesh<T>>(
      vertex_indices, solver_.get_num_vertices()));
  int object_id =
      solver_.AddUndeformedObject(vertex_indices, initial_positions, config);
  solver_.SetInitialStates(object_id, position_transform, velocity_transform);
  if (boundary_condition != nullptr) {
    solver_.SetBoundaryCondition(object_id, boundary_condition);
  }
}

template <typename T>
void FemSystem<T>::AddRectangularBlock(
    const int nx, const int ny, const int nz, const T h,
    const MaterialConfig& config,
    std::function<void(int, EigenPtr<Matrix3X<T>>)> position_transform,
    std::function<void(int, EigenPtr<Matrix3X<T>>)> velocity_transform,
    std::function<bool(int, const Matrix3X<T>&)> boundary_condition) {
  DRAKE_DEMAND(!finalized_);
  // Build vertex positions.
  Matrix3X<T> initial_positions =
      MeshUtility<T>::AddRectangularBlockVertices(nx, ny, nz, h);
  // Build Mesh connectivity.
  std::vector<Vector4<int>> vertex_indices =
      MeshUtility<T>::AddRectangularBlockMesh(nx, ny, nz);
  meshes_.emplace_back(std::make_unique<FemTetMesh<T>>(
      vertex_indices, solver_.get_num_vertices()));
  // Create the object in the underlying solver.
  int object_id =
      solver_.AddUndeformedObject(vertex_indices, initial_positions, config);
  solver_.SetInitialStates(object_id, position_transform, velocity_transform);
  if (boundary_condition != nullptr) {
    solver_.SetBoundaryCondition(object_id, boundary_condition);
  }
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
void FemSystem<T>::Finalize() {
  DRAKE_DEMAND(!finalized_);
  const auto& Q = solver_.get_q();
    const VectorX<T>& initial_position = Eigen::Map<const VectorX<T>>(
            Q.data(),
            Q.rows() * Q.cols());
    systems::BasicVector<T> initial_state(initial_position);
    this->DeclareDiscreteState(initial_state);
  this->DeclareVectorOutputPort(
      "vertex_positions", systems::BasicVector<T>(get_num_position_dofs()),
      &FemSystem::CopyPositionStateOut);
  this->DeclarePeriodicDiscreteUpdateEvent(dt_, 0.,
                                           &FemSystem::AdvanceOneTimeStep);
  finalized_ = true;
}
}  // namespace fem
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::FemSystem)
