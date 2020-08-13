#pragma once

#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/fem_solver.h"
#include "drake/fem/vtk_parser.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace fem {

template <typename T>
class FemSystem final : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemSystem)
  FemSystem(T dt) : solver_(dt), dt_(dt) {}

  void CopyPositionStateOut(const systems::Context<T>& context,
                            systems::BasicVector<T>* output) const {
    output->SetFromVector(
        context.get_discrete_state().get_vector().get_value());
  }

  void AddObjectFromVtkFile(
      const std::string& vtk, const FemConfig& config,
      const std::function<void(int, EigenPtr<Matrix3X<T>>)> position_transform,
      const std::function<void(int, EigenPtr<Matrix3X<T>>)>
          velocity_transform,
      std::function<void(int, const Matrix3X<T>&, EigenPtr<Matrix3X<T>>)> boundary_condition) {
    // We only allow one object in FEM sim so far.
    DRAKE_DEMAND(object_added_ == false);

    std::vector<Vector4<int>> vertex_indices;
    Matrix3X<T> initial_positions;
    parser_.Parse(vtk);
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
      const std::function<void(int, EigenPtr<Matrix3X<T>>)>
          velocity_transform,
      std::function<void(int, const Matrix3X<T>&, EigenPtr<Matrix3X<T>>)> boundary_condition) {
    // We only allow one object in FEM sim so far.
    DRAKE_DEMAND(object_added_ == false);
    // Build vertex positions.
    Matrix3X<T> initial_positions = AddRectangularBlockVertices(nx, ny, nz, h);
    // Build Mesh connectivity.
    std::vector<Vector4<int>> vertex_indices =
        AddRectangularBlockMesh(nx, ny, nz);
    // Create the object in the underlying solver.
    int object_id =
        solver_.AddUndeformedObject(vertex_indices, initial_positions, config);
    solver_.SetInitialStates(object_id, position_transform, velocity_transform);
    solver_.SetBoundaryCondition(object_id, boundary_condition);
    DeclareStatePortUpdate(initial_positions);
    object_added_ = true;
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

  int get_num_position_dofs() const {
      return solver_.get_num_position_dofs();
  }

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

  Matrix3X<T> AddRectangularBlockVertices(const int nx, const int ny,
                                          const int nz, T h) const {
    const int n_points = (nx + 1) * (ny + 1) * (nz + 1);
    Matrix3X<T> vertices(3, n_points);
    int position_count = 0;
    Vector3<T> position;
    for (int x = 0; x <= nx; ++x) {
      position(0) = h * x;
      for (int y = 0; y <= ny; ++y) {
        position(1) = h * y;
        for (int z = 0; z <= nz; ++z) {
          position(2) = h * z;
          vertices.col(position_count++) = position;
        }
      }
    }
    return vertices;
  }

  std::vector<Vector4<int>> AddRectangularBlockMesh(const int nx, const int ny,
                                                    const int nz) const {
    std::vector<Vector4<int>> vertex_indices;
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
          // For each block, the 8 corners are numerated as:
          //     4*-----*7
          //     /|    /|
          //    / |   / |
          //  5*-----*6 |
          //   | 0*--|--*3
          //   | /   | /
          //   |/    |/
          //  1*-----*2

          //    j ^
          //      |
          //      |
          //      *-----> i
          //     /
          //    /
          //   k

          const int p0 = (i * (ny + 1) + j) * (nz + 1) + k;
          const int p1 = p0 + 1;
          const int p3 = ((i + 1) * (ny + 1) + j) * (nz + 1) + k;
          const int p2 = p3 + 1;
          const int p7 = ((i + 1) * (ny + 1) + (j + 1)) * (nz + 1) + k;
          const int p6 = p7 + 1;
          const int p4 = (i * (nx + 1) + (j + 1)) * (nz + 1) + k;
          const int p5 = p4 + 1;

          // Ensure that neighboring tetras are sharing faces, and within a
          // single tetrahedron, if the indices are ordered like [a,b,c,d], then
          // the normal given by right hand rule applied to the face [a,b,c]
          // points to the node d.
          if ((i + j + k) % 2 == 1) {
            vertex_indices.emplace_back(p2, p1, p6, p3);
            vertex_indices.emplace_back(p6, p3, p4, p7);
            vertex_indices.emplace_back(p4, p1, p6, p5);
            vertex_indices.emplace_back(p3, p1, p4, p0);
            vertex_indices.emplace_back(p6, p1, p4, p3);
          } else {
            vertex_indices.emplace_back(p0, p2, p5, p1);
            vertex_indices.emplace_back(p7, p2, p0, p3);
            vertex_indices.emplace_back(p5, p2, p7, p6);
            vertex_indices.emplace_back(p7, p0, p5, p4);
            vertex_indices.emplace_back(p0, p2, p7, p5);
          }
        }
      }
    }
    return vertex_indices;
  }

  mutable FemSolver<T> solver_;
  VtkParser<T> parser_;
  T dt_{0.01};
  // This flag is turned on when an object has been added to FemSystem.
  bool object_added_{false};
};
}  // namespace fem
}  // namespace drake
