#pragma once

#include "drake/fem/vtk_parser.h"
#include "drake/fem/fem_solver.h"

#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace fem {
template <typename T>
class FemSystem final : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemSystem)
  FemSystem(T dt) : solver_(dt), parser_(this), dt_(dt) {}

  void CopyPositionStateOut(const systems::Context<T>& context,
                            systems::BasicVector<T>* output) const {
    output->SetFromVector(
        context.get_discrete_state().get_vector().get_value());
  }

  void AddObjectFromVtkFile(const std::string& vtk) {
    // We only allow one object in FEM sim so far.
    DRAKE_DEMAND(object_added_ == false);
    parser_.Parse(vtk);
    DeclareStatePortUpdate();
    object_added_ = true;
  }

  void AddRectangularBlock(const int I, const int J, const int K, const T h) {
      // We only allow one object in FEM sim so far.
      DRAKE_DEMAND(object_added_ == false);
      const int n_points = (I + 1) * (J + 1) * (K + 1);
      Q_.resize(3, n_points);
      int position_count = 0;
      Vector3<T> position;
      for (int x = 0; x <= I; ++x) {
          position(0) = h * x;
          for (int y = 0; y <= J; ++y) {
              position(1) = h * y;
              for (int z = 0; z <= K; ++z) {
                  position(2) = h * z;
                  Q_.col(position_count++) = position;
              }
          }
      }

      for (int i = 0; i < I; ++i) {
          for (int j = 0; j < J; ++j) {
              for (int k = 0; k < K; ++k) {
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

                  const int p0 = (i * (J + 1) + j) * (K + 1) + k;
                  const int p1 = p0 + 1;
                  const int p3 = ((i + 1) * (J + 1) + j) * (K + 1) + k;
                  const int p2 = p3 + 1;
                  const int p7 = ((i + 1) * (J + 1) + (j + 1)) * (K + 1) + k;
                  const int p6 = p7 + 1;
                  const int p4 = (i * (J + 1) + (j + 1)) * (K + 1) + k;
                  const int p5 = p4 + 1;

                  // Ensure that neighboring tetras are sharing faces, and within a
                  // single tetrahedron, if the indices are ordered like [a,b,c,d], then
                  // the normal given by right hand rule applied to the face [a,b,c]
                  // points to the node d.
                  if ((i + j + k) % 2 == 1) {
                      vertex_indices_.emplace_back(p2, p1, p6, p3);
                      vertex_indices_.emplace_back(p6, p3, p4, p7);
                      vertex_indices_.emplace_back(p4, p1, p6, p5);
                      vertex_indices_.emplace_back(p3, p1, p4, p0);
                      vertex_indices_.emplace_back(p6, p1, p4, p3);
                  } else {
                      vertex_indices_.emplace_back(p0, p2, p5, p1);
                      vertex_indices_.emplace_back(p7, p2, p0, p3);
                      vertex_indices_.emplace_back(p5, p2, p7, p6);
                      vertex_indices_.emplace_back(p7, p0, p5, p4);
                      vertex_indices_.emplace_back(p0, p2, p7, p5);
                  }
              }
          }
      }
      int object_id = solver_.AddUndeformedObject(vertex_indices_, Q_, 1000.0);
      auto initial_position = [](int vertex_index, EigenPtr<Matrix3X<T>> pos) {
          pos->col(vertex_index) -= Vector3<T>(0.05, 0.05, 0);
      };
      auto initial_velocity = [](int vertex_index, EigenPtr<Matrix3X<T>> vel) {
          vel->col(vertex_index).setZero();
      };
    solver_.SetInitialStates(object_id, initial_position, initial_velocity);
    DeclareStatePortUpdate();
    object_added_ = true;
  }

  void AdvanceOneTimeStep(const systems::Context<T>& context,
                          systems::DiscreteValues<T>* next_states) const {
      const VectorX<T>& positions = context.get_discrete_state().get_vector().get_value();
      const Matrix3X<T>& q = Eigen::Map<const Matrix3X<T>>(positions.data(), 3, positions.size()/3);
      solver_.set_q(q);
      solver_.AdvanceOneTimeStep();
    const auto& new_q = solver_.get_q();
    const VectorX<T>& q_vec =
        Eigen::Map<const VectorX<T>>(new_q.data(), new_q.rows() * new_q.cols());
    next_states->get_mutable_vector().SetFromVector(q_vec);
  }

  void set_dt(T dt) { dt_ = dt; }
  T get_dt() const { return dt_; }
  bool is_object_added() { return object_added_; }

  const Matrix3X<T>& get_Q() const { return Q_; }
  Matrix3X<T>& get_mutable_Q() { return Q_; }
  const std::vector<Vector4<int>>& get_indices() const {
    return vertex_indices_;
  }
  std::vector<Vector4<int>>& get_mutable_indices() { return vertex_indices_; }

 private:
  void DeclareStatePortUpdate() {
    VectorX<T> tmp = Eigen::Map<VectorX<T>>(Q_.data(), Q_.rows() * Q_.cols());
    systems::BasicVector<T> initial_state(tmp);
    this->DeclareDiscreteState(initial_state);
    this->DeclareVectorOutputPort(
        "vertex_positions", systems::BasicVector<T>(Q_.rows() * Q_.cols()),
        &FemSystem::CopyPositionStateOut);
    this->DeclarePeriodicDiscreteUpdateEvent(dt_, 0.,
                                             &FemSystem::AdvanceOneTimeStep);
  }

  mutable FemSolver<T> solver_;
  VtkParser<T> parser_;
  // vertex_indices[i][j] gives the index of the j-th node in the i-th element.
  std::vector<Vector4<int>> vertex_indices_;
  // Initial position of the vertices.
  Matrix3X<T> Q_;
  T dt_{0.1};
  // This flag is turned on when an object has been added to FemSystem.
  bool object_added_{false};
};
}  // namespace fem
}  // namespace drake
