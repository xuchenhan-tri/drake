#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem//fem_tetmesh_base.h"

namespace drake {
namespace fem {
template <typename T>
class FemTetMesh : public FemTetMeshBase {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemTetMesh);
  FemTetMesh(const std::vector<Vector4<int>>& tet_mesh, int vertex_offset)
      : FemTetMeshBase(tet_mesh, vertex_offset) {
    surface_vertex_normals_.resize(get_surface_vertex_count());
    surface_vertex_positions_.resize(get_surface_vertex_count());
  }

  /** Evaluates the unit outward normal at face `face_index` at barycentric
   coordinate `V_barycentric`. The unit outward normal is the normalized
   barycentric average of the vertex normals of the vertices that form the face.
   @throws std::runtime_error if the barycentric average of the vertex normals
   is degenerate. We say the normal is degenerate if it has magnitude smaller
   than `std::numeric_limits<T>::epsilon()`.*/
  Vector3<T> EvalNormal(int face_index, const Vector3<T>& V_barycentric) {
    const auto tri = get_surface_triangles()[face_index];
    Vector3<T> normal = Vector3<T>::Zero();
    for (int i = 0; i < 3; ++i) {
      normal += V_barycentric(i) * surface_vertex_normals_[tri[i]];
    }
    ThrowIfDegenerate(normal, __func__);
    normal.normalize();
    return normal;
  }

  /** Update surface vertex positions and surface vertex normals given the
     positions of all vertices. `UpdatePosition` needs to be called every time
     vertex position changes before `EvalNormal` can be called.
     @throws std::runtime_error if the vertex normal of any surface vertex is
     degenerate. The vertex normal of vertex v is the area-weighted average of
     the normals of the face that v belongs to. We say the vertex normal is
     degenerate if it has magnitude smaller than
     `std::numeric_limits<T>::epsilon()`.*/
  void UpdatePosition(const Eigen::Ref<const Matrix3X<T>>& q) {
    const auto& surface_to_volume = get_surface_to_volume_vertices();
    for (int i = 0; i < get_surface_vertex_count(); ++i) {
      const int volume_vertex =
          surface_to_volume[i] + get_volume_vertex_offset();
      DRAKE_DEMAND(volume_vertex >= 0);
      DRAKE_DEMAND(volume_vertex < q.cols());
      surface_vertex_positions_[i] = q.col(volume_vertex);
    }
    // Clear the data from previous frame.
    for (auto& n : surface_vertex_normals_) {
      n.setZero();
    }
    for (const auto& tri : get_surface_triangles()) {
      const Vector3<T>& q_WV0 = surface_vertex_positions_[tri[0]];
      const Vector3<T>& q_WV1 = surface_vertex_positions_[tri[1]];
      const Vector3<T>& q_WV2 = surface_vertex_positions_[tri[2]];
      Vector3<T> normal = (q_WV2 - q_WV0).cross(q_WV1 - q_WV0);
      for (int i = 0; i < 3; ++i) {
        surface_vertex_normals_[tri[i]] += normal;
      }
    }
    for (int i = 0; i < get_surface_vertex_count(); ++i) {
      // If the accumulated normal is zero at a vertex, the vertex normal is not
      // well-defined. We throw if the normal is close to 0.
      ThrowIfDegenerate(surface_vertex_normals_[i], __func__);
      surface_vertex_normals_[i].normalize();
    }
  }

 private:
  // Helper to throw a specific exception when a normal vector is degenerate.
  void ThrowIfDegenerate(const Vector3<T>& n, const char* source_method) const {
    if (n.norm() < std::numeric_limits<T>::epsilon()) {
      throw std::runtime_error(
          std::string(source_method) +
          "encountered a degenerate normal. The surface mesh is either too "
          "distorted or needs to be smoothed.");
    }
  }

  std::vector<Vector3<T>> surface_vertex_normals_;
  std::vector<Vector3<T>> surface_vertex_positions_;
};
}  // namespace fem
}  // namespace drake
