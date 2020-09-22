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
  }

  Vector3<T> EvalNormal(int face_index, const Vector3<T>& V_barycentric);

  void CalcVertexNormal(const Eigen::Ref<const Matrix3X<T>>& vertex_positions){
      DRAKE_DEMAND(vertex_positions.cols() = this->get_volume_vertex_count());
  }

 private:
  std::vector<Vector3<T>> vertex_normals_;
};
}  // namespace fem
}  // namespace drake
