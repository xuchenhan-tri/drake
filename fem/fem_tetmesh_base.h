#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {
class FemTetMeshBase {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemTetMeshBase);
  FemTetMeshBase(const std::vector<Vector4<int>>& tet_mesh, int vertex_offset)
      : volume_vertex_offset_(vertex_offset) {
    AnalyzeTets(tet_mesh);
  }

  int get_volume_vertex_count() const {
      return volume_vertex_count_;
  }

  const std::vector<int>& get_surface_to_volume_vertices() const {
      return surface_to_volume_vertices_;
  }

  const std::vector<Vector3<int>>& get_surface_triangles() const {
      return surface_triangles_;
  }

  int get_volume_vertex_offset() const {
      return volume_vertex_offset_;
  }

 private:
  /* Analyzes the tet mesh topology to do the following:
    1. Build a surface mesh from the volume mesh.
    2. Create a mapping from local surface vertex to local volume vertex.
    3. Record the expected number of vertices referenced by the tet mesh.  */
  void AnalyzeTets(const std::vector<Vector4<int>>& tet_mesh);

  /* An *implicit* map from *surface* vertex indices to volume vertex indices.
   The iᵗʰ surface vertex corresponds to the volume vertex with index
   `surface_to_volume_vertices_[i]`.  */
  std::vector<int> surface_to_volume_vertices_;

  /* A surface mesh representing the topology of the volume mesh's surface.  */
  std::vector<Vector3<int>> surface_triangles_;

  // The total number of vertices expected on the input port as implied by the
  // tetrahedra definitions.
  int volume_vertex_count_{};

  // Offset into global volume vertex indices. If iᵗʰ vertex in this volume mesh
  // has global index  `vertex_offset_ + i`.
  int volume_vertex_offset_{};
};
}  // namespace fem
}  // namespace drake
