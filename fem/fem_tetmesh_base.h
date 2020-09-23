#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {
/** FemTetMeshBase handles the non-templatized part of FemTetMesh. */
class FemTetMeshBase {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemTetMeshBase);
    /** Constructs a FemTetMeshBase. The mesh topology
     is given by a collection of tets (each represented by 4 indices into an
     *implied* ordered set of vertex positions).

     @param tet_mesh       The definition of the tetrahedral mesh topology (using
                           indices starting from 0).
     @param vertex_offset  The offset of local volume vertex indices into global
                           indices. In other words, for a given vertex in the tet
                           mesh, its global index is equal to its local index plus
                           `vertex_offset`. See fem_data.h for definition of local
                           vs. global indices. */
  FemTetMeshBase(const std::vector<Vector4<int>>& tet_mesh, int vertex_offset)
      : tet_mesh_(tet_mesh), volume_vertex_offset_(vertex_offset) {
    AnalyzeTets(tet_mesh);
  }

  /// @name                   Getter methods
  //@{
  int get_volume_vertex_count() const { return volume_vertex_count_; }
  int get_surface_vertex_count() const {
    return surface_to_volume_vertices_.size();
  }
  const std::vector<int>& get_surface_to_volume_vertices() const {
    return surface_to_volume_vertices_;
  }
  const std::vector<Vector3<int>>& get_surface_triangles() const {
    return surface_triangles_;
  }
  const std::vector<Vector4<int>>& get_tet_mesh() const { return tet_mesh_; }
  int get_volume_vertex_offset() const { return volume_vertex_offset_; }
  //@}
 private:
  /* Analyzes the tet mesh topology to do the following:
    1. Build a surface mesh from the volume mesh.
    2. Create a mapping from local surface vertex to local volume vertex.
    3. Record the expected number of vertices referenced by the tet mesh.  */
  void AnalyzeTets(const std::vector<Vector4<int>>& tet_mesh);

  // Local indices of the tetmesh.
  std::vector<Vector4<int>> tet_mesh_;

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
