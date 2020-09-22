#include "drake/fem/fem_tetmesh_base.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>

#include "drake/geometry/proximity/sorted_triplet.h"

namespace drake {
namespace fem {
using geometry::internal::SortedTriplet;
using std::array;
using std::map;
using std::set;
using std::vector;
void FemTetMeshBase::AnalyzeTets(const vector<Vector4<int>>& tet_mesh) {
  /* Extract all the border faces. Those are the triangles that are only
   referenced by a single face. So, for every tet, we examine its four faces
   and determine if any other tet shares it. Any face that is only referenced
   once is a border face.

   Each face has a unique key: a SortedTriplet (so the ordering of the
   vertices won't matter). The first time we see a face, we add it to a map.
   The second time we see the face, we remove it. When we're done, the keys in
   the map will be those faces only referenced once.

   The values in the map represent the triangle, with the vertices ordered so
   that they point *out* of the tetrahedron. Therefore, they will also point
   outside of the mesh.

   A typical tetrahedral element looks like:
       p2 *
          |
          |
       p3 *---* p0
         /
        /
    p1 *
   The index order for a particular tetrahedron has the order [p0, p1, p2,
   p3]. These local indices enumerate each of the tet faces with
   outward-pointing normals with respect to the right-hand rule.  */
  const array<array<int, 3>, 4> local_indices{
      {{{1, 0, 2}}, {{3, 0, 1}}, {{3, 1, 2}}, {{2, 0, 3}}}};

  // While visiting all of the referenced vertices, identify the largest
  // index.
  int largest_index = -1;
  map<SortedTriplet<int>, array<int, 3>> border_faces;
  for (const Vector4<int>& tet : tet_mesh) {
    for (const array<int, 3>& tet_face : local_indices) {
      const array<int, 3> face{tet(tet_face[0]), tet(tet_face[1]),
                               tet(tet_face[2])};
      largest_index = std::max({largest_index, face[0], face[1], face[2]});
      const SortedTriplet face_key(face[0], face[1], face[2]);
      if (auto itr = border_faces.find(face_key); itr != border_faces.end()) {
        border_faces.erase(itr);
      } else {
        border_faces[face_key] = face;
      }
    }
  }
  /* We're assuming that all of the mesh vertices are used in the topology.
   So, the expected number of vertices is the largest index + 1. This is the
   third documented responsibility of this function.  */
  volume_vertex_count_ = largest_index + 1;

  /* Using a set because the vertices will be nicely ordered. Ideally, we'll
   be extracting a subset of the vertex positions from the input port. We
   optimize cache coherency if we march in a monotonically increasing pattern.
   So, we'll map triangle vertex indices to volume vertex indices in a
   strictly monotonically increasing relationship.  */
  set<int> unique_vertices;
  for (const auto& [face_key, face] : border_faces) {
    unused(face_key);
    for (int i = 0; i < 3; ++i) unique_vertices.insert(face[i]);
  }
  /* This is the *second* documented responsibility of this function: populate
   the mapping from surface to volume so that we can efficiently extract the
   *surface* vertex positions from the *volume* vertex input.  */
  surface_to_volume_vertices_.clear();  // just to be safe.
  surface_to_volume_vertices_.insert(surface_to_volume_vertices_.begin(),
                                     unique_vertices.begin(),
                                     unique_vertices.end());

  /* The border faces all include indices into the volume vertices. To turn
   them into surface triangles, they need to include indices into the surface
   vertices. Create the volume index --> surface map to facilitate the
   transformation.  */
  const int surface_vertex_count =
      static_cast<int>(surface_to_volume_vertices_.size());
  map<int, int> volume_to_surface;
  for (int i = 0; i < surface_vertex_count; ++i) {
    volume_to_surface[surface_to_volume_vertices_[i]] = i;
  }

  /* This is the *first* documented responsibility: create the topology of the
   surface triangle mesh. Each triangle consists of three indices into the
   set of *surface* vertex positions.  */
  surface_triangles_.clear();
  surface_triangles_.reserve(border_faces.size());
  for (auto& [face_key, face] : border_faces) {
    unused(face_key);
    surface_triangles_.emplace_back(volume_to_surface[face[0]],
                                    volume_to_surface[face[1]],
                                    volume_to_surface[face[2]]);
  }
}

}  // namespace fem
}  // namespace drake
