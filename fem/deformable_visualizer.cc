#include "drake/fem/deformable_visualizer.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <utility>

#include "drake/geometry/proximity/sorted_triplet.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/lcmt_deformable_tri.hpp"
#include "drake/lcmt_deformable_tri_mesh_init.hpp"
#include "drake/lcmt_deformable_tri_mesh_update.hpp"

namespace drake {
namespace fem {

using geometry::internal::SortedTriplet;
using std::array;
using std::map;
using std::set;
using std::vector;
using systems::Context;
using systems::EventStatus;

DeformableVisualizer::DeformableVisualizer(double update_period,
                                           std::string mesh_name,
                                           const vector<Vector4<int>>& tet_mesh,
                                           lcm::DrakeLcmInterface* lcm)
    : lcm_(lcm), mesh_name_(std::move(mesh_name)) {
  if (lcm == nullptr) {
    owned_lcm_ = std::make_unique<lcm::DrakeLcm>();
    lcm_ = owned_lcm_.get();
  }

  AnalyzeTets(tet_mesh);

  this->DeclareInputPort("vertex_positions", systems::kVectorValued,
                         volume_vertex_count_ * 3);
  this->DeclareInitializationPublishEvent(
      &DeformableVisualizer::PublishMeshInit);
  this->DeclarePeriodicPublishEvent(update_period, 0.0,
                                    &DeformableVisualizer::PublishMeshUpdate);
}

void DeformableVisualizer::AnalyzeTets(const vector<Vector4<int>>& tet_mesh) {
  /* Extract all the border faces. Those are the triangles that are only
   referenced by a single face. So, for every tet, we examine its four faces and
   determine if any other tet shares it. Any face that is only referenced once
   is a border face.

   Each face has a unique key: a SortedTriplet (so the ordering of the vertices
   won't matter). The first time we see a face, we add it to a map. The second
   time we see the face, we remove it. When we're done, the keys in the map
   will be those faces only referenced once.

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
   The index order for a particular tetrahedron has the order [p0, p1, p2, p3].
   These local indices enumerate each of the tet faces with outward-pointing
   normals with respect to the right-hand rule.  */
  const array<array<int, 3>, 4> local_indices{
      {{{1, 0, 2}}, {{3, 0, 1}}, {{3, 1, 2}}, {{2, 0, 3}}}};

  // While visiting all of the referenced vertices, identify the largest index.
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
  /* We're assuming that all of the mesh vertices are used in the topology. So,
   the expected number of vertices is the largest index + 1. This is the third
   documented responsibility of this function.  */
  volume_vertex_count_ = largest_index + 1;

  /* Using a set because the vertices will be nicely ordered. Ideally, we'll be
   extracting a subset of the vertex positions from the input port. We optimize
   cache coherency if we march in a monotonically increasing pattern. So, we'll
   map triangle vertex indices to volume vertex indices in a strictly
   monotonically increasing relationship.  */
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

void DeformableVisualizer::SendMeshInit() const {
  lcmt_deformable_tri_mesh_init message;

  message.mesh_name = mesh_name_;
  message.num_vertices = static_cast<int>(surface_to_volume_vertices_.size());
  message.num_tris = static_cast<int>(surface_triangles_.size());
  message.tris.resize(message.num_tris);
  for (int t = 0; t < message.num_tris; ++t) {
    message.tris[t].vertices[0] = surface_triangles_[t](0);
    message.tris[t].vertices[1] = surface_triangles_[t](1);
    message.tris[t].vertices[2] = surface_triangles_[t](2);
  }

  lcm::Publish(lcm_, "DEFORMABLE_MESH_INIT", message);
}

EventStatus DeformableVisualizer::PublishMeshInit(
    const Context<double>&) const {
  SendMeshInit();
  return EventStatus::Succeeded();
}

void DeformableVisualizer::PublishMeshUpdate(
    const Context<double>& context) const {
  lcmt_deformable_tri_mesh_update message;
  message.timestamp =
      static_cast<int64_t>(ExtractDoubleOrThrow(context.get_time()) * 1e6);
  message.mesh_name = mesh_name_;
  const int v_count = static_cast<int>(surface_to_volume_vertices_.size());
  message.data_size = v_count * 3;
  message.data.resize(message.data_size);

  // The volume vertex positions are one flat vector. Such that the position of
  // the ith volume vertex is in entries j, j + 1, and j + 2, j = 3i.
  const auto& vertex_state = vertex_positions_input_port()
                                 .Eval<systems::BasicVector<double>>(context)
                                 .get_value();
  for (int v = 0; v < v_count; ++v) {
    const int out_index = v * 3;
    const int state_index = surface_to_volume_vertices_[v] * 3;
    message.data[out_index] = vertex_state(state_index);
    message.data[out_index + 1] = vertex_state(state_index + 1);
    message.data[out_index + 2] = vertex_state(state_index + 2);
  }
  lcm::Publish(lcm_, "DEFORMABLE_MESH_UPDATE", message);
}

}  // namespace fem
}  // namespace drake
