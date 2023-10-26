#include "drake/geometry/mesh_deformation_interpolator.h"

#include <array>
#include <map>
#include <set>

#include "drake/geometry/proximity/sorted_triplet.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Vector4i;
using Eigen::VectorXd;
using std::array;

BarycentricInterpolator::BarycentricInterpolator(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& positions,
    const VolumeMesh<double>& control_mesh)
    : num_control_vertices_(positions.rows()) {
  const double kTol = 1e-6;
  for (int v = 0; v < num_control_vertices_; ++v) {
    const Vector3d p_WV = positions.row(v);
    bool matched = false;
    for (int e = 0; e < control_mesh.num_elements(); ++e) {
      const Vector4d bary = control_mesh.CalcBarycentric(p_WV, e);
      if ((bary.array() >= -kTol).all()) {
        barycentric_coordinates_.push_back(bary);
        vertex_indices_.emplace_back(control_mesh.element(e).vertex(0),
                                     control_mesh.element(e).vertex(1),
                                     control_mesh.element(e).vertex(2),
                                     control_mesh.element(e).vertex(3));
        matched = true;
        break;
      }
    }
    if (!matched) {
      throw std::runtime_error("Passive vertex outside of the control mesh.");
    }
  }
}

VectorXd BarycentricInterpolator::operator()(const VectorXd& q) const {
  DRAKE_THROW_UNLESS(q.size() != 3 * num_control_vertices_);
  VectorXd result(3 * vertex_indices_.size());
  for (int i = 0; i < ssize(vertex_indices_); ++i) {
    Vector3d p_WV = Vector3d::Zero();
    const Vector4d& bary = barycentric_coordinates_[i];
    const Vector4i& indices = vertex_indices_[i];
    for (int j = 0; j < 4; ++j) {
      p_WV += bary[j] * q.segment<3>(3 * indices[j]);
    }
    result.segment<3>(3 * i) = p_WV;
  }
  return result;
}

VectorXd VertexSelector::operator()(const VectorXd& q) const {
  DRAKE_THROW_UNLESS(q.size() >= 3 * ssize(selected_vertices));
  VectorXd result(3 * selected_vertices.size());
  for (int i = 0; i < ssize(selected_vertices); ++i) {
    result.segment<3>(3 * i) = q.segment<3>(3 * selected_vertices[i]);
  }
  return result;
}

MeshDeformationInterpolator::MeshDeformationInterpolator(
    const std::vector<RenderMesh>& driven_meshes,
    const VolumeMesh<double>& control_mesh) {
  for (const auto& mesh : driven_meshes) {
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& positions =
        mesh.positions;
    data_.push_back(BarycentricInterpolator(positions, control_mesh));
  }
}

/* Constructor for a single driven mesh whose vertices form a subset of the
 set of vertices of the control mesh. */
MeshDeformationInterpolator::MeshDeformationInterpolator(
    VertexSelector selector) {
  data_.emplace_back(std::move(selector));
}

std::vector<VectorXd> MeshDeformationInterpolator::Interpolate(
    const VectorXd& q) const {
  std::vector<VectorXd> result;
  result.reserve(data_.size());
  for (const auto& interpolator : data_) {
    result.emplace_back(std::visit(
        [&q](const auto& f) {
          return f(q);
        },
        interpolator));
  }
  return result;
}

std::pair<TriangleSurfaceMesh<double>, MeshDeformationInterpolator>
ExtractSurfaceMeshAndInterpolator(const VolumeMesh<double>& control_mesh) {
  /* For each tet mesh, extract all the border triangles. Those are the
   triangles that are only referenced by a single tet. So, for every tet, we
   examine its four constituent triangle and determine if any other tet
   shares it. Any triangle that is only referenced once is a border triangle.
   Each triangle has a unique key: a SortedTriplet (so the ordering of the
   triangle vertex indices won't matter). The first time we see a triangle, we
   add it to a map. The second time we see the triangle, we remove it. When
   we're done, the keys in the map will be those triangles referenced only
   once. The values in the map represent the triangle, with the vertex indices
   ordered so that they point *out* of the tetrahedron. Therefore,
   they will also point outside of the mesh. A typical tetrahedral element
   looks like:

       p2 *
          |
          |
       p3 *---* p0
         /
        /
    p1 *

   The index order for a particular tetrahedron has the order [p0, p1, p2,
   p3]. These local indices enumerate each of the tet triangles with
   outward-pointing normals with respect to the right-hand rule. */
  const array<array<int, 3>, 4> local_indices{
      {{{1, 0, 2}}, {{3, 0, 1}}, {{3, 1, 2}}, {{2, 0, 3}}}};

  std::map<SortedTriplet<int>, array<int, 3>> border_triangles;
  for (const VolumeElement& tet : control_mesh.tetrahedra()) {
    for (const array<int, 3>& tet_triangle : local_indices) {
      const array<int, 3> tri{tet.vertex(tet_triangle[0]),
                              tet.vertex(tet_triangle[1]),
                              tet.vertex(tet_triangle[2])};
      const SortedTriplet triangle_key(tri[0], tri[1], tri[2]);
      // Here we rely on the fact that at most two tets would share a common
      // triangle.
      if (auto itr = border_triangles.find(triangle_key);
          itr != border_triangles.end()) {
        border_triangles.erase(itr);
      } else {
        border_triangles[triangle_key] = tri;
      }
    }
  }
  /* Record the expected minimum number of vertex positions to be received.
   For simplicity we choose a generous upper bound: the total number of
   vertices in the tetrahedral mesh, even though we really only need the
   positions of the vertices on the surface. */
  const int volume_vertex_count = control_mesh.num_vertices();

  /* Using a set because the vertices will be nicely ordered. Ideally, we'll
   be extracting a subset of the vertex positions from the input port. We
   optimize cache coherency if we march in a monotonically increasing pattern.
   So, we'll map triangle vertex indices to volume vertex indices in a
   strictly monotonically increasing relationship. */
  std::set<int> unique_vertices;
  for (const auto& [triangle_key, triangle] : border_triangles) {
    unused(triangle_key);
    for (int j = 0; j < 3; ++j) unique_vertices.insert(triangle[j]);
  }

  /* This is the *second* documented responsibility of this function: Populate
   the mapping from surface to volume so that we can efficiently extract the
   *surface* vertex positions from the *volume* vertex input. */
  std::vector<int> surface_to_volume_vertices;
  surface_to_volume_vertices.insert(surface_to_volume_vertices.begin(),
                                    unique_vertices.begin(),
                                    unique_vertices.end());

  /* The border triangles all include indices into the volume vertices. To
   turn them into surface triangles, they need to include indices into the
   surface vertices. Create the volume index --> surface map to facilitate the
   transformation. */
  const int surface_vertex_count =
      static_cast<int>(surface_to_volume_vertices.size());
  std::map<int, int> volume_to_surface;
  for (int j = 0; j < surface_vertex_count; ++j) {
    volume_to_surface[surface_to_volume_vertices[j]] = j;
  }

  /* This is the *first* documented responsibility: Create the topology of the
   surface triangle mesh for each volume mesh. Each triangle consists of three
   indices into the set of *surface* vertex positions. */
  std::vector<SurfaceTriangle> surface_triangles;
  surface_triangles.reserve(border_triangles.size());
  for (auto& [triangle_key, face] : border_triangles) {
    unused(triangle_key);
    surface_triangles.emplace_back(volume_to_surface[face[0]],
                                   volume_to_surface[face[1]],
                                   volume_to_surface[face[2]]);
  }

  VectorXd q(3 * volume_vertex_count);
  for (int v = 0; v < volume_vertex_count; ++v) {
    q.segment<3>(3 * v) = control_mesh.vertex(v);
  }
  const VertexSelector selector{.selected_vertices =
                                    std::move(surface_to_volume_vertices)};
  const VectorXd driven_qs = selector(q);
  std::vector<Vector3<double>> vertex_positions(driven_qs.size() / 3);
  for (int i = 0; i < ssize(vertex_positions); ++i) {
    vertex_positions[i] = driven_qs.segment<3>(3 * i);
  }
  TriangleSurfaceMesh<double> triangle_mesh(std::move(surface_triangles),
                                            std::move(vertex_positions));

  return {std::move(triangle_mesh),
          MeshDeformationInterpolator(std::move(selector))};
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
