#pragma once

#include <utility>
#include <variant>
#include <vector>

#include "drake/common/ssize.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/render/render_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

/* Given a volume mesh (the control mesh) and a list of (passively driven)
 points embedded in the mesh, BarycentricInterpolator uses the vertex positions
 of the control mesh to compute the positions of the embedded points based on
 their barycentric coordinates in the tetrahedron containing them. */
class BarycentricInterpolator {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BarycentricInterpolator)

  /* Constructs a BarycentricInterpolator.
   @param[in] positions     The positions of the passively driven points
                            embedded in the control mesh in the mesh frame. Each
                            row of the matrix represents the position of a
                            single point.
   @param[in] control mesh  The volume mesh driving the embedded points.
   @throws std::exception if any passively driven point is outside of the
   control mesh. */
  BarycentricInterpolator(const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>& positions,
                          const VolumeMesh<double>& control_mesh);

  /* Given the positions q of the vertices of the control mesh in some frame F,
   returns the positions of the passively driven points in the same frame F.
   Both the input and the returned value are ordered as flat Eigen vectors
   composing of (x₀, y₀, z₀, x₁, y₁, z₁, ...). The input is ordered the same as
   the vertices in the input control mesh. The output is ordered the same as the
   order of the positions of the points at construction.
   @pre q.size() is 3 times the number of vertices of the control mesh given at
   construction. */
  Eigen::VectorXd operator()(const Eigen::VectorXd& q) const;

 private:
  std::vector<Eigen::Vector4i> vertex_indices_;
  std::vector<Eigen::Vector4d> barycentric_coordinates_;
  int num_total_vertices_{};
};

/* Given an Eigen vector representing the positions of a control mesh, computes
 the positions of a subset of vertices of the control mesh in the same frame. */
class VertexSelector {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(VertexSelector)

  /* Constructs a VertexSelector.
   @pre `selected_vertices` is non-empty, sorted, doesn't contain duplicates,
   and its values fall in [0, control_mesh.num_vertices()). */
  VertexSelector(std::vector<int> selected_vertices,
                 const VolumeMesh<double>& control_mesh)
      : selected_vertices_(std::move(selected_vertices)),
        num_total_vertices_(control_mesh.num_vertices()) {
    DRAKE_THROW_UNLESS(!selected_vertices_.empty());
    DRAKE_THROW_UNLESS(selected_vertices_[0] >= 0);
    // No duplicates.
    DRAKE_THROW_UNLESS(std::adjacent_find(selected_vertices_.begin(),
                                          selected_vertices_.end()) ==
                       selected_vertices_.end());
    DRAKE_THROW_UNLESS(
        std::is_sorted(selected_vertices_.begin(), selected_vertices_.end()));
    DRAKE_THROW_UNLESS(num_total_vertices_ > selected_vertices_.back());
  }

  /* Given the positions q of the vertices of the control mesh in some
   frame F, returns the positions of the passively driven points in the
   same frame F. Both the input and the returned value are ordered as flat
   Eigen vectors composing of (x₀, y₀, z₀, x₁, y₁, z₁, ...). The input is
   ordered the same as the vertices in the input control mesh. The output
   is ordered the same as the order of the positions of the points at
   construction.
   @pre q.size() is 3 times the number of vertices of the control mesh
   given at construction. */
  Eigen::VectorXd operator()(const Eigen::VectorXd& q) const;

 private:
  std::vector<int> selected_vertices_;
  int num_total_vertices_{};
};

/* Given a control (volume) mesh and one or more driven meshes that deform
 passively with the control mesh, this class provides an `Interpolate()`
 function that maps the positions of the control mesh vertices (as a flat
 Eigen vector) to the vertex positions of the passively driven meshes. */
class MeshDeformationInterpolator {
 public:
  /* Constructor for a vector of RenderMesh as the driven meshes. The vertex
   positions of the driven meshes are interpolated using their barycentric
   coordinates in the containing control mesh's tetrahedron.
   @pre each vertex of the driven mesh is inside the control mesh. */
  MeshDeformationInterpolator(const std::vector<RenderMesh>& driven_meshes,
                              const VolumeMesh<double>& control_mesh);

  /* Constructor for a single driven mesh whose vertices form a subset of the
   set of vertices of the control mesh. */
  explicit MeshDeformationInterpolator(VertexSelector selector);

  /* Returns the vertex positions of driven meshes in a std::vector, following
   the same order of the meshes at construction. The vertex positions of the
   each driven mesh is represented as a flat Eigen vector. */
  std::vector<Eigen::VectorXd> Interpolate(const Eigen::VectorXd& q) const;

 private:
  std::vector<std::variant<BarycentricInterpolator, VertexSelector>>
      interpolators_;
};

/* Given a volume control mesh, returns its surface mesh along with the
 MeshDeformationInterpolator that maps the control meshes' vertex positions to
 the surface meshes' vertex positions. In particular, the vertex indices of the
 returned surface mesh respects the original vertex indices of the volume mesh:
 if vertex i and vertex j are vertices of the resulting surface mesh and their
 indices in the control volume mesh are f(i) and f(j) respectively, then i < j
 iff f(i) < f(j). */
std::pair<TriangleSurfaceMesh<double>, MeshDeformationInterpolator>
ExtractSurfaceMeshAndInterpolator(const VolumeMesh<double>& control_mesh);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
