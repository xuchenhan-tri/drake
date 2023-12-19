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

/* Interpolates the positions of passive vertices by linearly combining the
 vertex positions of the control mesh based on the barycentric coordinate of
 the passive vertex in the element of the control mesh containing the passive
 vertex. */
class BarycentricInterpolator {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BarycentricInterpolator)

  BarycentricInterpolator(const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                              Eigen::RowMajor>& positions,
                          const VolumeMesh<double>& control_mesh);

  /* The interpolation function. */
  Eigen::VectorXd operator()(const Eigen::VectorXd& q) const;

 private:
  std::vector<Eigen::Vector4i> vertex_indices_;
  std::vector<Eigen::Vector4d> barycentric_coordinates_;
  int num_control_vertices_{};
};

/* Computes the positions of passive vertices by concatenating the positions of
 a subest of the control mesh vertices. */
struct VertexSelector {
  /* The interpolatoin function. */
  Eigen::VectorXd operator()(const Eigen::VectorXd& q) const;
  std::vector<int> selected_vertices;
};

/* Given a control (volume) mesh and one or more driven meshes that deforms
 passively with the control mesh, this class provides an `Interpolate`
 function that maps the positions of the control mesh vertices (as a flat
 Eigen vector) to the vertex positions of the passively driven meshes. */
class MeshDeformationInterpolator {
 public:
  /* Constructor for a vector of RenderMesh as the driven meshes. */
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
  std::vector<std::variant<BarycentricInterpolator, VertexSelector>> data_;
};

std::pair<TriangleSurfaceMesh<double>, MeshDeformationInterpolator>
ExtractSurfaceMeshAndInterpolator(const VolumeMesh<double>& control_mesh);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
