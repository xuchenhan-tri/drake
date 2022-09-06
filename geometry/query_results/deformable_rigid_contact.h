#pragma once

#include <unordered_set>
#include <vector>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/polygon_surface_mesh.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"

namespace drake {
namespace geometry {
namespace internal {

/* DeformableGeometryInContact characterizes a deformable geometry in contact.
 The characterization includes:
   - The geometry id for the deformable geometry.
   - The *abstract* concept of contact "points". We get one point for each
     polygon across all per-geometry-pair contact meshes. The point is the
     polygon's centroid.
     - The number of contact points generated due to the deforamble geometry.
     - The number of vertices in the geometry incident to the tetrahedra that
       contain a contact point.
   - Partial and/or full permutation of the vertices of the deformable geometry
     and their associated degrees of freedom based on "contact participation",
     illustrated in the schematics below.

                          v3       v4       v5
                           ●--------●--------●
                           |\       |       /|
                           | \      |      / |
                           |  \  D  |     /  |
                           |   \    |    /   |
                           |    \   |   /    |
                           |     \  |  /     |
                           |      \ | /   X  |
                           |       \|/       |
                           ●--------●--------●
                          v0       v1       v2
 A 2D analog of a deformable geometry D in contact. The deformable mesh has 6
 vertices with indexes v0-v5. Vertices v1, v2, and v5 are said to be
 "participating in contact" as the element that they are incident to contains a
 contact point, marked with "X".

 @tparam_double_only */
template <typename T>
class DeformableGeometryInContact {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(DeformableGeometryInContact)

  /* Constructs a DeformableGeometryInContact for deformable geometry with the
   given id and the given number of vertices in its mesh representation.
   @pre deformable_id is valid.
   @pre num_vertices > 0. */
  DeformableGeometryInContact(GeometryId deformable_id, int num_vertices);

  /* Returns the GeometryId of the deformable body in contact. */
  GeometryId deformable_id() const { return deformable_id_; }

  /* Mark the given vertices as participating in contact.
   @param[in] participating_vertices
      Each contact point is completely contained within one tetrahedron of the
      deformable mesh. `participating_vertices` contains the indexes of vertices
      incident to all such tetrahedra.
   @pre each entry in `participating_vertices` is non-negative and less than
      `num_vertices` supplied in the constructor. */
  void Append(const std::unordered_set<int>& participating_vertices);

  /* Returns the permutation p such that p(i) gives the permuted vertex
   index for vertex i. The vertex indexes are permuted in a way
   characterized by the following properties:
      1. The permuted index of any vertex participating in contact is smaller
         than the permuted index of any vertex not participating in contact.
      2. If vertices with original indexes i and j (with i < j) are both
         participating in contact or both not participating in contact, then
         the permuted indexes satisfy p(i) < p(j).

   In the example shown in the class doc, v1, v2, and v5 are participating in
   contact and thus have new indexes 0, 1 and 2. v0, v3, and v4 are not
   participating in contact and have new indexes 3, 4, and 5.

   Hence, the returned vector would be {3, 0, 1, 4, 5, 2}, which means the
   permutation from the original vertex index to the permuted vertex index
   follows this table.

   |   Original       |   Permuted       |   Participating   |
   |   vertex index   |   vertex index   |   in contact      |
   | :--------------: | :--------------: | :---------------: |
   |        0         |        3         |       no          |
   |        1         |        0         |       yes         |
   |        2         |        1         |       yes         |
   |        3         |        4         |       no          |
   |        4         |        5         |       no          |
   |        5         |        2         |       yes         |

   If no contact exists, returns the identity permutation. */
  multibody::contact_solvers::internal::PartialPermutation
  CalcVertexPermutation() const;

  /* Returns the partial permutation that only considers vertices that do
   * participating in contact. */
  multibody::contact_solvers::internal::PartialPermutation
  CalcVertexPartialPermutation() const;

  /* Suppose p is the partial vertex permutation, this permutation q is defined
   such that q(3*i + d) = 3*p(i) + d for all i in [0, num_vertices) and d = 0,
   1, 2. */
  multibody::contact_solvers::internal::PartialPermutation
  CalcDofFullPermutation() const;

  /* The partial permutation version of CalcDofFullPermutation. */
  multibody::contact_solvers::internal::PartialPermutation
  CalcDofPartialPermutation() const;

  /* Returns the number of vertices of the deformable body that participate in
   contact. */
  int num_vertices_in_contact() const { return num_vertices_in_contact_; }

 private:
  GeometryId deformable_id_;
  /* participation_[i] indicates whether the i-th vertex participates in
   contact. */
  std::vector<bool> participation_;
  int num_vertices_in_contact_{0};
};

template <typename T>
class DeformableContact {
 public:
  const std::vector<DeformableContactSurface<T>>& contact_surfaces() const {
    return contact_surfaces;
  }

  const DeformableGeometryInContact& geometry(GeometryId deformable_id) const {
    return geometries_.at(deforamble_id);
  }

  /* Add a contact surface between the deformable geometry and a rigid geometry.
  @param[in] rigid_id
     The GeometryId of the rigid geometry.
  @param[in] participating_vertices
     Each contact polygon in `contact_mesh_W` is completely contained within
     one tetrahedron of the deformable mesh. `participating_vertices` contains
     the indexes of vertices incident to all such tetrahedra.
  @param[in] contact_mesh_W
     The contact surface mesh expressed in World frame. The normals of the
     mesh point out of the rigid geometry.
  @param[in] signed_distances
     _Approximate_ signed distances of penetration sampled on `contact_mesh_W`.
     These values are non-positive.
  @param[in] tetrahedra_indexes
     The indexes of the tetrahedra containing each contact point with the same
     index semantics as `signed_distances`.
  @param[in] barycentric_coordinates
     Barycentric coordinates of centroids of contact polygons with respect to
     their containing tetrahedra with the same index semantics as
     `signed_distances`.
  @pre A deformable geometry with the given `deformable_id` has been registered
       via `RegisterDeforambleGeometry()`.
  @pre contact_mesh_W.num_faces() == signed_distances.size().
  @pre contact_mesh_W.num_faces() == tetrahedra_indexes.size().
  @pre contact_mesh_W.num_faces() == barycentric_coordinates.size().
  @pre each entry in `participating_vertices` is non-negative and less than
  `num_vertices` supplied in the constructor.
  @note Some variables are passed by r-value reference (as opposed to value) to
  facilitate efficient construction and moving of local variables. */
  void AddDeformableRigidContactSurface(
      GeometryId deformable_id, GeometryId rigid_id,
      const std::unordered_set<int>& participating_vertices,
      PolygonSurfaceMesh<T>&& contact_mesh_W, std::vector<T>&& signed_distances,
      std::vector<int>&& tetrahedra_indexes,
      std::vector<Vector4<T>>&& barycentric_coordinates) {
    const auto iter = deformable_geometries_.find(deformable_id);
    DRAKE_THROW_UNLESS(iter != deformable_geometries_.end());
    iter->second.Append(participating_vertices);
    contact_surfaces_.emplace_back(
        deformable_id, rigid_id, move(contact_mesh_W), move(signed_distances),
        move(tetrahedra_indexes), move(barycentric_coordinates));
  }

  void RegisterDeformableGeometry(GeometryId deformable_id, int num_vertices) {
    geometries_.emplace(deformable_id, deformable_id, num_vertices);
  }

 private:
  std::unordered_map<GeometryId, DeformableGeometryInContact<T>> geometries_;
  std::vector<DeformableContactSurface<T>> contact_surfaces_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
