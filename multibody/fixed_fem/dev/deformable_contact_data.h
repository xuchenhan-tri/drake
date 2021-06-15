#pragma once
#include <set>
#include <utility>
#include <vector>

#include "drake/multibody/fixed_fem/dev/deformable_rigid_contact_pair.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
/* DeformbaleContactData stores all the contact query information related to a
 particular deformable body. In addition, it stores information about the
 indexes of vertices participating in contact for this deformable body. See
 below for an example. */
template <typename T>
class DeformableContactData {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(DeformableContactData)
  /* Constructs the DeformableContactData for a deformable body given all
   deformable-rigid contact pairs involving the deformable body and the number
   of vertices for the deformable mesh. */
  DeformableContactData(
      std::vector<DeformableRigidContactPair<T>> contact_pairs,
      const geometry::VolumeMesh<T>& deformable_mesh)
      : contact_pairs_(contact_pairs),
        permuted_vertex_indexes_(deformable_mesh.num_vertices(), -1),
        permuted_to_original_indexes_(deformable_mesh.num_vertices()) {
    num_contact_points_ = 0;
    if (!contact_pairs.empty()) {
      CalcParticipatingVertices(deformable_mesh);
      for (const auto& contact_pair : contact_pairs) {
        num_contact_points_ += contact_pair.num_contact_points();
      }
    } else {
      std::iota(std::begin(permuted_vertex_indexes_),
                std::end(permuted_vertex_indexes_), 0);
      std::iota(std::begin(permuted_to_original_indexes_),
                std::end(permuted_to_original_indexes_), 0);
    }
  }

  /* A 2D anologue of a deformable mesh D in contact with a rigid body R. The
   deformable mesh has 6 vertices with indexes v0-v5. Vertices v1, v2, and v5
   are participating in contact.

                          v3       v4       v5
                           ●--------●--------●
                           |\       |       /|
                           | \      |      / |
                           |  \  D  |     /  |
                           |   \    |    /   |
                           |    \   |   /    |
                           |     \  |  /     |
                           |      \ | /   ●--+----●
                           |       \|/    |  |    |
                           ●--------●-----+--●    |
                          v0       v1     | v2    |
                                          |       |
                                          |  R    |
                                          ●-------●                */

  /* Returns a vector v such that v[i] gives the permuted vertex index for
   vertex i. The vertex indexes in the way characterized by the following
   properties:
      1. The new index of any vertex participating in contact is smaller than
         the new index of any vertex not participating in contact, and
      2. If vertex with old indexes i and j are both participating in
         contact/not participating in contact and i < j, then the new indexes
         satisfy v[i] < v[j].
      3. The mapping v is a bijection on {0, ..., number of vertices in the
         deformable mesh - 1}.

   In the example shown above, v1, v2, and v5 are participating in contact and
   thus have new indexes 0, 1 and 2. v0, v3, and v4 are not participating in
   contact and have new indexes 3, 4, and 5.

   Hence the returned vector would be {3, 0, 1, 4, 5, 2}. */
  const std::vector<int>& permuted_vertex_indexes() const {
    return permuted_vertex_indexes_;
  }

  const std::vector<int>& permuted_to_original_indexes() const {
    return permuted_to_original_indexes_;
  }

  /* Returns the number of vertices of the deformable body that participate in
   contact. */
  int num_vertices_in_contact() const { return num_vertices_in_contact_; }

  /* Returns the total number of contact points that have the deformable body as
   one of the bodies in contact. */
  int num_contact_points() const { return num_contact_points_; }

  /* Returns the number of contact pairs that involve the deformable body of
   interest. */
  int num_contact_pairs() const { return contact_pairs_.size(); }

  /* Returns all deformable-rigid contact pairs that involve the deformable body
   of interest with no particular order. */
  const std::vector<DeformableRigidContactPair<T>>& contact_pairs() const {
    return contact_pairs_;
  }

 private:
  /* Populates the data member `permuted_vertex_indexes_`. Only called by the
   constructor when there exists at least one contact pair. */
  void CalcParticipatingVertices(
      const geometry::VolumeMesh<T>& deformable_mesh) {
    constexpr int kNumVerticesInTetrahedron = 4;
    /* Accumulate indexes of all vertices participating in contact into an
     ordered set. */
    std::set<int> participating_vertices;
    for (int i = 0; i < num_contact_pairs(); ++i) {
      const DeformableContactSurface<T>& contact_surface =
          contact_pairs_[i].contact_surface;
      for (int j = 0; j < contact_surface.num_polygons(); ++j) {
        geometry::VolumeElementIndex tet_in_contact =
            contact_surface.polygon_data(j).tet_index;
        for (int k = 0; k < kNumVerticesInTetrahedron; ++k) {
          participating_vertices.insert(
              deformable_mesh.element(tet_in_contact).vertex(k));
        }
      }
    }
    num_vertices_in_contact_ = participating_vertices.size();

    /* Builds the permuted_vertex_indexes_. All entries are already initialized
     to -1 in the constructor. */
    int new_index = 0;
    /* Vertices participating in contact. */
    for (int old_index : participating_vertices) {
      permuted_vertex_indexes_[old_index] = new_index++;
    }
    /* Vertices not participating in contact. */
    for (int i = 0; i < static_cast<int>(permuted_vertex_indexes_.size());
         ++i) {
      if (permuted_vertex_indexes_[i] == -1) {
        permuted_vertex_indexes_[i] = new_index++;
      }
    }
    /* Sanity check that the old and new indexes go up to the same number. */
    DRAKE_DEMAND(new_index ==
                 static_cast<int>(permuted_vertex_indexes_.size()));

    /* Build the inverse map. */
    for (int i = 0; i < static_cast<int>(permuted_vertex_indexes_.size());
         ++i) {
      permuted_to_original_indexes_[permuted_vertex_indexes_[i]] = i;
    }
  }

  /* All contact pairs involving the deformable body of interest. */
  std::vector<DeformableRigidContactPair<T>> contact_pairs_{};
  /* Maps vertex indexes to "permuted vertex indexes". See the getter method for
   more info. */
  std::vector<int> permuted_vertex_indexes_{};
  std::vector<int> permuted_to_original_indexes_{};
  int num_contact_points_{0};
  int num_vertices_in_contact_{0};
};
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
