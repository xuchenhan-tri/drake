#include "drake/geometry/query_results/deformable_rigid_contact.h"

#include <algorithm>
#include <iterator>
#include <set>
#include <utility>

namespace drake {
namespace geometry {
namespace internal {

using multibody::contact_solvers::internal::PartialPermutation;

namespace {

/* Extends a partial permutation to a full permutation. */
void ExtendToFullPermutation(PartialPermutation* permutation) {
  for (int i = 0; i < static_cast<int>(permutation->domain_size()); ++i) {
    /* The call to permutation.push() only conditionally adds i. */
    permutation->push(i);
  }
}

}  // namespace

template <typename T>
DeformableGeometryInContact<T>::DeformableGeometryInContact(
    GeometryId deformable_id, int num_vertices)
    : deformable_id_(deformable_id), participation_(num_vertices, false) {}

template <typename T>
void DeformableGeometryInContact<T>::Append(
    const std::unordered_set<int>& participating_vertices) {
  for (int v : participating_vertices) {
    DRAKE_DEMAND(0 <= v && v < static_cast<int>(participation_.size()));
    if (!participation_[v]) {
      ++num_vertices_in_contact_;
      participation_[v] = true;
    }
  }
}

template <typename T>
PartialPermutation DeformableGeometryInContact<T>::CalcVertexPermutation()
    const {
  /* Build the partial permutation. */
  PartialPermutation permutation = CalcVertexPartialPermutation();
  ExtendToFullPermutation(&permutation);
  return permutation;
}

template <typename T>
PartialPermutation
DeformableGeometryInContact<T>::CalcVertexPartialPermutation() const {
  int permuted_vertex_index = -1;  // We'll pre-increment before using.
  std::vector<int> permuted_vertex_indexes(participation_.size(), -1);
  for (int v = 0; v < static_cast<int>(participation_.size()); ++v) {
    if (participation_[v]) {
      permuted_vertex_indexes[v] = ++permuted_vertex_index;
    }
  }
  return PartialPermutation(std::move(permuted_vertex_indexes));
}

template <typename T>
PartialPermutation DeformableGeometryInContact<T>::CalcDofPermutation() const {
  PartialPermutation permutation = CalcDofPartialPermutation();
  ExtendToFullPermutation(&permutation);
  return permutation;
}

template <typename T>
PartialPermutation DeformableGeometryInContact<T>::CalcDofPartialPermutation()
    const {
  /* Build the partial permutation. */
  int permuted_vertex_index = 0;
  std::vector<int> permuted_dof_indexes(3 * participation_.size(), -1);
  for (int v = 0; v < static_cast<int>(participation_.size()); ++v) {
    if (participation_[v]) {
      permuted_dof_indexes[3 * v] = 3 * permuted_vertex_index;
      permuted_dof_indexes[3 * v + 1] = 3 * permuted_vertex_index + 1;
      permuted_dof_indexes[3 * v + 2] = 3 * permuted_vertex_index + 2;
      ++permuted_vertex_index;
    }
  }
  return PartialPermutation(std::move(permuted_dof_indexes));
}

template class DeformableGeometryInContact<double>;

}  // namespace internal
}  // namespace geometry
}  // namespace drake
