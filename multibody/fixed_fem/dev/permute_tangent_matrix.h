#pragma once

#include <utility>
#include <vector>

#include <Eigen/SparseCore>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
/* Data structure to hold the result of PermuteTangentMatrix(). See below. */
template <typename T>
struct BlockTangentMatrix {
  MatrixX<T> participating_block;
  MatrixX<T> off_diagonal_block;
  MatrixX<T> nonparticipating_block;
};

/* Reindexes the deformable dofs so that the tangent matrix assumes a block
 structure depending on contact participation.
 In the context of deformable contact, we may reindex the deformable dofs so
 that the deformable tangent matrix (see FemModel) form a particular pattern
 that facilitates subsequent computations. In particular, PermuteTangentMatrix()
 reindexes the dofs according to whether or not it's participating in contact
 and calculates the tangent matrix with the new indexes. We define a deformable
 dof to be participating in contact if it belongs to a vertex on a tetrahedron
 intersecting a rigid mesh.

 Suppose there are `nv` deformable dofs in a given deformable body and `npv`
 dofs are participating in contact. We reindex the `npv` participating dofs,
 ordered by their original indexes, as 0, 1, ..., npv-1, and reindex the
 `nv-npv` nonparticipating dofs, ordered by their original indexes, as `npv,
 npv+1, ..., nv`. As a result, the tangent matrix assumes a block structure; we
 call the top-left npv-by-npv block the "participating block", the bottom right
 `nv-npv`-by-`nv-npv` block the nonparticipating block, and the top-right
 npv-by-`nv-npv` block the "off-diagonal block". The bottom-left `nv-npv`-by-npv
 block is guaranteed to be the transpose of the top-right block by symmetry of
 the tangent matrix.
 @param[in] tangent_matrix    The tangent matrix calculated with the original
 deformable dof indexes.
 @param[in] vertex_mapping    The mapping from old vertex indexes to the new
 ones. vertex_mapping[i] gives the new index after permutation according to
 contact participation of the vertex whose original index was `i`.
 @param[in] num_participating_vertices    The number of vertices that
 participate in contact.
 @pre tangent_matrix.rows() == tangent_matrix.cols().
 @pre vertex_mapping is a permutation of 0, 1, ..., tangent_matrix.cols()/3-1.
 @pre 0 <= num_participating_vertices <= vertex_mapping.size(). */
template <typename T>
BlockTangentMatrix<T> PermuteTangentMatrix(
    const Eigen::SparseMatrix<T>& tangent_matrix,
    const std::vector<int>& vertex_mapping, int num_participating_vertices) {
  DRAKE_ASSERT(tangent_matrix.rows() == tangent_matrix.cols());
  DRAKE_ASSERT(static_cast<int>(vertex_mapping.size()) * 3 ==
               tangent_matrix.cols());
  DRAKE_ASSERT(num_participating_vertices <=
               static_cast<int>(vertex_mapping.size()));
  const int nv = tangent_matrix.rows();
  const int npv = num_participating_vertices * 3;
  MatrixX<T> participating_block = MatrixX<T>::Zero(npv, npv);
  MatrixX<T> off_diagonal_block = MatrixX<T>::Zero(npv, nv - npv);
  MatrixX<T> nonparticipating_block = MatrixX<T>::Zero(nv - npv, nv - npv);
  using InnerIterator = typename Eigen::SparseMatrix<T>::InnerIterator;
  for (int i = 0; i < tangent_matrix.outerSize(); ++i) {
    for (InnerIterator it(tangent_matrix, i); it; ++it) {
      const int row_dim = it.row() % 3;
      const int col_dim = it.col() % 3;
      const int row_vertex = it.row() / 3;
      const int col_vertex = it.col() / 3;
      const int new_row_vertex = vertex_mapping[row_vertex];
      const int new_col_vertex = vertex_mapping[col_vertex];
      const int new_row = new_row_vertex * 3 + row_dim;
      const int new_col = new_col_vertex * 3 + col_dim;
      if (new_row_vertex < num_participating_vertices) {
        if (new_col_vertex < num_participating_vertices) {
          participating_block(new_row, new_col) = it.value();
        } else {
          off_diagonal_block(new_row, new_col - npv) = it.value();
        }
      } else {
        if (new_col_vertex >= num_participating_vertices) {
          nonparticipating_block(new_row - npv, new_col - npv) = it.value();
        }
      }
    }
  }
  return {std::move(participating_block), std::move(off_diagonal_block),
          std::move(nonparticipating_block)};
}
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
