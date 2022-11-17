#include "drake/multibody/fem/block_sparse_cholesky_solver.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <cholmod.h>

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

using std::set;
using std::unordered_set;
using Vector4i = Vector4<int>;
using std::vector;

vector<set<int>> BuildAdjacencyGraph(int num_verts,
                                     const vector<Vector4i>& elements) {
  vector<set<int>> adj(num_verts);
  for (const Vector4i& e : elements) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        if (e(i) >= e(j)) {
          adj[e(j)].insert(e(i));
        }
      }
    }
  }
  return adj;
}

std::vector<int> CalcEliminationOrdering(
    const std::vector<std::set<int>>& adjacency_graph) {
  /* Size of the matrix. */
  const int N = adjacency_graph.size();
  int nnz = 0;
  for (const auto& col : adjacency_graph) {
    nnz += col.size();
  }
  // TODO(xuchenhan-tri): Figure out if the upper triangular part of the matrix
  // needs to be filled out.
  /* Build a matrix A for symbolic analysis. */
  cholmod_common cm;
  cholmod_start(&cm);
  auto A = std::unique_ptr<cholmod_sparse>(cholmod_allocate_sparse(
      N, N,
      /* max number of nonzeros */
      nnz,
      /* sorted */ true,
      /* packed */ true,
      /* ignore top right corner */ -1, CHOLMOD_REAL, &cm));
  /* Fill out A->i (inner index), A->p (outer index), and A->x (values). See
   CHOLMOD guide for definitions. */
  int value_index = 0; /* index for nonzero values */
  int* Ap = static_cast<int*>(A->p);
  int* Ai = static_cast<int*>(A->i);
  int* Ax = static_cast<int*>(A->x);
  Ap[0] = 0;
  for (int col = 0; col < static_cast<int>(adjacency_graph.size()); ++col) {
    Ap[col + 1] = Ap[col] + adjacency_graph[col].size();
    for (const int r : adjacency_graph[col]) {
      Ai[value_index] = r;
      /* We set the value to an arbitrary dummy value because we only need to
       perform symbolic analysis. */
      Ax[value_index] = 1.0;
      ++value_index;
    }
  }
  auto L = std::unique_ptr<cholmod_factor>(cholmod_analyze(A.get(), &cm));
  // std::cout << "-------------------------- " << std::endl;
  // std::cout << "n = " << L->n << std::endl;
  // std::cout << "nnz= " << L->nzmax << std::endl;
  // std::cout << "supernodal nnz= " << L->xsize << std::endl;
  // std::cout << "ordering= " << L->ordering << std::endl;
  // std::cout << "is_ll= " << L->is_ll << std::endl;
  // std::cout << "is_super= " << L->is_super << std::endl;
  // std::cout << "is_monotonic= " << L->is_monotonic << std::endl;
  // std::cout << "-------------------------- " << std::endl;
  std::vector<int> permutation(N);
  memcpy(permutation.data(), L->Perm,
         permutation.size() * sizeof(permutation[0]));
  /* Clean up memory */
  auto* A_ptr = A.release();
  cholmod_free_sparse(&A_ptr, &cm);
  auto* L_ptr = L.release();
  cholmod_free_factor(&L_ptr, &cm);
  cholmod_finish(&cm);

  return permutation;
}

vector<int> RestrictOrdering(const vector<int>& perm,
                             const vector<int>& D_indices) {
  const int N = perm.size();
  /* perm maps new index into old index. */
  unordered_set<int> D_set;
  for (int d : D_indices) {
    D_set.insert(d);
  }

  /* The new index for non-participating vertices. */
  int pd = 0;
  /* The new index participating. */
  int pa = D_indices.size();
  /* The resulting permutation that respects participation mapping from old to
   new. */
  vector<int> old_to_new(N);
  for (int i = 0; i < N; ++i) {
    int old_index = perm[i];
    if (D_set.count(old_index) > 0) {
      old_to_new[old_index] = pd++;
    } else {
      old_to_new[old_index] = pa++;
    }
  }
  /* We want to return the permutation mapping from new to old*/
  vector<int> result(N);
  for (int i = 0; i < N; ++i) {
    result[old_to_new[i]] = i;
  }
  return result;
}

std::vector<std::vector<int>> CalcSparsityPattern(
    const std::vector<std::set<int>>& adjacency_graph,
    const std::vector<int>& elimination_ordering) {
  int N = elimination_ordering.size();
  DRAKE_DEMAND(static_cast<int>(adjacency_graph.size()) == N);
  std::vector<int> old_to_new(N);
  for (int i = 0; i < N; ++i) {
    old_to_new[elimination_ordering[i]] = i;
  }

  /* Computes the adjacency graph for the new ordering. */
  std::vector<std::set<int>> new_graph(N);
  for (int i = 0; i < N; ++i) {
    for (int v : adjacency_graph[i]) {
      int a = old_to_new[i];
      int b = old_to_new[v];
      if (a >= b) {
        new_graph[b].insert(a);
      } else {
        new_graph[a].insert(b);
      }
    }
  }

  /* children[p] is a vector of sort children of p. */
  std::vector<std::vector<int>> children(N);
  std::vector<std::vector<int>> result(N);
  for (int i = 0; i < N; ++i) {
    /* Merge the neighbors of i and all neighbors of children of i. */
    std::set<int> neighbor_i = new_graph[i];
    const auto& children_i = children[i];
    for (int c : children_i) {
      const auto& neighbor_c = result[c];
      for (int n : neighbor_c) {
        if (n > i) neighbor_i.insert(n);
      }
    }
    /* Turn set into vector. */
    for (int n : neighbor_i) result[i].emplace_back(n);
    /* Record the parent of i if i isn't already the root. */
    if (result[i].size() > 1) {
      const int p = result[i][1];
      children[p].emplace_back(i);
    }
  }
  return result;
}

std::vector<std::vector<int>> GetFillInGraph(
    int num_verts, const std::vector<Vector4<int>>& cliques) {
  std::vector<std::unordered_set<int>> fill_in_graph(num_verts);
  const auto original_graph = BuildAdjacencyGraph(num_verts, cliques);
  int i = 0;
  for (const auto& neighbors : original_graph) {
    fill_in_graph[i].reserve(num_verts);
    for (const int& n : neighbors) {
      if (i <= n) fill_in_graph[i].insert(n);
    }
    ++i;
  }

  // Add fill-ins.
  for (int v = 0; v < num_verts; ++v) {
    const auto& one_ring = fill_in_graph[v];
    for (int n1 : one_ring) {
      if (n1 <= v) continue;
      for (int n2 : one_ring) {
        if (n1 > v && n2 > v) {
          fill_in_graph[std::min(n1, n2)].insert(std::max(n1, n2));
        }
      }
    }
  }

  // Turn set into vector.
  std::vector<std::vector<int>> results(num_verts);
  for (int v = 0; v < num_verts; ++v) {
    for (int n : fill_in_graph[v]) {
      if (n >= v) results[v].push_back(n);
    }
  }
  return results;
}

BlockSparseCholeskySolver::BlockSparseCholeskySolver(
    std::vector<std::vector<int>> sparsity_pattern)
    : block_cols_(sparsity_pattern.size()),
      L_(std::move(sparsity_pattern)),
      L_diag_(block_cols_) {}

MatrixX<double> BlockSparseCholeskySolver::CalcSchurComplement(
    int num_eliminated_blocks) {
  /* If the matrix has been factored the original matrix is already gone. */
  DRAKE_DEMAND(0 <= num_eliminated_blocks &&
               num_eliminated_blocks <= block_cols_);
  // DRAKE_DEMAND(!is_factored_);
  FactorImpl(num_eliminated_blocks);
  return L_.MakeDenseBottomRightCorner(block_cols_ - num_eliminated_blocks);
}

void BlockSparseCholeskySolver::SolveInPlace(VectorX<double>* y) const {
  DRAKE_DEMAND(is_factored_);
  DRAKE_DEMAND(y != nullptr);
  DRAKE_DEMAND(y->size() == size()); /* Solve Lz = y in place. */
  for (int j = 0; j < block_cols_; ++j) {
    /* Solve for the j-th block entry. */
    y->segment<3>(3 * j) =
        L_diag_[j].triangularView<Eigen::Lower>().solve(y->segment<3>(3 * j));
    const auto& yj = y->segment<3>(3 * j);
    /* Eliminate for the j-th block entry from the system. */
    const auto& blocks_in_col_j = L_.get_col_blocks(j);
    for (int flat = 1; flat < static_cast<int>(blocks_in_col_j.size());
         ++flat) {
      const int i = blocks_in_col_j[flat];
      y->segment<3>(3 * i) -= L_.get_block(i, j) * yj;
    }
  }

  VectorX<double>* z = y;
  /* Solve Lᵀx = z in place. */
  for (int j = block_cols_ - 1; j >= 0; --j) {
    /* Eliminate all solved variables. */
    const auto& blocks_in_col_j = L_.get_col_blocks(j);
    for (int flat = 1; flat < static_cast<int>(blocks_in_col_j.size());
         ++flat) {
      const int i = blocks_in_col_j[flat];
      z->segment<3>(3 * j) -=
          L_.get_block(i, j).transpose() * z->segment<3>(3 * i);
    }
    /* Solve for the j-th block entry. */
    z->segment<3>(3 * j) =
        L_diag_[j].transpose().triangularView<Eigen::Upper>().solve(
            z->segment<3>(3 * j));
  }
}

VectorX<double> BlockSparseCholeskySolver::Solve(
    const VectorX<double>& y) const {
  VectorX<double> x(y);
  SolveInPlace(&x);
  return x;
}

void BlockSparseCholeskySolver::FactorImpl(int block_cols_to_factorize) {
  // DRAKE_DEMAND(!is_factored_);
  for (int j = 0; j < block_cols_to_factorize; ++j) {
    /* Update diagonal. */
    const Matrix3<double>& Ajj = L_.get_diagonal_block(j);
    const auto llt = Eigen::LLT<Matrix3<double>>(Ajj);
    DRAKE_DEMAND(llt.info() == Eigen::Success);
    L_diag_[j] = llt.matrixL();
    /* Technically, there's no need to spell out the diagonal block of the L
     matrix, but we do it for completeness here. */
    L_.SetBlock(j, j, L_diag_[j]);

    /* Update column.
     | a₁₁  *  | = | λ₁₁  0 | * | λ₁₁ᵀ L₂₁ᵀ |
     | a₂₁ a₂₂ |   | L₂₁ L₂₂|   |  0   L₂₂ᵀ |
     So we have
      L₂₁λ₁₁ᵀ = a₂₁, and thus
      λ₁₁L₂₁ᵀ = a₂₁ᵀ */
    const std::vector<int>& blocks_in_col_j = L_.get_col_blocks(j);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int flat = 1; flat < static_cast<int>(blocks_in_col_j.size());
         ++flat) {
      const auto& L_diag_j = L_diag_[j].triangularView<Eigen::Lower>();
      const int i = blocks_in_col_j[flat];
      const Matrix3<double>& Aij = L_.get_block(i, j);
      const Matrix3<double> Lij = L_diag_j.solve(Aij.transpose()).transpose();
      L_.SetBlock(i, j, Lij);
    }
    RightLookingSymmetricRank1Update(j);
  }
  is_factored_ = true;
}

void BlockSparseCholeskySolver::RightLookingSymmetricRank1Update(int j) {
  const std::vector<int>& blocks_in_col_j = L_.get_col_blocks(j);
  /* We start from f1 = 1 here to skip the j,j entry. */
#if defined(_OPENMP)
omp_set_num_threads(16);
#pragma omp parallel for collapse(2)
#endif
  for (int f1 = 0; f1 < static_cast<int>(blocks_in_col_j.size() - 1); ++f1) {
    for (int f2 = 0; f2 < static_cast<int>(blocks_in_col_j.size()); ++f2) {
      /* The loop is equivalent to
       for (int f1 = 1; f1 < blocks_in_col_j.size(); ++f1) {
         for (int f2 = f1; f2 < blocks_in_col_j.size(); ++f2) {
           ...
         }
       }
      The weird formatting is to satisfy OpenMP. */
      int flat1 = f1 + 1;
      if (f2 >= flat1) {
        const int col = blocks_in_col_j[flat1];
        const int row = blocks_in_col_j[f2];
        L_.AddToBlock(row, col,
                      -L_.get_block(row, j) * L_.get_block(col, j).transpose());
      }
    }
  }
}

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
