#include "drake/multibody/fem/block_sparse_cholesky_solver.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

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

std::vector<int> CalcPermutationFromCholmod(
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
  std::cout << "nnz= " << L->nzmax << std::endl;
  std::cout << "nnz= " << L->xsize << std::endl;
  std::cout << "ordering= " << L->ordering << std::endl;
  std::cout << "is_ll= " << L->is_ll << std::endl;
  std::cout << "is_super= " << L->is_super << std::endl;
  std::cout << "is_monotonic= " << L->is_monotonic << std::endl;
  std::vector<int> permutation(N);
  memcpy(permutation.data(), L->Perm,
         permutation.size() * sizeof(permutation[0]));
  // std::cout << "permutation " << std::endl;
  // for (int i : permutation) std::cout << i << " " << std::endl;
  // std::cout << std::endl;
  /* Clean up memory */
  auto* A_ptr = A.release();
  cholmod_free_sparse(&A_ptr, &cm);
  auto* L_ptr = L.release();
  cholmod_free_factor(&L_ptr, &cm);
  cholmod_finish(&cm);

  vector<int> r(permutation.size());
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    r[permutation[i]] = i;
  }
  return r;
}

vector<int> CalcPermutationForSchurComplement(
    const vector<int>& perfect_ordering, const vector<int>& D_indices) {
  vector<int> perfect_ordering_inverse(perfect_ordering.size());
  for (int i = 0; i < static_cast<int>(perfect_ordering.size()); ++i) {
    perfect_ordering_inverse[perfect_ordering[i]] = i;
  }
  unordered_set<int> D_set;
  for (int d : D_indices) {
    D_set.insert(d);
  }

  /* The start of permutation of D indices. */
  int pd = 0;
  /* The start of permutation of A indices. */
  int pa = D_indices.size();
  vector<int> result(perfect_ordering.size());
  for (int n : perfect_ordering_inverse) {
    if (D_set.count(n) > 0) {
      result[n] = pd++;
    } else {
      result[n] = pa++;
    }
  }
  return result;
}

std::vector<std::vector<int>> CalcSparsityPattern(
    const std::vector<std::set<int>>& adjacency_graph,
    std::vector<int> elimination_ordering) {
  /* Size of the matrix. */
  const int N = adjacency_graph.size();
  int nnz = 0;
  for (const auto& col : adjacency_graph) {
    nnz += col.size();
  }
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

  std::vector<int> workspace1(N);
  std::vector<int> workspace2(N);
  std::vector<int> parent(N);
  std::vector<int> postordering(N);
  std::vector<int> col_nnz(N);
  cholmod_analyze_ordering(A.get(), /*only analyze the provided ordering*/ 0,
                           elimination_ordering.data(),
                           /* Perform analysis on full A */ nullptr, 0,
                           parent.data(), postordering.data(), col_nnz.data(),
                           workspace1.data(), workspace2.data(), &cm);

  /* Build tril(A). */
  std::vector<std::unordered_set<int>> sparsity(N);
  for (int i = 0; i < N; ++i) {
    for (const int n : adjacency_graph[i]) {
      sparsity[i].emplace(n);
    }
  }
  /* Traverse elimination tree and parent nodes union child nonzeros. */
  for (int c : postordering) {
    const int p = parent[c];
    /* p < 0 means p is the root. */
    if (p >= 0) {
      for (int n : sparsity[c]) {
        if (n > p) {
          sparsity[p].insert(n);
        }
      }
    }
  }

  /* Verify that the sparsity pattern created match the number of nonzero
   entries per column computed by CHOLMOD. */
  for (int i = 0; i < N; ++i) {
    DRAKE_DEMAND(static_cast<int>(sparsity[i].size()) == col_nnz[i]);
  }

  /* Turn set into vector. */
  vector<vector<int>> result(N);
  for (int v = 0; v < N; ++v) {
    for (int n : sparsity[v]) {
      result[v].push_back(n);
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
  DRAKE_DEMAND(!is_factored_);
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
  DRAKE_DEMAND(!is_factored_);
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
  for (int f1 = 1; f1 < static_cast<int>(blocks_in_col_j.size()); ++f1) {
    for (int f2 = f1; f2 < static_cast<int>(blocks_in_col_j.size()); ++f2) {
      const int col = blocks_in_col_j[f1];
      const int row = blocks_in_col_j[f2];
      const Matrix3<double> diff =
          -L_.get_block(row, j) * L_.get_block(col, j).transpose();
      L_.AddToBlock(row, col, diff);
    }
  }
}

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
