#include "drake/multibody/fem/block_sparse_cholesky_solver.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

std::vector<std::unordered_set<int>> BuildAdjacencyGraph(
    int num_verts, const std::vector<Vector4<int>>& elements) {
  using std::unordered_set;
  using Vector4i = Vector4<int>;
  using std::vector;
  vector<unordered_set<int>> adj(num_verts);
  for (const Vector4i& e : elements) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        adj[e(i)].insert(e(j));
      }
    }
  }
  return adj;
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
    const std::vector<Vector4<int>>& cliques, int block_cols)
    : block_cols_(block_cols),
      L_(GetFillInGraph(block_cols_, cliques)),
      L_diag_(block_cols) {}

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
   // namespace drake