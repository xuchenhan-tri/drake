#pragma once

#include <optional>
#include <unordered_map>
#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/common/sorted_pair.h"
#include "drake/multibody/tree/multibody_tree_topology.h"
#include "drake/multibody/contact_solvers/block_sparse_matrix.h"
  
namespace drake {
namespace multibody {
namespace internal {

#if 0
class PermutationBuilder { 
  public:
   // Builder for a permutation from a domain of size n.
   PermutationBuilder() = default;

   // Register permutation from i to i_permuted.
   void register_permutation(int i, int i_permuted) {
     DRAKE_DEMAND(i >= 0);
     DRAKE_DEMAND(i_permuted >= 0);
     // The permutation MUST be a bijection.
     DRAKE_DEMAND(permutation_pairs_.count(i) == 0);
     permutation_pairs_[i] = i_permuted;
     max_domain_element_ = max(max_domain_element_, i);
     max_codomain_element_ = max(max_codomain_element_, i_permuted);
   }

   // This object is left in an invalid state after this call.
   Permutation Build() const {
     std::vector<int> permutation(max_domain_element_ + 1, -1);
     for (const auto& p : permutation_pairs_) {
       permutation[p.first] = p.second;
     }

     // Make inverse permutation.
     std::vector<int> inverse_permutation_(max_codomain_element_ + 1);

   }

  private:
  int max_domain_element_{0};
   int max_codomain_element_{0};
   std::unordered_map<int, int> permutation_pairs_;
};

// A class that represents a permutation of only k <= n items out of n total
// items. In particular, k = n for a full permutation.
class PartialPermutation {
 public:
  // i_permuted = permutation[i], i=0:n-1.
  // i_permuted < 0 if i is not considered in the permutation.
  PartialPermutation(const std::vector<int>& permutation) {
    domain_size_ = permutation.size();
    permutation_ = permutation;
    const int n = permutation.size();
    inverse_permutation_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      if (permutation[i] >= 0) inverse_permutation_.push_back(i);
    }
    codomain_size_ = inverse_permutation_.size();
  }

  // Return ns, the size of the domain.
  int domain_size() const { return domain_size_; }

  // Returns m, the size of the codomain.
  int codomain_size() const { return codomain_size_; }

  // Returns i_permuted for the original element i.
  // i_permuted < 0 if the original element i does not participate in the
  // permutation.
  int forward(int i) const { return permutation_[i]; }


  // Returns the original index i given the permuted index i_permuted.
  int inverse(int i_permuted) const { return inverse_permutation_[i_permuted]; }

 private:
  int domain_size_;

  // Number of elements considered in the permutation.
  int codomain_size_;

  // i_permuted = permutation[i], i=0:n-1.
  // i_permuted < 0 if i is not considered in the permutation.
  std::vector<int> permutation_;

  // i = inverse_permutation[i_permuted], i_permuted=1:num_permuted-1.
  // i is always a valid positive index.
  std::vector<int> inverse_permutation_;
};
#endif

// Computes the permutation to go from velocities in a BFS order in
// tree_topology to DFS order, so that all dofs for a tree are contiguous.
// Important note: We use a "reverse" DFS order (that is, deeper dofs are first)
// since for complex tree structures with a free floating base (consider the
// Allegro hand for instance) the matrix will have an arrow sparsity pattern
// (pointing to the bottom right). With a traditional DFS ordering we also get
// an arrow pattern, but pointing to the upper left. The distinction here is
// when performing Cholesky on the resulting mass matrix. The "reverse" DFS is
// "optimal", in that it'll produce a minimum number of zeros in the
// factorization. While more often than not, the traditional DFS ordering will
// lead to a fully dense factorization, unless a permutation is applied (extra
// work).
//
// Summarizing.
// v = velocity_permutation[t][vt]
// where:
//   t: tree index.
//   vt: local velocity index in tree t with DFT order.
//   v: original velocity index in tree_topology.
// @param[out] body_to_tree_map t = body_to_tree_map[body_index]. t < 0 if body
// is anchored to the wold. Therefore body_to_tree_map[0] < 0 always.
void ComputeBfsToDfsPermutation(
    const MultibodyTreeTopology& tree_topology,
    std::vector<std::vector<int>>* velocity_permutation,
    std::vector<int>* body_to_tree_map);

// In this graph tress are the nodes and patches are the edges.
// Represents the the graph G such that:
//   - Patch p conects trees G.patches[p].t1 and G.patches[p].t2.
//   - The list of k contacts is for patch p is G.patches[p].contacts.
struct ContactGraph {
  struct ContactPatch {
    int t1;
    int t2;
    std::vector<int> contacts;
  };
  std::vector<ContactPatch> patches;
};

// Forms a reduced graph, that only containts participating trees, from a list
// of contact pairs between trees in the range 1:num_trees-1.
// The graph uses a "reduced" ordering, such that the p-th patch between tr1 and
// tr2, corresponds to contact between original trees t1 =
// participating_trees[tr1] and t2 = participating_trees[tr2].
// The number of trees in contact is participating_trees.size().
//
// `contacts[k]` contains pairs (t1, t2) for the two trees in contact for the
// k-th contact pair.
//
// k = patches[p][i], with i = 0:r_p-1. Where:
//  - k is the index in the original list `contacts`.
//  - r_p = patches[p].size() is the number of contacts in patch p.
ContactGraph ComputeContactGraph(int num_trees,
                                 const std::vector<SortedPair<int>>& contacts,
                                 std::vector<int>* participating_trees);

// v = velocity_permutation[t][vt]
// where:
//   t: tree index.
//   vt: local velocity index in tree t with DFT order.
//   v: original velocity index in the model.
template <typename T>
drake::multibody::contact_solvers::internal::BlockSparseMatrix<T>
ExtractBlockDiagonalMassMatrix(
    const MatrixX<T>& M,
    const std::vector<std::vector<int>>& velocity_permutation,
    const std::optional<std::vector<int>>& participating_trees = std::nullopt) {
  int nv = 0;
  for (const auto& tree_velocities : velocity_permutation) {
    nv += tree_velocities.size();
  }

  DRAKE_DEMAND(M.rows() == nv);
  DRAKE_DEMAND(M.cols() == nv);

  // Number of participating trees. All of them if participating_trees is not
  // present.
  const int num_trees = participating_trees ? participating_trees->size()
                                            : velocity_permutation.size();
  drake::multibody::contact_solvers::internal::BlockSparseMatrixBuilder<T>
      builder(num_trees, num_trees, num_trees /*nnz*/);
  for (int tr = 0; tr < num_trees; ++tr) {
    const int t = participating_trees ? participating_trees.value()[tr] : tr;
    const int nt = velocity_permutation[t].size();
    MatrixX<T> Mt(nt, nt);
    for (int it = 0; it < nt; ++it) {
      const int i = velocity_permutation[t][it];
      for (int jt = 0; jt < nt; ++jt) {
        const int j = velocity_permutation[t][jt];        
        Mt(it, jt) = M(i, j);
      }
    }
    builder.PushBlock(tr, tr, Mt);
  }
  return builder.Build();
}

// Extract dense blocks Jc_pt.
// k = patches[p][i], with i = 0:r_p-1. Where:
//  - k is the index in the original list `contacts`.
//  - r_p = patches[p].size() is the number of contacts in patch p.
// v = velocity_permutation[t][vt]
//  - v: index in the original topology.
//  - t: tree index.
//  - vt: local velocity index in reverse DFT order.
template <typename T>
drake::multibody::contact_solvers::internal::BlockSparseMatrix<T>
ExtractBlockJacobian(const MatrixX<T>& Jc, const ContactGraph& graph,
                     const std::vector<std::vector<int>>& velocity_permutation,
                     const std::vector<int>& participating_trees) {
  const std::vector<ContactGraph::ContactPatch>& patches = graph.patches;
  // Helper lambda to extract block (p, t) from dense Jacobian Jc.
  auto extract_block = [&velocity_permutation, &patches, &participating_trees,
                        &Jc](int p, int tr) {
    const int t = participating_trees[tr];
    const int rp = patches[p].contacts.size();
    const int nt = velocity_permutation[t].size();
    MatrixX<T> Jpt(3 * rp, nt);
    for (int kp = 0; kp < rp; ++kp) {  // local patch index.
      const int k = patches[p].contacts[kp];
      for (int vt = 0; vt < nt; ++vt) {  // local velocity index.
        const int v = velocity_permutation[t][vt];
        Jpt.block(3 * kp, vt, 3, 1) = Jc.block(3 * k, v, 3, 1);
      }
    }
    return Jpt;
  };

  const int num_patches = patches.size();
  const int num_trees = participating_trees.size();
  // N.B. We reseve the maximum number of blocks, which is two blocks per patch
  // connectint two trees. There might be only one block per patch when in
  // contact with the world.
  drake::multibody::contact_solvers::internal::BlockSparseMatrixBuilder<T>
      builder(num_patches, num_trees, 2 * num_patches);
  for (int p = 0; p < num_patches; ++p) {
    // Extract J_pt, unless t is the world.
    const int t1 = patches[p].t1;
    const int t2 = patches[p].t2;

    // A tree can be in contact with itself (consider a robot arm folding into
    // itself). Therefore we can have t1 = t2. In that case, the block must be
    // added only once.
    if (t1 >= 0) builder.PushBlock(p, t1, extract_block(p, t1));
    if (t2 >= 0 && t1 != t2) builder.PushBlock(p, t2, extract_block(p, t2));
  }
  return builder.Build();
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake