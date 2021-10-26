#include "drake/multibody/plant/contact_permutation_utils.h"

#include <unordered_map>
#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/common/sorted_pair.h"
#include "drake/multibody/tree/multibody_tree_topology.h"

#include <iostream>
#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;

#define PRINT_VEC(v)       \
  std::cout << #v ":\n";   \
  for (auto& e : v) {      \
    std::cout << e << " "; \
  }                        \
  std::cout << std::endl;

namespace drake {
namespace multibody {
namespace internal {

// Helper to deep traverse a tree from its base at node_index.
// @pre node_index != world_index().
void TreeDepthFirstTraversal(const MultibodyTreeTopology& tree_topology,
                             BodyNodeIndex node_index,
                             std::vector<int>* velocity_permutation, std::vector<BodyIndex>* tree_bodies) {
  const BodyNodeTopology& node = tree_topology.get_body_node(node_index);
  tree_bodies->push_back(node.body);

  for (BodyNodeIndex child_index : node.child_nodes) {
    TreeDepthFirstTraversal(tree_topology, child_index, velocity_permutation,
                            tree_bodies);
  }

  // Push the velocity indexes for this node.
  for (int i = 0; i < node.num_mobilizer_velocities; ++i) {
    const int m = node.mobilizer_velocities_start_in_v + i;
    velocity_permutation->push_back(m);
  }
}

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
    std::vector<int>* body_to_tree_map) {
  velocity_permutation->clear();
  body_to_tree_map->clear();

  const BodyNodeTopology& world_node =
      tree_topology.get_body_node(BodyNodeIndex(0));

  // Ivalid (negative) index for the world. It does not belong to any tree in
  // particular, but it's the "floor" of the forest. Also any "tree" that is
  // anchored to the world is labeled with -1.
  body_to_tree_map->resize(tree_topology.num_bodies(), -1);

  // We deep traverse one tree at a time.
  for (BodyNodeIndex base_index : world_node.child_nodes) {
    // Bodies of the tree with base_index at the base.
    std::vector<BodyIndex> tree_bodies;
    // the permutation for the tree with base_index at its base.
    std::vector<int> tree_permutation;  
    TreeDepthFirstTraversal(tree_topology, base_index, &tree_permutation,
                            &tree_bodies);
    const int tree_num_velocities = tree_permutation.size();
    // Trees with zero dofs are not considered.
    if (tree_num_velocities != 0) {
      const int t = velocity_permutation->size();
      velocity_permutation->push_back(tree_permutation);
      for (BodyIndex body_index : tree_bodies) {
        (*body_to_tree_map)[body_index] = t;
      }
    }
  }
}

// `contacts[k]` contains pairs (t1, t2) for the two trees in contact for the
// k-th contact pair.
//
// k = patches[p][i], with i = 0:r_p-1. Where:
//  - k is the index in the original list `contacts`.
//  - r_p = patches[p].size() is the number of contacts in patch p.
ContactGraph ComputeContactGraph(
    int num_trees,
    const std::vector<SortedPair<int>>& contacts,
    std::vector<int>* participating_trees) {
  ContactGraph graph;  
  std::vector<ContactGraph::ContactPatch>& patches = graph.patches;

  // We'll group contact by "patches".
  // We define patch as a group of contacts that have the same pair of
  // trees.
  std::unordered_map<SortedPair<int>, std::vector<int>> patch_set;
  for (size_t k = 0; k < contacts.size(); ++k) {
    const auto& pair = contacts[k];
    patch_set[pair].push_back(k);  // store index to original list of contacts.
  }

  std::vector<bool> tree_participates;
  tree_participates.resize(num_trees, false);
  for (const auto& patch_pair : patch_set) {
    const int t1 = patch_pair.first.first();
    if (t1 >= 0) tree_participates[t1] = true;
    const int t2 = patch_pair.first.second();
    if (t2 >= 0) tree_participates[t2] = true;
  }

  // Form list of participating trees.
  std::vector<int> reduced_index(num_trees, -1);
  participating_trees->clear();
  participating_trees->reserve(num_trees);  // reserve maximum.
  for (int t = 0; t < num_trees; ++t) {
    if (tree_participates[t]) {
      const int tr = participating_trees->size();  // reduced index.
      participating_trees->push_back(t);
      reduced_index[t] = tr;
    }
  }

  // patch_num_contacts->reserve(patch_set.size());
  // patch_permutation->reserve(contacts.size());
  patches.reserve(patch_set.size());
  for (const auto& patch_pair : patch_set) {
    const int t1 = patch_pair.first.first();
    const int t2 = patch_pair.first.second();    
    const auto& p = patch_pair.second;

    // Get reduced numbering of participating trees.
    const int t1r = t1 >= 0 ? reduced_index[t1] : t1;
    const int t2r = t2 >= 0 ? reduced_index[t2] : t2;

    // The permutation places contact pairs ordered by patch, one patch after
    // the other.
    // patch_permutation->insert(patch_permutation->end(), p.begin(), p.end());
    // patch_num_contacts->push_back(p.size());
    patches.push_back({t1r, t2r,p});
  }
  return graph;
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake