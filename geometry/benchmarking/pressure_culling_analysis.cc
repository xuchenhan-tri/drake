#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/geometry/proximity/bvh.h"
#include "drake/geometry/proximity/field_intersection.h"
#include "drake/geometry/proximity/make_ellipsoid_field.h"
#include "drake/geometry/proximity/make_ellipsoid_mesh.h"
#include "drake/geometry/proximity/make_sphere_field.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;
using math::RigidTransformd;

const double kElasticModulus = 1.0e5;
const double kSphereDimension = 3.;
const Vector3d kEllipsoidDimension{3.01, 3.5, 4.};
const double kResolutionHint[4] = {4., 3., 2., 1.};
const Vector3d kContactOverlapTranslation[4] = {
    Vector3d{7, 7, 7}, Vector3d{4, 4, 4}, Vector3d{3.5, 3.5, 3.5},
    Vector3d{1.2, 1.2, 1.2}};

struct PressureRange {
  double min_val;
  double max_val;
};

using NodeType = BvNode<Obb, VolumeMesh<double>>;
using NodePressureMap = std::unordered_map<const NodeType*, PressureRange>;

PressureRange ComputeNodePressureRange(
    const NodeType& node, const VolumeMeshFieldLinear<double, double>& field,
    NodePressureMap* node_ranges) {
  if (node.is_leaf()) {
    const int num_elements = node.num_element_indices();
    double lo = std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::lowest();
    for (int i = 0; i < num_elements; ++i) {
      const int tet = node.element_index(i);
      lo = std::min(lo, field.EvaluateMin(tet));
      hi = std::max(hi, field.EvaluateMax(tet));
    }
    PressureRange range{lo, hi};
    (*node_ranges)[&node] = range;
    return range;
  }
  const PressureRange left_range =
      ComputeNodePressureRange(node.left(), field, node_ranges);
  const PressureRange right_range =
      ComputeNodePressureRange(node.right(), field, node_ranges);
  PressureRange range{std::min(left_range.min_val, right_range.min_val),
                      std::max(left_range.max_val, right_range.max_val)};
  (*node_ranges)[&node] = range;
  return range;
}

bool RangesOverlap(const PressureRange& a, const PressureRange& b) {
  return a.max_val >= b.min_val && b.max_val >= a.min_val;
}

struct BvhTraversalStats {
  int total_node_pairs_tested = 0;
  int pruned_by_spatial = 0;
  int prunable_by_pressure_at_node_level = 0;
  int leaf_pairs_reached = 0;
};

void AnalyzeBvhTraversal(const Bvh<Obb, VolumeMesh<double>>& bvh_A,
                         const NodePressureMap& pressure_A,
                         const Bvh<Obb, VolumeMesh<double>>& bvh_B,
                         const NodePressureMap& pressure_B,
                         const RigidTransformd& X_AB,
                         BvhTraversalStats* stats) {
  using NodePair = std::pair<const NodeType&, const NodeType&>;
  std::stack<NodePair, std::vector<NodePair>> node_pairs;
  node_pairs.emplace(bvh_A.root_node(), bvh_B.root_node());

  while (!node_pairs.empty()) {
    const auto& [node_a, node_b] = node_pairs.top();
    node_pairs.pop();

    stats->total_node_pairs_tested++;

    if (!Obb::HasOverlap(node_a.bv(), node_b.bv(), X_AB)) {
      stats->pruned_by_spatial++;
      continue;
    }

    // Spatial overlap passed. Would pressure culling prune this?
    const auto it_a = pressure_A.find(&node_a);
    const auto it_b = pressure_B.find(&node_b);
    DRAKE_DEMAND(it_a != pressure_A.end());
    DRAKE_DEMAND(it_b != pressure_B.end());
    if (!RangesOverlap(it_a->second, it_b->second)) {
      stats->prunable_by_pressure_at_node_level++;
      // Don't descend -- this would be pruned with the optimization.
      // But for counting, we still want to know how many leaf pairs
      // this corresponds to, so we don't skip. We just count and continue
      // the traversal to count descendants too.
      // Actually, for the most useful metric, let's count this as prunable
      // and NOT descend (simulating what the optimization would do).
      continue;
    }

    if (node_a.is_leaf() && node_b.is_leaf()) {
      stats->leaf_pairs_reached++;
    } else if (node_b.is_leaf()) {
      node_pairs.emplace(node_a.left(), node_b);
      node_pairs.emplace(node_a.right(), node_b);
    } else if (node_a.is_leaf()) {
      node_pairs.emplace(node_a, node_b.left());
      node_pairs.emplace(node_a, node_b.right());
    } else {
      node_pairs.emplace(node_a.left(), node_b.left());
      node_pairs.emplace(node_a.right(), node_b.left());
      node_pairs.emplace(node_a.left(), node_b.right());
      node_pairs.emplace(node_a.right(), node_b.right());
    }
  }
}

struct TetPairStats {
  int total_candidates = 0;
  int cullable_by_pressure_range = 0;
  int cullable_by_gradient_direction = 0;
  int cullable_by_either = 0;
  int rejected_at_equilibrium_plane = 0;
  int rejected_at_gradient_check_tet0 = 0;
  int rejected_at_gradient_check_tet1 = 0;
  int rejected_at_intersect_tetrahedra = 0;
  int accepted = 0;
};

void AnalyzeTetPairs(const VolumeMeshFieldLinear<double, double>& field0_M,
                     const Bvh<Obb, VolumeMesh<double>>& bvh0_M,
                     const VolumeMeshFieldLinear<double, double>& field1_N,
                     const Bvh<Obb, VolumeMesh<double>>& bvh1_N,
                     const RigidTransformd& X_MN, TetPairStats* stats) {
  std::vector<std::pair<int, int>> candidate_tetrahedra;
  auto callback = [&candidate_tetrahedra](int tet0,
                                          int tet1) -> BvttCallbackResult {
    candidate_tetrahedra.emplace_back(tet0, tet1);
    return BvttCallbackResult::Continue;
  };
  bvh0_M.Collide(bvh1_N, X_MN, callback);

  stats->total_candidates = static_cast<int>(candidate_tetrahedra.size());

  const math::RotationMatrix<double> R_NM = X_MN.rotation().inverse();

  const math::RotationMatrix<double> R_MN = X_MN.rotation();

  // Gradient direction culling threshold: gradients must be at least
  // this "opposing" for contact. cos(120°) = -0.5 means the angle
  // between gradient directions must be > 120° (surfaces face each
  // other within 60° of head-on).
  constexpr double kGradientCosThreshold = -0.5;  // cos(120°)

  for (const auto& [tet0, tet1] : candidate_tetrahedra) {
    // Check if pressure ranges overlap.
    const double min0 = field0_M.EvaluateMin(tet0);
    const double max0 = field0_M.EvaluateMax(tet0);
    const double min1 = field1_N.EvaluateMin(tet1);
    const double max1 = field1_N.EvaluateMax(tet1);
    const bool pressure_overlap = (max0 >= min1) && (max1 >= min0);
    if (!pressure_overlap) {
      stats->cullable_by_pressure_range++;
    }

    // Check if gradient directions are roughly opposing (surfaces
    // face each other). Gradient direction = inward surface normal.
    const Vector3d grad0_M = field0_M.EvaluateGradient(tet0);
    const Vector3d grad1_N = field1_N.EvaluateGradient(tet1);
    const Vector3d grad1_M = R_MN * grad1_N;
    const double grad0_norm = grad0_M.norm();
    const double grad1_norm = grad1_M.norm();
    bool gradient_opposing = true;
    if (grad0_norm > 1e-10 && grad1_norm > 1e-10) {
      const double cos_angle = grad0_M.dot(grad1_M) / (grad0_norm * grad1_norm);
      // If cos_angle > threshold, gradients are NOT opposing enough.
      if (cos_angle > kGradientCosThreshold) {
        gradient_opposing = false;
        stats->cullable_by_gradient_direction++;
      }
    }

    if (!pressure_overlap || !gradient_opposing) {
      stats->cullable_by_either++;
    }

    // Trace through CalcContactPolygon's early exits.
    Plane<double> equilibrium_plane_M{Vector3d::UnitZ(), Vector3d::Zero()};
    if (!CalcEquilibriumPlane(tet0, field0_M, tet1, field1_N, X_MN,
                              &equilibrium_plane_M)) {
      stats->rejected_at_equilibrium_plane++;
      continue;
    }

    Vector3d polygon_nhat_M = equilibrium_plane_M.unit_normal();
    if (!IsPlaneNormalAlongPressureGradient(polygon_nhat_M, tet0, field0_M)) {
      stats->rejected_at_gradient_check_tet0++;
      continue;
    }

    Vector3d reverse_polygon_nhat_N = R_NM * (-polygon_nhat_M);
    if (!IsPlaneNormalAlongPressureGradient(reverse_polygon_nhat_N, tet1,
                                            field1_N)) {
      stats->rejected_at_gradient_check_tet1++;
      continue;
    }

    const auto [polygon_vertices_M, faces] =
        IntersectTetrahedra(tet0, field0_M.mesh(), tet1, field1_N.mesh(), X_MN,
                            equilibrium_plane_M);

    if (polygon_vertices_M.size() < 3) {
      stats->rejected_at_intersect_tetrahedra++;
      continue;
    }

    stats->accepted++;
  }
}

void RunAnalysis(int resolution, int overlap) {
  const Ellipsoid ellipsoid{kEllipsoidDimension[0], kEllipsoidDimension[1],
                            kEllipsoidDimension[2]};
  const Sphere sphere{kSphereDimension};

  const double resolution_hint = kResolutionHint[resolution];

  auto mesh_S =
      std::make_unique<VolumeMesh<double>>(MakeEllipsoidVolumeMesh<double>(
          ellipsoid, resolution_hint,
          TessellationStrategy::kDenseInteriorVertices));
  auto field_S = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeEllipsoidPressureField<double>(ellipsoid, mesh_S.get(),
                                         kElasticModulus));
  auto mesh_R =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphere, resolution_hint,
          TessellationStrategy::kDenseInteriorVertices));
  auto field_R = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField<double>(sphere, mesh_R.get(), kElasticModulus));

  Bvh<Obb, VolumeMesh<double>> bvh_S(*mesh_S);
  Bvh<Obb, VolumeMesh<double>> bvh_R(*mesh_R);

  const RigidTransformd X_SR{kContactOverlapTranslation[overlap]};

  std::printf(
      "\n===== Resolution %d, Overlap %d =====\n"
      "Ellipsoid: %d tets, %d vertices\n"
      "Sphere:    %d tets, %d vertices\n",
      resolution, overlap, mesh_S->num_elements(), mesh_S->num_vertices(),
      mesh_R->num_elements(), mesh_R->num_vertices());

  // --- BVH node-level analysis ---
  NodePressureMap pressure_map_S;
  ComputeNodePressureRange(bvh_S.root_node(), *field_S, &pressure_map_S);
  NodePressureMap pressure_map_R;
  ComputeNodePressureRange(bvh_R.root_node(), *field_R, &pressure_map_R);

  // Also run without pressure culling to get baseline leaf pair count.
  BvhTraversalStats bvh_stats_no_culling;
  {
    using NodePair = std::pair<const NodeType&, const NodeType&>;
    std::stack<NodePair, std::vector<NodePair>> node_pairs;
    node_pairs.emplace(bvh_S.root_node(), bvh_R.root_node());
    while (!node_pairs.empty()) {
      const auto& [node_a, node_b] = node_pairs.top();
      node_pairs.pop();
      bvh_stats_no_culling.total_node_pairs_tested++;
      if (!Obb::HasOverlap(node_a.bv(), node_b.bv(), X_SR)) {
        bvh_stats_no_culling.pruned_by_spatial++;
        continue;
      }
      if (node_a.is_leaf() && node_b.is_leaf()) {
        bvh_stats_no_culling.leaf_pairs_reached++;
      } else if (node_b.is_leaf()) {
        node_pairs.emplace(node_a.left(), node_b);
        node_pairs.emplace(node_a.right(), node_b);
      } else if (node_a.is_leaf()) {
        node_pairs.emplace(node_a, node_b.left());
        node_pairs.emplace(node_a, node_b.right());
      } else {
        node_pairs.emplace(node_a.left(), node_b.left());
        node_pairs.emplace(node_a.right(), node_b.left());
        node_pairs.emplace(node_a.left(), node_b.right());
        node_pairs.emplace(node_a.right(), node_b.right());
      }
    }
  }

  BvhTraversalStats bvh_stats_with_culling;
  AnalyzeBvhTraversal(bvh_S, pressure_map_S, bvh_R, pressure_map_R, X_SR,
                      &bvh_stats_with_culling);

  std::printf(
      "\n--- BVH Traversal (no pressure culling) ---\n"
      "  Node pairs tested:       %d\n"
      "  Pruned by spatial:       %d\n"
      "  Leaf pairs reached:      %d\n",
      bvh_stats_no_culling.total_node_pairs_tested,
      bvh_stats_no_culling.pruned_by_spatial,
      bvh_stats_no_culling.leaf_pairs_reached);

  std::printf(
      "\n--- BVH Traversal (with pressure culling) ---\n"
      "  Node pairs tested:       %d\n"
      "  Pruned by spatial:       %d\n"
      "  Pruned by pressure:      %d\n"
      "  Leaf pairs reached:      %d\n"
      "  Leaf pair reduction:     %.1f%%\n",
      bvh_stats_with_culling.total_node_pairs_tested,
      bvh_stats_with_culling.pruned_by_spatial,
      bvh_stats_with_culling.prunable_by_pressure_at_node_level,
      bvh_stats_with_culling.leaf_pairs_reached,
      bvh_stats_no_culling.leaf_pairs_reached > 0
          ? 100.0 * (1.0 - static_cast<double>(
                               bvh_stats_with_culling.leaf_pairs_reached) /
                               bvh_stats_no_culling.leaf_pairs_reached)
          : 0.0);

  // --- Tet pair analysis ---
  if (overlap >= 2) {
    TetPairStats tet_stats;
    AnalyzeTetPairs(*field_S, bvh_S, *field_R, bvh_R, X_SR, &tet_stats);

    std::printf(
        "\n--- Tet Pair Analysis ---\n"
        "  Total BVH candidates:            %d\n"
        "  Cullable by pressure range:      %d (%.1f%%)\n"
        "  Cullable by gradient direction:  %d (%.1f%%)\n"
        "  Cullable by either:              %d (%.1f%%)\n"
        "  Rejected at CalcEquilibPlane:    %d (%.1f%%)\n"
        "  Rejected at gradient check tet0: %d (%.1f%%)\n"
        "  Rejected at gradient check tet1: %d (%.1f%%)\n"
        "  Rejected at IntersectTetrahedra: %d (%.1f%%)\n"
        "  Accepted (produced polygon):     %d (%.1f%%)\n",
        tet_stats.total_candidates, tet_stats.cullable_by_pressure_range,
        100.0 * tet_stats.cullable_by_pressure_range /
            std::max(1, tet_stats.total_candidates),
        tet_stats.cullable_by_gradient_direction,
        100.0 * tet_stats.cullable_by_gradient_direction /
            std::max(1, tet_stats.total_candidates),
        tet_stats.cullable_by_either,
        100.0 * tet_stats.cullable_by_either /
            std::max(1, tet_stats.total_candidates),
        tet_stats.rejected_at_equilibrium_plane,
        100.0 * tet_stats.rejected_at_equilibrium_plane /
            std::max(1, tet_stats.total_candidates),
        tet_stats.rejected_at_gradient_check_tet0,
        100.0 * tet_stats.rejected_at_gradient_check_tet0 /
            std::max(1, tet_stats.total_candidates),
        tet_stats.rejected_at_gradient_check_tet1,
        100.0 * tet_stats.rejected_at_gradient_check_tet1 /
            std::max(1, tet_stats.total_candidates),
        tet_stats.rejected_at_intersect_tetrahedra,
        100.0 * tet_stats.rejected_at_intersect_tetrahedra /
            std::max(1, tet_stats.total_candidates),
        tet_stats.accepted,
        100.0 * tet_stats.accepted / std::max(1, tet_stats.total_candidates));

    // Cross-reference: of the rejected pairs, how many would have been
    // caught by pressure culling?
    const int total_rejected = tet_stats.total_candidates - tet_stats.accepted;
    std::printf(
        "\n--- Cross-reference ---\n"
        "  Total rejected pairs:            %d\n"
        "  Of which, cullable by pressure:  %d (%.1f%%)\n"
        "  NOT cullable by pressure:        %d (%.1f%%)\n",
        total_rejected, tet_stats.cullable_by_pressure_range,
        100.0 * tet_stats.cullable_by_pressure_range /
            std::max(1, total_rejected),
        total_rejected - tet_stats.cullable_by_pressure_range,
        100.0 * (total_rejected - tet_stats.cullable_by_pressure_range) /
            std::max(1, total_rejected));
  } else {
    std::printf(
        "\n--- Tet Pair Analysis ---\n"
        "  Skipped (no contact expected at overlap %d)\n",
        overlap);
  }
}

int main() {
  // Run analysis for overlap levels 2 (small contact) and 3 (intermediate).
  // Run at all resolutions.
  for (int resolution = 0; resolution <= 3; ++resolution) {
    for (int overlap = 2; overlap <= 3; ++overlap) {
      RunAnalysis(resolution, overlap);
    }
  }
  return 0;
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake

int main() {
  return drake::geometry::internal::main();
}
