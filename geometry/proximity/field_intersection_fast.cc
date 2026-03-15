#include "drake/geometry/proximity/field_intersection_fast.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace {

using TetrahedronEdge = std::pair<int, int>;
constexpr std::array<std::pair<int, int>, 6> kTetEdges = {
    TetrahedronEdge{0, 1}, TetrahedronEdge{1, 2}, TetrahedronEdge{2, 0},
    TetrahedronEdge{0, 3}, TetrahedronEdge{1, 3}, TetrahedronEdge{2, 3}};

// clang-format off
constexpr std::array<std::array<int, 4>, 16> kMarchingTetsEdgeTable = {
    std::array<int, 4>{-1, -1, -1, -1},
    std::array<int, 4>{0, 3, 2, -1},
    std::array<int, 4>{0, 1, 4, -1},
    std::array<int, 4>{4, 3, 2, 1},
    std::array<int, 4>{1, 2, 5, -1},
    std::array<int, 4>{0, 3, 5, 1},
    std::array<int, 4>{0, 2, 5, 4},
    std::array<int, 4>{3, 5, 4, -1},
    std::array<int, 4>{3, 4, 5, -1},
    std::array<int, 4>{4, 5, 2, 0},
    std::array<int, 4>{1, 5, 3, 0},
    std::array<int, 4>{1, 5, 2, -1},
    std::array<int, 4>{1, 2, 3, 4},
    std::array<int, 4>{0, 4, 1, -1},
    std::array<int, 4>{0, 2, 3, -1},
    std::array<int, 4>{-1, -1, -1, -1}};

constexpr std::array<std::array<int, 4>, 16> kMarchingTetsFaceTable = {
    std::array<int, 4>{-1, -1, -1, -1},
    std::array<int, 4>{2, 1, 3, -1},
    std::array<int, 4>{3, 0, 2, -1},
    std::array<int, 4>{2, 1, 3, 0},
    std::array<int, 4>{3, 1, 0, -1},
    std::array<int, 4>{2, 1, 0, 3},
    std::array<int, 4>{3, 1, 0, 2},
    std::array<int, 4>{1, 0, 2, -1},
    std::array<int, 4>{2, 0, 1, -1},
    std::array<int, 4>{0, 1, 3, 2},
    std::array<int, 4>{0, 1, 2, 3},
    std::array<int, 4>{0, 1, 3, -1},
    std::array<int, 4>{3, 1, 2, 0},
    std::array<int, 4>{2, 0, 3, -1},
    std::array<int, 4>{3, 1, 2, -1},
    std::array<int, 4>{-1, -1, -1, -1}};
// clang-format on

int RemoveDuplicates(
    std::array<Vector3d, FastVolumeIntersector::kMaxPolygonSize>* polygon,
    int size) {
  if (size <= 1) return size;
  constexpr double kEpsSquared = 1e-14 * 1e-14;
  int write = 0;
  for (int read = 0; read < size; ++read) {
    const int next = (read + 1) % size;
    const double dist_sq = ((*polygon)[read] - (*polygon)[next]).squaredNorm();
    if (dist_sq > kEpsSquared) {
      (*polygon)[write++] = (*polygon)[read];
    }
  }
  return write;
}

}  // namespace

bool FastVolumeIntersector::CalcEquilibriumPlaneInline(
    int element0, const VolumeMeshFieldLinear<double, double>& field0_M,
    int element1, const VolumeMeshFieldLinear<double, double>& field1_N,
    const Matrix3d& R_MN, const Vector3d& t_MN, const Matrix3d& R_NM,
    Vector3d* nhat_M, double* displacement) {
  const Vector3d& grad_f0_M = field0_M.EvaluateGradient(element0);
  const double f0_Mo = field0_M.EvaluateAtMo(element0);

  const Vector3d& grad_f1_N = field1_N.EvaluateGradient(element1);
  const Vector3d grad_f1_M = R_MN * grad_f1_N;

  const Vector3d p_NMo = -(R_NM * t_MN);
  const double f1_Mo = field1_N.EvaluateCartesian(element1, p_NMo);

  const Vector3d n_M = grad_f0_M - grad_f1_M;
  const double magnitude = n_M.norm();
  if (magnitude <= 0.0) return false;

  *nhat_M = n_M / magnitude;
  *displacement = (f1_Mo - f0_Mo) / magnitude;
  return true;
}

bool FastVolumeIntersector::IsPlaneNormalAlongPressureGradientInline(
    const Vector3d& nhat_M, int tetrahedron,
    const VolumeMeshFieldLinear<double, double>& field_M) {
  const Vector3d& grad_p_M = field_M.EvaluateGradient(tetrahedron);
  const double grad_norm = grad_p_M.norm();
  if (grad_norm <= 0.0) return false;
  const double cos_theta = nhat_M.dot(grad_p_M) / grad_norm;
  constexpr double kAlpha = 5. * M_PI / 8.;
  static const double kCosAlpha = std::cos(kAlpha);
  return cos_theta > kCosAlpha;
}

int FastVolumeIntersector::IntersectTetrahedraInline(
    int element0, const VolumeMesh<double>& mesh0_M, int element1,
    const VolumeMesh<double>& mesh1_N, const Matrix3d& R_MN,
    const Vector3d& t_MN, const Vector3d& nhat_M, double displacement,
    ThreadLocalData* local) {
  auto& poly_a = local->polygon_buffer_a;
  auto& poly_b = local->polygon_buffer_b;
  auto& face_a = local->face_buffer_a;
  auto& face_b = local->face_buffer_b;
  int& size_a = local->polygon_size_a;
  int& size_b = local->polygon_size_b;
  int& fsize_a = local->face_size_a;
  int& fsize_b = local->face_size_b;
  size_a = 0;
  size_b = 0;
  fsize_a = 0;
  fsize_b = 0;

  const auto& elem0 = mesh0_M.element(element0);
  double dist[4];
  int intersection_code = 0;
  for (int i = 0; i < 4; ++i) {
    const Vector3d& v = mesh0_M.vertex(elem0.vertex(i));
    dist[i] = nhat_M.dot(v) - displacement;
    if (dist[i] > 0.0) intersection_code |= 1 << i;
  }

  if (kMarchingTetsEdgeTable[intersection_code][0] == -1) return 0;

  for (int i = 0; i < 4; ++i) {
    const int edge_index = kMarchingTetsEdgeTable[intersection_code][i];
    if (edge_index == -1) break;

    const auto& tet_edge = kTetEdges[edge_index];
    const Vector3d& p0 = mesh0_M.vertex(elem0.vertex(tet_edge.first));
    const Vector3d& p1 = mesh0_M.vertex(elem0.vertex(tet_edge.second));
    const double d0 = dist[tet_edge.first];
    const double d1 = dist[tet_edge.second];
    const double t = d0 / (d0 - d1);
    poly_a[size_a++] = p0 + t * (p1 - p0);
    face_a[fsize_a++] = kMarchingTetsFaceTable[intersection_code][i];
  }

  size_a = RemoveDuplicates(&poly_a, size_a);
  if (size_a < 3) return 0;

  const auto& elem1 = mesh1_N.element(element1);
  Vector3d p_MVs[4];
  for (int i = 0; i < 4; ++i) {
    const Vector3d& v_N = mesh1_N.vertex(elem1.vertex(i));
    p_MVs[i].noalias() = R_MN * v_N + t_MN;
  }

  Vector3d outward_normals_M[4];
  for (int face = 0; face < 4; ++face) {
    outward_normals_M[face].noalias() =
        -(R_MN * mesh1_N.inward_normal(element1, face));
  }

  auto* current_poly = &poly_a;
  auto* clipped_poly = &poly_b;
  auto* current_face = &face_a;
  auto* clipped_face = &face_b;
  int* current_size = &size_a;
  int* clipped_size = &size_b;
  int* current_fsize = &fsize_a;
  int* clipped_fsize = &fsize_b;

  std::array<double, kMaxPolygonSize> distances;

  for (int face = 0; face < 4; ++face) {
    *clipped_size = 0;
    *clipped_fsize = 0;

    const Vector3d& p_MA = p_MVs[(face + 1) % 4];
    const Vector3d& normal_M = outward_normals_M[face];
    const double plane_d = normal_M.dot(p_MA);

    const int sz = *current_size;
    for (int i = 0; i < sz; ++i) {
      distances[i] = normal_M.dot((*current_poly)[i]) - plane_d;
    }

    for (int i = 0; i < sz; ++i) {
      const int j = (i + 1) % sz;
      if (distances[i] <= 0) {
        (*clipped_poly)[(*clipped_size)++] = (*current_poly)[i];
        (*clipped_face)[(*clipped_fsize)++] = (*current_face)[i];
        if (distances[j] > 0) {
          const double wa = distances[j] / (distances[j] - distances[i]);
          const double wb = 1.0 - wa;
          (*clipped_poly)[(*clipped_size)++] =
              wa * (*current_poly)[i] + wb * (*current_poly)[j];
          (*clipped_face)[(*clipped_fsize)++] = face + 4;
        }
      } else if (distances[j] <= 0) {
        const double wa = distances[j] / (distances[j] - distances[i]);
        const double wb = 1.0 - wa;
        (*clipped_poly)[(*clipped_size)++] =
            wa * (*current_poly)[i] + wb * (*current_poly)[j];
        (*clipped_face)[(*clipped_fsize)++] = (*current_face)[i];
      }
    }
    std::swap(current_poly, clipped_poly);
    std::swap(current_face, clipped_face);
    std::swap(current_size, clipped_size);
    std::swap(current_fsize, clipped_fsize);

    if (*current_size == 0) return 0;
  }

  *current_size = RemoveDuplicates(current_poly, *current_size);
  if (*current_size < 3) return 0;

  if (current_poly != &poly_a) {
    const int n = *current_size;
    for (int i = 0; i < n; ++i) {
      poly_a[i] = (*current_poly)[i];
      face_a[i] = (*current_face)[i];
    }
    size_a = n;
    fsize_a = n;
  }
  return size_a;
}

void FastVolumeIntersector::ProcessSingleCandidate(
    const VolumeMeshFieldLinear<double, double>& field0_M,
    const VolumeMeshFieldLinear<double, double>& field1_N, const Matrix3d& R_MN,
    const Vector3d& t_MN, const Matrix3d& R_NM,
    const std::pair<int, int>& candidate, ThreadLocalData* local) {
  const auto& [tet0, tet1] = candidate;

  Vector3d nhat_M;
  double displacement;
  if (!CalcEquilibriumPlaneInline(tet0, field0_M, tet1, field1_N, R_MN, t_MN,
                                  R_NM, &nhat_M, &displacement)) {
    return;
  }

  if (!IsPlaneNormalAlongPressureGradientInline(nhat_M, tet0, field0_M)) {
    return;
  }
  const Vector3d reverse_nhat_N = R_NM * (-nhat_M);
  if (!IsPlaneNormalAlongPressureGradientInline(reverse_nhat_N, tet1,
                                                field1_N)) {
    return;
  }

  const int num_vertices =
      IntersectTetrahedraInline(tet0, field0_M.mesh(), tet1, field1_N.mesh(),
                                R_MN, t_MN, nhat_M, displacement, local);
  if (num_vertices < 3) return;

  const int vertex_offset = static_cast<int>(local->vertices.size());
  local->face_data.push_back(num_vertices);
  for (int i = 0; i < num_vertices; ++i) {
    const Vector3d& p_MV = local->polygon_buffer_a[i];
    local->vertices.push_back(p_MV);
    local->pressures.push_back(field0_M.EvaluateCartesian(tet0, p_MV));
    local->face_data.push_back(vertex_offset + i);
  }
  local->polygon_count++;
  local->grad_e_per_face.push_back(field0_M.EvaluateGradient(tet0));
  local->tet0_indices.push_back(tet0);
  local->tet1_indices.push_back(tet1);
}

void FastVolumeIntersector::MergeThreadResults(
    int num_threads, std::unique_ptr<PolygonSurfaceMesh<double>>* surface_M,
    std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>>* e_M) {
  int total_vertices = 0;
  int total_polygons = 0;
  int total_face_data = 0;
  for (int t = 0; t < num_threads; ++t) {
    const auto& td = thread_data_[t];
    total_vertices += static_cast<int>(td.vertices.size());
    total_polygons += td.polygon_count;
    total_face_data += static_cast<int>(td.face_data.size());
  }

  if (total_polygons == 0) return;

  std::vector<Vector3d> all_vertices;
  all_vertices.reserve(total_vertices);
  std::vector<double> all_pressures;
  all_pressures.reserve(total_vertices);
  std::vector<int> all_face_data;
  all_face_data.reserve(total_face_data);
  std::vector<Vector3d> all_gradients;
  all_gradients.reserve(total_polygons);

  tet0_of_contact_polygon_.clear();
  tet1_of_contact_polygon_.clear();
  tet0_of_contact_polygon_.reserve(total_polygons);
  tet1_of_contact_polygon_.reserve(total_polygons);

  for (int t = 0; t < num_threads; ++t) {
    const auto& td = thread_data_[t];
    const int vertex_offset = static_cast<int>(all_vertices.size());
    all_vertices.insert(all_vertices.end(), td.vertices.begin(),
                        td.vertices.end());
    all_pressures.insert(all_pressures.end(), td.pressures.begin(),
                         td.pressures.end());

    size_t i = 0;
    while (i < td.face_data.size()) {
      const int n = td.face_data[i];
      all_face_data.push_back(n);
      for (int j = 1; j <= n; ++j) {
        all_face_data.push_back(td.face_data[i + j] + vertex_offset);
      }
      i += n + 1;
    }

    all_gradients.insert(all_gradients.end(), td.grad_e_per_face.begin(),
                         td.grad_e_per_face.end());
    tet0_of_contact_polygon_.insert(tet0_of_contact_polygon_.end(),
                                    td.tet0_indices.begin(),
                                    td.tet0_indices.end());
    tet1_of_contact_polygon_.insert(tet1_of_contact_polygon_.end(),
                                    td.tet1_indices.begin(),
                                    td.tet1_indices.end());
  }

  *surface_M = std::make_unique<PolygonSurfaceMesh<double>>(
      std::move(all_face_data), std::move(all_vertices));

  *e_M = std::make_unique<PolygonSurfaceMeshFieldLinear<double, double>>(
      std::move(all_pressures), surface_M->get(), std::move(all_gradients));
}

void FastVolumeIntersector::IntersectFields(
    const VolumeMeshFieldLinear<double, double>& field0_M,
    const Bvh<Aabb, VolumeMesh<double>>& aabb_bvh0_M,
    const VolumeMeshFieldLinear<double, double>& field1_N,
    const Bvh<Aabb, VolumeMesh<double>>& aabb_bvh1_N,
    const math::RigidTransformd& X_MN,
    std::unique_ptr<PolygonSurfaceMesh<double>>* surface_M,
    std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>>* e_M) {
  surface_M->reset();
  e_M->reset();

  candidates_.clear();
  CollectCandidatesAabbFast(aabb_bvh0_M, aabb_bvh1_N, field0_M, field1_N, X_MN,
                            &candidates_);

  if (candidates_.empty()) return;

  const Matrix3d R_MN_mat = X_MN.rotation().matrix();
  const Vector3d t_MN_vec = X_MN.translation();
  const Matrix3d R_NM = R_MN_mat.transpose();

  const int num_candidates = static_cast<int>(candidates_.size());
  const int num_threads = Parallelism::Max().num_threads();

  // Ensure we have enough ThreadLocalData entries.
  if (static_cast<int>(thread_data_.size()) < num_threads) {
    thread_data_.resize(num_threads);
  }
  for (int t = 0; t < num_threads; ++t) {
    thread_data_[t].ClearOutputs();
  }

  // Pre-reserve output vectors based on expected hit rate (~16%).
  const int expected_polygons = std::max(1, num_candidates / (6 * num_threads));
  for (int t = 0; t < num_threads; ++t) {
    thread_data_[t].vertices.reserve(expected_polygons * 4);
    thread_data_[t].pressures.reserve(expected_polygons * 4);
    thread_data_[t].face_data.reserve(expected_polygons * 5);
    thread_data_[t].grad_e_per_face.reserve(expected_polygons);
    thread_data_[t].tet0_indices.reserve(expected_polygons);
    thread_data_[t].tet1_indices.reserve(expected_polygons);
  }

#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads) schedule(static, 1)
#endif
  for (int idx = 0; idx < num_candidates; ++idx) {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    ProcessSingleCandidate(field0_M, field1_N, R_MN_mat, t_MN_vec, R_NM,
                           candidates_[idx], &thread_data_[tid]);
  }

  if (num_threads == 1) {
    auto& td = thread_data_[0];
    if (td.polygon_count == 0) return;

    tet0_of_contact_polygon_.swap(td.tet0_indices);
    tet1_of_contact_polygon_.swap(td.tet1_indices);

    *surface_M = std::make_unique<PolygonSurfaceMesh<double>>(
        std::move(td.face_data), std::move(td.vertices));

    *e_M = std::make_unique<PolygonSurfaceMeshFieldLinear<double, double>>(
        std::move(td.pressures), surface_M->get(),
        std::move(td.grad_e_per_face));
  } else {
    MergeThreadResults(num_threads, surface_M, e_M);
  }
}

void FastVolumeIntersector::AabbOverlapPrecomputed::Init(
    const math::RigidTransformd& X_MN) {
  R = X_MN.rotation().matrix();
  t = X_MN.translation();
  abs_R = R.cwiseAbs().array() + std::numeric_limits<double>::epsilon();
}

bool FastVolumeIntersector::AabbOverlapPrecomputed::TestOverlap(
    const Aabb& a, const Aabb& b) const {
  const Vector3d t_AB = R * b.center() + t - a.center();
  const Vector3d& ha = a.half_width();
  const Vector3d& hb = b.half_width();

  for (int i = 0; i < 3; ++i) {
    if (std::abs(t_AB[i]) > ha[i] + abs_R.row(i).dot(hb)) return false;
  }
  for (int i = 0; i < 3; ++i) {
    if (std::abs(t_AB.dot(R.col(i))) > hb[i] + abs_R.col(i).dot(ha))
      return false;
  }
  if (std::abs(t_AB[2] * R(1, 0) - t_AB[1] * R(2, 0)) >
      ha[1] * abs_R(2, 0) + ha[2] * abs_R(1, 0) + hb[1] * abs_R(0, 2) +
          hb[2] * abs_R(0, 1))
    return false;
  if (std::abs(t_AB[2] * R(1, 1) - t_AB[1] * R(2, 1)) >
      ha[1] * abs_R(2, 1) + ha[2] * abs_R(1, 1) + hb[0] * abs_R(0, 2) +
          hb[2] * abs_R(0, 0))
    return false;
  if (std::abs(t_AB[2] * R(1, 2) - t_AB[1] * R(2, 2)) >
      ha[1] * abs_R(2, 2) + ha[2] * abs_R(1, 2) + hb[0] * abs_R(0, 1) +
          hb[1] * abs_R(0, 0))
    return false;
  if (std::abs(t_AB[0] * R(2, 0) - t_AB[2] * R(0, 0)) >
      ha[0] * abs_R(2, 0) + ha[2] * abs_R(0, 0) + hb[1] * abs_R(1, 2) +
          hb[2] * abs_R(1, 1))
    return false;
  if (std::abs(t_AB[0] * R(2, 1) - t_AB[2] * R(0, 1)) >
      ha[0] * abs_R(2, 1) + ha[2] * abs_R(0, 1) + hb[0] * abs_R(1, 2) +
          hb[2] * abs_R(1, 0))
    return false;
  if (std::abs(t_AB[0] * R(2, 2) - t_AB[2] * R(0, 2)) >
      ha[0] * abs_R(2, 2) + ha[2] * abs_R(0, 2) + hb[0] * abs_R(1, 1) +
          hb[1] * abs_R(1, 0))
    return false;
  if (std::abs(t_AB[1] * R(0, 0) - t_AB[0] * R(1, 0)) >
      ha[0] * abs_R(1, 0) + ha[1] * abs_R(0, 0) + hb[1] * abs_R(2, 2) +
          hb[2] * abs_R(2, 1))
    return false;
  if (std::abs(t_AB[1] * R(0, 1) - t_AB[0] * R(1, 1)) >
      ha[0] * abs_R(1, 1) + ha[1] * abs_R(0, 1) + hb[0] * abs_R(2, 2) +
          hb[2] * abs_R(2, 0))
    return false;
  if (std::abs(t_AB[1] * R(0, 2) - t_AB[0] * R(1, 2)) >
      ha[0] * abs_R(1, 2) + ha[1] * abs_R(0, 2) + hb[0] * abs_R(2, 1) +
          hb[1] * abs_R(2, 0))
    return false;

  return true;
}

void FastVolumeIntersector::CollectCandidatesAabbFast(
    const Bvh<Aabb, VolumeMesh<double>>& bvh0_M,
    const Bvh<Aabb, VolumeMesh<double>>& bvh1_N,
    const VolumeMeshFieldLinear<double, double>& field0_M,
    const VolumeMeshFieldLinear<double, double>& field1_N,
    const math::RigidTransformd& X_MN,
    std::vector<std::pair<int, int>>* candidates) {
  using NodeType = typename Bvh<Aabb, VolumeMesh<double>>::NodeType;
  using NodePair = std::pair<const NodeType*, const NodeType*>;

  AabbOverlapPrecomputed overlap_test;
  overlap_test.Init(X_MN);

  const auto& min0 = field0_M.min_values();
  const auto& max0 = field0_M.max_values();
  const auto& min1 = field1_N.min_values();
  const auto& max1 = field1_N.max_values();

  std::array<NodePair, kMaxBvhStackDepth> stack_data;
  int stack_top = 0;
  stack_data[stack_top++] = {&bvh0_M.root_node(), &bvh1_N.root_node()};

  while (stack_top > 0) {
    const auto [node_a, node_b] = stack_data[--stack_top];

    if (!overlap_test.TestOverlap(node_a->bv(), node_b->bv())) {
      continue;
    }

    if (node_a->is_leaf() && node_b->is_leaf()) {
      const int num_a = node_a->num_element_indices();
      const int num_b = node_b->num_element_indices();
      for (int a = 0; a < num_a; ++a) {
        const int tet0 = node_a->element_index(a);
        for (int b = 0; b < num_b; ++b) {
          const int tet1 = node_b->element_index(b);
          if (min0[tet0] <= max1[tet1] && min1[tet1] <= max0[tet0]) {
            candidates->emplace_back(tet0, tet1);
          }
        }
      }
    } else if (node_b->is_leaf()) {
      stack_data[stack_top++] = {&node_a->left(), node_b};
      stack_data[stack_top++] = {&node_a->right(), node_b};
    } else if (node_a->is_leaf()) {
      stack_data[stack_top++] = {node_a, &node_b->left()};
      stack_data[stack_top++] = {node_a, &node_b->right()};
    } else {
      if (node_a->bv().CalcVolume() >= node_b->bv().CalcVolume()) {
        stack_data[stack_top++] = {&node_a->left(), node_b};
        stack_data[stack_top++] = {&node_a->right(), node_b};
      } else {
        stack_data[stack_top++] = {node_a, &node_b->left()};
        stack_data[stack_top++] = {node_a, &node_b->right()};
      }
    }
  }
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
