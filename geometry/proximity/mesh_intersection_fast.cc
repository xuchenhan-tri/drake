#include "drake/geometry/proximity/mesh_intersection_fast.h"

#include <algorithm>
#include <array>
#include <cmath>
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

int RemoveDuplicates(
    std::array<Vector3d, FastSurfaceVolumeIntersector::kMaxPolygonSize + 1>*
        polygon,
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

bool FastSurfaceVolumeIntersector::IsFaceNormalAlongPressureGradientInline(
    const Vector3d& grad_p_S, const Vector3d& face_normal_S) {
  const double grad_sq = grad_p_S.squaredNorm();
  if (grad_sq <= 0.0) return false;
  const double dot = grad_p_S.dot(face_normal_S);
  // cos(theta) = dot / (|grad| * |face_normal|). We check cos(theta) >
  // cos(5*pi/8). Since cos(5*pi/8) < 0, if dot >= 0, the condition is
  // always true because the face normal is unit-length (within tolerance).
  if (dot >= 0.0) return true;
  // Both dot and cos(alpha) are negative. Squaring flips the inequality:
  // dot^2 / (grad_sq * fn_sq) < cos_alpha^2.
  constexpr double kAlpha = 5.0 * M_PI / 8.0;
  static const double kCosAlphaSq = std::cos(kAlpha) * std::cos(kAlpha);
  const double fn_sq = face_normal_S.squaredNorm();
  return dot * dot < kCosAlphaSq * grad_sq * fn_sq;
}

int FastSurfaceVolumeIntersector::ClipTriangleByTetrahedronInline(
    int tet_index, const VolumeMesh<double>& volume_S, int tri_index,
    const TriangleSurfaceMesh<double>& surface_R, const Matrix3d& R_SR,
    const Vector3d& t_SR, ThreadLocalData* local) {
  auto& poly_a = local->clip_buffer_a;
  auto& poly_b = local->clip_buffer_b;
  int& size_a = local->clip_size_a;
  int& size_b = local->clip_size_b;

  // Initialize polygon from the triangle's vertices, transformed to S frame.
  size_a = 3;
  for (int i = 0; i < 3; ++i) {
    const int v = surface_R.element(tri_index).vertex(i);
    const Vector3d& p_RV = surface_R.vertex(v);
    poly_a[i].noalias() = R_SR * p_RV + t_SR;
  }

  // Get tetrahedron vertices in S frame (they're already in S).
  const auto& elem = volume_S.element(tet_index);
  Vector3d p_SVs[4];
  for (int i = 0; i < 4; ++i) {
    p_SVs[i] = volume_S.vertex(elem.vertex(i));
  }

  // Four faces of the tetrahedron with outward-pointing normals.
  constexpr int faces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

  auto* current_poly = &poly_a;
  auto* clipped_poly = &poly_b;
  int* current_size = &size_a;
  int* clipped_size = &size_b;

  std::array<double, kMaxPolygonSize + 1> distances;

  for (int f = 0; f < 4; ++f) {
    *clipped_size = 0;

    const Vector3d& p_SA = p_SVs[faces[f][0]];
    const Vector3d& p_SB = p_SVs[faces[f][1]];
    const Vector3d& p_SC = p_SVs[faces[f][2]];
    // Unnormalized outward normal of this face.
    const Vector3d normal_S = (p_SB - p_SA).cross(p_SC - p_SA);
    // plane_d = normal.dot(point_on_plane)
    const double plane_d = normal_S.dot(p_SA);

    const int sz = *current_size;
    for (int i = 0; i < sz; ++i) {
      // Signed distance: positive means outside the tet (above the plane).
      distances[i] = normal_S.dot((*current_poly)[i]) - plane_d;
    }

    for (int i = 0; i < sz; ++i) {
      const int j = (i + 1) % sz;
      if (distances[i] <= 0) {
        // Current vertex is inside.
        (*clipped_poly)[(*clipped_size)++] = (*current_poly)[i];
        if (distances[j] > 0) {
          // Next vertex is outside -- compute intersection.
          const double wa = distances[j] / (distances[j] - distances[i]);
          const double wb = 1.0 - wa;
          (*clipped_poly)[(*clipped_size)++] =
              wa * (*current_poly)[i] + wb * (*current_poly)[j];
        }
      } else if (distances[j] <= 0) {
        // Current outside, next inside -- compute intersection.
        const double wa = distances[j] / (distances[j] - distances[i]);
        const double wb = 1.0 - wa;
        (*clipped_poly)[(*clipped_size)++] =
            wa * (*current_poly)[i] + wb * (*current_poly)[j];
      }
    }

    std::swap(current_poly, clipped_poly);
    std::swap(current_size, clipped_size);

    if (*current_size == 0) return 0;
  }

  *current_size = RemoveDuplicates(current_poly, *current_size);
  if (*current_size < 3) return 0;

  // Ensure result is in poly_a.
  if (current_poly != &poly_a) {
    const int n = *current_size;
    for (int i = 0; i < n; ++i) {
      poly_a[i] = (*current_poly)[i];
    }
    size_a = n;
  }
  return size_a;
}

void FastSurfaceVolumeIntersector::ProcessSingleCandidate(
    const VolumeMeshFieldLinear<double, double>& volume_field_S,
    const TriangleSurfaceMesh<double>& surface_R, const Matrix3d& R_SR,
    const Vector3d& t_SR, int tet_index, int tri_index,
    ThreadLocalData* local) {
  // Face normal direction check.
  const Vector3d& grad_p_S = volume_field_S.EvaluateGradient(tet_index);
  // Rotate the triangle face normal from R frame to S frame.
  const Vector3d face_normal_S = R_SR * surface_R.face_normal(tri_index);

  if (!IsFaceNormalAlongPressureGradientInline(grad_p_S, face_normal_S)) {
    return;
  }

  // Clip triangle by tetrahedron.
  const int num_vertices =
      ClipTriangleByTetrahedronInline(tet_index, volume_field_S.mesh(),
                                      tri_index, surface_R, R_SR, t_SR, local);
  if (num_vertices < 3) return;

  // Add polygon to thread-local output.
  const int vertex_offset = static_cast<int>(local->vertices.size());
  local->face_data.push_back(num_vertices);
  for (int i = 0; i < num_vertices; ++i) {
    const Vector3d& p_SV = local->clip_buffer_a[i];
    local->vertices.push_back(p_SV);
    local->pressures.push_back(
        volume_field_S.EvaluateCartesian(tet_index, p_SV));
    local->face_data.push_back(vertex_offset + i);
  }
  local->polygon_count++;
  local->grad_e_per_face.push_back(grad_p_S);
}

void FastSurfaceVolumeIntersector::MergeThreadResults(
    int num_threads, std::unique_ptr<PolygonSurfaceMesh<double>>* surface_SR,
    std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>>* field_SR) {
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

  grad_eS_S_.clear();
  grad_eS_S_.reserve(total_polygons);

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

    grad_eS_S_.insert(grad_eS_S_.end(), td.grad_e_per_face.begin(),
                      td.grad_e_per_face.end());
  }

  *surface_SR = std::make_unique<PolygonSurfaceMesh<double>>(
      std::move(all_face_data), std::move(all_vertices));

  std::vector<Vector3d> field_gradients(grad_eS_S_);
  *field_SR = std::make_unique<PolygonSurfaceMeshFieldLinear<double, double>>(
      std::move(all_pressures), surface_SR->get(), std::move(field_gradients));
}

void FastSurfaceVolumeIntersector::CollectCandidates(
    const Bvh<Obb, VolumeMesh<double>>& bvh_S,
    const Bvh<Obb, TriangleSurfaceMesh<double>>& bvh_R,
    const math::RigidTransformd& X_SR,
    std::vector<std::pair<int, int>>* candidates) {
  using NodeTypeS = typename Bvh<Obb, VolumeMesh<double>>::NodeType;
  using NodeTypeR = typename Bvh<Obb, TriangleSurfaceMesh<double>>::NodeType;
  using NodePair = std::pair<const NodeTypeS*, const NodeTypeR*>;

  std::array<NodePair, kMaxBvhStackDepth> stack_data;
  int stack_top = 0;
  stack_data[stack_top++] = {&bvh_S.root_node(), &bvh_R.root_node()};

  while (stack_top > 0) {
    const auto [node_s, node_r] = stack_data[--stack_top];

    if (!Obb::HasOverlap(node_s->bv(), node_r->bv(), X_SR)) {
      continue;
    }

    if (node_s->is_leaf() && node_r->is_leaf()) {
      const int num_s = node_s->num_element_indices();
      const int num_r = node_r->num_element_indices();
      for (int s = 0; s < num_s; ++s) {
        const int tet = node_s->element_index(s);
        for (int r = 0; r < num_r; ++r) {
          const int tri = node_r->element_index(r);
          candidates->emplace_back(tet, tri);
        }
      }
    } else if (node_r->is_leaf()) {
      stack_data[stack_top++] = {&node_s->left(), node_r};
      stack_data[stack_top++] = {&node_s->right(), node_r};
    } else if (node_s->is_leaf()) {
      stack_data[stack_top++] = {node_s, &node_r->left()};
      stack_data[stack_top++] = {node_s, &node_r->right()};
    } else {
      stack_data[stack_top++] = {&node_s->left(), &node_r->left()};
      stack_data[stack_top++] = {&node_s->right(), &node_r->left()};
      stack_data[stack_top++] = {&node_s->left(), &node_r->right()};
      stack_data[stack_top++] = {&node_s->right(), &node_r->right()};
    }
  }
}

void FastSurfaceVolumeIntersector::IntersectSurfaceVolume(
    const VolumeMeshFieldLinear<double, double>& volume_field_S,
    const Bvh<Obb, VolumeMesh<double>>& bvh_S,
    const TriangleSurfaceMesh<double>& surface_R,
    const Bvh<Obb, TriangleSurfaceMesh<double>>& bvh_R,
    const math::RigidTransformd& X_SR,
    std::unique_ptr<PolygonSurfaceMesh<double>>* surface_SR,
    std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>>* field_SR) {
  surface_SR->reset();
  field_SR->reset();
  grad_eS_S_.clear();

  candidates_.clear();
  CollectCandidates(bvh_S, bvh_R, X_SR, &candidates_);

  if (candidates_.empty()) return;

  const Matrix3d R_SR_mat = X_SR.rotation().matrix();
  const Vector3d t_SR_vec = X_SR.translation();

  const int num_candidates = static_cast<int>(candidates_.size());
  const int num_threads = Parallelism::Max().num_threads();

  if (static_cast<int>(thread_data_.size()) < num_threads) {
    thread_data_.resize(num_threads);
  }
  for (int t = 0; t < num_threads; ++t) {
    thread_data_[t].ClearOutputs();
  }

  const int expected_polygons = std::max(1, num_candidates / (6 * num_threads));
  for (int t = 0; t < num_threads; ++t) {
    thread_data_[t].vertices.reserve(expected_polygons * 4);
    thread_data_[t].pressures.reserve(expected_polygons * 4);
    thread_data_[t].face_data.reserve(expected_polygons * 5);
    thread_data_[t].grad_e_per_face.reserve(expected_polygons);
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
    const auto& [tet_index, tri_index] = candidates_[idx];
    ProcessSingleCandidate(volume_field_S, surface_R, R_SR_mat, t_SR_vec,
                           tet_index, tri_index, &thread_data_[tid]);
  }

  if (num_threads == 1) {
    auto& td = thread_data_[0];
    if (td.polygon_count == 0) return;

    grad_eS_S_.swap(td.grad_e_per_face);

    *surface_SR = std::make_unique<PolygonSurfaceMesh<double>>(
        std::move(td.face_data), std::move(td.vertices));

    std::vector<Vector3d> field_gradients(grad_eS_S_);
    *field_SR = std::make_unique<PolygonSurfaceMeshFieldLinear<double, double>>(
        std::move(td.pressures), surface_SR->get(), std::move(field_gradients));
  } else {
    MergeThreadResults(num_threads, surface_SR, field_SR);
  }
}

std::unique_ptr<ContactSurface<double>>
ComputeContactSurfaceFromSoftVolumeRigidSurfaceFast(
    GeometryId id_S, const VolumeMeshFieldLinear<double, double>& field_S,
    const Bvh<Obb, VolumeMesh<double>>& bvh_S,
    const math::RigidTransformd& X_WS, GeometryId id_R,
    const TriangleSurfaceMesh<double>& mesh_R,
    const Bvh<Obb, TriangleSurfaceMesh<double>>& bvh_R,
    const math::RigidTransformd& X_WR) {
  thread_local FastSurfaceVolumeIntersector fast_intersector;

  const math::RigidTransformd X_SR = X_WS.InvertAndCompose(X_WR);

  std::unique_ptr<PolygonSurfaceMesh<double>> surface_SR;
  std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>> field_SR;

  fast_intersector.IntersectSurfaceVolume(field_S, bvh_S, mesh_R, bvh_R, X_SR,
                                          &surface_SR, &field_SR);

  if (surface_SR == nullptr) return nullptr;

  surface_SR->TransformVertices(X_WS);
  field_SR->Transform(X_WS);

  const auto& grad_eS_S = fast_intersector.grad_eS_S();
  auto grad_eS_W = std::make_unique<std::vector<Vector3d>>();
  grad_eS_W->reserve(grad_eS_S.size());
  for (const auto& grad : grad_eS_S) {
    grad_eS_W->emplace_back(X_WS.rotation() * grad);
  }

  return std::make_unique<ContactSurface<double>>(
      id_S, id_R, std::move(surface_SR), std::move(field_SR),
      std::move(grad_eS_W), nullptr);
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
