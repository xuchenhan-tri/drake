#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/parallelism.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/bvh.h"
#include "drake/geometry/proximity/obb.h"
#include "drake/geometry/proximity/polygon_surface_mesh.h"
#include "drake/geometry/proximity/polygon_surface_mesh_field.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

std::unique_ptr<ContactSurface<double>>
ComputeContactSurfaceFromSoftVolumeRigidSurfaceFast(
    GeometryId id_S, const VolumeMeshFieldLinear<double, double>& field_S,
    const Bvh<Obb, VolumeMesh<double>>& bvh_S,
    const math::RigidTransformd& X_WS, GeometryId id_R,
    const TriangleSurfaceMesh<double>& mesh_R,
    const Bvh<Obb, TriangleSurfaceMesh<double>>& bvh_R,
    const math::RigidTransformd& X_WR);

class FastSurfaceVolumeIntersector {
 public:
  FastSurfaceVolumeIntersector() = default;

  void IntersectSurfaceVolume(
      const VolumeMeshFieldLinear<double, double>& volume_field_S,
      const Bvh<Obb, VolumeMesh<double>>& bvh_S,
      const TriangleSurfaceMesh<double>& surface_R,
      const Bvh<Obb, TriangleSurfaceMesh<double>>& bvh_R,
      const math::RigidTransformd& X_SR,
      std::unique_ptr<PolygonSurfaceMesh<double>>* surface_SR,
      std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>>* field_SR);

  const std::vector<Vector3<double>>& grad_eS_S() const { return grad_eS_S_; }

  static constexpr int kMaxPolygonSize = 7;

 private:
  static constexpr int kMaxBvhStackDepth = 64;

  struct ThreadLocalData {
    std::array<Eigen::Vector3d, kMaxPolygonSize + 1> clip_buffer_a;
    int clip_size_a{0};
    std::array<Eigen::Vector3d, kMaxPolygonSize + 1> clip_buffer_b;
    int clip_size_b{0};

    std::vector<Eigen::Vector3d> vertices;
    std::vector<double> pressures;
    std::vector<int> face_data;
    int polygon_count{0};
    std::vector<Eigen::Vector3d> grad_e_per_face;

    void ClearOutputs() {
      vertices.clear();
      pressures.clear();
      face_data.clear();
      polygon_count = 0;
      grad_e_per_face.clear();
    }
  };

  void ProcessSingleCandidate(
      const VolumeMeshFieldLinear<double, double>& volume_field_S,
      const TriangleSurfaceMesh<double>& surface_R, const Eigen::Matrix3d& R_SR,
      const Eigen::Vector3d& t_SR, int tet_index, int tri_index,
      ThreadLocalData* local);

  static bool IsFaceNormalAlongPressureGradientInline(
      const Eigen::Vector3d& grad_p_S, const Eigen::Vector3d& face_normal_S);

  int ClipTriangleByTetrahedronInline(
      int tet_index, const VolumeMesh<double>& volume_S, int tri_index,
      const TriangleSurfaceMesh<double>& surface_R, const Eigen::Matrix3d& R_SR,
      const Eigen::Vector3d& t_SR, ThreadLocalData* local);

  void CollectCandidates(const Bvh<Obb, VolumeMesh<double>>& bvh_S,
                         const Bvh<Obb, TriangleSurfaceMesh<double>>& bvh_R,
                         const math::RigidTransformd& X_SR,
                         std::vector<std::pair<int, int>>* candidates);

  void MergeThreadResults(
      int num_threads, std::unique_ptr<PolygonSurfaceMesh<double>>* surface_SR,
      std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>>* field_SR);

  std::vector<std::pair<int, int>> candidates_;
  std::vector<ThreadLocalData> thread_data_;
  std::vector<Vector3<double>> grad_eS_S_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
