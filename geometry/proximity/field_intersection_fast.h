#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/parallelism.h"
#include "drake/geometry/proximity/aabb.h"
#include "drake/geometry/proximity/bvh.h"
#include "drake/geometry/proximity/polygon_surface_mesh.h"
#include "drake/geometry/proximity/polygon_surface_mesh_field.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

class FastVolumeIntersector {
 public:
  FastVolumeIntersector() = default;

  void IntersectFields(
      const VolumeMeshFieldLinear<double, double>& field0_M,
      const Bvh<Aabb, VolumeMesh<double>>& aabb_bvh0_M,
      const VolumeMeshFieldLinear<double, double>& field1_N,
      const Bvh<Aabb, VolumeMesh<double>>& aabb_bvh1_N,
      const math::RigidTransformd& X_MN,
      std::unique_ptr<PolygonSurfaceMesh<double>>* surface_M,
      std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>>* e_M);

  const std::vector<int>& tet0_of_contact_polygon() const {
    return tet0_of_contact_polygon_;
  }
  const std::vector<int>& tet1_of_contact_polygon() const {
    return tet1_of_contact_polygon_;
  }

  static constexpr int kMaxPolygonSize = 8;

 private:
  static constexpr int kMaxBvhStackDepth = 64;

  struct ThreadLocalData {
    std::array<Vector3<double>, kMaxPolygonSize> polygon_buffer_a;
    int polygon_size_a{0};
    std::array<Vector3<double>, kMaxPolygonSize> polygon_buffer_b;
    int polygon_size_b{0};
    std::array<int, kMaxPolygonSize> face_buffer_a;
    int face_size_a{0};
    std::array<int, kMaxPolygonSize> face_buffer_b;
    int face_size_b{0};

    std::vector<Vector3<double>> vertices;
    std::vector<double> pressures;
    std::vector<int> face_data;
    int polygon_count{0};
    std::vector<Vector3<double>> grad_e_per_face;
    std::vector<int> tet0_indices;
    std::vector<int> tet1_indices;

    void ClearOutputs() {
      vertices.clear();
      pressures.clear();
      face_data.clear();
      polygon_count = 0;
      grad_e_per_face.clear();
      tet0_indices.clear();
      tet1_indices.clear();
    }
  };

  void ProcessSingleCandidate(
      const VolumeMeshFieldLinear<double, double>& field0_M,
      const VolumeMeshFieldLinear<double, double>& field1_N,
      const Eigen::Matrix3d& R_MN, const Eigen::Vector3d& t_MN,
      const Eigen::Matrix3d& R_NM, const std::pair<int, int>& candidate,
      ThreadLocalData* local);

  bool CalcEquilibriumPlaneInline(
      int element0, const VolumeMeshFieldLinear<double, double>& field0_M,
      int element1, const VolumeMeshFieldLinear<double, double>& field1_N,
      const Eigen::Matrix3d& R_MN, const Eigen::Vector3d& t_MN,
      const Eigen::Matrix3d& R_NM, Eigen::Vector3d* nhat_M,
      double* displacement);

  bool IsPlaneNormalAlongPressureGradientInline(
      const Eigen::Vector3d& nhat_M, int tetrahedron,
      const VolumeMeshFieldLinear<double, double>& field_M);

  int IntersectTetrahedraInline(int element0, const VolumeMesh<double>& mesh0_M,
                                int element1, const VolumeMesh<double>& mesh1_N,
                                const Eigen::Matrix3d& R_MN,
                                const Eigen::Vector3d& t_MN,
                                const Eigen::Vector3d& nhat_M,
                                double displacement, ThreadLocalData* local);

  struct AabbOverlapPrecomputed {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix3d abs_R;
    void Init(const math::RigidTransformd& X_MN);
    bool TestOverlap(const Aabb& a, const Aabb& b) const;
  };

  void CollectCandidatesAabbFast(
      const Bvh<Aabb, VolumeMesh<double>>& bvh0_M,
      const Bvh<Aabb, VolumeMesh<double>>& bvh1_N,
      const VolumeMeshFieldLinear<double, double>& field0_M,
      const VolumeMeshFieldLinear<double, double>& field1_N,
      const math::RigidTransformd& X_MN,
      std::vector<std::pair<int, int>>* candidates);

  void MergeThreadResults(
      int num_threads, std::unique_ptr<PolygonSurfaceMesh<double>>* surface_M,
      std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>>* e_M);

  // Persistent scratch buffers (survive across calls, capacity retained).
  std::vector<std::pair<int, int>> candidates_;
  std::vector<ThreadLocalData> thread_data_;

  std::vector<int> tet0_of_contact_polygon_;
  std::vector<int> tet1_of_contact_polygon_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
