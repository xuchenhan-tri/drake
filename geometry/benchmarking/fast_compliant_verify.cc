#include <cmath>
#include <iostream>
#include <memory>

#include "drake/geometry/proximity/bvh.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/field_intersection.h"
#include "drake/geometry/proximity/field_intersection_fast.h"
#include "drake/geometry/proximity/make_ellipsoid_field.h"
#include "drake/geometry/proximity/make_ellipsoid_mesh.h"
#include "drake/geometry/proximity/make_sphere_field.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;
using math::RigidTransformd;

namespace {

int RunVerification() {
  const double kElasticModulus = 1.0e5;
  const Ellipsoid ellipsoid{3.01, 3.5, 4.};
  const Sphere sphere{3.};
  const double kResolutionHint[4] = {4., 3., 2., 1.};
  const Vector3d kOverlap[4] = {Vector3d{7, 7, 7}, Vector3d{4, 4, 4},
                                Vector3d{3.5, 3.5, 3.5},
                                Vector3d{1.2, 1.2, 1.2}};

  bool all_ok = true;

  for (int res = 0; res < 4; ++res) {
    for (int overlap = 2; overlap < 4; ++overlap) {
      const double hint = kResolutionHint[res];
      auto mesh_S =
          std::make_unique<VolumeMesh<double>>(MakeEllipsoidVolumeMesh<double>(
              ellipsoid, hint, TessellationStrategy::kDenseInteriorVertices));
      auto field_S = std::make_unique<VolumeMeshFieldLinear<double, double>>(
          MakeEllipsoidPressureField<double>(ellipsoid, mesh_S.get(),
                                             kElasticModulus));
      auto mesh_R =
          std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
              sphere, hint, TessellationStrategy::kDenseInteriorVertices));
      auto field_R = std::make_unique<VolumeMeshFieldLinear<double, double>>(
          MakeSpherePressureField<double>(sphere, mesh_R.get(),
                                          kElasticModulus));

      Bvh<Obb, VolumeMesh<double>> obb_bvh_S(*mesh_S);
      Bvh<Obb, VolumeMesh<double>> obb_bvh_R(*mesh_R);
      Bvh<Aabb, VolumeMesh<double>> aabb_bvh_S(*mesh_S);
      Bvh<Aabb, VolumeMesh<double>> aabb_bvh_R(*mesh_R);

      const RigidTransformd X_SR{kOverlap[overlap]};

      // Baseline.
      VolumeIntersector<PolyMeshBuilder<double>, Obb> baseline;
      std::unique_ptr<PolygonSurfaceMesh<double>> baseline_mesh;
      std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>>
          baseline_field;
      baseline.IntersectFields(*field_S, obb_bvh_S, *field_R, obb_bvh_R, X_SR,
                               &baseline_mesh, &baseline_field);

      // Fast path.
      FastVolumeIntersector fast;
      std::unique_ptr<PolygonSurfaceMesh<double>> fast_mesh;
      std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>> fast_field;
      fast.IntersectFields(*field_S, aabb_bvh_S, *field_R, aabb_bvh_R, X_SR,
                           &fast_mesh, &fast_field);

      bool ok = true;
      if (!baseline_mesh && !fast_mesh) {
        // Both null, ok
      } else if (!baseline_mesh || !fast_mesh) {
        ok = false;
      } else {
        const int b_poly = baseline_mesh->num_elements();
        const int f_poly = fast_mesh->num_elements();
        const int b_vert = baseline_mesh->num_vertices();
        const int f_vert = fast_mesh->num_vertices();

        double b_area = baseline_mesh->total_area();
        double f_area = fast_mesh->total_area();

        double area_ratio = b_area > 0 ? f_area / b_area : 1.0;
        if (std::abs(area_ratio - 1.0) > 0.01) ok = false;

        std::cout << "Res=" << res << " Overlap=" << overlap << " Fast"
                  << ": polys=" << f_poly << "/" << b_poly
                  << " verts=" << f_vert << "/" << b_vert
                  << " area_ratio=" << area_ratio << (ok ? " OK" : " FAIL")
                  << std::endl;
      }
      if (!ok) all_ok = false;
    }
  }

  if (all_ok) {
    std::cout << "\nAll correctness checks PASSED." << std::endl;
  } else {
    std::cout << "\nSome correctness checks FAILED!" << std::endl;
  }
  return all_ok ? 0 : 1;
}

}  // namespace

}  // namespace internal
}  // namespace geometry
}  // namespace drake

int main() {
  return drake::geometry::internal::RunVerification();
}
