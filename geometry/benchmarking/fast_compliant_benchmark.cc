#include <memory>
#include <tuple>

#include <benchmark/benchmark.h>

#include "drake/geometry/proximity/bvh.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/field_intersection.h"
#include "drake/geometry/proximity/field_intersection_fast.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/make_ellipsoid_field.h"
#include "drake/geometry/proximity/make_ellipsoid_mesh.h"
#include "drake/geometry/proximity/make_sphere_field.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;
using math::RigidTransformd;

namespace {

const double kElasticModulus = 1.0e5;
const double kSphereDimension = 3.;
const Vector3d kEllipsoidDimension{3.01, 3.5, 4.};
const double kResolutionHint[4] = {4., 3., 2., 1.};
const Vector3d kContactOverlapTranslation[4] = {
    Vector3d{7, 7, 7}, Vector3d{4, 4, 4}, Vector3d{3.5, 3.5, 3.5},
    Vector3d{1.2, 1.2, 1.2}};

struct TestMeshes {
  std::unique_ptr<VolumeMesh<double>> mesh_S;
  std::unique_ptr<VolumeMeshFieldLinear<double, double>> field_S;
  std::unique_ptr<VolumeMesh<double>> mesh_R;
  std::unique_ptr<VolumeMeshFieldLinear<double, double>> field_R;
  std::unique_ptr<Bvh<Obb, VolumeMesh<double>>> obb_bvh_S;
  std::unique_ptr<Bvh<Obb, VolumeMesh<double>>> obb_bvh_R;
  std::unique_ptr<Bvh<Aabb, VolumeMesh<double>>> aabb_bvh_S;
  std::unique_ptr<Bvh<Aabb, VolumeMesh<double>>> aabb_bvh_R;
  RigidTransformd X_SR;
};

TestMeshes MakeTestMeshes(int resolution, int contact_overlap) {
  TestMeshes tm;
  Ellipsoid ellipsoid{kEllipsoidDimension[0], kEllipsoidDimension[1],
                      kEllipsoidDimension[2]};
  Sphere sphere{kSphereDimension};
  const double hint = kResolutionHint[resolution];

  tm.mesh_S =
      std::make_unique<VolumeMesh<double>>(MakeEllipsoidVolumeMesh<double>(
          ellipsoid, hint, TessellationStrategy::kDenseInteriorVertices));
  tm.field_S = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeEllipsoidPressureField<double>(ellipsoid, tm.mesh_S.get(),
                                         kElasticModulus));
  tm.mesh_R = std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
      sphere, hint, TessellationStrategy::kDenseInteriorVertices));
  tm.field_R = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField<double>(sphere, tm.mesh_R.get(),
                                      kElasticModulus));

  tm.obb_bvh_S = std::make_unique<Bvh<Obb, VolumeMesh<double>>>(*tm.mesh_S);
  tm.obb_bvh_R = std::make_unique<Bvh<Obb, VolumeMesh<double>>>(*tm.mesh_R);
  tm.aabb_bvh_S = std::make_unique<Bvh<Aabb, VolumeMesh<double>>>(*tm.mesh_S);
  tm.aabb_bvh_R = std::make_unique<Bvh<Aabb, VolumeMesh<double>>>(*tm.mesh_R);

  tm.X_SR = RigidTransformd{kContactOverlapTranslation[contact_overlap]};
  return tm;
}

}  // namespace

class BaselineBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    const int resolution = state.range(0);
    const int overlap = state.range(1);
    tm_ = MakeTestMeshes(resolution, overlap);
  }
  TestMeshes tm_;
};

BENCHMARK_DEFINE_F(BaselineBenchmark, Baseline)
(benchmark::State& state) {
  for (auto _ : state) {
    VolumeIntersector<PolyMeshBuilder<double>, Obb> intersector;
    std::unique_ptr<PolygonSurfaceMesh<double>> surface;
    std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>> field;
    intersector.IntersectFields(*tm_.field_S, *tm_.obb_bvh_S, *tm_.field_R,
                                *tm_.obb_bvh_R, tm_.X_SR, &surface, &field);
    benchmark::DoNotOptimize(surface);
  }
}

class FastBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    const int resolution = state.range(0);
    const int overlap = state.range(1);
    tm_ = MakeTestMeshes(resolution, overlap);
  }
  TestMeshes tm_;
  FastVolumeIntersector intersector_;
};

BENCHMARK_DEFINE_F(FastBenchmark, Fast)
(benchmark::State& state) {
  for (auto _ : state) {
    std::unique_ptr<PolygonSurfaceMesh<double>> surface;
    std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>> field;
    intersector_.IntersectFields(*tm_.field_S, *tm_.aabb_bvh_S, *tm_.field_R,
                                 *tm_.aabb_bvh_R, tm_.X_SR, &surface, &field);
    benchmark::DoNotOptimize(surface);
  }
}

// clang-format off
BENCHMARK_REGISTER_F(BaselineBenchmark, Baseline)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.02)
    ->MinWarmUpTime(0.01)
    ->Args({3, 2})->Args({3, 3})
    ->Args({2, 2})->Args({2, 3});

BENCHMARK_REGISTER_F(FastBenchmark, Fast)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.02)
    ->MinWarmUpTime(0.01)
    ->Args({3, 2})->Args({3, 3})
    ->Args({2, 2})->Args({2, 3});
// clang-format on

}  // namespace internal
}  // namespace geometry
}  // namespace drake
