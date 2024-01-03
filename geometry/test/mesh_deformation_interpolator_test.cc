#include "drake/geometry/mesh_deformation_interpolator.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;
using Eigen::VectorXd;

VolumeMesh<double> MakeSingleTetVolumeMesh() {
  std::vector<VolumeElement> elements{VolumeElement{0, 1, 2, 3}};
  std::vector<Vector3d> vertices{Vector3d{0, 0, 0}, Vector3d::UnitX(),
                                 Vector3d::UnitY(), Vector3d::UnitZ()};
  return VolumeMesh(std::move(elements), std::move(vertices));
}

/* Makes an octahedron volume mesh.
 The octahedron looks like this in its geometry frame, F.
                  +Fz   -Fx
                   |   /
                   v5 v3
                   | /
                   |/
   -Fy---v4------v0+------v2---+ Fy
                  /| Fo
                 / |
               v1  v6
               /   |
             +Fx   |
                  -Fz
*/
VolumeMesh<double> MakeOctahedronVolumeMesh() {
  return geometry::internal::MakeSphereMeshLevel0<double>().first;
}

GTEST_TEST(BarycentricInterpolatorTest, ConstructAndInterpolate) {
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> positions;
  positions.resize(2, 3);
  positions.row(0) = Vector3d::Zero();  // barycentric coordinate (1, 0, 0, 0).
  positions.row(1) = Vector3d(
      0.25, 0.25, 0.25);  // barycentric coordinate (0.25, 0.25, 0.25, 0.25).
  const BarycentricInterpolator interpolator(positions,
                                             MakeSingleTetVolumeMesh());
  // Arbirtrary q for 4 vertices of the control mesh.
  const VectorXd q = VectorXd::LinSpaced(12, 0.0, 1.0);
  const VectorXd interpolated_q = interpolator(q);
  EXPECT_EQ(interpolated_q.size(), 6);
  EXPECT_TRUE(CompareMatrices(interpolated_q.head(3), q.head(3)));
  Vector3d expected_q_for_second_point = Vector3d::Zero();
  for (int i = 0; i < 4; ++i) {
    expected_q_for_second_point += 0.25 * q.segment<3>(3 * i);
  }
  EXPECT_TRUE(
      CompareMatrices(interpolated_q.tail(3), expected_q_for_second_point));
  // Throws if the size of q for the control mesh is the wrong size.
  EXPECT_THROW(interpolator(VectorXd::LinSpaced(9, 0.0, 1.0)), std::exception);
}

GTEST_TEST(BarycentricInterpolatorTest, PassivePointOutOfBound) {
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> positions;
  positions.resize(1, 3);
  // Point outside of the tet.
  positions.row(0) = Vector3d(-1e-4, 0, 0);
  DRAKE_EXPECT_THROWS_MESSAGE(
      BarycentricInterpolator(positions, MakeSingleTetVolumeMesh()),
      ".*Passive point outside.*");
}

GTEST_TEST(VertexSelector, ConstructAndInterpolate) {
  const VolumeMesh<double> control_mesh = MakeOctahedronVolumeMesh();
  const VertexSelector selector(std::vector<int>{1, 3}, control_mesh);
  // Arbitrary q for 7 vertices of the control mesh.
  const VectorXd q = VectorXd::LinSpaced(21, 0.0, 1.0);
  const VectorXd interpolated_q = selector(q);
  EXPECT_TRUE(CompareMatrices(interpolated_q.head(3), q.segment<3>(3 * 1)));
  EXPECT_TRUE(CompareMatrices(interpolated_q.tail(3), q.segment<3>(3 * 3)));
  // Throws if the size of q for the control mesh is the wrong size.
  EXPECT_THROW(selector(VectorXd::LinSpaced(9, 0.0, 1.0)), std::exception);
}

GTEST_TEST(VertexSelector, ConstructorFailures) {
  const VolumeMesh<double> control_mesh = MakeOctahedronVolumeMesh();
  // No vertex selected.
  EXPECT_THROW(VertexSelector(std::vector<int>{}, control_mesh),
               std::exception);
  // Duplicated vertices.
  EXPECT_THROW(VertexSelector(std::vector<int>{1, 1}, control_mesh),
               std::exception);
  // Vertices out of bound.
  EXPECT_THROW(VertexSelector(std::vector<int>{-1}, control_mesh),
               std::exception);
  EXPECT_THROW(VertexSelector(std::vector<int>{7}, control_mesh),
               std::exception);
  // Vertices not sorted.
  EXPECT_THROW(VertexSelector(std::vector<int>{3, 2, 1}, control_mesh),
               std::exception);
}

GTEST_TEST(MeshDeformationInterpolator, RenderMeshes) {
  const VolumeMesh<double> control_mesh = MakeOctahedronVolumeMesh();
  RenderMesh mesh0;
  mesh0.positions.resize(1, 3);
  mesh0.positions.row(0) = Vector3d::Zero();
  RenderMesh mesh1;
  mesh1.positions.resize(1, 3);
  mesh1.positions.row(0) = Vector3d(0.25, 0.25, 0.25);
  const MeshDeformationInterpolator interpolator(
      std::vector<RenderMesh>{mesh0, mesh1}, control_mesh);
  // Arbitrary q for 7 vertices of the control mesh.
  const VectorXd q = VectorXd::LinSpaced(21, 0.0, 1.0);
  const std::vector<VectorXd> interpolated_qs = interpolator.Interpolate(q);
  ASSERT_EQ(interpolated_qs.size(), 2);
  EXPECT_TRUE(CompareMatrices(interpolated_qs[0], q.head(3)));
  Vector3d expected_q_for_second_mesh = Vector3d::Zero();
  // The only point in the second render mesh has barycentric coordinate (0.25,
  // 0.25, 0.25, 0.25) in the element formed by vertices (0, 1, 2, 5).
  expected_q_for_second_mesh += 0.25 * q.segment<3>(3 * 0);
  expected_q_for_second_mesh += 0.25 * q.segment<3>(3 * 1);
  expected_q_for_second_mesh += 0.25 * q.segment<3>(3 * 2);
  expected_q_for_second_mesh += 0.25 * q.segment<3>(3 * 5);
  EXPECT_TRUE(CompareMatrices(interpolated_qs[1], expected_q_for_second_mesh));
}

GTEST_TEST(MeshDeformationInterpolator, VertexSelector) {
  const VolumeMesh<double> control_mesh = MakeOctahedronVolumeMesh();
  const MeshDeformationInterpolator interpolator(
      VertexSelector(std::vector<int>{0}, control_mesh));
  // Arbitrary q for 7 vertices of the control mesh.
  const VectorXd q = VectorXd::LinSpaced(21, 0.0, 1.0);
  const std::vector<VectorXd> interpolated_qs = interpolator.Interpolate(q);
  ASSERT_EQ(interpolated_qs.size(), 1);
  EXPECT_TRUE(CompareMatrices(interpolated_qs[0].head(3), q.head(3)));
}

GTEST_TEST(ExtractSurfaceMeshAndInterpolator, OctahedronMesh) {
  const VolumeMesh<double> control_mesh = MakeOctahedronVolumeMesh();
  const auto [tri_mesh, interpolator] =
      ExtractSurfaceMeshAndInterpolator(control_mesh);
  // The mapping of vertex indices from the surface mesh to the volume mesh
  // should be (0,1,2,3,4,5) -> (1,2,3,4,5,6).
  const std::vector<int> expected_mapping{1, 2, 3, 4, 5, 6};
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(control_mesh.vertex(expected_mapping[i]), tri_mesh.vertex(i));
  }
  // We expect 8 faces from the surface of the octahedron (see illustration in
  // MakeOctrahedronVolumeMesh()), and each triangle is an equilateral triangle
  // with side length √2. We test the area of the triangles as a proxy for the
  // tri-mesh is correct.
  EXPECT_EQ(tri_mesh.num_elements(), 8);
  for (int t = 0; t < 8; ++t) {
    EXPECT_NEAR(tri_mesh.area(t), std::sqrt(3.0) / 2.0,
                4.0 * std::numeric_limits<double>::epsilon());
  }

  const VectorXd q = VectorXd::LinSpaced(21, 0.0, 1.0);
  const VectorXd interpolated_q = interpolator.Interpolate(q)[0];
  ASSERT_EQ(interpolated_q.size(), 18);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(q.segment<3>(3 * expected_mapping[i]),
              interpolated_q.segment<3>(3 * i));
  }
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
