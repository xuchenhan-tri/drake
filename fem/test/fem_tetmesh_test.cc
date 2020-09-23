#include "drake/fem/fem_tetmesh.h"

#include <limits>
#include <memory>
#include <unordered_set>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace fem {

/* There are two tetrahedral meshes in FemTetMesh.
   Both mesh consist of a single tetrahedron with vertices at (-1, -1, 1), (1,
   1, 1), (1, -1, -1) and (-1, 1, -1). The following ASCII art visualizes the
   cube centered at origin with side length 2 that the tetrahedron lives in.

    0: (-1, -1, 1)  X--------+
                   /|       /|
                  / |      / |
                 +--+-----X  | 1: (1, 1, 1)
                 |  |     |  |
                 |  +-----+--X 3: (-1, 1, -1)
                 | /      | /
                 |/       |/
                 X--------+
         2: (1, -1, -1)
 */
class FemTetMeshTest : public ::testing::Test {
 public:
  void SetUp() override {
    q_.resize(3, 8);
    q_.col(0) = Vector3<double>(-1, -1, 1);
    q_.col(1) = Vector3<double>(1, 1, 1);
    q_.col(2) = Vector3<double>(1, -1, -1);
    q_.col(3) = Vector3<double>(-1, 1, -1);
    // The last four vertex positions are the same as the first four.
    for (int i = 0; i < 4; ++i) {
      q_.col(i + 4) = q_.col(i);
    }
    std::vector<Vector4<int>> tet_mesh = {Vector4<int>(0, 1, 2, 3)};
    fem_tetmeshes_.clear();
    fem_tetmeshes_.emplace_back(
        std::make_unique<FemTetMesh<double>>(tet_mesh, 0));
    fem_tetmeshes_.emplace_back(
        std::make_unique<FemTetMesh<double>>(tet_mesh, 4));
  }

 protected:
  const std::vector<Vector3<double>>& get_surface_vertex_normals(int i) {
    return fem_tetmeshes_[i]->surface_vertex_normals_;
  }
  std::vector<std::unique_ptr<FemTetMesh<double>>> fem_tetmeshes_;
  Matrix3X<double> q_;
};

namespace {
TEST_F(FemTetMeshTest, AnalyticVertexNormal) {
  fem_tetmeshes_[0]->UpdatePosition(q_);
  fem_tetmeshes_[1]->UpdatePosition(q_);
  EXPECT_EQ(get_surface_vertex_normals(0), get_surface_vertex_normals(1));
  // For a regular tetrahedron centered at zero, the vertex normal for vertex V
  // should be in the direction of p_WV.
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(CompareMatrices(get_surface_vertex_normals(0)[i],
                                q_.col(i).normalized()));
  }
}

TEST_F(FemTetMeshTest, FaceCenterNormal) {
  // The face normal at the center of a face should be the opposite of the
  // vertex normal of the vertex opposing the face.
  fem_tetmeshes_[0]->UpdatePosition(q_);
  fem_tetmeshes_[1]->UpdatePosition(q_);
  Vector3<double> bary = 1.0 / 3.0 * Vector3<double>::Ones();
  for (int m = 0; m < 2; ++m) {
    const auto& surface_triangles = fem_tetmeshes_[m]->get_surface_triangles();
    const auto& vertex_normals = get_surface_vertex_normals(m);
    for (int t = 0; t < static_cast<int>(surface_triangles.size()); ++t) {
      auto face_normal = fem_tetmeshes_[m]->EvalNormal(t, bary);
      std::unordered_set<int> surface_vertices = {0, 1, 2, 3};
      for (int i = 0; i < 3; ++i) {
        surface_vertices.erase(surface_triangles[t][i]);
      }
      // The left-over vertex is the one not on the triangle t.
      int opposing_index = *surface_vertices.begin();
      auto vertex_normal = vertex_normals[opposing_index];
      EXPECT_TRUE(CompareMatrices(face_normal, -vertex_normal,
                                  std::numeric_limits<double>::epsilon()));
    }
  }
}

TEST_F(FemTetMeshTest, VertexNormalSameAsFaceNormal) {
  // Vertex normal should be the same as face normals when the barycentric
  // coordinate lands at a vertex.
  fem_tetmeshes_[0]->UpdatePosition(q_);
  fem_tetmeshes_[1]->UpdatePosition(q_);
  std::vector<Vector3<double>> bary = {Vector3<double>(1, 0, 0),
                                       Vector3<double>(0, 1, 0),
                                       Vector3<double>(0, 0, 1)};
  for (int m = 0; m < 2; ++m) {
    const auto& surface_triangles = fem_tetmeshes_[m]->get_surface_triangles();
    const auto& vertex_normals = get_surface_vertex_normals(m);
    for (int t = 0; t < static_cast<int>(surface_triangles.size()); ++t) {
      for (int b = 0; b < 3; ++b) {
        auto face_normal = fem_tetmeshes_[m]->EvalNormal(t, bary[b]);
        EXPECT_TRUE(CompareMatrices(face_normal,
                                    vertex_normals[surface_triangles[t](b)]));
      }
    }
  }
}
}  // namespace
}  // namespace fem
}  // namespace drake
