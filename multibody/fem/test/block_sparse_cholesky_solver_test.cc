#include "drake/multibody/fem/block_sparse_cholesky_solver.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/unused.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/fem/schur_complement.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace {

using Eigen::MatrixXd;
using Eigen::Vector4i;
using Eigen::VectorXd;
using geometry::GeometryId;
using geometry::GeometryInstance;
using geometry::SceneGraph;
using geometry::SceneGraphInspector;
using geometry::SourceId;
using geometry::Sphere;
using geometry::VolumeMesh;
using math::RigidTransformd;
using std::make_unique;
using std::unique_ptr;
using std::vector;

// Make an arbitrary SPD element matrix sized 12x12.
Eigen::Matrix<double, 12, 12> dummy_matrix12x12() {
  Eigen::Matrix<double, 12, 12> A;
  for (int i = 0; i < 12; ++i) {
    for (int j = 0; j < 12; ++j) {
      A(i, j) = 3.14 * i + 2.7 * j;
    }
  }
  Eigen::Matrix<double, 12, 12> I =
      5 * Eigen::Matrix<double, 12, 12>::Identity();
  return A * A.transpose() + I;
}

unique_ptr<GeometryInstance> make_sphere_instance(double radius = 1.0) {
  return make_unique<GeometryInstance>(RigidTransformd::Identity(),
                                       make_unique<Sphere>(radius), "sphere");
}

std::unique_ptr<BlockSparseCholeskySolver> MakeSolver() {
  SceneGraph<double> scene_graph;
  SourceId s_id = scene_graph.RegisterSource();
  constexpr double kRezHint = 1.5;
  unique_ptr<GeometryInstance> geometry_instance = make_sphere_instance();
  GeometryId g_id = scene_graph.RegisterDeformableGeometry(
      s_id, scene_graph.world_frame_id(), std::move(geometry_instance),
      kRezHint);
  const SceneGraphInspector<double>& inspector = scene_graph.model_inspector();
  const VolumeMesh<double>* reference_mesh = inspector.GetReferenceMesh(g_id);
  DRAKE_DEMAND(reference_mesh != nullptr);

  vector<Vector4i> elements;
  for (const auto element : reference_mesh->tetrahedra()) {
    elements.emplace_back(element.vertex(0), element.vertex(1),
                          element.vertex(2), element.vertex(3));
  }
  auto solver = std::make_unique<BlockSparseCholeskySolver>(
      elements, reference_mesh->num_vertices());

  SymmetricBlockSparseMatrix<double>& A = solver->GetMutableMatrix();
  const Eigen::Matrix<double, 12, 12> element_matrix = dummy_matrix12x12();
  /* Add in element matrices */
  for (const Vector4i& element : elements) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        if (element(a) >= element(b)) {
          A.AddToBlock(element(a), element(b),
                       element_matrix.block<3, 3>(3 * a, 3 * b));
        }
      }
    }
  }

  return solver;
}

GTEST_TEST(CholmodSparseMatrixTest, Solve) {
  auto solver = MakeSolver();
  const MatrixXd dense_A = solver->GetMutableMatrix().MakeDenseMatrix();

  solver->Factor();
  const VectorXd b = VectorXd::LinSpaced(dense_A.rows(), 0.0, 1.0);
  const VectorXd x = solver->Solve(b);
  const VectorXd expected_x = Eigen::LLT<MatrixXd>(dense_A).solve(b);
  EXPECT_TRUE(CompareMatrices(x, expected_x, 1e-13));
}

GTEST_TEST(CholmodSparseMatrixTest, SchurComplement) {
  auto solver = MakeSolver();
  const MatrixXd dense_M = solver->GetMutableMatrix().MakeDenseMatrix();
  const int m_size = dense_M.cols();
  constexpr int num_eliminated = 4;
  const MatrixXd S = solver->CalcSchurComplement(num_eliminated);

  /* Compute expected Schur complement. */
  constexpr int n = num_eliminated;
  const MatrixXd A = dense_M.topLeftCorner<3 * n, 3 * n>();
  const MatrixXd D = dense_M.bottomRightCorner(m_size - 3 * n, m_size - 3 * n);
  const MatrixXd B = dense_M.topRightCorner(3 * n, m_size - 3 * n);
  const MatrixXd expected_S =
      D - B.transpose() * Eigen::LLT<MatrixXd>(A).solve(B);
  EXPECT_TRUE(CompareMatrices(S, expected_S, 1e-9));
}

}  // namespace
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
