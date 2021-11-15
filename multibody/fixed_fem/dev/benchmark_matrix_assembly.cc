/**
A benchmark to compare FEM solvers.

Run with

```
bazel run --copt -DENABLE_TIMERS
multibody/fixed_fem/dev:benchmark_matrix_assembly
```

to enable performance stats. */
#include <Eigen/SparseCholesky>
#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/profiler.h"
#include "drake/common/unused.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/multibody/fixed_fem/dev/block_sparse_solver.h"
#include "drake/multibody/fixed_fem/dev/mesh_utilities.h"
#include "drake/multibody/fixed_fem/dev/parse_vtk.h"
#include "drake/multibody/fixed_fem/dev/symmetric_block_sparse_matrix.h"
#include "drake/multibody/fixed_fem/dev/symmetric_block_sparse_matrix_impl.h"

namespace drake {
namespace multibody {
namespace fem {

using Eigen::MatrixXd;
using Eigen::Vector4i;
using Eigen::VectorXd;
using geometry::VolumeMesh;
using internal::SymmetricBlockSparseMatrix;
using internal::SymmetricBlockSparseMatrixImpl;
using std::vector;

DEFINE_int32(max_solve_iterations, 1,
             "The number of times to run Cholesky solves.");
DEFINE_int32(max_assemble_iterations, 1,
             "The number of times to run matrix assembly.");
DEFINE_bool(use_mesh, true,
            "Use a mesh to generate sparsity pattern if true. Otherwise use a "
            "simple, contrived pattern.");

// Make an arbitrary vector for testing.
VectorX<double> MakeVector(int num_nodes) {
  VectorX<double> x(3 * num_nodes);
  for (int i = 0; i < 3 * num_nodes; ++i) {
    x(i) = 3.14 * std::sin(i - 0.512);
  }
  return x;
}

// Build the sparsity pattern for block sparse matrix.
std::unique_ptr<SymmetricBlockSparseMatrix<double>> MakeBlockSparseMatrix(
    const vector<Vector4i>& elements, int num_nodes) {
  const int num_elements = elements.size();
  std::vector<std::vector<int>> index_to_row(num_nodes);
  for (int e = 0; e < num_elements; ++e) {
    const Vector4i& element = elements[e];
    for (int a = 0; a < 4; ++a) {
      const int block_row = element(a);
      for (int b = 0; b < 4; ++b) {
        const int block_col = element(b);
        if (block_row <= block_col)
          index_to_row[block_col].push_back(block_row);
      }
    }
  }
  return std::make_unique<SymmetricBlockSparseMatrixImpl<double, 3>>(
      std::move(index_to_row));
}

// Build the sparsity pattern for Eigen::SparseMatrix.
Eigen::SparseMatrix<double> MakeEigenSparseMatrix(
    const vector<Vector4i>& elements, int num_nodes) {
  const int num_elements = elements.size();
  vector<Eigen::Triplet<double>> non_zero_entries;
  for (int e = 0; e < num_elements; ++e) {
    const Vector4i& element = elements[e];
    for (int a = 0; a < 4; ++a) {
      for (int i = 0; i < 3; ++i) {
        const int row = 3 * element(a) + i;
        for (int b = 0; b < 4; ++b) {
          for (int j = 0; j < 3; ++j) {
            const int col = 3 * element(b) + j;
            non_zero_entries.emplace_back(row, col, 0);
          }
        }
      }
    }
  }
  Eigen::SparseMatrix<double> matrix(3 * num_nodes, 3 * num_nodes);
  matrix.setFromTriplets(non_zero_entries.begin(), non_zero_entries.end());
  matrix.makeCompressed();
  return matrix;
}

// Make an arbitrary SPD element matrix sized 12x12.
Eigen::Matrix<double, 12, 12> dummy_matrix12x12() {
  Eigen::Matrix<double, 12, 12> A;
  for (int i = 0; i < 12; ++i) {
    for (int j = 0; j < 12; ++j) {
      A(i, j) = 3.14 * i + 2.7 * j;
    }
  }
  return A * A.transpose() + Eigen::Matrix<double, 12, 12>::Identity();
}

Eigen::Matrix<double, 12, 12> matrix12x12(double s) {
  return s * Eigen::Matrix<double, 12, 12>::Ones();
}

// Assemble an Eigen::SparseMatrix with correctly allocated sparsity pattern
// using arbitrary but SPD element matrices.
void AssembleEigenSparseMatrix(const vector<Vector4i>& elements,
                               Eigen::SparseMatrix<double>* matrix) {
  const Eigen::Matrix<double, 12, 12> element_matrix = dummy_matrix12x12();
  static const common::TimerIndex sparse_timer =
      addTimer("Sparse Matrix Build");
  startTimer(sparse_timer);
  for (int iter = 0; iter < FLAGS_max_assemble_iterations; ++iter) {
    // Clear old values;
    using Iterator = typename Eigen::SparseMatrix<double>::InnerIterator;
    for (int k = 0; k < matrix->outerSize(); ++k) {
      for (Iterator it(*matrix, k); it; ++it) {
        it.valueRef() = 0;
      }
    }
    // Add in element matrices
    for (const Vector4i& element : elements) {
      for (int a = 0; a < 4; ++a) {
        for (int i = 0; i < 3; ++i) {
          for (int b = 0; b < 4; ++b) {
            for (int j = 0; j < 3; ++j) {
              matrix->coeffRef(element(a) * 3 + i, element(b) * 3 + j) +=
                  element_matrix(a * 3 + i, b * 3 + j);
            }
          }
        }
      }
    }
  }
  lapTimer(sparse_timer);
}

// Assemble an Eigen::MatrixXd using arbitrary but SPD element matrices.
void AssembleDenseMatrix(const vector<Vector4i>& elements, MatrixXd* matrix) {
  const Eigen::Matrix<double, 12, 12> element_matrix = dummy_matrix12x12();
  static const common::TimerIndex dense_timer = addTimer("Dense Matrix Build");
  startTimer(dense_timer);
  for (int it = 0; it < FLAGS_max_assemble_iterations; ++it) {
    // Clear old values;
    matrix->setZero();
    // Add in element matrices
    for (const Vector4i& element : elements) {
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          matrix->template block<3, 3>(3 * element(a), 3 * element(b)) +=
              element_matrix.template block<3, 3>(3 * a, 3 * b);
        }
      }
    }
  }
  lapTimer(dense_timer);
}

// Assemble the underlying matrix in the SuperNodalSolver using arbitrary but
// SPD element matrices.
BlockSparseSolver AssembleSuperNodalMatrix(int num_vertices,
                                           std::vector<Vector4i> elements) {
  const Eigen::Matrix<double, 12, 12> element_matrix = dummy_matrix12x12();
  BlockSparseSolver s(num_vertices, elements);
  static const common::TimerIndex supernodal_timer =
      addTimer("Supernodal Matrix Assembly");
  startTimer(supernodal_timer);

  static const common::TimerIndex clear =
      addTimer("Supernodal Matrix `SetZero()`");
  static const common::TimerIndex set =
      addTimer("Supernodal Matrix `SetClique()`");
  static const common::TimerIndex build =
      addTimer("Supernodal Matrix `BuildMatrix()`");

  for (int it = 0; it < FLAGS_max_assemble_iterations; ++it) {
    // Clear old values;
    startTimer(clear);
    s.SetMatrixToZero();
    lapTimer(clear);
    startTimer(set);
    for (int i = 0; i < static_cast<int>(elements.size()); ++i)
      s.SetElementTangentMatrix(i, element_matrix);
    lapTimer(set);
    startTimer(build);
    s.BuildMatrix();
    lapTimer(build);
  }
  lapTimer(supernodal_timer);
  return s;
}

// Assemble a block sparse matrix with correctly allocated sparsity pattern
// using arbitrary but SPD element matrices.
void AssembleBlockSparseMatrix(const vector<Vector4i>& elements,
                               SymmetricBlockSparseMatrix<double>* matrix) {
  Eigen::Matrix<double, 12, 12> element_matrix = dummy_matrix12x12();
  static const common::TimerIndex block_timer = addTimer("Block Matrix Build");
  auto concrete_matrix =
      static_cast<SymmetricBlockSparseMatrixImpl<double, 3>*>(matrix);
  startTimer(block_timer);
  for (int it = 0; it < FLAGS_max_assemble_iterations; ++it) {
    // Clear old values;
    concrete_matrix->SetZero();
    // Add in element matrices
    for (const Vector4i& element : elements) {
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          if (element(a) <= element(b))
            concrete_matrix->AddToBlock(
                element(a), element(b),
                element_matrix.template block<3, 3>(3 * a, 3 * b));
        }
      }
    }
  }
  lapTimer(block_timer);
}

VectorXd EigenSparseMultiply(const Eigen::SparseMatrix<double>& A,
                             const VectorX<double>& x) {
  VectorX<double> y;
  y.resize(A.rows());
  static const common::TimerIndex sparse_multiply_timer =
      addTimer("Eigen Sparse Multiply");
  startTimer(sparse_multiply_timer);
  for (int it = 0; it < FLAGS_max_assemble_iterations; ++it) {
    y = A * x;
  }
  lapTimer(sparse_multiply_timer);
  return y;
}

VectorXd BlockSparseMultiply(const SymmetricBlockSparseMatrix<double>& A,
                             const VectorX<double>& x) {
  VectorX<double> y;
  y.resize(A.rows());
  static const common::TimerIndex block_multiply_timer =
      addTimer("Block Sparse Multiply");
  startTimer(block_multiply_timer);
  for (int it = 0; it < FLAGS_max_assemble_iterations; ++it) {
    A.Multiply(x, &y);
  }
  lapTimer(block_multiply_timer);
  return y;
}

void EigenDenseLDLT(const Eigen::SparseMatrix<double>& A,
                    const VectorX<double>& b, VectorX<double>* x) {
  static const common::TimerIndex eigen_dense_ldlt_timer =
      addTimer("Eigen dense LDLT");
  startTimer(eigen_dense_ldlt_timer);
  const MatrixX<double> A_dense = A;
  for (int it = 0; it < FLAGS_max_solve_iterations; ++it) {
    *x = A_dense.ldlt().solve(b);
  }
  lapTimer(eigen_dense_ldlt_timer);
}

void EigenSparseLDLT(const Eigen::SparseMatrix<double>& A,
                     const VectorX<double>& b, VectorX<double>* x) {
  using SparseLDLT = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
  SparseLDLT ldlt;
  ldlt.analyzePattern(A);
  static const common::TimerIndex eigen_sparse_ldlt_timer =
      addTimer("Eigen sparse LDLT");
  startTimer(eigen_sparse_ldlt_timer);
  for (int it = 0; it < FLAGS_max_solve_iterations; ++it) {
    ldlt.factorize(A);
    *x = ldlt.solve(b);
  }
  lapTimer(eigen_sparse_ldlt_timer);
}

void EigenDenseCholesky(const Eigen::SparseMatrix<double>& A,
                        const VectorX<double>& b, VectorX<double>* x) {
  static const common::TimerIndex eigen_dense_llt_timer =
      addTimer("Eigen dense Cholesky");
  startTimer(eigen_dense_llt_timer);
  for (int it = 0; it < FLAGS_max_solve_iterations; ++it) {
    const MatrixX<double> A_dense = A;
    *x = A_dense.llt().solve(b);
  }
  lapTimer(eigen_dense_llt_timer);
}

void SuperNodalCholesky(BlockSparseSolver& A, const VectorX<double>& b,
                        VectorX<double>* x) {
  static const common::TimerIndex supernodal_llt_timer =
      addTimer("Super nodal Cholesky");
  startTimer(supernodal_llt_timer);
  for (int it = 0; it < FLAGS_max_solve_iterations; ++it) {
    A.Factor();
    *x = A.Solve(b);
  }
  lapTimer(supernodal_llt_timer);
}

void BlockLDLT(const SymmetricBlockSparseMatrix<double>& A,
               const VectorX<double>& b, VectorX<double>* x) {
  Eigen::SparseMatrix<double> A_sparse = A.MakeEigenSparseMatrix();
  using SparseLDLT = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
  static const common::TimerIndex block_ldlt_timer =
      addTimer("Block dense LDLT");
  SparseLDLT ldlt;
  ldlt.analyzePattern(A_sparse);
  startTimer(block_ldlt_timer);
  for (int it = 0; it < FLAGS_max_solve_iterations; ++it) {
    Eigen::SparseMatrix<double> A_ = A.MakeEigenSparseMatrix();
    ldlt.factorize(A_);
    *x = ldlt.solve(b);
  }
  lapTimer(block_ldlt_timer);
}

void BlockCholesky(const SymmetricBlockSparseMatrix<double>& A,
                   const VectorX<double>& b, VectorX<double>* x) {
  static const common::TimerIndex block_llt_timer =
      addTimer("Block dense Cholesky");
  startTimer(block_llt_timer);
  const MatrixX<double> A_dense = A.MakeDenseMatrix();
  for (int it = 0; it < FLAGS_max_solve_iterations; ++it) {
    *x = A_dense.llt().solve(b);
  }
  lapTimer(block_llt_timer);
}

constexpr double eps = 1e-10;
void BenchmarkPerformance() {
  int num_nodes = 0;
  vector<Vector4i> elements;

  if (FLAGS_use_mesh) {
    // Read in a mesh.
    internal::ReferenceDeformableGeometry<double> geo =
        MakeDiamondCubicBoxDeformableGeometry(
            geometry::Box(2, 1, 1), 1.0, math::RigidTransformd::Identity());
    const auto& mesh = geo.mesh();
    elements.resize(mesh.num_elements());
    for (geometry::VolumeElementIndex i(0); i < mesh.num_elements(); ++i) {
      const auto& e = mesh.element(i);
      for (int j = 0; j < 4; ++j) {
        elements[i](j) = static_cast<int>(e.vertex(j));
      }
    }
    num_nodes = mesh.num_vertices();
  } else {
    num_nodes = 12;
    elements.emplace_back(Vector4i(0, 2, 3, 7));
    elements.emplace_back(Vector4i(0, 4, 5, 7));
    elements.emplace_back(Vector4i(0, 1, 2, 5));
    elements.emplace_back(Vector4i(2, 5, 6, 7));
    elements.emplace_back(Vector4i(0, 2, 5, 7));

    elements.emplace_back(Vector4i(4, 5, 7, 8));
    elements.emplace_back(Vector4i(5, 6, 7, 10));
    elements.emplace_back(Vector4i(5, 8, 9, 10));
    elements.emplace_back(Vector4i(7, 8, 10, 11));
    elements.emplace_back(Vector4i(5, 7, 8, 10));
  }

  // Build block sparse matrix.
  auto block_sparse_matrix = MakeBlockSparseMatrix(elements, num_nodes);
  AssembleBlockSparseMatrix(elements, block_sparse_matrix.get());

  // Build Eigen::SparseMatrix.
  Eigen::SparseMatrix<double> eigen_sparse_matrix =
      MakeEigenSparseMatrix(elements, num_nodes);
  AssembleEigenSparseMatrix(elements, &eigen_sparse_matrix);

  // Build Eigen::MatrixXd.
  MatrixXd dense_matrix = MatrixXd(3 * num_nodes, 3 * num_nodes);
  AssembleDenseMatrix(elements, &dense_matrix);

  // Build SuperNodalSolver.
  BlockSparseSolver s = AssembleSuperNodalMatrix(num_nodes, elements);

  // Check assembly correctness.
  const MatrixXd A = eigen_sparse_matrix;
  const MatrixXd B = block_sparse_matrix->MakeDenseMatrix();
  const MatrixXd C = s.MakeDenseMatrix();
  DRAKE_DEMAND((A - B).norm() < eps * std::min(A.norm(), B.norm()));
  DRAKE_DEMAND((A - C).norm() < eps * std::min(A.norm(), C.norm()));

  VectorX<double> x = MakeVector(num_nodes);
  VectorX<double> x2 = MakeVector(num_nodes);
  // VectorXd y1 = EigenSparseMultiply(eigen_sparse_matrix, x);
  // VectorXd y2 = BlockSparseMultiply(*block_sparse_matrix, x);
  // DRAKE_DEMAND((y1 - y2).norm() <
  //              eps * A.norm() * std::min(y1.norm(), y2.norm()));

  // Check solve correctness.
  VectorX<double> b = MakeVector(num_nodes);
  EigenSparseLDLT(eigen_sparse_matrix, b, &x);
  EigenDenseLDLT(eigen_sparse_matrix, b, &x);
  SuperNodalCholesky(s, b, &x2);
  std::cout << "SuperNodal Error = " << (A * x2 - b).norm() << std::endl;
  std::cout << "Eigen Error = " << (A * x - b).norm() << std::endl;
}

int DoMain() {
  BenchmarkPerformance();
  std::cout << TableOfAverages() << "\n";
  return 0;
}
}  // namespace fem
}  // namespace multibody
}  // namespace drake

int main(int argc, char** argv) {
  gflags::SetUsageMessage("Matrix assembly comparison.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::fem::DoMain();
}
