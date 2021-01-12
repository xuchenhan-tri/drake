#include "drake/fem/eigen_conjugate_gradient_solver.h"

#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/fem_config.h"
#include "drake/fem/fem_data.h"
#include "drake/fem/fem_force.h"
#include "drake/fem/mesh_utility.h"

namespace drake {
namespace fem {
namespace {

class EigenConjugateGradientSolverTest : public ::testing::Test {
 public:
  void SetUp() override {
    data_ = std::make_unique<FemData<double>>(0.1);
    force_ = std::make_unique<FemForce<double>>(data_->get_elements());
    objective_ = std::make_unique<BackwardEulerObjective<double>>(data_.get(),
                                                                  force_.get());
    solver_ =
        std::make_unique<EigenConjugateGradientSolver<double>>(*objective_);
    FemConfig config;
    config.density = 1e3;
    config.youngs_modulus = 1e4;
    config.poisson_ratio = 0.4;
    config.mass_damping = 0.3;
    config.stiffness_damping = 0.5;
    const int nx = 2;
    const int ny = 2;
    const int nz = 2;
    const double h = 0.1;
    const auto positions =
        MeshUtility<double>::AddRectangularBlockVertices(nx, ny, nz, h);
    const auto indices =
        MeshUtility<double>::AddRectangularBlockMesh(nx, ny, nz);
    data_->AddUndeformedObject(indices, positions, config);
  }

 protected:
  std::unique_ptr<FemForce<double>> force_;
  std::unique_ptr<FemData<double>> data_;
  std::unique_ptr<BackwardEulerObjective<double>> objective_;
  std::unique_ptr<EigenConjugateGradientSolver<double>> solver_;
};

TEST_F(EigenConjugateGradientSolverTest, MatrixVsMatrixFree) {
  // Move vertices to random positions.
  Matrix3X<double> positions =
      Matrix3X<double>::Random(3, data_->get_num_vertices());
  // Set the physics states of the system.
  data_->get_mutable_q_hat() = positions;
  // x is the unknown of the linear system.
  VectorX<double> x_direct(positions.size());
  VectorX<double> x_matrix_free(positions.size());
  x_direct.setZero();
  x_matrix_free.setZero();
  // dv is required to update the physics states. The test should pass no matter
  // what value it takes.
  VectorX<double> dv = VectorX<double>::Random(positions.size());
  objective_->Update(dv);
  // b is the right hand side of the linear system.
  VectorX<double> b = VectorX<double>::Zero(positions.size());
  objective_->CalcResidual(&b);
  solver_->set_accuracy(std::numeric_limits<double>::epsilon());
  // Rounding errors would accumulate over CG iterations, so we limit the number
  // of iterations to 1.
  solver_->set_max_iterations(1);
  std::cerr << "solver1" << std::endl;
  // Get the matrix free solution.
  solver_->set_matrix_free(true);
  solver_->SetUp();
  solver_->Solve(b, &x_matrix_free);
  // Get the matrix solution.
  solver_->set_matrix_free(false);
  solver_->SetUp();
  solver_->Solve(b, &x_direct);
  double scale = x_direct.norm() + x_matrix_free.norm();
  EXPECT_NEAR((x_direct - x_matrix_free).norm(), 0.0,
              std::numeric_limits<double>::epsilon() * scale);
}
}  // namespace
}  // namespace fem
}  // namespace drake
