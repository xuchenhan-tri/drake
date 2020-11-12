#include "drake/fem/newton_solver.h"

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

class NewtonSolverTest : public ::testing::Test {
 public:
  void SetUp() override {
    data_ = std::make_unique<FemData<double>>(0.01);
    force_ = std::make_unique<FemForce<double>>(data_->get_elements());
    objective_ = std::make_unique<BackwardEulerObjective<double>>(data_.get(),
                                                                  force_.get());
    solver_ = std::make_unique<NewtonSolver<double>>(objective_.get());
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
  std::unique_ptr<NewtonSolver<double>> solver_;
};

// Linear system should converge in one newton iteration.
TEST_F(NewtonSolverTest, LinearSystemTest) {
  // Set random velocities and random positions.
  const int nv = data_->get_num_vertices();
  //  Matrix3X<double> positions = Matrix3X<double>::Random(3, nv);
  //  Matrix3X<double> velocities = Matrix3X<double>::Random(3, nv);
  Matrix3X<double> positions = data_->get_q();
  Matrix3X<double> velocities = Matrix3X<double>::Zero(3, nv);
  data_->get_mutable_q() = positions;
  data_->get_mutable_v() = velocities;
  data_->get_mutable_dv() = Matrix3X<double>::Zero(3, nv);
  data_->get_mutable_q_hat() = positions + data_->get_dt() * velocities;
  auto& elements = data_->get_mutable_elements();
  for (auto& e : elements) {
    e.UpdateTimeNPositionBasedState(positions);
  }
  // dv is the unknown of the linear system.
  Matrix3X<double>& dv_tmp = data_->get_mutable_dv();
  Eigen::Map<VectorX<double>> dv(dv_tmp.data(), dv_tmp.size());
  // Solve the linear solve to machine epsilon. This tolerance measures
  // |Ax-b|/|b|.
  solver_->set_linear_solver_accuracy(std::numeric_limits<double>::epsilon());
  solver_->set_max_iterations(1);
  // Set a tight accuracy requirement: 1 nm/s.
  constexpr double newton_tolerance = 1e-9;
  solver_->set_tolerance(newton_tolerance);
  NewtonSolver<double>::NewtonSolverStatus solver_status = solver_->Solve(&dv);
  EXPECT_EQ(solver_status, NewtonSolver<double>::NewtonSolverStatus::Success);
  // Check that the momentum balance equation holds: Mdv = fdt.
  Matrix3X<double> momentum_change(3, nv);
  VectorX<double> mass = data_->get_mass();
  for (int i = 0; i < nv; ++i) {
    momentum_change.col(i) = mass(i) * dv_tmp.col(i);
  }
  Matrix3X<double> impulse = Matrix3X<double>::Zero(3, nv);
  force_->AccumulateScaledElasticForce(data_->get_dt(), &impulse);
  Matrix3X<double> new_velocities =
      velocities + Eigen::Map<Matrix3X<double>>(dv.data(), 3, nv);
  force_->AccumulateScaledDampingForce(data_->get_dt(), new_velocities,
                                       &impulse);
  for (int i = 0; i < nv; ++i) {
    impulse.col(i) += data_->get_dt() * data_->get_gravity() * mass(i);
  }
  // The terms in the momentum equation have the same units as momentum, so we
  // scale the newton_accuracy (that has the unit of velocity) by the mass.
  const double scale = mass.norm();
  EXPECT_NEAR((impulse - momentum_change).norm(), 0.0,
              scale * newton_tolerance);
}
}  // namespace
}  // namespace fem
}  // namespace drake
