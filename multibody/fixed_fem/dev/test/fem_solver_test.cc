#include "drake/multibody/fixed_fem/dev/fem_solver.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/multibody/fixed_fem/dev/acceleration_newmark_scheme.h"
#include "drake/multibody/fixed_fem/dev/test/dummy_model.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace test {
namespace {

/* Parameters for the Newmark-beta integration scheme. */
constexpr double kDt = 0.01;
constexpr double kGamma = 0.5;
constexpr double kBeta = 0.25;

class FemSolverTest : public ::testing::Test {
 protected:
  DummyModel model_{};
  AccelerationNewmarkScheme<double> integrator_{kDt, kGamma, kBeta};
  FemSolver<double> solver_{&model_, &integrator_};
};

TEST_F(FemSolverTest, Tolerancse) {
  /* Default values. */
  EXPECT_EQ(solver_.relative_tolerance(), 1e-6);
  EXPECT_EQ(solver_.absolute_tolerance(), 1e-3);
  EXPECT_EQ(solver_.linear_solve_tolerance(), 1e-4);
  /* Test Setters. */
  constexpr double kEps = 1e-8;
  solver_.set_relative_tolerance(kEps);
  solver_.set_absolute_tolerance(kEps);
  solver_.set_linear_solve_tolerance(kEps);
  EXPECT_EQ(solver_.relative_tolerance(), kEps);
  EXPECT_EQ(solver_.absolute_tolerance(), kEps);
  EXPECT_EQ(solver_.linear_solve_tolerance(), kEps);
}

TEST_F(FemSolverTest, AdvanceOneTimeStep) {
  std::unique_ptr<FemStateBase<double>> state0 = model_.MakeFemStateBase();
  std::unique_ptr<FemStateBase<double>> state = model_.MakeFemStateBase();
  std::unique_ptr<FemStateBase<double>> expected_state =
      model_.MakeFemStateBase();
  solver_.AdvanceOneTimeStep(*state0, state.get());

  /* The expected result from AdvanceOneTimeStep(). */
  Eigen::SparseMatrix<double> tangent_matrix =
      model_.MakeEigenSparseTangentMatrix();
  model_.CalcTangentMatrix(*state0, integrator_.weights(), &tangent_matrix);
  VectorX<double> residual(model_.num_dofs());
  model_.CalcResidual(*state0, &residual);
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cg;
  cg.compute(tangent_matrix);
  const VectorX<double> dz = cg.solve(-residual);
  integrator_.UpdateStateFromChangeInUnknowns(dz, expected_state.get());
  EXPECT_TRUE(
      CompareMatrices(expected_state->GetPositions(), state->GetPositions()));
}

}  // namespace
}  // namespace test
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
