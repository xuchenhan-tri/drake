#include "drake/solvers/conex_solver.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/test/mathematical_program_test_util.h"

namespace drake {
namespace solvers {
namespace test {

namespace {



// SCS uses `eps = 1e-5` by default.  For testing, we'll allow for some
// small cumulative error beyond that.
constexpr double kTol = 1e-4;

}  // namespace


GTEST_TEST(TestConex, EqualityConstrainedQuadraticMinimization0) {
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();

  Eigen::MatrixXd Q(2, 2);
  Eigen::MatrixXd c(2, 1);
  // clang-format off
  Q << 1, .1,
      .1, 4;
  c << 1,
       2;

  // clang-format on
  prog.AddQuadraticCost(Q, c, x);

  ConexSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(prog, {}, {});
    EXPECT_NEAR((Q * result.GetSolution(x) + c).norm(), 0, kTol);
  }
}

GTEST_TEST(TestConex, EqualityConstrainedQuadraticMinimization1) {
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  prog.AddQuadraticCost(x(0) * x(0) + x(1) * x(1) + x(0) + x(1));

  ConexSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(prog, {}, {});
    const Eigen::Vector2d x_expected(-.5, -.5);
    EXPECT_TRUE(CompareMatrices(result.GetSolution(x), x_expected, kTol,
                                  MatrixCompareType::absolute));
  }
}

GTEST_TEST(TestConex, EqualityConstrainedQuadraticMinimization2) {
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  prog.AddQuadraticCost(x(0) * x(0) + x(1) * x(1) + x(0) + x(1));
  // prog.AddConstraint(x(0)  <= 10);
  // prog.AddConstraint(x(1)  >= -10);

  prog.AddLorentzConeConstraint(2, x(0) * x(0) + x(1) * x(1));

  ConexSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(prog, {}, {});
    const Eigen::Vector2d x_expected(-.5, -.5);
    EXPECT_TRUE(CompareMatrices(result.GetSolution(x), x_expected, kTol,
                                  MatrixCompareType::absolute));
  }
}

}  // namespace test
}  // namespace solvers
}  // namespace drake
