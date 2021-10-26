#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/contact_solvers/numerical_inverse_dynamics.h"
#include "drake/multibody/contact_solvers/analytical_inverse_dynamics.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/nlopt_solver.h"
#include "drake/solvers/snopt_solver.h"

#include <iostream>
#define PRINT_VAR(a) std::cout << #a": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a":\n" << a << std::endl;

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

using Eigen::Vector3d;

struct ProjectionParameters {
  double Rt{1.0};
  double Rn{1.0};
  double mu{1.0};
};

void CompareAnalyticaVsNumericalProjection(const ProjectionParameters& p,
                                           const Vector3d& y) {
  NumericalInverseDynamics solver(p.Rt, p.Rn, p.mu);
  const Vector3d gamma = solver.Solve(y, drake::solvers::IpoptSolver::id());
  PRINT_VAR(gamma.transpose());

  const Vector3d gamma_analytical = AnalyticalProjection(y, p.Rt, p.Rn, p.mu);
  PRINT_VAR(gamma_analytical.transpose());

  EXPECT_TRUE(CompareMatrices(
        gamma, gamma_analytical,
        1e-5, MatrixCompareType::absolute));
}

GTEST_TEST(InverseDynamics, InsideFrictionCone) {
  const Vector3d y(1.0, 0, -0.05);
  CompareAnalyticaVsNumericalProjection({0.1, 1.0, 0.5}, y);  
}
}
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
