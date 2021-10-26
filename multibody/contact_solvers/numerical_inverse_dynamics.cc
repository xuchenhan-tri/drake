#include "drake/multibody/contact_solvers/numerical_inverse_dynamics.h"

#include <memory>

#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/nlopt_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/multibody/contact_solvers/friction_cone_constraint.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

using drake::solvers::Binding;
using drake::solvers::IpoptSolver;
//using drake::solvers::LorentzConeConstraint;
using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::QuadraticCost;
using drake::solvers::SolverId;
using Eigen::Matrix3d;
using Eigen::Vector3d;

NumericalInverseDynamics::NumericalInverseDynamics(double Rt, double Rn,
                                                   double mu)
    : prog_{},
      gamma_{prog_.NewContinuousVariables<3>()},
      R_{Rt, Rt, Rn},
      mu_{mu} {
#if 0
    prog_.AddQuadraticCost(2 * Eigen::Matrix<double, 3, 3>::Identity(),
                           Eigen::Matrix<double, 3, 1>::Zero(),
                           0.5 * pt_.squaredNorm(), gamma_);
    prog_.AddLinearCost(-2 * pt_, 0.5 * pt_.squaredNorm(), gamma_);
#endif
  const Matrix3d Q = R_.asDiagonal();
  Binding<QuadraticCost> cost_binding =
      prog_.AddQuadraticErrorCost(Q, Vector3d::Zero(), gamma_);
  cost_ = cost_binding.evaluator().get();

  // Add constraint ‖γₜ‖ ≤ μ⋅γₙ
#if 0  
  Matrix3d A;
  A.setZero();
  A(0, 2) = mu;
  A.bottomLeftCorner<2, 2>().setIdentity();
  const Vector3d b = Vector3d::Zero();
#endif  
  auto cone_constraint = std::make_shared<FrictionConeConstraint>(mu);
  Binding<FrictionConeConstraint> binding(cone_constraint, gamma_);
  prog_.AddConstraint(binding);

  // Solver options.
  // Level 5 is the default for Ipopt.
  prog_.SetSolverOption(IpoptSolver::id(), "print_level", 5);
  prog_.SetSolverOption(IpoptSolver::id(), "print_user_options", "yes");
  //PRINT_VAR(prog.solver_options());
}

Vector3d NumericalInverseDynamics::Solve(
    const Vector3d& y, 
    const std::optional<SolverId>& opt_id) const {

  const Matrix3d Q = R_.asDiagonal();
  const Vector3d b = -Q * y;
  cost_->UpdateCoefficients(Q, b);

  const SolverId id = opt_id ? *opt_id : ChooseBestSolver(prog_);
  std::unique_ptr<drake::solvers::SolverInterface> solver = MakeSolver(id);
  if (!solver->available()) {
    throw std::runtime_error("Solver '" + id.name() + "' not available.");
  }
  // We use default initial guess (most likely gamma = 0).
  MathematicalProgramResult result;
  const Vector3d gamma_init(0.0, 0.0, 1.0);  // we'll place it inside the cone.
  solver->Solve(prog_, gamma_init, {}, &result);
  DRAKE_DEMAND(result.is_success());
  // Solution for all decision variables.
  return result.GetSolution();
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
