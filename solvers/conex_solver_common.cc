/* clang-format off to disable clang-format-includes */
#include "drake/solvers/conex_solver.h"
/* clang-format on */

#include "drake/common/never_destroyed.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {

ConexSolver::ConexSolver()
    : SolverBase(&id, &is_available, &is_enabled,
                 &ProgramAttributesSatisfied) {}

ConexSolver::~ConexSolver() = default;

SolverId ConexSolver::id() {
  static const never_destroyed<SolverId> singleton{"SCS"};
  return singleton.access();
}

bool ConexSolver::is_enabled() { return true; }

bool ConexSolver::ProgramAttributesSatisfied(const MathematicalProgram& prog) {
  return AreRequiredAttributesSupported(
      prog.required_capabilities(),
      ProgramAttributes({ProgramAttribute::kLinearEqualityConstraint,
                         ProgramAttribute::kLinearConstraint,
                         ProgramAttribute::kLorentzConeConstraint,
                         ProgramAttribute::kRotatedLorentzConeConstraint,
                         ProgramAttribute::kPositiveSemidefiniteConstraint,
                         ProgramAttribute::kExponentialConeConstraint,
                         ProgramAttribute::kLinearCost,
                         ProgramAttribute::kQuadraticCost}));
}

}  // namespace solvers
}  // namespace drake
