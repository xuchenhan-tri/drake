#pragma once

#include <memory>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solver_interface.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

class NumericalInverseDynamics {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(NumericalInverseDynamics);

  NumericalInverseDynamics(double Rt, double Rn, double mu);

  drake::solvers::MathematicalProgram& get_mutable_prog() { return prog_; }

  const Eigen::Matrix<symbolic::Variable, 3, 1>& continuous_variables() const {
    return gamma_;
  }

  Eigen::Vector3d Solve(const Eigen::Vector3d& y,
                        const std::optional<drake::solvers::SolverId>& opt_id =
                            std::nullopt) const;

#if 0
  void SolveAndCheckSolution(const SolverInterface& solver, double tol) const {
    if (solver.available()) {
      MathematicalProgramResult result;
      solver.Solve(prog_, {}, {}, &result);
      EXPECT_TRUE(result.is_success());
      const Eigen::Matrix<double, 3, 1> gamma_sol = result.GetSolution(gamma_);
      // If pt is inside the sphere, then the optimal solution is x=pt.
      if ((pt_ - center_).norm() <= radius_) {
        EXPECT_TRUE(CompareMatrices(gamma_sol, pt_, tol));
        EXPECT_NEAR(result.get_optimal_cost(), 0, tol);
      } else {
        // pt should be the intersection of the ray from center to pt, and the
        // sphere surface.
        Eigen::Matrix<double, 3, 1> ray = pt_ - center_;
        EXPECT_TRUE(CompareMatrices(
            gamma_sol, center_ + radius_ * (ray.normalized()), tol));
      }
    }
  }
#endif

 private:
  drake::solvers::MathematicalProgram prog_;
  Eigen::Matrix<symbolic::Variable, 3, 1> gamma_;
  Eigen::Vector3d R_;
  double mu_;
  drake::solvers::QuadraticCost* cost_{nullptr};
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
