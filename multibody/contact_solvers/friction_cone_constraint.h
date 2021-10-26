#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/common/symbolic.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/decision_variable.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

class FrictionConeConstraint : public drake::solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FrictionConeConstraint)

  /** Implements the smooth friction cone constraint specified by
   sₑ(γ) = μγₙ − ‖γₜ‖ₛ + ε ≥ 0. 
   We use this smooth constraint since:
    1. Both gradient and Hessian are well defined even at γ = 0. 
    2. The domain defined by sₑ(γ) ≥ 0 IS convex.
    3. γₙ ≥ 0 
  */  
  FrictionConeConstraint(double mu, double epsilon = 1.0e-7);

  ~FrictionConeConstraint() override {}

  double mu() const { return mu_; }
  double epsilon() const { return epsilon_; }

 private:
  template <typename DerivedX, typename ScalarY>
  void DoEvalGeneric(const Eigen::MatrixBase<DerivedX>& x,
                     VectorX<ScalarY>* y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override;

  double mu_;
  double epsilon_;  
  double epsilon2_;  // = epsilon * epsilon
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
