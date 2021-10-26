#include "drake/multibody/contact_solvers/friction_cone_constraint.h"

#include "drake/common/eigen_types.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

const double kInf = std::numeric_limits<double>::infinity();

FrictionConeConstraint::FrictionConeConstraint(double mu, double epsilon)
    : Constraint(1 /* num constraints */, 3 /* num variables */,
                 Vector1d::Zero(), Vector1d::Constant(kInf)),
      mu_(mu),
      epsilon_(epsilon),
      epsilon2_(epsilon * epsilon) {
  DRAKE_DEMAND(mu >= 0);
  DRAKE_DEMAND(epsilon > 0);
}

template <typename DerivedX, typename ScalarY>
void FrictionConeConstraint::DoEvalGeneric(const Eigen::MatrixBase<DerivedX>& x,
                                           VectorX<ScalarY>* y) const {
  using std::sqrt;
  y->resize(num_constraints());
  // sₑ(γ) = μγₙ − ‖γₜ‖ₛ + ε ≥ 0.
  ScalarY& se = (*y)(0);
  const ScalarY& gn = x(2);  // Normal component.
  // N.B. Here we make a copy instead of using "auto" because otherwise
  // squaredNorm() below does not compile for symbolic::Expression.
  const Vector2<ScalarY> gt = x.template head<2>();  // Tangential component.
  const auto gt_soft_norm = sqrt(gt.squaredNorm() + epsilon2_);
  se = mu_ * gn - gt_soft_norm + epsilon_;
}

void FrictionConeConstraint::DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
                                    Eigen::VectorXd* y) const {
  DoEvalGeneric(x, y);
}

void FrictionConeConstraint::DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
                                    AutoDiffVecXd* y) const {
  DoEvalGeneric(x, y);
}

void FrictionConeConstraint::DoEval(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    VectorX<symbolic::Expression>* y) const {
  DoEvalGeneric(x, y);
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
