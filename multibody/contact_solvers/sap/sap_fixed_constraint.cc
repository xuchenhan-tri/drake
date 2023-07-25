#include "drake/multibody/contact_solvers/sap/sap_fixed_constraint.h"

#include <limits>
#include <utility>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

template <typename T>
SapFixedConstraint<T>::SapFixedConstraint(
    FixedConstraintKinematics<T> kinematics)
    : SapHolonomicConstraint<T>(
          typename SapHolonomicConstraint<T>::Kinematics(
              std::move(kinematics.p_PQs_W), std::move(kinematics.J),
              VectorX<T>::Zero(kinematics.p_BQs_W.size())),
          MakeSapHolonomicConstraintParameters(kinematics.p_BQs_W.size()),
          {kinematics.objectA, kinematics.objectB}),
      num_constraint_points_(kinematics.p_BQs_W.size() / 3),
      p_APs_W_(std::move(kinematics.p_APs_W)),
      p_BQs_W_(std::move(kinematics.p_BQs_W)) {}

template <typename T>
typename SapHolonomicConstraint<T>::Parameters
SapFixedConstraint<T>::MakeSapHolonomicConstraintParameters(
    int num_constraint_equations) {
  // "Near-rigid" regime parameter, see [Castro et al., 2022].
  // TODO(amcastro-tri): consider exposing this parameter.
  constexpr double kBeta = 0.1;

  // Fixed constraints do not have impulse limits, they are bi-lateral
  // constraints. Each fixed point pair introduces three constraint
  // equations.
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  VectorX<T> gamma_lower =
      VectorX<T>::Constant(num_constraint_equations, -kInfinity);
  VectorX<T> gamma_upper =
      VectorX<T>::Constant(num_constraint_equations, kInfinity);

  VectorX<T> stiffness =
      VectorX<T>::Constant(num_constraint_equations, kInfinity);
  VectorX<T> relaxation_time = VectorX<T>::Zero(num_constraint_equations);

  return typename SapHolonomicConstraint<T>::Parameters{
      std::move(gamma_lower), std::move(gamma_upper), std::move(stiffness),
      std::move(relaxation_time), kBeta};
}

template <typename T>
void SapFixedConstraint<T>::DoAccumulateSpatialImpulses(
    int i, const Eigen::Ref<const VectorX<T>>& gamma,
    SpatialForce<T>* F) const {
  // To interpret γ as a spatial impulse and determine the point of application
  // for this formulation, let's consider the case where both A and B are free
  // bodies for simplicity. In this case the generalized velocities v are just
  // the spatial velocities of each body in the world frame stacked:
  //   v = [w_WA, v_WA, w_WB, v_WB]
  // We know that J⋅v = v_W_PQ = v_WQ - v_WP. Thus J is just the operator that
  // takes the difference of the body's translational velocities shifted to P
  // and Q:
  //   J = [[p_AP]ₓ -[I] -[p_BQ]ₓ [I]]
  // We also know from the optimality condition of the SAP formulation
  // (essentially the balance of momentum condition):
  //   A⋅(v - v*) - Jᵀ⋅γ = 0
  // Therefore the generalized impulse Jᵀ⋅γ corresponds to a spatial impulse on
  // body A and a spatial impulse on body B stacked:
  //   Jᵀ⋅γ = [Γ_Ao_W, Γ_Bo_W]
  // Where:
  //   Γ_Ao_W = (-[p_PA]ₓ⋅-γ, -γ) and Γ_Bo_W = (-[p_QB]ₓ⋅γ, γ)
  // Therefore Jᵀ can be understood as the operator that shifts a spatial
  // impulse (0, -γ) applied at P to Ao and shifts the equal and opposite
  // spatial impulse (0, γ) applied at Q to Bo. Thus, this constraint can be
  // interpreted as applying an impulse γ at point Q on B and an impulse -γ
  // at point P on A. As a consequence the constraint satisfies Newton's 3rd
  // law, but does introduce a small moment of order O(‖γ‖⋅‖p_PQ‖) when P and Q
  // are not coincident.
  if (i == 0) {
    // Object A.
    // -gamma = gamma_Ap_W
    // Shift gamma_Ap_W = to Ao and add in.
    for (int c = 0; c < num_constraint_points_; ++c) {
      const SpatialForce<T> gamma_Ap_W(Vector3<T>::Zero(),
                                       -gamma.template segment<3>(3 * c));
      *F += gamma_Ap_W.Shift(p_APs_W_.template segment<3>(3 * c));
    }
  } else {
    DRAKE_DEMAND(i == 1);
    // Object B.
    // gamma = gamma_Bq_W
    // Shift gamma_Bq_W to Bo and add in.
    for (int c = 0; c < num_constraint_points_; ++c) {
      const SpatialForce<T> gamma_Bq_W(Vector3<T>::Zero(),
                                       gamma.template segment<3>(3 * c));
      *F += gamma_Bq_W.Shift(p_BQs_W_.template segment<3>(3 * c));
    }
  }
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::SapFixedConstraint)
