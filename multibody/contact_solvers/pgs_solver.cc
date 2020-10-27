#include "drake/multibody/contact_solvers/pgs_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

template <typename T>
ContactSolverStatus PgsSolver<T>::SolveWithGuess(
    const T& time_step, const SystemDynamicsData<T>& dynamics_data,
    const PointContactData<T>& contact_data, const VectorX<T>& v_guess,
    ContactSolverResults<T>* result) {
  PreProcessData(time_step, dynamics_data, contact_data);
  result->Resize(dynamics_data.num_velocities(), contact_data.num_contacts());
  // Aliases to pre-processed (const) data.
  const auto& v_star = pre_proc_data_.v_star;
  const auto& vc_star = pre_proc_data_.vc_star;
  const auto& Dinv = pre_proc_data_.Dinv;

  // Aliases so solver's state.
  auto& v = state_.mutable_v();
  auto& gamma = state_.mutable_gamma();

  // Aliases to parameters.
  const int max_iters = parameters_.max_iterations;
  const T omega = parameters_.relaxation;

  // Set initial guess.
  v = v_guess;
  gamma.setZero();  // we don't know any better.

  // Below we use index k to denote the iteration. Hereinafter we'll adopt the
  // convention of appending a trailing _kp ("k plus") to refer to the next
  // iterate k+1.
  State state_kp(state_);  // Next iteration, k+1, state.
  VectorX<T>& v_kp = state_kp.mutable_v();
  VectorX<T>& gamma_kp = state_kp.mutable_gamma();

  // State dependent quantities.
  vc_ = vc_star;  // Contact velocity at state_, intialized to when gamma = 0.
  VectorX<T> vc_kp(
      3 * contact_data.num_contacts());  // Contact velocity at state_kp.
  for (int k = 0; k < max_iters; ++k) {
    // N.B. This is more of a "Projected Jacobi" update since we are not using
    // the already updated values. A small variation from PGS ok for testing
    // purposes.
    gamma_kp = gamma - omega * Dinv.asDiagonal() * vc_;
    ProjectAllImpulses(contact_data, vc_, &gamma_kp);
    // Update generalized velocities; v = v* + M⁻¹⋅Jᵀ⋅γ.
    contact_data.get_Jc().MultiplyByTranspose(gamma_kp,
                                              &tau_c_);  // tau_c = Jᵀ⋅γ
    dynamics_data.get_Ainv().Multiply(tau_c_, &v_kp);  // v_kp = M⁻¹⋅Jᵀ⋅γ
    v_kp += v_star;  // v_kp = v* + M⁻¹⋅Jᵀ⋅γ
    // Update contact velocities; vc = J⋅v.
    contact_data.get_Jc().Multiply(v_kp, &vc_kp);

    // Verify convergence.
    const bool converged =
        VerifyConvergenceCriteria(contact_data, vc_, vc_kp, gamma, gamma_kp,
                                  &stats_.vc_err, &stats_.gamma_err);
    stats_.iterations++;

    // Update state for the next iteration.
    state_ = state_kp;
    vc_ = vc_kp;
    if (converged) {
      std::cout << "PGS iterations = " << k + 1 << std::endl;
      // TODO(xuchenhan-tri): write other contact results as well.
      result->v_next = state_.v();
      return ContactSolverStatus::kSuccess;
    }
  }
  std::cout << "PGS iterations = " << max_iters << std::endl;
  // TODO(xuchenhan-tri): write other contact results as well.
  result->v_next = state_.v();
  return ContactSolverStatus::kFailure;
}

template <typename T>
void PgsSolver<T>::PreProcessData(T dt,
                                  const SystemDynamicsData<T>& dynamics_data,
                                  const PointContactData<T>& contact_data) {
  const int nc = contact_data.num_contacts();
  const int nv = dynamics_data.num_velocities();
  state_.Resize(nv, nc);
  pre_proc_data_.Resize(nv, nc);
  tau_c_.resize(nv);
  vc_.resize(3 * nc);

  // Generalized velocities when contact forces are zero.
  auto& v_star = pre_proc_data_.v_star;
  v_star = dynamics_data.get_v_star();

  if (nc != 0) {
    // Contact velocities when contact forces are zero.
    auto& vc_star = pre_proc_data_.vc_star;
    contact_data.get_Jc().Multiply(v_star, &vc_star);

    auto& N = pre_proc_data_.N;
    this->FormDelassusOperatorMatrix(contact_data.get_Jc(),
                                     dynamics_data.get_Ainv(),
                                     contact_data.get_Jc(), &N);

    // Compute scaling factors, one per contact.
    auto& Nii_norm = pre_proc_data_.Nii_norm;
    auto& Dinv = pre_proc_data_.Dinv;
    for (int i = 0; i < nc; ++i) {
      // 3x3 diagonal block. It might be singular, but definitely non-zero.
      // That's why we use an rms norm.
      const auto& Nii = N.block(3 * i, 3 * i, 3, 3);
      Matrix3<T> tmp = Nii;
      Matrix3<T> tmp_inv = tmp.inverse();
      Nii_norm(i) = Nii.norm() / 3;  // 3 = sqrt(9).
      Dinv.template segment<3>(3 * i).setConstant(1.0 / Nii_norm(i));
      Dinv.template segment<3>(3 * i) =
          Vector3<T>(tmp_inv(0, 0), tmp_inv(1, 1), tmp_inv(2, 2));
    }
  }
}
template <typename T>
bool PgsSolver<T>::VerifyConvergenceCriteria(
    const PointContactData<T>& contact_data, const VectorX<T>& vc,
    const VectorX<T>& vc_kp, const VectorX<T>& gamma,
    const VectorX<T>& gamma_kp, T* vc_err, T* gamma_err) const {
  using std::max;
  const auto& Nii_norm = pre_proc_data_.Nii_norm;
  bool converged = true;
  *vc_err = 0;
  *gamma_err = 0;
  for (int ic = 0; ic < contact_data.num_contacts(); ++ic) {
    auto within_error_bounds = [& p = parameters_](const T& error,
                                                   const T& scale) {
      const T bound(p.abs_tolerance + p.rel_tolerance * scale);
      return error < bound;
    };
    // Check velocity convergence.
    const auto vci = vc.template segment<3>(3 * ic);
    const auto vci_kp = vc_kp.template segment<3>(3 * ic);
    const T vc_norm = vci.norm();
    const T vci_err = (vci_kp - vci).norm();
    *vc_err = max(*vc_err, vci_err);
    if (!within_error_bounds(vci_err, vc_norm)) {
      converged = false;
    }

    // Check impulse convergence. Scaled to velocity so that its convergence
    // metric is compatible with that of contact velocity.
    const auto gi = gamma.template segment<3>(3 * ic);
    const auto gi_kp = gamma_kp.template segment<3>(3 * ic);
    const T g_norm = gi.norm() / Nii_norm(ic);
    T g_err = (gi_kp - gi).norm();
    *gamma_err = max(*gamma_err, g_err);
    g_err /= Nii_norm(ic);
    if (!within_error_bounds(g_err, g_norm)) {
      converged = false;
    }
  }
  return converged;
}

template <typename T>
void PgsSolver<T>::ProjectAllImpulses(const PointContactData<T>& contact_data,
                                      const VectorX<T>& vc,
                                      VectorX<T>* gamma_inout) const {
  VectorX<T>& gamma = *gamma_inout;
  const auto& mu = contact_data.get_mu();
  for (int ic = 0; ic < contact_data.num_contacts(); ++ic) {
    auto vci = vc.template segment<3>(3 * ic);
    auto gi = gamma.template segment<3>(3 * ic);
    gi = ProjectImpulse(vci, gi, mu(ic));
  }
}

template <typename T>
Vector3<T> PgsSolver<T>::ProjectImpulse(
    const Eigen::Ref<const Vector3<T>>& vc,
    const Eigen::Ref<const Vector3<T>>& gamma, const T& mu) const {
  const T& pi = gamma(2);                    // Normal component.
  if (pi <= 0.0) return Vector3<T>::Zero();  // No contact.

  const auto beta = gamma.template head<2>();  // Tangential component.
  if (beta.norm() <= mu * pi) return gamma;    // Inside the cone.

  // Non-zero impulse lies outside the cone. We'll project it.
  using std::sqrt;
  // We use the absolute tolerance as a velocity scale to use in a velocity
  // soft norm.
  const T v_eps = parameters_.abs_tolerance;
  const T v_eps2 = v_eps * v_eps;
  // Alias to tangential velocity.
  const auto vt = vc.template head<2>();
  // Compute a direction.
  const T vt_soft_norm = sqrt(vt.squaredNorm() + v_eps2);
  const Vector2<T> that = vt / vt_soft_norm;
  // Project. Principle of maximum dissipation.
  const Vector2<T> projected_beta = -mu * pi * that;
  return Vector3<T>(projected_beta(0), projected_beta(1), pi);
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::PgsSolver)
