#include "drake/multibody/contact_solvers/test/pgs_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

template <typename T>
ContactSolverStatus PgsSolver<T>::SolveWithGuess(
    const T& time_step, const SystemDynamicsData<T>& dynamics_data,
    const PointContactData<T>& contact_data, const VectorX<T>& v_guess,
    ContactSolverResults<T>* results) {
  PreProcessData(dynamics_data, contact_data);

  const int nv = dynamics_data.num_velocities();
  const int nc = contact_data.num_contacts();

  // Aliases to data.
  const auto& Ainv = dynamics_data.get_Ainv();
  const auto& v_star = dynamics_data.get_v_star();
  const auto& Jc = contact_data.get_Jc();
  const auto& mu = contact_data.get_mu();

  // Aliases to pre-processed (const) data.
  const auto& vc_star = pre_proc_data_.vc_star;
  const auto& W = pre_proc_data_.W;
  const auto& Dinv = pre_proc_data_.Dinv;

  // Aliases to solver's state.
  auto& v = state_.mutable_v();
  auto& gamma = state_.mutable_gamma();

  // Aliases to parameters.
  const int max_iters = parameters_.max_iterations;
  const double omega = parameters_.relaxation;

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
  VectorX<T> vc_kp(3 * num_contacts());  // Contact velocity at state_kp.
  stats_ = {};                           // Reset stats.
  stats_.num_contacts = num_contacts();
  for (int k = 0; k < max_iters; ++k) {
    // N.B. This is more of a "Projected Jacobi" update since we are not using
    // the already updated values. A small variation from PGS ok for testing
    // purposes.
    //gamma_kp = gamma - omega * Dinv.asDiagonal() * vc_;
    //ProjectAllImpulses(vc_, mu, &gamma_kp);

    // Gauss-Seidel
    gamma_kp = gamma;
    for (int ic = 0, ic3 = 0; ic < num_contacts(); ic++, ic3 += 3) {
      auto gamma_ic = gamma_kp.template segment<3>(ic3);      

      // W rows for the ic-th contact.
//      const auto& Wic = W.block(3, 3 * nc, ic3, 0);

      // Update of contact velocity with last values of impulses.
      // N.B. gamma_kp(0:ic-1) have been updated while gamma_kp(ic:nc-1) have
      // the previous iteration values.
      const Vector3<T> vc_ic =
          W.middleRows(ic3, 3) * gamma_kp + vc_star.template segment<3>(ic3);

      // Update ic-th contact impulse.
      gamma_ic = gamma_ic - omega * Dinv(ic3) * vc_ic;
      gamma_ic = ProjectImpulse(vc_ic, gamma_ic, mu(ic));
    }

    // Update generalized velocities; v = v* + M⁻¹⋅Jᵀ⋅γ.
    Jc.MultiplyByTranspose(gamma_kp, &jc_);  // tau_c = Jᵀ⋅γ
    Ainv.Multiply(jc_, &v_kp);               // v_kp = M⁻¹⋅Jᵀ⋅γ
    v_kp += v_star;                          // v_kp = v* + M⁻¹⋅Jᵀ⋅γ
    // Update contact velocities; vc = J⋅v.
    Jc.Multiply(v_kp, &vc_kp);

    // Verify convergence and update stats.
    PgsErrorMetrics error_metrics;
    const bool converged = VerifyConvergenceCriteria(
        vc_, vc_kp, gamma, gamma_kp, omega, &error_metrics.vc_err,
        &error_metrics.gamma_err);
    stats_.iteration_errors.push_back(error_metrics);
    stats_.iterations++;

    // Update state for the next iteration.
    state_ = state_kp;
    vc_ = vc_kp;
    if (converged) {
      // Pack results into ContactSolverResults.
      results->Resize(nv, nc);
      results->v_next = state_.v();
      ExtractNormal(vc_, &results->vn);
      ExtractTangent(vc_, &results->vt);
      ExtractNormal(state_.gamma(), &results->fn);
      ExtractTangent(state_.gamma(), &results->ft);
      // N.B. While contact solver works with impulses, results are reported as
      // forces.
      results->fn /= time_step;
      results->ft /= time_step;
      results->tau_contact = jc_ / time_step;

      return ContactSolverStatus::kSuccess;
    }
  }

  // Pack results into ContactSolverResults.
  results->Resize(nv, nc);
  results->v_next = state_.v();
  ExtractNormal(vc_, &results->vn);
  ExtractTangent(vc_, &results->vt);
  ExtractNormal(state_.gamma(), &results->fn);
  ExtractTangent(state_.gamma(), &results->ft);
  // N.B. While contact solver works with impulses, results are reported as
  // forces.
  results->fn /= time_step;
  results->ft /= time_step;
  results->tau_contact = jc_ / time_step;

  // N.B. We always return success for PGS so that we can look at the solution
  // even if not converged.
  return ContactSolverStatus::kSuccess;
}

template <typename T>
void PgsSolver<T>::PreProcessData(const SystemDynamicsData<T>& dynamics_data,
                                  const PointContactData<T>& contact_data) {
  const int nc = contact_data.num_contacts();
  const int nv = dynamics_data.num_velocities();
  state_.Resize(nv, nc);
  pre_proc_data_.Resize(nv, nc);
  jc_.resize(nv);
  vc_.resize(3 * nc);

  // Aliases to data.
  const auto& Ainv = dynamics_data.get_Ainv();
  const auto& v_star = dynamics_data.get_v_star();
  const auto& Jc = contact_data.get_Jc();

  if (nc != 0) {
    Jc.Multiply(v_star, &pre_proc_data_.vc_star);

    auto& W = pre_proc_data_.W;
    this->FormDelassusOperatorMatrix(Jc, Ainv, Jc, &W);
    PRINT_VARn(W);

    // Compute scaling factors, one per contact.
    auto& Wii_norm = pre_proc_data_.Wii_norm;
    auto& Dinv = pre_proc_data_.Dinv;
    for (int i = 0; i < nc; ++i) {
      // 3x3 diagonal block. It might be singular, but definitely non-zero.
      // That's why we use an rms norm.
      const auto& Wii = W.block(3 * i, 3 * i, 3, 3);
      Wii_norm(i) = Wii.norm() / 3;  // 3 = sqrt(9).
      Dinv.template segment<3>(3 * i).setConstant(1.0 / Wii_norm(i));
    }
    PRINT_VAR(Wii_norm.transpose());
    PRINT_VAR(Dinv.transpose());
  }
}

template <typename T>
bool PgsSolver<T>::VerifyConvergenceCriteria(const VectorX<T>& vc,
                                             const VectorX<T>& vc_kp,
                                             const VectorX<T>& gamma,
                                             const VectorX<T>& gamma_kp,
                                             double omega, double* vc_err,
                                             double* gamma_err) const {
  using std::max;
  const auto& Wii_norm = pre_proc_data_.Wii_norm;
  bool converged = true;
  *vc_err = 0;
  *gamma_err = 0;
  for (int ic = 0; ic < num_contacts(); ++ic) {
    auto within_error_bounds = [&p = parameters_](const T& error,
                                                  const T& scale) {
      const T bounds = p.abs_tolerance + p.rel_tolerance * scale;
      return error < bounds;
    };
    // Check velocity convergence.
    const auto vci = vc.template segment<3>(3 * ic);
    const auto vci_kp = vc_kp.template segment<3>(3 * ic);
    const T vc_norm = vci.norm();
    const T vci_err = (vci_kp - vci).norm() / omega;
    *vc_err = max(*vc_err, vci_err);
    if (!within_error_bounds(vci_err, vc_norm)) {
      converged = false;
    }

    // Check impulse convergence. Scaled to velocity so that its convergence
    // metric is compatible with that of contact velocity.
    const auto gi = gamma.template segment<3>(3 * ic);
    const auto gi_kp = gamma_kp.template segment<3>(3 * ic);
    const T g_norm = gi.norm() * Wii_norm(ic);
    T g_err = (gi_kp - gi).norm() * Wii_norm(ic) / omega;
    *gamma_err = max(*gamma_err, g_err);
    if (!within_error_bounds(g_err, g_norm)) {
      converged = false;
    }
  }
  return converged;
}

template <typename T>
void PgsSolver<T>::ProjectAllImpulses(const VectorX<T>& vc,
                                      const VectorX<T>& mu,
                                      VectorX<T>* gamma_inout) const {
  VectorX<T>& gamma = *gamma_inout;
  for (int ic = 0; ic < num_contacts(); ++ic) {
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
  const double v_eps = parameters_.abs_tolerance;
  const double v_eps2 = v_eps * v_eps;
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

template class ::drake::multibody::contact_solvers::internal::PgsSolver<double>;
