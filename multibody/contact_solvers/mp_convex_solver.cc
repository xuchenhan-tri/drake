#include "drake/multibody/contact_solvers/mp_convex_solver.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#include "fmt/format.h"

#include "drake/multibody/contact_solvers/friction_cone_constraint.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/nlopt_solver.h"
#include "drake/solvers/snopt_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

using drake::solvers::Binding;
using drake::solvers::GurobiSolver;
using drake::solvers::MosekSolver;
using drake::solvers::IpoptSolver;
using drake::solvers::LorentzConeConstraint;
using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::QuadraticCost;
using drake::solvers::SolverId;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;

using Eigen::SparseMatrix;
using Eigen::SparseVector;

using clock = std::chrono::steady_clock;

template <typename T>
MpConvexSolver<T>::MpConvexSolver() {}

template <typename T>
ContactSolverStatus MpConvexSolver<T>::SolveWithGuess(
    const T& time_step, const SystemDynamicsData<T>& dynamics_data,
    const PointContactData<T>& contact_data, const VectorX<T>& v_guess,
    ContactSolverResults<T>* result) {
  throw std::logic_error("Only T = double is supported.");
}

template <>
ContactSolverStatus MpConvexSolver<double>::SolveWithGuess(
    const double& time_step, const SystemDynamicsData<double>& dynamics_data,
    const PointContactData<double>& contact_data,
    const VectorX<double>& v_guess, ContactSolverResults<double>* results) {
  using std::abs;
  using std::max;

  const auto global_start_time = clock::now();

  const int nv = dynamics_data.num_velocities();
  const int nc = contact_data.num_contacts();
  const int nc3 = 3 * nc;

  // Quick exit when there is no contact.
  stats_ = {};
  if (nc == 0) {
    results->Resize(nv, nc);
    results->tau_contact.setZero();
    results->v_next = dynamics_data.get_v_star();
    stats_.iteration_errors = MpConvexSolverErrorMetrics{};
    stats_history_.push_back(stats_);
    const auto global_end_time = clock::now();
    stats_.total_time =
        std::chrono::duration<double>(global_end_time - global_start_time)
            .count();
    return ContactSolverStatus::kSuccess;
  }

  const auto preproc_start_time = clock::now();
  pre_proc_data_ = PreProcessData(time_step, dynamics_data, contact_data);
  const auto preproc_end_time = clock::now();
  stats_.preproc_time =
      std::chrono::duration<double>(preproc_end_time - preproc_start_time)
          .count();
  state_.Resize(pre_proc_data_.nv, pre_proc_data_.nc);

  // Aliases to data.  
  const auto& v_star = dynamics_data.get_v_star();
  const auto& Jc = contact_data.get_Jc();
  const auto& mu = contact_data.get_mu();

  // Aliases to pre-processed (const) data.
  const auto& vc_hat = pre_proc_data_.vc_hat;
  const auto& Rinv = pre_proc_data_.Rinv;
  const auto& mu_tilde = pre_proc_data_.mu_tilde;

  const auto mp_setup_start_time = clock::now();
  drake::solvers::MathematicalProgram prog_;
  SetUpSolver(&prog_);
  const auto mp_setup_end_time = clock::now();
  stats_.mp_setup_time =
      std::chrono::duration<double>(mp_setup_end_time - mp_setup_start_time)
          .count();

  // Make solver based on id.
  std::optional<SolverId> optional_id = parameters_.solver_id;
  const SolverId best_solver_id = drake::solvers::ChooseBestSolver(prog_);
  const SolverId solver_id = optional_id ? *optional_id : best_solver_id;
  std::unique_ptr<drake::solvers::SolverInterface> solver =
      MakeSolver(solver_id);
  if (!solver->available()) {
    throw std::runtime_error("Solver '" + solver_id.name() +
                             "' not available.");
  }

  // Compute initial guess. We user our analytical inverse dynamics.
  VectorX<double>& gamma = state_.mutable_gamma();
  VectorX<double>& vc = state_.mutable_cache().vc;
  Jc.Multiply(v_guess, &vc);
  CalcInverseDynamics(pre_proc_data_, vc, &gamma);

  const auto solver_start_time = clock::now();
  MathematicalProgramResult result;
  VectorXd x_guess;
  if (parameters_.primal) {
    x_guess.resize(nv + nc3);
    x_guess << v_guess, gamma;
  } else {    
    x_guess.resize(nc3);
    x_guess = gamma;
  }
  solver->Solve(prog_, x_guess, {}, &result);
  const auto solver_end_time = clock::now();
  stats_.solver_time =
      std::chrono::duration<double>(solver_end_time - solver_start_time)
          .count();
  if (!result.is_success()) return ContactSolverStatus::kFailure;

  // Update generalized velocities and contact velocities.
  results->Resize(nv, nc);
  VectorX<double>& tau_c = results->tau_contact;

  if (!parameters_.primal) {    
    state_.mutable_gamma() = result.GetSolution();    
    VectorX<double>& v = results->v_next;    

    // Given the impulses, update generalized and contact velocities.
    // TODO: Use this velocity for inverse dynamics instead of separately doing
    // vc = W*gamma + vc_star above (thus twice).
    // Update generalized velocities; v = v* + M⁻¹⋅Jᵀ⋅γ.
    Jc.MultiplyByTranspose(gamma, &tau_c);  // tau_c = Jᵀ⋅γ    
    const auto& Ainv = dynamics_data.get_Ainv();
    Ainv.Multiply(tau_c, &v);               // v = M⁻¹⋅Jᵀ⋅γ
    v += v_star;                            // v = v* + M⁻¹⋅Jᵀ⋅γ
    Jc.Multiply(v, &vc);                    // vc = J⋅v
  } else {
    // Primal. Update: v, gamma, vc, tau_c.
    VectorX<double>& v = results->v_next;
    v = result.GetSolution(v_variable_);
    gamma = result.GetSolution(sigma_variable_);
    Jc.Multiply(v, &vc);
    Jc.MultiplyByTranspose(gamma, &tau_c);
  }

  // Compute analytical inverse dynamics to verify the solution.
  VectorX<double> gamma_id(nc3);
  CalcInverseDynamics(pre_proc_data_, vc, &gamma_id);

  // Verify convergence and update stats.
  MpConvexSolverErrorMetrics error_metrics;
  error_metrics.id_abs_error = (gamma_id - gamma).norm();
  error_metrics.gamma_norm = gamma.norm();
  error_metrics.id_rel_error =
      error_metrics.id_abs_error / error_metrics.gamma_norm;

  // N.B. For testing only. This will be removed in production code.
  // Compute the maximum effective stiction tolerance.
  using std::max;
  error_metrics.vs_max = -1.0;  // Value for all sliding contacts.
  const VectorX<double> y = -(Rinv.asDiagonal() * (vc - vc_hat));
  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    const auto g_ic = gamma.template segment<3>(ic3);
    const auto gt = g_ic.template head<2>();
    const double gn = g_ic(2);

    const auto y_ic = y.template segment<3>(ic3);
    const auto yt = y_ic.template head<2>();
    const double yn = y_ic(2);

    // const auto vc_ic = vc.template segment<3>(ic3);
    const auto vt = vc.template head<2>();

    const double gr = gt.norm();
    const double vr = vt.norm();
    const double yr = yt.norm();
    if (yr < mu_tilde(ic) * yn) {  // In stiction.
      const double vs = mu(ic) * gn / (gr + 1.0e-14) * vr;
      error_metrics.vs_max = max(vs, error_metrics.vs_max);
    }
  }

  // Pack results into ContactSolverResults.
  ExtractNormal(vc, &results->vn);
  ExtractTangent(vc, &results->vt);
  ExtractNormal(gamma, &results->fn);
  ExtractTangent(gamma, &results->ft);
  // N.B. While contact solver works with impulses, results are reported as
  // forces.
  results->fn /= time_step;
  results->ft /= time_step;
  results->tau_contact = tau_c / time_step;

  const auto global_end_time = clock::now();
  stats_.total_time =
      std::chrono::duration<double>(global_end_time - global_start_time)
          .count();

  // Update solution statistics.
  stats_.num_contacts = nc;
  stats_.iteration_errors = error_metrics;
  stats_history_.push_back(stats_);

  return ContactSolverStatus::kSuccess;
}

template <typename T>
typename MpConvexSolver<T>::PreProcessedData MpConvexSolver<T>::PreProcessData(
    const T& time_step, const SystemDynamicsData<T>& dynamics_data,
    const PointContactData<T>& contact_data) const {
  const int nc = contact_data.num_contacts();
  const int nv = dynamics_data.num_velocities();
  PreProcessedData data;
  data.Resize(nv, nc);

  // Aliases to data.  
  const auto& v_star = dynamics_data.get_v_star();
  const auto& Jc = contact_data.get_Jc();

  if (nc != 0) {
    // Common data.
    data.mu = contact_data.get_mu();  // just a handy copy.

    // Both primal and dual use compliance information.
    // We use compliance to compute R.
    auto& Rt = data.Rt;
    auto& Rn = data.Rn;
    auto& R = data.R;
    auto& Rinv = data.Rinv;
    auto& mu_tilde = data.mu_tilde;
    auto& vc_hat = data.vc_hat;
    const auto& phi0 = contact_data.get_phi0();
    const auto& stiffness = contact_data.get_stiffness();
    const auto& dissipation = contact_data.get_dissipation();
    const auto& mu = contact_data.get_mu();
    for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
      // Regularization.
      auto Ric = R.template segment<3>(ic3);
      const T& k = stiffness(ic);
      const T& c = dissipation(ic);
      const T taud = (c == 0) ? 0.0 : c / k;  // Damping rate.
      Rn(ic) = 1.0 / (time_step * time_step * k * (1.0 + taud / time_step));
      DRAKE_DEMAND(Rn(ic) > 0);
      Rt(ic) = parameters_.Rt_factor * Rn(ic);
      Ric = Vector3<T>(Rt(ic), Rt(ic), Rn(ic));
      mu_tilde(ic) = sqrt(Rt(ic) / Rn(ic)) * mu(ic);

      // Stabilization velocity.
      const T vn_hat = -phi0(ic) / (time_step + taud);
      vc_hat.template segment<3>(ic3) = Vector3<T>(0, 0, vn_hat);
    }
    Rinv = R.cwiseInverse();    

    if (parameters_.primal) {
      // Preprocess for primal formulation.
      dynamics_data.get_A().AssembleMatrix(&data.M);
      contact_data.get_Jc().AssembleMatrix(&data.Jc);
      data.v_star = v_star;
    } else {
      // Preprocess for dual formulation.
      Jc.Multiply(v_star, &data.vc_star);
      data.r = data.vc_star - vc_hat;

      const auto& Ainv = dynamics_data.get_Ainv();
      auto& W = data.W;
      this->FormDelassusOperatorMatrix(Jc, Ainv, Jc, &W);
      // Compute scaling factors, one per contact.
      // Notice mi and gi contain the diagonal component R.
      auto& gi = data.gi;
      auto& mi = data.mi;
      for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
        // 3x3 diagonal block. It might be singular, but definitely non-zero.
        // That's why we use an rms norm.
        const auto& Wii = W.block(ic3, ic3, 3, 3);
        gi(ic) = Matrix3<T>(Wii).trace() / 3;
        mi(ic) = 1.0 / gi(ic);
      }

      // Add regularization to Delassus operator into N = W + diag(R).
      auto& N = data.N;
      // N.B. We add full 3x3 diagonal blocks so that the pattern of N matches
      // the pattern of the Hessian even if those blocks are missing in N.
      std::vector<Matrix3<T>> diagonal_blocks(nc, Matrix3<T>::Zero());
      for (int ic = 0; ic < nc; ic++) {
        diagonal_blocks[ic] = Vector3<T>(Rt(ic), Rt(ic), Rn(ic)).asDiagonal();
      }
      InsertSparseBlockDiagonal(diagonal_blocks, &N);
      N += W;
      N.makeCompressed();
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    // THIS CODE EMULATE "RIGID" CONTACT. WE HOWEVER USE COMPLIANCE INSTEAD.
    ///////////////////////////////////////////////////////////////////////////
#if 0        
    // Estimate regularization parameters.
    // We estimate a diagonal regularization Rn as a factor of the gᵢ =
    // trace(Wii)/3.
    // That is, Rn = eps * gᵢ.
    // The value of eps is estimated so that the numerical compliance time
    // scale is the contact_period below.

    // We note that compliance frequency is ω₀² = k/mᵢ.
    // We propose a period T₀ = α⋅dt and frequency ω₀ = 2π/T₀.
    // Then the stiffness will be k = ω₀² mᵢ.
    // Since Rn = (dt²k)⁻¹ we have that: Rn = gᵢ/(ω₀dt)² = ε⋅gᵢ.
    // With ε = 1/(ω₀dt)².
    const T contact_period = parameters_.alpha * time_step;
    const T w0 = 2.0 * M_PI / contact_period;
    const T w0_hat = w0 * time_step;
    const T eps = 1.0 / (w0_hat * w0_hat);
    // For the tangential compliance we use Rt = Rn * Rt_factor.
    // Usually we look for Rt_factor << 1
    const auto& mu = data.mu;
    auto& Rt = data.Rt;
    auto& Rn = data.Rn;
    auto& R = data.R;
    auto& Rinv = data.Rinv;
    auto& mu_tilde = data.mu_tilde;
    for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
      Rn(ic) = eps * gi(ic);
      Rt(ic) = parameters_.Rt_factor * Rn(ic);
      R.template segment<3>(ic3) = Vector3<T>(Rt(ic), Rt(ic), Rn(ic));
      Rinv.template segment<3>(ic3) =
          Vector3<T>(Rt(ic), Rt(ic), Rn(ic)).cwiseInverse();
      mu_tilde(ic) = sqrt(Rt(ic) / Rn(ic)) * mu(ic);
    }
#endif
    ///////////////////////////////////////////////////////////////////////////
    // END OF "RIGID" CONTACT.
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////    
  }  // nc != 0

  return data;
}

template <typename T>
void MpConvexSolver<T>::SetUpSolver(
    drake::solvers::MathematicalProgram* prog) {
  const VectorX<T>& mu = pre_proc_data_.mu;
  const VectorX<T>& Rn = pre_proc_data_.Rn;
  const VectorX<T>& Rt = pre_proc_data_.Rt;
  const VectorX<T>& vc_hat = pre_proc_data_.vc_hat;

  const int nc = num_contacts();
  const int nv = num_velocities();

  if (parameters_.primal) {
    drake::solvers::VectorXDecisionVariable v =
        prog->NewContinuousVariables(nv);
    drake::solvers::VectorXDecisionVariable sigma =
        prog->NewContinuousVariables(3 * nc);
    v_variable_.resize(nv);
    sigma_variable_.resize(3 * nc);
    v_variable_ = v;
    sigma_variable_ = sigma;
    const auto& M = pre_proc_data_.M;
    const auto& Jc = pre_proc_data_.Jc;
    const auto& v_star = pre_proc_data_.v_star;

    // Add 0.5 (v-v_star)'*M*(v-v_star)
    // N.B. Notice we must include the 1/2 factor in Q.
    const MatrixX<T> Qv = 0.5 * M;
    prog->AddQuadraticErrorCost(Qv, v_star, v);

    // Add regularizer.
    const MatrixX<T> Qs = pre_proc_data_.R.asDiagonal();
    const VectorX<T> bs = VectorX<T>::Zero(3 * nc);
    prog->AddQuadraticCost(Qs, bs, sigma);

    // Add constraint g = J * v - v_hat + R * sigma ∈ ℱ*.
    MatrixXd A = MatrixXd::Zero(3, nv + 3);
    Vector3d b;
    for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {      
      A.setZero();
      const auto Jn = Jc.block(ic3 + 2, 0, 1, nv);
      A.topLeftCorner(1, nv) = Jn / mu(ic);
      A(0, nv + 2) = Rn(ic) / mu(ic);

      const auto Jt = Jc.block(ic3, 0, 2, nv);
      A.bottomLeftCorner(2, nv) = Jt;
      A(1, nv) = Rt(ic);
      A(2, nv + 1) = Rt(ic);

      const auto vc_hat_ic = vc_hat.template segment<3>(ic3);
      b = -Vector3d(vc_hat_ic(2) / mu(ic), vc_hat_ic(0), vc_hat_ic(1));

      const auto sigma_ic = sigma.template segment<3>(ic3);
      drake::solvers::VectorXDecisionVariable x_ic(nv + 3);
      x_ic << v, sigma_ic;
      auto cone_constraint = std::make_shared<LorentzConeConstraint>(
          A, b, LorentzConeConstraint::EvalType::kConvexSmooth);
      Binding<LorentzConeConstraint> binding(cone_constraint, x_ic);
      prog->AddConstraint(binding);      
    }

  } else {
    drake::solvers::VectorXDecisionVariable gamma_ =
        prog->NewContinuousVariables(3 * nc);

    const MatrixX<T> Q(pre_proc_data_.N);
    prog->AddQuadraticCost(Q, pre_proc_data_.r, gamma_);

    // Add constraint ‖γₜ‖ ≤ μ⋅γₙ
    for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
      const auto gamma_ic = gamma_.template segment<3>(ic3);
      // Add custom FrictionConeConstraint.
      // N.B. MP does not flag it in any special way and treats it as a generic
      // constraints. Therefore we use Lorentz constraint below.
      // auto cone_constraint =
      // std::make_shared<FrictionConeConstraint>(mu(ic));
      // Binding<FrictionConeConstraint> binding(cone_constraint, gamma_ic);
      Matrix3d A = Matrix3d::Zero();
      A(0, 2) = mu(ic);
      A.bottomLeftCorner<2, 2>().setIdentity();
      const Vector3d b = Vector3d::Zero();
      auto cone_constraint = std::make_shared<LorentzConeConstraint>(
          A, b, LorentzConeConstraint::EvalType::kConvexSmooth);
      Binding<LorentzConeConstraint> binding(cone_constraint, gamma_ic);
      prog->AddConstraint(binding);
    }
  }

  // Solver options.
  // Level 5 is the default for Ipopt.
  prog->SetSolverOption(IpoptSolver::id(), "print_level", 5);
  prog->SetSolverOption(IpoptSolver::id(), "print_user_options", "yes");

  // Default value:	1e-8
  // Minimum value:	0.0
  // Maximum value:	1.0
  prog->SetSolverOption(GurobiSolver::id(), "BarConvTol", 1.0e-8);

  // Default value:	1e-6
  // Minimum value:	1e-9
  // Maximum value:	1e-2
  prog->SetSolverOption(GurobiSolver::id(), "FeasibilityTol", 1.0e-6);

  // Default value:	1e-6
  // Minimum value:	1e-9
  // Maximum value:	1e-2
  prog->SetSolverOption(GurobiSolver::id(), "OptimalityTol", 1.0e-6);

  // Acquiring the license is an time consumming process. We do it once on the
  // first solve and keep a shared pointer to the license alive.
  if (parameters_.solver_id) {
    if (*parameters_.solver_id == GurobiSolver::id())
      gurobi_license_ = GurobiSolver::AcquireLicense();
    else if (*parameters_.solver_id == MosekSolver::id())
      mosek_license_ = MosekSolver::AcquireLicense();
  }
}

template <typename T>
void MpConvexSolver<T>::CalcInverseDynamics(const PreProcessedData& data,
                                            const VectorX<T>& vc,
                                            VectorX<T>* gamma) const {
  using std::abs;
  using std::sqrt;
  const VectorX<T>& Rn = data.Rn;
  const VectorX<T>& Rt = data.Rt;
  const VectorX<T>& mu = data.mu;
  const VectorX<T>& mu_tilde = data.mu_tilde;
  const VectorX<T>& vc_hat = data.vc_hat;

  for (int ic = 0, ic3 = 0; ic < num_contacts(); ic++, ic3 += 3) {
    auto vc_hat_ic = vc_hat.template segment<3>(ic3);
    auto vc_ic = vc.template segment<3>(ic3);
    const Vector3<T> vc_tilde_ic = vc_ic - vc_hat_ic;

    // y = -R⁻¹⋅vc_tilde_ic
    const Vector3<T> y_ic(-vc_tilde_ic(0) / Rt(ic), -vc_tilde_ic(1) / Rt(ic),
                          -vc_tilde_ic(2) / Rn(ic));
    Vector3<T> fperp;  // The direction of the projection.
    const Vector3<T> gamma_id =
        ProjectImpulse(mu(ic), mu_tilde(ic), Rt(ic) / Rn(ic), y_ic, &fperp);
    gamma->template segment<3>(ic3) = gamma_id;
  }
}

template <typename T>
void MpConvexSolver<T>::InsertSparseBlockDiagonal(
    const std::vector<Matrix3<T>>& blocks, SparseMatrix<T>* S) const {
  const int nb = blocks.size();
  const int n = 3 * nb;
  S->resize(n, n);
  S->reserve(Eigen::VectorXi::Constant(n, 3));  // 3 non-zeros per column.
  for (int b = 0; b < nb; b++) {
    const Matrix3<T>& B = blocks[b];
    for (int i = 0; i < 3; ++i) {
      const int ik = 3 * b + i;
      for (int j = 0; j < 3; ++j) {
        const int jk = 3 * b + j;
        S->insert(ik, jk) = B(i, j);
      }
    }
  }
}

template <typename T>
void MpConvexSolver<T>::AddSparseBlockDiagonal(
    const std::vector<Matrix3<T>>& blocks, SparseMatrix<T>* S) const {
  const int nb = blocks.size();
  const int n = 3 * nb;
  DRAKE_DEMAND(S->rows() == n);
  DRAKE_DEMAND(S->cols() == n);
  // S->resize(n, n);
  // S->reserve(Eigen::VectorXi::Constant(n, 3));  // 3 non-zeros per column.
  for (int b = 0; b < nb; b++) {
    const Matrix3<T>& B = blocks[b];
    for (int i = 0; i < 3; ++i) {
      const int ik = 3 * b + i;
      for (int j = 0; j < 3; ++j) {
        const int jk = 3 * b + j;
        S->coeffRef(ik, jk) += B(i, j);
      }
    }
  }
}

template <typename T>
Vector3<T> MpConvexSolver<T>::ProjectImpulse(
    const T& mu, const T& mu_tilde, const T& Rt_over_Rn,
    const Eigen::Ref<const Vector3<T>>& gamma, Vector3<T>* fperp) const {
  using std::sqrt;
  // Notation:
  //   gn: gamma(2)
  //   gt: gamma.head<2>();
  //   gr: gt.norm()
  // We use "tilde" to denote the change of variables:
  //   gamma_tilde = sqrt(R) * gamma
  // with R = diag({Rt, Rt, Rn})
  //
  // mu_tilde = mu * sqrt(Rt/Rn)
  const T& gn = gamma(2);
  const Vector2<T> gt = gamma.template head<2>();
  const T gr_squared = gt.squaredNorm();
  const T gr = sqrt(gr_squared);

  // Build the projection direction fperp.
  // constexpr double kEpsilonSquared = 1.0e-14;
  const T gr_soft = sqrt(gr_squared + kEpsilonSquared);
  const Vector2<T> that = gt / gr_soft;
  fperp->template head<2>() = that;
  (*fperp)(2) = -mu * Rt_over_Rn;

  // Region III: Inside the polar cone gn_tile < -mu_tilde * gr_tilde.
  // This is equivalent to: gn < -mu * Rt/Rn * gr
  // Notice that in the limit Rt/Rn --> 0 we recover the original Coulomb
  // cone.
  if (gn < -mu * Rt_over_Rn * gr) {
    return Vector3<T>::Zero();
  }

  // Region I: Inside the friction cone.
  // gr_tilde <= mu_tilde * gn_tilde is equivalent to gr < mu * gn.
  if (gr <= mu * gn) {
    return gamma;
  }

  // Region II: Inside the region between the cone and its polar. We need to
  // project.
  // We build G such that the projection is Π(γ) = G⋅γ.
  Matrix3<T> G;
  const T mu_tilde2 = mu_tilde * mu_tilde;
  G.template topLeftCorner<2, 2>() = mu_tilde2 * that * that.transpose();
  G.template topRightCorner<2, 1>() = mu * that;
  G.template bottomLeftCorner<1, 2>() = mu * Rt_over_Rn * that.transpose();
  G(2, 2) = 1.0;
  G /= (1.0 + mu_tilde2);
  // N.B. In the limit Rt_over_Rn --> 0, G projects on the Coulomb cone:
  // G = |0 μt̂ |
  //     |0  1 |

  // If we are here, then we are in Region II and we need to project:
  return G * gamma;
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

template class ::drake::multibody::contact_solvers::internal::MpConvexSolver<
    double>;
