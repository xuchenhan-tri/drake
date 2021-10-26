#include "drake/multibody/contact_solvers/mp_primal_solver.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>

#include "fmt/format.h"

#include "drake/common/test_utilities/limit_malloc.h"
#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#include "drake/multibody/contact_solvers/timer.h"
#include "drake/solvers/choose_best_solver.h"

#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

using drake::solvers::Binding;
using drake::solvers::GurobiSolver;
using drake::solvers::LorentzConeConstraint;
using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::MosekSolver;
using drake::solvers::QuadraticCost;
using drake::solvers::SolverId;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

template <typename T>
MpPrimalSolver<T>::MpPrimalSolver()
    : ConvexSolverBase<T>(
          {parameters_.Rt_factor, parameters_.alpha, parameters_.sigma}) {}

template <typename T>
ContactSolverStatus MpPrimalSolver<T>::DoSolveWithGuess(
    const typename ConvexSolverBase<T>::PreProcessedData& data,
    const VectorX<T>& v_guess, ContactSolverResults<T>* result) {
  throw std::logic_error("Only T = double is supported.");
}

template <>
ContactSolverStatus MpPrimalSolver<double>::DoSolveWithGuess(
    const ConvexSolverBase<double>::PreProcessedData& data,
    const VectorX<double>& v_guess, ContactSolverResults<double>* results) {
  using std::abs;
  using std::max;

  // Starts a timer for the overall execution time of the solver.
  Timer global_timer;

  const auto& dynamics_data = *data.dynamics_data;
  const auto& contact_data = *data.contact_data;

  const int nv = dynamics_data.num_velocities();
  const int nc = contact_data.num_contacts();
  const int nc3 = 3 * nc;

  // The primal method needs the inverse dynamics data.
  DRAKE_DEMAND(dynamics_data.has_inverse_dynamics());

  // We should not attempt solving zero sized problems for no reason since the
  // solution is trivially v = v*.
  DRAKE_DEMAND(nc != 0);

  // Workspace variables.
  VectorX<double> xc_work1(nc3);

  // Reset stats.
  stats_ = {};
  stats_.num_contacts = nc;
  // Log the time it took to pre-process data.
  stats_.preproc_time = this->pre_process_time();

  // Print stuff for debugging.
  // TODO: refactor into PrintProblemSize().
  if (parameters_.verbosity_level >= 1) {
    PRINT_VAR(nv);
    PRINT_VAR(nc);
    PRINT_VAR(data_.Mt.size());
    PRINT_VAR(data_.Mblock.rows());
    PRINT_VAR(data_.Mblock.cols());
    PRINT_VAR(data_.Mblock.num_blocks());

    PRINT_VAR(data_.Jblock.block_rows());
    PRINT_VAR(data_.Jblock.block_cols());
    PRINT_VAR(data_.Jblock.rows());
    PRINT_VAR(data_.Jblock.cols());
    PRINT_VAR(data_.Jblock.num_blocks());
  }
  // TODO: refactor into PrintProblemStructure().
  if (parameters_.verbosity_level >= 2) {
    for (const auto& [p, t, Jb] : data_.Jblock.get_blocks()) {
      std::cout << fmt::format("(p,t) = ({:d},{:d}). {:d}x{:d}.\n", p, t,
                               Jb.rows(), Jb.cols());
    }
  }

  // TODO: make prog_ a member that SetUpProgram modifies.
  // TODO: add timing stats withing that method.
  Timer timer;
  drake::solvers::MathematicalProgram prog;
  drake::solvers::VectorXDecisionVariable v_variable;
  drake::solvers::VectorXDecisionVariable sigma_variable;
  SetUpProgram(data, &prog, &v_variable, &sigma_variable);
  stats_.mp_setup_time = timer.Elapsed();

  // Make solver based on id.
  std::optional<SolverId> optional_id = parameters_.solver_id;
  const SolverId solver_id =
      optional_id ? *optional_id : drake::solvers::ChooseBestSolver(prog);
  std::unique_ptr<drake::solvers::SolverInterface> solver =
      MakeSolver(solver_id);
  if (!solver->available()) {
    throw std::runtime_error("Solver '" + solver_id.name() +
                             "' not available.");
  }

  // Compute initial guess. We use our analytical inverse dynamics.
  State state(nv, nc);
  {
    VectorX<double>& gamma = state.mutable_cache().gamma_id;
    VectorX<double>& vc = state.mutable_cache().vc;
    const auto& Jc = data.contact_data->get_Jc();
    Jc.Multiply(v_guess, &vc);
    this->CalcAnalyticalInverseDynamics(parameters_.soft_tolerance, vc, &gamma);

    state.mutable_v() = v_guess;
    state.mutable_sigma() = gamma;
  }

  timer.Reset();
  MathematicalProgramResult result;
  VectorXd x_guess;
  x_guess.resize(nv + nc3);
  x_guess << v_guess, gamma;
  solver->Solve(prog, x_guess, {}, &result);
  stats_.solver_time = timer.Elapsed();
  if (!result.is_success()) return ContactSolverStatus::kFailure;

  if (*parameters_.solver_id == GurobiSolver::id()) {
    const solvers::GurobiSolverDetails& details =
        result.get_solver_details<GurobiSolver>();
    stats_.gurobi_time = details.optimizer_time;
    stats_.gurobi_barrier_iterations = details.num_barrier_iterations;
  }

  // Load the solution from MathematicalProgramResult into the state.
  state.mutable_v() = result.GetSolution(v_variable);
  state.mutable_sigma() = result.GetSolution(sigma_variable);

  // Update contact velocities and analytical inverse dynamics to verify the
  // solution.
  {
    VectorX<double>& gamma = state.mutable_cache().gamma_id;
    VectorX<double>& vc = state.mutable_cache().vc;
    const auto& Jc = data.contact_data->get_Jc();
    Jc.Multiply(state.v(), &vc);
    this->CalcAnalyticalInverseDynamics(parameters_.soft_tolerance, vc, &gamma);
  }

  // Verify convergence and update stats.
  MpPrimalSolverErrorMetrics& error_metrics = stats_.error_metrics;
  const VectorX<double>& gamma = state.sigma();
  const VectorX<double>& gamma_id = state.cache().gamma_id;
  error_metrics.id_abs_error = (gamma_id - gamma).norm();
  error_metrics.id_rel_max =
      ((gamma_id - gamma).cwiseAbs().array() / gamma.cwiseAbs().array())
          .maxCoeff();
  error_metrics.gamma_norm = gamma.norm();
  error_metrics.id_rel_error = error_metrics.id_abs_error /
                               max(error_metrics.gamma_norm, gamma_id.norm());
  std::tie(error_metrics.mom_l2, error_metrics.mom_max) =
      this->CalcScaledMomentumError(data, state.v(), state.sigma());
  std::tie(error_metrics.mom_rel_l2, error_metrics.mom_rel_max) =
      this->CalcRelativeMomentumError(data, state.v(), state.sigma());
  this->CalcEnergyMetrics(data, state.v(), state.sigma(), &error_metrics.Ek,
                          &error_metrics.costM, &error_metrics.costR,
                          &error_metrics.cost);
  error_metrics.opt_cond =
      this->CalcOptimalityCondition(data_, state.v(), state.sigma(), &xc_work1);

  PackContactResults(data, state.v(), state.cache().vc, state.sigma(), results);

  stats_.total_time = global_timer.Elapsed();
  // Update stats history.
  if (parameters_.log_stats) stats_history_.push_back(stats_);

  total_time_ += global_timer.Elapsed();

  return ContactSolverStatus::kSuccess;
}

template <typename T>
void MpPrimalSolver<T>::SetUpProgram(
    const typename ConvexSolverBase<T>::PreProcessedData& data,
    drake::solvers::MathematicalProgram* prog,
    drake::solvers::VectorXDecisionVariable* v_variable,
    drake::solvers::VectorXDecisionVariable* sigma_variable) {
  // Aliases to data.
  const int nc = data.nc;
  const int nv = data.nv;
  const VectorX<T>& mu = data.contact_data->get_mu();
  const VectorX<T>& R = data.R;
  const VectorX<T>& v_star = data.dynamics_data->get_v_star();
  const VectorX<T>& vc_stab = data.vc_stab;

  auto& v = *v_variable;
  auto& sigma = *sigma_variable;
  v = prog->NewContinuousVariables(nv);
  sigma = prog->NewContinuousVariables(3 * nc);

  // const auto& M = pre_proc_data_.M;
  // const auto& Jc = pre_proc_data_.Jc;
  // const auto& v_star = pre_proc_data_.v_star;

  // TODO: add one block at a time if this turns out to be way too expensive.
  const MatrixX<T> M = data.Mblock.MakeDenseMatrix();

  // Add 0.5 (v-v_star)'*M*(v-v_star)
  // N.B. Notice we must include the 1/2 factor in Q.
  const MatrixX<T> Qv = 0.5 * M;
  prog->AddQuadraticErrorCost(Qv, v_star, v);

  // Add regularizer.
  const MatrixX<T> Qs = R.asDiagonal();
  const VectorX<T> bs = VectorX<T>::Zero(3 * nc);
  prog->AddQuadraticCost(Qs, bs, sigma);

  // TODO: use individual blocks if this is measured to cost too much.
  const MatrixX<T> Jc = data.Jblock.MakeDenseMatrix();

  // Add constraint g = J * v - v_hat + R * sigma ∈ ℱ*.
  MatrixXd A = MatrixXd::Zero(3, nv + 3);
  Vector3d b;
  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    const auto Ric = R.template segment<3>(ic3);
    const T& Rt = Ric(0);
    const T& Rn = Ric(2);

    A.setZero();
    const auto Jn = Jc.block(ic3 + 2, 0, 1, nv);
    A.topLeftCorner(1, nv) = Jn / mu(ic);
    A(0, nv + 2) = Rn / mu(ic);

    const auto Jt = Jc.block(ic3, 0, 2, nv);
    A.bottomLeftCorner(2, nv) = Jt;
    A(1, nv) = Rt;
    A(2, nv + 1) = Rt;

    const auto vc_stab_ic = vc_stab.template segment<3>(ic3);
    b = -Vector3d(vc_stab_ic(2) / mu(ic), vc_stab_ic(0), vc_stab_ic(1));

    const auto sigma_ic = sigma.template segment<3>(ic3);
    drake::solvers::VectorXDecisionVariable x_ic(nv + 3);
    x_ic << v, sigma_ic;
    auto cone_constraint = std::make_shared<LorentzConeConstraint>(
        A, b, LorentzConeConstraint::EvalType::kConvexSmooth);
    Binding<LorentzConeConstraint> binding(cone_constraint, x_ic);
    prog->AddConstraint(binding);
  }

  // Default value:	1e-8
  // Minimum value:	0.0
  // Maximum value:	1.0
  prog->SetSolverOption(GurobiSolver::id(), "BarConvTol", 1.0e-8);

  // Default value:	1e-6
  // Minimum value:	1e-9
  // Maximum value:	1e-2
  prog->SetSolverOption(GurobiSolver::id(), "FeasibilityTol", 1.0e-6);
  if (parameters_.verbosity_level >= 1) {
    //prog->SetSolverOption(GurobiSolver::id(), "LogToConsole", 1);
    prog->SetSolverOption(GurobiSolver::id(), "OutputFlag", 1);
    prog->SetSolverOption(GurobiSolver::id(), "LogFile", parameters_.log_file);
    PRINT_VAR(parameters_.log_file);
  }

  // Default value:	1e-6
  // Minimum value:	1e-9
  // Maximum value:	1e-2
  prog->SetSolverOption(GurobiSolver::id(), "OptimalityTol", 1.0e-6);

  // Default value: 0; It will generally use all of the cores in the machine,
  // but it may choose to use fewer.
  prog->SetSolverOption(GurobiSolver::id(), "Threads", 1);

  // Acquiring the license is an time consumming process. We do it once on the
  // first solve and keep a shared pointer to the license alive.
  if (parameters_.solver_id) {
    if (*parameters_.solver_id == GurobiSolver::id() && !gurobi_license_) {
      gurobi_license_ = GurobiSolver::AcquireLicense();
    } else if (*parameters_.solver_id == MosekSolver::id() && !mosek_license_) {
      mosek_license_ = MosekSolver::AcquireLicense();
    }
  }
}

template <typename T>
void MpPrimalSolver<T>::LogIterationsHistory(
    const std::string& file_name) const {
  std::ofstream file(file_name);
  const std::vector<MpPrimalSolverStats>& stats_hist = get_stats_history();
  file << fmt::format(
      "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
      "num_contacts",
      // ID errors.
      "id_rel_err", "id_rel_max", "id_abs_error", "gamma_norm",
      // Momentum errors.
      "mom_l2", "mom_max", "mom_rel_l2", "mom_rel_max",
      // Wall-clock timings.
      "total_time", "preproc_time", "mp_setup_time", "solver_time",
      // Gurobi specifics.
      "grb_time", "grb_barrier_iterations",
      // Energy metrics.
      "Ek", "ellM", "ellR", "ell", "opt_cond");
  for (const auto& s : stats_hist) {
    file << fmt::format(
        "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
        s.num_contacts,
        // ID errors.
        s.error_metrics.id_rel_error, s.error_metrics.id_rel_max,
        s.error_metrics.id_abs_error, s.error_metrics.gamma_norm,
        // Momentum errors.
        s.error_metrics.mom_l2, s.error_metrics.mom_max,
        s.error_metrics.mom_rel_l2, s.error_metrics.mom_rel_max,
        // Wall-clock timings.
        s.total_time, s.preproc_time, s.mp_setup_time, s.solver_time,
        // Gurobi specifics.
        s.gurobi_time, s.gurobi_barrier_iterations,
        // Energy metrics.
        s.error_metrics.Ek, s.error_metrics.costM, s.error_metrics.costR,
        s.error_metrics.cost, s.error_metrics.opt_cond);
  }
  file.close();
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

template class ::drake::multibody::contact_solvers::internal::MpPrimalSolver<
    double>;
