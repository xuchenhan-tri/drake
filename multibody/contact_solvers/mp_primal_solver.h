#pragma once

#include <iostream>
#include <memory>

#include <Eigen/SparseCore>

#include "drake/multibody/contact_solvers/block_sparse_matrix.h"
#include "drake/multibody/contact_solvers/contact_solver.h"
#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#include "drake/multibody/contact_solvers/convex_solver_base.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

struct MpPrimalSolverParameters {
  std::optional<drake::solvers::SolverId> solver_id{std::nullopt};

  // Tolerance used in impulse soft norms and soft cones. In Ns.
  // TODO(amcastro-tri): Consider this to have units of N and scale by time
  // step.
  double soft_tolerance{1.0e-7};

  // Tangential regularization factor. We make Rt = Rt_factor * Rn.
  double Rt_factor{1.0e-3};

  // Rigid approximation contant: Rₙ = α⋅Wᵢ when the contact frequency ωₙ is
  // below the limit ωₙ⋅dt ≤ 2π. That is, the period is Tₙ = α⋅dt.
  double alpha{1.0};

  // Dimensionless parameterization of the regularization of friction.
  // An approximation for the bound on the slip velocity is vₛ ≈ ε⋅δt⋅g.
  double sigma{1.0e-3};

  bool log_stats{false};

  // The verbosity level determines how much information to print into stdout.
  // These levels are additive. E.g.: level 2 also prints level 0 and 1 info.
  //  0: Nothing gets printed.
  //  1: Prints problem size and error at convergence.
  //  2: Prints sparsity structure.
  //  3: Prints stats at each iteration.
  int verbosity_level{1};
  // We'll dump the output form Gurobi into this file if verbosity_level >= 3.
  std::string log_file{"solver_log_file"};
};

struct MpPrimalSolverErrorMetrics {
  // Error in the impulses relative to the analytical inverse dynamics.
  // Relative, dimensionless.
  double id_rel_error{0.0};
  // Error in the impulses relative to the analytical inverse dynamics. Absolute
  // value, [Ns].
  double id_abs_error{0.0};
  double id_rel_max{0};
  // Norm of the impulses vector.
  double gamma_norm{0.0};

  // L2 norm of the scaled momentum equation (first optimality condition)
  double mom_l2{0};
  // Max norm of the scaled momentum equation (first optimality condition)
  double mom_max{0};

  double mom_rel_l2{0};
  double mom_rel_max{0};

  // Energy metrics.
  double Ek{0}, cost{0}, costM{0}, costR{0};
  double opt_cond{0};
};

struct MpPrimalSolverStats {
  int num_contacts{0};
  MpPrimalSolverErrorMetrics error_metrics;

  // Performance statistics. All these times are in seconds.

  // Total time for the last call to SolveWithGuess().
  double total_time{0};
  // Time used in pre-processing the data: forming the Delassus operator,
  // computing regularization parameters, etc.
  double preproc_time{0};
  // Time used to setup the MathematicalProgram.
  double mp_setup_time{0};
  // Time used by the underlying solver.
  double solver_time{0};

  // Wall-clock in seconds as reported by Gurobi, when using Gurobi.
  double gurobi_time{-1};
  // The number of iterations performed by the Gurobi barrier method.
  int gurobi_barrier_iterations{-1};
};

// This solver uses the regularized convex formulation from [Todorov 2014].
// This class must only implement the API ContactSolver::SolveWithGuess(),
// please refer to the documentation in ContactSolver for details.
//
// - [Todorov, 2014] Todorov, E., 2014, May. Convex and analytically-invertible
// dynamics with contacts and constraints: Theory and implementation in MuJoCo.
// In 2014 IEEE International Conference on Robotics and Automation (ICRA) (pp.
// 6054-6061). IEEE.
template <typename T>
class MpPrimalSolver final : public ConvexSolverBase<T> {
 public:
  MpPrimalSolver();

  virtual ~MpPrimalSolver() = default;

  void set_parameters(MpPrimalSolverParameters& parameters) {
    ConvexSolverBaseParameters base_parameters{
        1.0, parameters.Rt_factor, parameters.alpha, parameters.sigma};
    ConvexSolverBase<T>::set_parameters(base_parameters);
    parameters_ = parameters;
  }

  // Retrieves solver statistics since the last call to SolveWithGuess().
  const MpPrimalSolverStats& get_iteration_stats() const { return stats_; }

  // Retrieves the history of statistics during the lifetime of this solver for
  // each call to SolveWithGuess().
  const std::vector<MpPrimalSolverStats>& get_stats_history() const {
    return stats_history_;
  }

  // Returns the total time spent on calls to SolveWithGuess().
  double get_total_time() const { return total_time_; }

  void LogIterationsHistory(const std::string& file_name) const final;

 private:
  class State {
   public:
    // This is not a real cache in the CS sense (i.e. there is no tracking of
    // dependencies nor automatic validity check) but in the sense that this
    // objects stores computations that are function of the solver's state. It
    // is the responsability of the solver to keep these computations properly
    // in sync.
    struct Cache {
      void Resize(int /*nv*/, int nc) {
        const int nc3 = 3 * nc;
        vc.resize(nc3);
        gamma_id.resize(nc3);
      }
      VectorX<T> vc;        // vc = W⋅γ + vc_star.
      VectorX<T> gamma_id;  // Analytical inverse dynamics, gamma_id = gamma(vc)
    };

    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(State);
    State() = default;
    State(int nv, int nc) { Resize(nv, nc); }

    void Resize(int nv, int nc) {
      nv_ = nv;
      nc_ = nc;
      v_.resize(nv);
      sigma_.resize(3 * nc);
      cache_.Resize(nv, nc);
    }

    int num_velocities() const { return nv_; }
    int num_contacts() const { return nc_; }

    const VectorX<T>& v() const { return v_; }
    VectorX<T>& mutable_v() { return v_; }
    const VectorX<T>& sigma() const { return sigma_; }
    VectorX<T>& mutable_sigma() { return sigma_; }
    const Cache& cache() const { return cache_; }
    Cache& mutable_cache() const { return cache_; }

   private:
    int nv_;
    int nc_;

    // Generalized velocities.
    VectorX<T> v_;

    // Slack variable (the optimality conditions state that sigma = gamma).
    VectorX<T> sigma_;

    // Cached quantities are all function of gamma.
    mutable Cache cache_;
  };

  // This is the one and only API from ContactSolver that must be implemented.
  // Refere to ContactSolverBase's documentation for details.
  ContactSolverStatus DoSolveWithGuess(
      const typename ConvexSolverBase<T>::PreProcessedData& data,
      const VectorX<T>& v_guess, ContactSolverResults<T>* result) final;

  // Helper method to setup the underlying MathematicalProgram.
  void SetUpProgram(const typename ConvexSolverBase<T>::PreProcessedData& data,
                    drake::solvers::MathematicalProgram* prog,
                    drake::solvers::VectorXDecisionVariable* v_variable,
                    drake::solvers::VectorXDecisionVariable* sigma_variable);

  MpPrimalSolverParameters parameters_;
  MpPrimalSolverStats stats_;
  double total_time_{0};
  std::vector<MpPrimalSolverStats> stats_history_;

  // TODO: needed?
  struct Workspace {
    void Resize(int nv, int nc) {
      aux_v1.resize(nv);
      aux_v2.resize(nv);
    }
    VectorX<T> aux_v1;
    VectorX<T> aux_v2;
  };
  mutable Workspace workspace_;

  mutable std::shared_ptr<drake::solvers::GurobiSolver::License>
      gurobi_license_;
  mutable std::shared_ptr<drake::solvers::MosekSolver::License> mosek_license_;
};

template <>
ContactSolverStatus MpPrimalSolver<double>::DoSolveWithGuess(
    const ConvexSolverBase<double>::PreProcessedData&, const VectorX<double>&,
    ContactSolverResults<double>*);

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

extern template class ::drake::multibody::contact_solvers::internal::
    MpPrimalSolver<double>;
