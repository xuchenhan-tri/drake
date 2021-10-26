#pragma once

#include <iostream>
#include <memory>

#include <Eigen/SparseCore>

#include "drake/multibody/contact_solvers/contact_solver.h"
#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solver_interface.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

struct MpConvexSolverParameters {
  // We estimate a contact time constant as tc = alpha * time_step.
  double alpha{1.0};
  double Rt_factor{1.0};  // Rt = Rt_factor * Rn.
  bool primal{false};  // If true use primal formulation, otherwise dual.
  std::optional<drake::solvers::SolverId> solver_id{std::nullopt};
};

struct MpConvexSolverErrorMetrics {
  // Maximum effective stiction tolerance, [m/s]. A metric used to evaluate the
  // effectiveness of the formulation to model stiction.
  double vs_max{-1.0};
  // Error in the impulses relative to the analytical inverse dynamics.
  // Relative, dimensionless.
  double id_rel_error{0.0};
  // Error in the impulses relative to the analytical inverse dynamics. Absolute
  // value, [Ns].
  double id_abs_error{0.0};
  // Norm of the impulses vector.
  double gamma_norm{0.0};
};

struct MpConvexSolverStats {
  int num_contacts{0};
  MpConvexSolverErrorMetrics iteration_errors;

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
class MpConvexSolver final : public ContactSolver<T> {
 public:
  // The solver pre-processes SystemDynamicsData and PointContactData before a
  // solve and stores it here.
  // Computed with PreProcessData().
  struct PreProcessedData {
    int nv{0};
    int nc{0};

    VectorX<T> vc_star;
    VectorX<T> vc_hat;  // Anitescu uses vc_hat = -b = (0, 0, -phi0/dt).
    VectorX<T> r;        // = vc_star - vc_hat

    // Delassus operator and related quantities:
    Eigen::SparseMatrix<T> W;
    Eigen::SparseMatrix<T> N;  // N = W + diag(R).
    // gi = trace(Wii)/3. Trace of the 3x3 diagonal block of W.
    VectorX<T> gi;
    // We define gᵢ = tr(Wᵢᵢ)/3 the trace of the 3x3 diagonal block of
    // W. We define the effective mass of the i-th contact as
    // mᵢ = 1/gᵢ = 3/tr(Wᵢᵢ).
    VectorX<T> mi;

    // Friction and regularization:
    VectorX<T> mu;        // friction coefficients.
    VectorX<T> mu_tilde;  // mu_tilde = sqrt(Rt/Rn) * mu.
    VectorX<T> Rt;        // Regularization in the tangential direction.
    VectorX<T> Rn;        // Regularization in the normal direction.
    VectorX<T> R;         // R = (Rt, Rt, Rn)
    VectorX<T> Rinv;      // R = (1.0/Rt, 1.0/Rt, 1.0/Rn)

    // Data for primal.
    MatrixX<T> M;
    MatrixX<T> Jc;
    VectorX<T> v_star;

    void Resize(int num_velocities, int num_contacts) {
      nv = num_velocities;
      nc = num_contacts;
      const int nc3 = 3 * nc;
      W.resize(nc3, nc3);
      N.resize(nc3, nc3);
      vc_star.resize(nc3);
      gi.resize(nc);
      mi.resize(nc);
      mu.resize(nc);
      mu_tilde.resize(nc);
      Rt.resize(nc);
      Rn.resize(nc);
      R.resize(nc3);
      Rinv.resize(nc3);
      vc_hat.resize(nc3);
      r.resize(nc3);
      M.resize(nv, nv);
      Jc.resize(nc3, nv);
      v_star.resize(nv);
    }
  };

  class State {
   public:
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
    void Resize(int nv, int nc) {
      nv_ = nv;
      nc_ = nc;
      gamma_.resize(3 * nc);
      cache_.Resize(nv, nc);
    }

    int num_velocities() const { return nv_; }
    int num_contacts() const { return nc_; }
    const VectorX<T>& gamma() const { return gamma_; }
    VectorX<T>& mutable_gamma() { return gamma_; }
    const Cache& cache() const { return cache_; }
    Cache& mutable_cache() const { return cache_; }

   private:
    int nv_;
    int nc_;

    // This solver's state is fully defined by the impulses gamma.
    VectorX<T> gamma_;

    // Cached quantities are all function of gamma.
    mutable Cache cache_;
  };

  MpConvexSolver();

  virtual ~MpConvexSolver() = default;

  void set_parameters(MpConvexSolverParameters& parameters) {
    parameters_ = parameters;
  }

  // This is the one and only API from ContactSolver that must be implemented.
  // Refere to ContactSolver's documentation for details.
  ContactSolverStatus SolveWithGuess(const T& time_step,
                                     const SystemDynamicsData<T>& dynamics_data,
                                     const PointContactData<T>& contact_data,
                                     const VectorX<T>& v_guess,
                                     ContactSolverResults<T>* result) final;

  // Retrieves solver statistics since the last call to SolveWithGuess().
  const MpConvexSolverStats& get_iteration_stats() const { return stats_; }

  // Retrieves the history of statistics during the lifetime of this solver for
  // each call to SolveWithGuess().
  const std::vector<MpConvexSolverStats>& get_stats_history() const {
    return stats_history_;
  }

  const VectorX<T>& GetContactVelocities() const { return state_.cache().vc; }
  const VectorX<T>& GetImpulses() const { return state_.gamma(); }

  PreProcessedData PreProcessData(
      const T& time_step, const SystemDynamicsData<T>& dynamics_data,
      const PointContactData<T>& contact_data) const;

  /// Given generalized velocities v, this method computes the analytical
  /// solution to the inverse dynamics problem, i.e. the impulses gamma.
  void CalcInverseDynamics(const PreProcessedData& data, const VectorX<T>& vc,
                           VectorX<T>* gamma) const;

 private:
  int num_velocities() const { return state_.num_velocities(); }
  int num_contacts() const { return state_.num_contacts(); }

  void InsertSparseBlockDiagonal(const std::vector<Matrix3<T>>& blocks,
                                 Eigen::SparseMatrix<T>* S) const;

  // This assumes the coefficients do exist in S.
  void AddSparseBlockDiagonal(const std::vector<Matrix3<T>>& blocks,
                              Eigen::SparseMatrix<T>* S) const;

  void SetUpSolver(drake::solvers::MathematicalProgram* prog);

  // Project y into gamma.
  Vector3<T> ProjectImpulse(const T& mu, const T& mu_tilde, const T& Rt_over_Rn,
                            const Eigen::Ref<const Vector3<T>>& y,
                            Vector3<T>* fperp = nullptr) const;

  MpConvexSolverParameters parameters_;
  const SystemDynamicsData<T>* dynamics_data_{nullptr};
  const PointContactData<T>* contact_data_{nullptr};
  PreProcessedData pre_proc_data_;
  State state_;
  MpConvexSolverStats stats_;
  std::vector<MpConvexSolverStats> stats_history_;
  const double kEpsilon{1.0e-7};
  const double kEpsilonSquared{kEpsilon * kEpsilon};
  mutable std::shared_ptr<drake::solvers::GurobiSolver::License>
      gurobi_license_;
  mutable std::shared_ptr<drake::solvers::MosekSolver::License>
    mosek_license_;      

  // Primal variables.
  drake::solvers::VectorXDecisionVariable v_variable_;
  drake::solvers::VectorXDecisionVariable sigma_variable_;
};

template <>
ContactSolverStatus MpConvexSolver<double>::SolveWithGuess(
    const double&, const SystemDynamicsData<double>&,
    const PointContactData<double>&, const VectorX<double>&,
    ContactSolverResults<double>*);

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

extern template class ::drake::multibody::contact_solvers::internal::
    MpConvexSolver<double>;
