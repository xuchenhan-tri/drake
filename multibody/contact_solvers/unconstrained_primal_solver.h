#pragma once

#include <iostream>
#include <memory>

#include <Eigen/SparseCore>

#include "drake/multibody/contact_solvers/block_sparse_matrix.h"
#include "drake/multibody/contact_solvers/contact_solver.h"
#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#include "drake/multibody/contact_solvers/convex_solver_base.h"
#include "drake/multibody/contact_solvers/supernodal_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

struct UnconstrainedPrimalSolverParameters {
  enum class LineSearchMethod {
    // Inexact line search satisfying the Armijo rule.
    kArmijo,
    // Line search exact to machine precision.
    kExact
  };

  // Theta method parameter.
  double theta{1.0};

  // We monitor convergence of the contact velocities.
  double abs_tolerance{1.0e-6};  // m/s
  double rel_tolerance{1.0e-6};  // Unitless.
  int max_iterations{100};       // Maximum number of Newton iterations.

  // Line-search parameters.
  LineSearchMethod ls_method{LineSearchMethod::kExact};
  double ls_alpha_max{1.5};  // Maximum line-search parameter allowed.
  // The line search terminates if between two iterates the condition
  // |αᵏ⁺¹−αᵏ| < ls_tolerance is satisfied.
  // When ls_tolerance < 0 the line search is performed to machine precision.
  double ls_tolerance{-1};
  // Maximum number of line-search iterations. Only used for inexact methods.
  int ls_max_iterations{40};

  // Arimijo's method parameters.
  double ls_c{1.0e-4};  // Armijo's reduction parameter.
  double ls_rho{0.8};

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

  // Use supernodal algebra for the linear solver.
  bool use_supernodal_solver{true};

  // (Temporary hack). Use the Geodesic solver.
  bool use_geodesic_solver{false};

  // For debugging. Compare supernodal reconstructed Hessian with dense algebra
  // Hessian.
  bool compare_with_dense{false};

  // The verbosity level determines how much information to print into stdout.
  // These levels are additive. E.g.: level 2 also prints level 0 and 1 info.
  //  0: Nothing gets printed.
  //  1: Prints problem size and error at convergence.
  //  2: Prints sparsity structure.
  //  3: Prints stats at each iteration.
  int verbosity_level{1};

  bool log_stats{true};

  bool log_condition_number{false};
};

struct UnconstrainedPrimalSolverIterationMetrics {
  // vc_max_norm_error = ‖vcᵏ⁺¹ − vcᵏ‖∞.
  // Max norm of the contact velocity error. It is nice in that all components
  // have the same units.
  double vc_error_max_norm{0.0};

  // Max norm of the generalized velocities error.
  // v_max_norm_error = ‖vᵏ⁺¹ − vᵏ‖∞.
  double v_error_max_norm{0.0};

  // Max norm of the contact forces error.
  // gamma_max_norm_error = ‖γᵏ⁺¹−γᵏ‖∞.
  double gamma_error_max_norm{0.0};

  // L2 norm of the scaled momentum equation (first optimality condition)
  double mom_l2{0};
  // Max norm of the scaled momentum equation (first optimality condition)
  double mom_max{0};

  double mom_rel_l2{0};
  double mom_rel_max{0};

  // Optimality condition between g and gamma.
  double opt_cond{0.0};

  // Some norms.
  double vc_norm{0.0};
  double gamma_norm{0.0};

  // Regularization cost ℓᵣ(v).
  double ellR{0.0};

  // Total cost ℓ(v).
  double ell{0.0};

  // The gradient of the cost, ∇ℓ(v).
  double grad_ell_max_norm{0.0};

  // The search direction is dv = H⁻¹∇ℓ(v).
  double search_direction_max_norm{0.0};

  // Estimation of the reverse condition number.
  double rcond{-1};

  // Line search parameter.
  double ls_alpha;

  int ls_iters{0};

  // Energy metrics.
  double Ek{0}, costM{0}, costR{0}, cost{0};

  // Inverse dynamics realtive error.
  double id_rel_error{0};

  // Mean and rms value of the tangential velocities.
  double vt_mean{0.0};
  double vt_rms{0.0};

  // An estimate of the condition number. Zero if not logged, see
  // log_condition_number in the parameters.
  double cond_number{0.0};
};

// Intended for debugging only. Remove.
template <typename T>
struct SolutionData {
#if 0  
  SolutionData(int nv_, int nc_) : nc(nc_) {
    const int nc3 = 3 * nc;
    vc.resize(nc3);
    gamma.resize(nc3);
    mu.resize(nc);
    R.resize(nc3);
  }
#endif 
  int nc;
  VectorX<T> vc;
  VectorX<T> gamma;
  VectorX<T> mu;
  VectorX<T> R;
};

struct UnconstrainedPrimalSolverStats {
  int num_contacts{0};
  int num_iters;  // matches iteration_metrics.size() unless we use Geodesic.
  std::vector<UnconstrainedPrimalSolverIterationMetrics> iteration_metrics;

  // Performance statistics. All these times are in seconds.

  // Total time for the last call to SolveWithGuess().
  double total_time{0};
  double supernodal_construction_time{0};
  // Time used in pre-processing the data: forming the Delassus operator,
  // computing regularization parameters, etc.
  double preproc_time{0};
  // Time used to assembly the Hessian.
  double assembly_time{0};
  // Time used by the underlying linear solver.
  double linear_solver_time{0};
  // Time used in line search.
  double line_search_time{0};
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
class UnconstrainedPrimalSolver final : public ConvexSolverBase<T> {
 public:
  UnconstrainedPrimalSolver();

  virtual ~UnconstrainedPrimalSolver() = default;

  void set_parameters(UnconstrainedPrimalSolverParameters& parameters) {
    ConvexSolverBaseParameters base_parameters{parameters.theta,
                                               parameters.Rt_factor,
                                               parameters.alpha,
                                               parameters.sigma};
    ConvexSolverBase<T>::set_parameters(base_parameters);
    parameters_ = parameters;
  }

  // Retrieves solver statistics since the last call to SolveWithGuess().
  const UnconstrainedPrimalSolverStats& get_iteration_stats() const {
    return stats_;
  }

  // Retrieves the history of statistics during the lifetime of this solver for
  // each call to SolveWithGuess().
  const std::vector<UnconstrainedPrimalSolverStats>& get_stats_history() const {
    return stats_history_;
  }

  const std::vector<SolutionData<T>>& solution_history() const {
    return solution_history_;
  }  

  void LogIterationsHistory(const std::string& file_name) const final;

  void LogPerStepIterationsHistory(const std::string& file_name) const;

  void LogSolutionHistory(const std::string& file_name) const;

 private:
  // This is not a real cache in the CS sense (i.e. there is no tracking of
  // dependencies nor automatic validity check) but in the sense that this
  // objects stores computations that are function of the solver's state. It is
  // the responsability of the solver to keep these computations
  // properly in sync.
  struct Cache {
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Cache);

    Cache() = default;

    void Resize(int nv, int nc, bool dense = true) {
      const int nc3 = 3 * nc;
      vc.resize(nc3);
      gamma.resize(nc3);
      ellR_grad_y.resize(nc3);
      ellR_hessian_y.resize(nc);
      ell_grad_v.resize(nv);
      if (dense) ell_hessian_v.resize(nv, nv);
      dv.resize(nv);
      dp.resize(nv);
      dvc.resize(nc3);
      regions.resize(nc);
      dgamma_dy.resize(nc);
      // N.B. The supernodal solver needs MatrixX instead of Matrix3.
      G.resize(nc, Matrix3<T>::Zero());
    }

    void mark_invalid() {
      valid_contact_velocity_and_impulses = false;
      valid_cost_and_gradients = false;
      valid_dense_gradients = false;
      valid_search_direction = false;
      valid_line_search_quantities = false;
    }

    // Direct algebraic funtions of velocity.
    // CalcVelocityAndImpulses() updates these entries.
    bool valid_contact_velocity_and_impulses{false};
    VectorX<T> vc;     // Contact velocities.
    VectorX<T> gamma;  // Impulses.

    bool valid_cost_and_gradients{false};
    T ell;   // The total cost.
    T ellM;  // Mass matrix cost.
    T ellR;  // The regularizer cost.
    // N.B. The supernodal solver consumes G as a vector MatrixX instead of
    // Matrix3. That is why dgamma_dy uses Matrix3 and G uses MatrixX.
    std::vector<Matrix3<T>> dgamma_dy;  // ∂γ/∂y.
    std::vector<MatrixX<T>> G;          // G = -∂γ/∂vc.
    VectorX<T> ell_grad_v;              // Gradient of the cost in v.
    VectorX<int> regions;

    // TODO: needed?
    VectorX<T> ellR_grad_y;                  // Gradient of regularizer in y.
    std::vector<Matrix3<T>> ellR_hessian_y;  // Hessian of regularizer in y.

    // TODO: only for debugging. Remove these.
    bool valid_dense_gradients{false};
    MatrixX<T> ell_hessian_v;  // Hessian in v.

    // Search directions are also a function of state. Gradients (i.e.
    // valid_cost_and_gradients) must be valid in order for the computation to
    // be correct.
    bool valid_search_direction{false};
    VectorX<T> dv;       // search direction.
    VectorX<T> dvc;      // Search direction in contact velocities.
    T condition_number;  // An estimate of the Hessian's condition number.

    // One-dimensional quantities used in line-search.
    // These depend on Δv (i.e. on valid_search_direction).
    bool valid_line_search_quantities{false};
    VectorX<T> dp;     // Δp = M⋅Δv
    T d2ellM_dalpha2;  // d2ellM_dalpha2 = Δvᵀ⋅M⋅Δv
  };

  // Everything in this solver is a function of the generalized velocities v.
  // State stores generalized velocities v and cached quantities that are
  // function of v.
  class State {
   public:
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(State);

    State() = default;

    State(int nv, int nc, bool dense) { Resize(nv, nc, dense); }

    void Resize(int nv, int nc, bool dense) {
      v_.resize(nv);
      cache_.Resize(nv, nc, dense);
    }

    const VectorX<T>& v() const { return v_; }
    VectorX<T>& mutable_v() {
      // Mark all cache quantities as invalid since they all are a function of
      // velocity.
      cache_.mark_invalid();
      return v_;
    }

    const Cache& cache() const { return cache_; }
    Cache& mutable_cache() const { return cache_; }

   private:
    VectorX<T> v_;
    mutable Cache cache_;
  };

  // This is the one and only API from ContactSolver that must be implemented.
  // Refere to ContactSolverBase's documentation for details.
  ContactSolverStatus DoSolveWithGuess(
      const typename ConvexSolverBase<T>::PreProcessedData& data,
      const VectorX<T>& v_guess, ContactSolverResults<T>* result) final;

  // Update:
  //  - Contact velocities vc(v).
  //  - Contact impulses gamma(v).
  //  - Gradient ∂γ/∂y.
  void CalcVelocityAndImpulses(
      const State& state, VectorX<T>* vc, VectorX<T>* gamma,
      std::vector<Matrix3<T>>* dgamma_dy = nullptr) const;

  // Computes the cost ℓ(v) and gradients with respect to v.
  // If ell_hessian_v == nullptr we skip the expensive computation of the
  // Hessian.
  T CalcCostAndGradients(const State& state, VectorX<T>* ell_grad_v,
                         std::vector<MatrixX<T>>* G, T* ellM = nullptr,
                         T* ellR = nullptr,
                         MatrixX<T>* ell_hessian_v = nullptr) const;

  // Helper used for debugging. Given un-projected impulse y and search
  // direction dy, it detects all cone crossings along that direction.
  // N.B. This might not make it into production code, even when super useful.
  std::vector<T> FindAllContinuousIntervals(const State& state,
                                            const VectorX<T>& y,
                                            const VectorX<T>& dy) const;

  // Given velocities v and search direction dv stored in `state`, this method
  // computes ℓ(α) = ℓ(v+αΔv), for a given alpha (α), and first and second
  // derivatives dℓ/dα and d²ℓ/dα².
  T CalcLineSearchCostAndDerivatives(
      const State& state_v, const T& alpha, T* dell_dalpha, T* d2ell_dalpha2,
      State* state_alpha, T* ellM = nullptr, T* dellM_dalpha = nullptr,
      T* d2ellM_dalpha2 = nullptr, T* ellR = nullptr, T* dellR_dalpha = nullptr,
      T* d2ellR_dalpha2 = nullptr) const;

  // Given velocities v and search direction dv stored in `state` this method
  // computes the optimum alpha (α) such that ℓ(α) = ℓ(v+αΔv) is minimum. This
  // search is performed to machine precision to avoid having additional
  // parameters. Convergence to machine precision only cost a couple extra
  // iterations and therefore is well worth the price.
  int CalcLineSearchParameter(const State& state, T* alpha) const;

  // Approximation to the 1D minimization problem α = argmin ℓ(α)= ℓ(v + αΔv)
  // over α. We define ϕ(α) = ℓ₀ + α c ℓ₀', where ℓ₀ = ℓ(0) and ℓ₀' = dℓ/dα(0).
  // With this definition the Armijo condition reads ℓ(α) < ϕ(α).
  // This approximate method seeks to minimize ℓ(α) over a discrete set of
  // values given by the geometric progression αᵣ = ρʳαₘₐₓ with r an integer,
  // 0 < ρ < 1 and αₘₐₓ the maximum value of α allowed. That is, the exact
  // problem is replaced by
  //   α = argmin ℓ(α)= ℓ(v + αᵣΔv)
  //       αᵣ = ρʳαₘₐₓ
  //       s.t. ℓ(α) < ϕ(α), Armijo's condition.
  int CalcInexactLineSearchParameter(const State& state, T* alpha) const;

  // Computes iteration metrics between iterations k and k-1 at states s_k and
  // s_kp respectively.
  UnconstrainedPrimalSolverIterationMetrics CalcIterationMetrics(
      const State& s_k, const State& s_kp, int num_ls_iterations,
      double alpha, VectorX<T>* xc_work1) const;

  // Solves for dv using supernodal algebra.
  void CallSupernodalSolver(const State& s, VectorX<T>* dv,
                            SuperNodalSolver* solver);

  // Solves for dv usind dense algebra, for debugging.
  void CallDenseSolver(const State& s, VectorX<T>* dv);

  using ConvexSolverBase<T>::data_;
  UnconstrainedPrimalSolverParameters parameters_;
  UnconstrainedPrimalSolverStats stats_;
  double total_time_{0};
  std::vector<UnconstrainedPrimalSolverStats> stats_history_;
  std::vector<SolutionData<T>> solution_history_;

  struct Workspace {
    void Resize(int nv, int nc) {
      aux_v1.resize(nv);
      aux_v2.resize(nv);
    }
    VectorX<T> aux_v1;
    VectorX<T> aux_v2;
  };
  mutable Workspace workspace_;

  // Auxiliary state used by CalcLineSearchParameter().
  // TODO: either remove or make it an argument to CalcLineSearchParameter().
  mutable State aux_state_;
};

template <>
ContactSolverStatus UnconstrainedPrimalSolver<double>::DoSolveWithGuess(
    const ConvexSolverBase<double>::PreProcessedData&, const VectorX<double>&,
    ContactSolverResults<double>*);

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

extern template class ::drake::multibody::contact_solvers::internal::
    UnconstrainedPrimalSolver<double>;
