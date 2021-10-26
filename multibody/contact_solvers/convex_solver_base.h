#pragma once

#include "drake/multibody/contact_solvers/block_sparse_matrix.h"
#include "drake/multibody/contact_solvers/contact_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

/// Parameters for ConvexSolverBase. These will typically affect the computation
/// of precomputed quantities by NVI SolveWithGuess() before calling
/// DoSolveWithGuess().
struct ConvexSolverBaseParameters {
  double theta{1.0};

  // TODO(amcastro-tri): consider other ways of estimating Rg. Maybe a factor of
  // the Delassus operator?
  double Rt_factor{1.0e-3};  

  // Rigid approximation contant: Rₙ = α⋅Wᵢ when the contact frequency ωₙ is
  // below the limit ωₙ⋅dt ≤ 2π. That is, the period is Tₙ = α⋅dt.
  double alpha{1.0};

  // Dimensionless parameterization of the regularization of friction.
  // An approximation for the bound on the slip velocity is vₛ ≈ ε⋅δt⋅g.
  double sigma{1.0e-3};
};

// This solver uses the regularized convex formulation from [Todorov 2014]. This
// class provides data pre-processing to write this formulaton. Derived classes
// can use this data to either implement a primal or dual formulation as in
// [Todorov 2014]. Derived classes must only implement the API
// DoSolveWithGuess().
//
// - [Todorov, 2014] Todorov, E., 2014, May. Convex and analytically-invertible
//   dynamics with contacts and constraints: Theory and implementation in
//   MuJoCo. In 2014 IEEE International Conference on Robotics and Automation
//   (ICRA) (pp. 6054-6061). IEEE.
template <typename T>
class ConvexSolverBase : public ContactSolver<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ConvexSolverBase);
  ConvexSolverBase(const ConvexSolverBaseParameters& parameters)
      : parameters_(parameters) {}
  virtual ~ConvexSolverBase() = default;

  // set parameters used by this class to pre-process data when calling
  // SolveWithGuess.
  void set_parameters(const ConvexSolverBaseParameters& parameters) {
    parameters_ = parameters;
  }

  // This is the one and only API from ContactSolver that must be implemented.
  // Refere to ContactSolver's documentation for details.
  // This now becomes the NVI to DoSolveWithGuess() that must be implemented by
  // child classes.
  ContactSolverStatus SolveWithGuess(const T& time_step,
                                     const SystemDynamicsData<T>& dynamics_data,
                                     const PointContactData<T>& contact_data,
                                     const VectorX<T>& v_guess,
                                     ContactSolverResults<T>* result) final;

  // Reteturns the time it took to preprocess data when calling
  // SolveWithGuess(), in seconds.
  double pre_process_time() const { return pre_process_time_; }

  // Returns the total time spent on calls to SolveWithGuess().
  double get_total_time() const { return total_time_; }

  virtual void LogIterationsHistory(const std::string&) const {
    throw std::runtime_error("Implement this capability.");
  }

 protected:
  // TODO: make into a proper class.
  struct PreProcessedData {
    void Resize(int nv_in, int nc_in) {
      nv = nv_in;
      nc = nc_in;
      const int nc3 = 3 * nc;
      R.resize(nc3);
      Rinv.resize(nc3);
      vc_stab.resize(nc3);
      Djac.resize(nv);
      p_star.resize(nv);
      Wdiag.resize(nc);
    }
    T time_step;
    const SystemDynamicsData<T>* dynamics_data{nullptr};
    const PointContactData<T>* contact_data{nullptr};
    int nv;
    int nc;
    VectorX<T> R;        // Regularization parameters, of size 3nc.
    VectorX<T> Rinv;     // Inverse of regularization parameters, of size 3nc.
    VectorX<T> vc_stab;  // Constraints stabilization velocity, see paper.
    BlockSparseMatrix<T> Jblock;  // Jacobian as block-structured matrix.
    BlockSparseMatrix<T> Mblock;  // Mass mastrix as block-structured matrix.
    std::vector<MatrixX<T>> Mt;  // Per-tree diagonal blocks of the mass matrix.
    // Jacobi pre-conditioner for the mass matrix.
    // Djac = diag(M)^(-0.5)
    VectorX<T> Djac;
    VectorX<T> p_star;
    VectorX<T> Wdiag;  // Delassus operator diagonal approximation.
  };

  // Derived classes must implement this method to
  virtual ContactSolverStatus DoSolveWithGuess(
      const PreProcessedData& data, const VectorX<T>& v_guess,
      ContactSolverResults<T>* result) = 0;

  // TODO: make ti return data instead and remove member data from here.
  void PreProcessData(const T& time_step,
                      const SystemDynamicsData<T>& dynamics_data,
                      const PointContactData<T>& contact_data,
                      double theta,
                      double Rt_factor, double alpha, double sigma);

  // Utility to compute the "soft norm" ‖x‖ₛ defined by ‖x‖ₛ² = ‖x‖² + ε², where
  // ε = soft_tolerance.
  T SoftNorm(const Eigen::Ref<const VectorX<T>>& x,
             double soft_tolerance) const {
    using std::sqrt;
    return sqrt(x.squaredNorm() + soft_tolerance * soft_tolerance);
  }

  // Compute the analytical inverse dynamics γ = γ(vc).
  // @param[in] soft_norm_tolerance tolerance used to compute the norm of the
  // tangential unprojected impulse yt, with units of Ns.
  // @param[in] vc contact velocities. On input vc.segment<3>(3*i) contains
  // velocity for the i-th contact point.
  // @param[out] gamma contact impulses. On output gamma.segment<3>(3*i)
  // contains the impulse for the i-th contact point.
  // @param[out] dgamma_dy Gradient of gamma wrt y. Not computed if nullptr.
  // @param[out] regions not computed if dgamma_dy = nullptr. Must be
  // non-nullptr if dgamma_dy is not nullptr.
  void CalcAnalyticalInverseDynamics(
      double soft_norm_tolerance, const VectorX<T>& vc, VectorX<T>* gamma,
      std::vector<Matrix3<T>>* dgamma_dy = nullptr,
      VectorX<int>* regions = nullptr) const;

  bool CheckConvergenceCriteria(const VectorX<T>& vc, const VectorX<T>& dvc,
                                double abs_tolerance,
                                double rel_tolerance) const;

  // Pack solution into ContactSolverResults.
  void PackContactResults(const PreProcessedData& data, const VectorX<T>& v,
                          const VectorX<T>& vc, const VectorX<T>& gamma,
                          ContactSolverResults<T>* result) const;

  // Returns the pair {‖m‖₂,‖m‖∞} with the L2 and max norms of the scaled
  // momentum equations, respectively.
  std::pair<T, T> CalcScaledMomentumError(const PreProcessedData& data,
                               const VectorX<T>& v,
                               const VectorX<T>& gamma) const;

  void CalcScaledMomentumAndScales(const PreProcessedData& data,
                                   const VectorX<T>& v, const VectorX<T>& gamma,
                                   T* scaled_momentum_error, T* momentum_scale,
                                   T* Ek, T* ellM, T* ellR, T* ell,
                                   VectorX<T>* v_work1, VectorX<T>* v_work2,
                                   VectorX<T>* v_work3) const;

  // We define the momentum error (optimality condition) as r(v) = M⋅(v−v*)−Jᵀγ.
  // We define the momentum p = M⋅v and generalized impulse j = Jᵀγ.
  // We define the relative momentum error in 2-norm as:
  //   ε₂ = ‖r‖/max(‖p‖,‖j‖).
  // We define the "max-norm" relative error as:
  //   εₘₐₓ = max({εᵢ}), with εᵢ = |rᵢ|/max(|pᵢ|,|jᵢ|),
  // where i spans the dofs of the system.
  // 
  // This method returns the pair (ε₂, εₘₐₓ).
  std::pair<T, T> CalcRelativeMomentumError(const PreProcessedData& data,
                                            const VectorX<T>& v,
                                            const VectorX<T>& gamma) const;

  // Computes energy metrics that can be used for logging or as reference scales
  // in dimensionless quantities.
  //   Ek: Kinetic energy. M norm of v.
  //   ellM: Kinetic energy. M norm of (v - v*).
  //   ellR: R norm of gamma.
  //   ell: Total cost, ellM + ellR.
  void CalcEnergyMetrics(const PreProcessedData& data, const VectorX<T>& v,
                         const VectorX<T>& gamma, T* Ek, T* ellM, T* ellR,
                         T* ell) const;

  void CalcSlipMetrics(const PreProcessedData& data, const VectorX<T>& vc,
                       T* vt_mean, T* vt_rms) const;

  // Computes the optimality condition m = 1/nc∑|gᵢ⋅γᵢ|
  // where g = J⋅v − v̂ + R⋅γ.
  // For this problem the optimality conditions read
  // ℱ* ∋ gᵢ ⊥ γᵢ ∈ ℱ
  // This metric then takes the average over all contact points of the absolute
  // value of gᵢ⋅γᵢ.
  T CalcOptimalityCondition(const PreProcessedData& data, const VectorX<T>& v,
                            const VectorX<T>& gamma, VectorX<T>* xc_work) const;

  PreProcessedData data_;

 private:
  // Parameters that define the projection gamma = P(y) on the friction cone ℱ
  // using the R norm.
  struct ProjectionParams {
    // Friction coefficient. It defines the friction cone ℱ.
    T mu;
    // Regularization parameters. Define the R norm.
    T Rt;  // Tangential direction.
    T Rn;  // Normal direction.
  };

  // Computes gamma = P(y) and its gradient dPdy (if requested).
  // In addition to passing y as an argument we also pass the triplet {yr,
  // yn, that}. This allow us to reuse these quantities if already computed.
  // TODO: make static?
  Vector3<T> CalcProjection(const ProjectionParams& params,
                            const Eigen::Ref<const Vector3<T>>& y, const T& yr,
                            const T& yn,
                            const Eigen::Ref<const Vector2<T>>& that,
                            int* region, Matrix3<T>* dPdy = nullptr) const;

  void CalcDelassusDiagonalApproximation(int nc,
                                         const std::vector<MatrixX<T>>& Mt,
                                         const BlockSparseMatrix<T>& Jblock,
                                         VectorX<T>* Wdiag) const;

  ConvexSolverBaseParameters parameters_;
  double pre_process_time_{0};
  double total_time_{0};
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

extern template class ::drake::multibody::contact_solvers::internal::
    ConvexSolverBase<double>;
