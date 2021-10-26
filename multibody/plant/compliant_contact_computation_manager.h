#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/contact_solvers/block_sparse_matrix.h"
#include "drake/multibody/plant/discrete_update_manager.h"
#include "drake/multibody/plant/contact_permutation_utils.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/context.h"

namespace drake {
namespace multibody {

struct ContactManagerStats {
  double time;
  int num_contacts;
  double total_time;
  double geometry_time{0};
  double vstar_time{0};
  double graph_time{0};
  double linear_dynamics_time{0};
  double contact_jacobian_time{0};
  double contact_solver_time{0};
  double pack_results_time{0};
};

namespace internal {
template <typename T>
struct ContactJacobianCache {
  MatrixX<T> Jc;
  std::vector<drake::math::RotationMatrix<T>> R_WC_list;
};
}  // namespace internal

// Forward declare MultibodyPlant.
// template <typename T>
// class MultibodyPlant;

// This class manages the interaction between MultibodyPlant and ContactSolver.
// It uses MultibodyPlant to setup the ContactSolver problem, solves the contact
// problem, and reports results as ContactSolverResults.
// @tparam_default_scalar
template <typename T>
class CompliantContactComputationManager
    : public internal::DiscreteUpdateManager<T> {
 public:
  // TODO(amcastro-tri): consider owning a proper contact solver here.
  CompliantContactComputationManager(
      std::unique_ptr<contact_solvers::internal::ContactSolver<T>>
          contact_solver);

  ~CompliantContactComputationManager() = default;

  // TODO: changes these continuous parameters simply by an enum with common
  // options:
  // - Semi-explicit Euler.
  // - Semi-implicit Euler.
  // - Implicit Euler.
  // - Explicit Euler.
  // - Trapezoidal rule.

  // How v is computed in the equation for q.
  void set_theta_q(double theta) { theta_q_ = theta; }

  // How q is computed in the equation for v (momentum).
  void set_theta_qv(double theta) { theta_qv_ = theta; }

  // How v is computed in the equation for v (momentum).
  void set_theta_v(double theta) { theta_v_ = theta; }

  // Sets the underlying contact solver to be used by the manager.
  // @param solver The contact solver to be used for simulations of discrete
  // models with frictional contact. CalcContactSolverResults() will use this
  // solver after this call.
  // @pre solver must not be nullptr.
  // @pre SolverType must be a subclass of
  // multibody::contact_solvers::internal::ContactSolver.
  // @returns a mutable reference to `solver`, now owned by `this`
  // CompliantContactComputationManager.
  template <class SolverType>
  SolverType& set_contact_solver(std::unique_ptr<SolverType> solver) {
    DRAKE_DEMAND(solver != nullptr);
    static_assert(std::is_base_of<contact_solvers::internal::ContactSolver<T>,
                                  SolverType>::value,
                  "SolverType must be a sub-class of ContactSolver.");
    SolverType* solver_ptr = solver.get();
    contact_solver_ = std::move(solver);
    return *solver_ptr;
  }

  template <template <typename>
            class SolverType = contact_solvers::internal::ContactSolver>
  SolverType<T>& mutable_contact_solver() {
    DRAKE_DEMAND(contact_solver_ != nullptr);
    SolverType<T>* result = dynamic_cast<SolverType<T>*>(contact_solver_.get());
    DRAKE_DEMAND(result != nullptr);
    return *result;
  }

  const std::vector<ContactManagerStats>& get_stats_history() const {
    return stats_;
  }

  // Total wall-clock time used in subsequent calls to
  // CalcContactSolverResults().
  double total_time() const { return total_time_; }

  void LogStats(const std::string& log_file_name) const;

 private:
  struct CacheIndexes {
    systems::CacheIndex contact_jacobian;
  };

  using internal::DiscreteUpdateManager<T>::plant;

  void ExtractModelInfo() final;

  void DoCalcContactSolverResults(
      const systems::Context<T>&,
      contact_solvers::internal::ContactSolverResults<T>*) const final;

  void DoCalcAccelerationKinematicsCache(
      const systems::Context<T>&,
      multibody::internal::AccelerationKinematicsCache<T>*) const final;

  void DoCalcDiscreteValues(
      const drake::systems::Context<T>& context0,
      drake::systems::DiscreteValues<T>* updates) const final;

  void DoCalcContactResults(const systems::Context<T>& context,
                            ContactResults<T>* contact_results) const final;

  void DeclareCacheEntries() final;

  // TODO: change signature so that it returns a new contact graph instead.
  void CalcContactGraph(
      const geometry::QueryObject<T>& query_object,
      const std::vector<internal::DiscreteContactPair<T>>& contact_pairs) const;

  // Solves for v*.
  // @param[out] vstar velocity v* consistent with the MultibodyPlant model.
  // @param[out] vstar_reduced A possibly smaller (reduced) vector containing
  // only results for those trees in the forest that participate through
  // contact. This vector can also be permuted, do not use directly with
  // MultibodyPlant but as an intermediate results of
  // CompliantContactComputationManager.
  void CalcVelocityUpdateWithoutConstraints(
      const systems::Context<T>& context, VectorX<T>* vstar,
      VectorX<T>* vstar_reduced, VectorX<T>* v_guess,
      VectorX<T>* participating_v_guess) const;

  void CalcVelocityUpdateWithoutConstraintsUsingThetaMethod(
      const systems::Context<T>& context, VectorX<T>* vstar,
      VectorX<T>* vstar_reduced, VectorX<T>* v_guess,
      VectorX<T>* participating_v_guess,
      systems::Context<T>* context_star) const;

  // Helper to compute a = M⁻¹(q)τ, with the mass matrix evaluated at the
  // configuration q stored in `context`.
  void MinvOperator(const systems::Context<T>& context, const VectorX<T>& tau,
                    VectorX<T>* a) const;

  // Helper to compute xdot(context{x, u}).
  VectorX<T> CalcXdot(const systems::Context<T>& context) const;

  // Computes the lienarized dynamics matrix A in A(v-v*) = Jᵀγ.
  void CalcLinearDynamics(
      const systems::Context<T>& context,
      contact_solvers::internal::BlockSparseMatrix<T>* A) const;

  void CalcContactQuantities(
      const systems::Context<T>& context,
      const std::vector<internal::DiscreteContactPair<T>>& contact_pairs,
      contact_solvers::internal::BlockSparseMatrix<T>* Jc, VectorX<T>* phi0,
      VectorX<T>* vc0, VectorX<T>* mu, VectorX<T>* stiffness,
      VectorX<T>* linear_damping) const;

  void CalcContactJacobian(
      const systems::Context<T>& context,
      const std::vector<internal::DiscreteContactPair<T>>& contact_pairs,
      MatrixX<T>* Jc_ptr, std::vector<math::RotationMatrix<T>>* R_WC_set) const;

  const internal::ContactJacobianCache<T>& EvalContactJacobianCache(
      const systems::Context<T>& context) const {
    return plant().get_cache_entry(cache_indexes_.contact_jacobian)
        .template Eval<internal::ContactJacobianCache<T>>(context);
  }

  // forward = true --> vp(ip) = v(i).
  // forward = false --> v(i) = vp(ip).
  void PermuteVelocities(bool forward, VectorX<T>* v, VectorX<T>* vp) const;

  void PermuteFullToParticipatingVelocities(const VectorX<T>& v,
                                            VectorX<T>* vp) const;

  void PermuteParticipatingToFullVelocities(const VectorX<T>& vp,
                                            VectorX<T>* v) const;

  // forward = true --> xp(kp) = x(k).
  // forward = false --> x(k) = xp(kp).
  void PermuteContacts(bool forward, int stride, VectorX<T>* x,
                       VectorX<T>* xp) const;

  void PermuteIntoPatches(int stride, const VectorX<T>& x,
                          VectorX<T>* xp) const;

  void PermuteFromPatches(int stride, const VectorX<T>& xp,
                          VectorX<T>* x) const;

  std::unique_ptr<contact_solvers::internal::ContactSolver<T>> contact_solver_;

  // Tree structure and velocities permutation. Computed at construction and
  // const afterwards.
  // TODO: struct ForestTopology?
  int num_trees_;
  int num_velocities_;
  std::vector<std::vector<int>> velocities_permutation_;
  std::vector<int> body_to_tree_map_;

  // Contact graph and reduced system information.
  // TODO: struct ContactTopology?
  // TODO: remove all these "mutable" and make them available only in local
  // scope.
  mutable int num_participating_trees_;  // Number of tree that are in contact.
  mutable int num_participating_velocities_;
  mutable std::vector<int> participating_trees_;
  // v_participating = participating_tree_permutation_[tp][vt].
  mutable std::vector<std::vector<int>> participating_velocities_permutation_;
  // Contact graph. It makes reference to the "reduced tree indexes" tr.
  // The original tree can be obtained with t = participating_trees_[tr].
  mutable internal::ContactGraph graph_;  

  // Scratch workspace.
  struct Workspace {
    MatrixX<T> M;
    MatrixX<T> Jc;
    std::unique_ptr<systems::Context<T>> aux_plant_context_;
  };
  mutable Workspace workspace_;

  // Hack. We save here the previous value of the generalized contact forces.
  // We'd need to place it in its own cache entry.
  mutable VectorX<T> tau_c_;

  CacheIndexes cache_indexes_;

  mutable std::vector<ContactManagerStats> stats_;
  mutable double total_time_{0};
  double theta_q_{1.0};   // how v is computed in the equation for q.
  double theta_v_{0.0};   // how v is computed in the equation for v.
  double theta_qv_{0.0};  // how q is computed in the equation for v.
};

}  // namespace multibody
}  // namespace drake