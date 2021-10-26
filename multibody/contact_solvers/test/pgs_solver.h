#pragma once

#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include "drake/multibody/contact_solvers/contact_solver.h"
#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

struct PgsSolverParameters {
  // Over-relaxation paramter, in (0, 1]
  double relaxation{0.5};
  // Absolute contact velocity tolerance, m/s.
  double abs_tolerance{1.0e-8};
  // Relative contact velocity tolerance, unitless.
  double rel_tolerance{1.0e-6};
  // Maximum number of PGS iterations.
  int max_iterations{100};
};

struct PgsErrorMetrics {
  double vc_err{0.0};     // Error in the contact velocities, [m/s].
  double gamma_err{0.0};  // Error in the contact impulses, [Ns].
};

struct PgsSolverStats {
  int iterations{0};
  std::vector<PgsErrorMetrics> iteration_errors;
  int num_contacts{0};
};

template <typename T>
class PgsSolver final : public ContactSolver<T> {
 public:
  class State {
   public:
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(State);
    State() = default;
    void Resize(int nv, int nc) {
      nv_ = nv;
      nc_ = nc;
      v_.resize(nv);
      gamma_.resize(3 * nc);
    }
    int num_velocities() const { return nv_; }
    int num_contacts() const { return nc_; }
    const VectorX<T>& v() const { return v_; }
    VectorX<T>& mutable_v() { return v_; }
    const VectorX<T>& gamma() const { return gamma_; }
    VectorX<T>& mutable_gamma() { return gamma_; }

   private:
    int nv_;
    int nc_;
    VectorX<T> v_;
    VectorX<T> gamma_;
  };

  PgsSolver() = default;

  virtual ~PgsSolver() = default;

  void set_parameters(PgsSolverParameters& parameters) {
    parameters_ = parameters;
  }

  ContactSolverStatus SolveWithGuess(const T& time_step,
                                     const SystemDynamicsData<T>& dynamics_data,
                                     const PointContactData<T>& contact_data,
                                     const VectorX<T>& v_guess,
                                     ContactSolverResults<T>* result) final;

  const PgsSolverStats& get_iteration_stats() const { return stats_; }

  const VectorX<T>& GetContactVelocities() const { return vc_; }
  const VectorX<T>& GetImpulses() const { return state_.gamma(); }

  // no-op at this point.
  void set_guess(const VectorX<T>& v, const VectorX<T>& gamma) {}

 private:
  // All this data must remain const after the call to PreProcessData().
  struct PreProcessedData {
    Eigen::SparseMatrix<T> W;
    VectorX<T> vc_star;
    // Norm of the 3x3 block diagonal block of matrix W, of size nc.
    VectorX<T> Wii_norm;
    // Approximation to the inverse of the diagonal of W, of size nc.
    VectorX<T> Dinv;
    void Resize(int nv, int nc) {
      W.resize(3 * nc, 3 * nc);
      vc_star.resize(3 * nc);
      Wii_norm.resize(nc);
      Dinv.resize(3 * nc);
    }
  };

  int num_velocities() const { return state_.num_velocities(); }
  int num_contacts() const { return state_.num_contacts(); }

  void PreProcessData(const SystemDynamicsData<T>& dynamics_data,
                      const PointContactData<T>& contact_data);

  // If relaxation is used, gamma_kp, vc_kp MUST be the velocities after
  // relaxation has been applied. That is, gamma_kp, vc_kp are the updates at
  // the end of the iteration.
  // Input parameter `omega` is the relaxation parameter and it is used to
  // correct for the fact that omega < 1 implies a smaller update.
  bool VerifyConvergenceCriteria(const VectorX<T>& vc, const VectorX<T>& vc_kp,
                                 const VectorX<T>& gamma,
                                 const VectorX<T>& gamma_kp, double omega,
                                 double* vc_err, double* gamma_err) const;
  Vector3<T> ProjectImpulse(const Eigen::Ref<const Vector3<T>>& vc,
                            const Eigen::Ref<const Vector3<T>>& gamma,
                            const T& mu) const;
  void ProjectAllImpulses(const VectorX<T>& vc, const VectorX<T>& mu,
                          VectorX<T>* gamma_inout) const;

  int nv_, nc_;  // Problem size.
  PgsSolverParameters parameters_;
  const SystemDynamicsData<T>* dynamics_data_{nullptr};
  const PointContactData<T>* contact_data_{nullptr};
  PreProcessedData pre_proc_data_;
  State state_;
  PgsSolverStats stats_;
  // Workspace (many of these could live in the state as "cached" data.)
  VectorX<T> jc_;  // Generalized contact impulses.
  VectorX<T> vc_;  // Contact velocities.
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

extern template class ::drake::multibody::contact_solvers::internal::PgsSolver<
    double>;
