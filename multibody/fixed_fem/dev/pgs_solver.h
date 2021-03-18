#pragma once

#include <iostream>
#include <memory>

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
  double relaxation{1};
  // Absolute contact velocity tolerance, m/s.
  double abs_tolerance{1.0e-4};
  // Relative contact velocity tolerance, unitless.
  double rel_tolerance{1.0e-4};
  // Maximum number of PGS iterations.
  int max_iterations{100};
};

struct PgsSolverStats {
  int iterations{0};      // Number of PGS iterations.
  double vc_err{0.0};     // Error in the contact velocities, [m/s].
  double gamma_err{0.0};  // Error in the contact impulses, [Ns].
};

template <typename T>
class PgsSolver final : public ContactSolver<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PgsSolver);
  class State {
   public:
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(State);
    State() = default;
    void Resize(int nv, int nc) {
      v_.resize(nv);
      gamma_.resize(3 * nc);
    }
    const VectorX<T>& v() const { return v_; }
    VectorX<T>& mutable_v() { return v_; }
    const VectorX<T>& gamma() const { return gamma_; }
    VectorX<T>& mutable_gamma() { return gamma_; }

   private:
    VectorX<T> v_;
    VectorX<T> gamma_;
  };

  PgsSolver() = default;

  virtual ~PgsSolver() = default;

  void set_parameters(const PgsSolverParameters& parameters) {
    parameters_ = parameters;
  }

  ContactSolverStatus SolveWithGuess(const T& time_step,
                                     const SystemDynamicsData<T>& dynamics_data,
                                     const PointContactData<T>& contact_data,
                                     const VectorX<T>& v_guess,
                                     ContactSolverResults<T>* result) final;

  const PgsSolverStats& get_iteration_stats() const { return stats_; }

 private:
  // All this data must remain const after the call to PreProcessData().
  struct PreProcessedData {
    Eigen::SparseMatrix<T> W;
    VectorX<T> vc_star;
    VectorX<T> v_star;
    // Norm of the 3x3 block diagonal block of matrix W, of size nc.
    VectorX<T> Wii_norm;
    // Approximation to the inverse of the diagonal of N, of size nc.
    VectorX<T> Dinv;
    void Resize(int nv, int nc) {
      W.resize(3 * nc, 3 * nc);
      vc_star.resize(3 * nc);
      v_star.resize(nv);
      Wii_norm.resize(nc);
      Dinv.resize(3 * nc);
    }
  };

  void CopyContactResults(ContactSolverResults<T>* result) const {
    const int nv = state_.v().size();
    const int nc = state_.gamma().size() / 3;
    result->Resize(nv, nc);
    result->v_next = state_.v();
    ExtractNormal(vc_, &(result->vn));
    ExtractTangent(vc_, &(result->vt));
    ExtractNormal(state_.gamma(), &(result->fn));
    ExtractTangent(state_.gamma(), &(result->ft));
    result->tau_contact = tau_c_;
  }

  void PreProcessData(const SystemDynamicsData<T>& dynamics_data,
                      const PointContactData<T>& contact_data);
  bool VerifyConvergenceCriteria(int num_contacts, const VectorX<T>& vc,
                                 const VectorX<T>& vc_kp,
                                 const VectorX<T>& gamma,
                                 const VectorX<T>& gamma_kp, double* vc_err,
                                 double* gamma_err) const;
  Vector3<T> ProjectImpulse(const Eigen::Ref<const Vector3<T>>& vc,
                            const Eigen::Ref<const Vector3<T>>& gamma,
                            const T& mu) const;

  PgsSolverParameters parameters_;
  PreProcessedData pre_proc_data_;
  State state_;
  PgsSolverStats stats_;
  // Workspace (many of these could live in the state as "cached" data.)
  VectorX<T> tau_c_;  // Generalized contact impulses.
  VectorX<T> vc_;     // Contact velocities.
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::PgsSolver);