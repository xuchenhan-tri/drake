#pragma once

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/fem_force.h"
#include "drake/fem/fem_solver.h"

namespace drake {
namespace fem {

template <typename T>
class FemSolver;
/**
 When doing Backward Euler, the objective we are solving for is

 G(x) = M*x - f(qⁿ⁺¹, vⁿ⁺¹) * dt

 where x = dv = vⁿ⁺¹ - vⁿ.
 Using qⁿ⁺¹ = qⁿ + dt * vⁿ⁺¹, we get

 G(x) = M*x - f(qⁿ + dt * (vⁿ + x), vⁿ + x) * dt.

 and dG/dx = M - ∂f/∂q * dt² - ∂f/∂v * dt.

 Update quantities depending on x to prepare for residual calculation and
 the linear solve. More specifically, we need v* = vⁿ + x and q* = qⁿ + dt *
 v*. The force can be decomposed into elastic force and damping force: f(q*,
 v*) = fₑ(q*) + fd(q*, v*).

 Elastic Force:
 The elastic force comes from an energy E(q), and fₑ(q*) = -dE/dq(q*). The
 discrete energy E takes the form E(q) = ∑ₑ ϕ(Fₑ(q)) Vₑ. Differentiating
 w.r.t. q, we get

     fₑ(q) = -∑ₑ dϕ/dF * dF/dq * Vₑ = ∑ₑ P(F) * dF/dq * Vₑ.

 Notice that dF/dq is a third order tensor. Using Einstein notation for
 dimension indices, the above equation reads:

     fₖ = ∑ₑ PᵢⱼdFᵢⱼ/dqₖ * Vₑ.

 Suppose the deformed shape function is q(ξ) and the undeformed shape
 function is Q(ξ), such that

     q(0,0,0) = q0, q(1,0,0) = q1, q(0,1,0) = q2, q(0,0,1) = q3,
     Q(0,0,0) = Q0, Q(1,0,0) = Q1, Q(0,1,0) = q2, Q(0,0,1) = Q3.

 then qi = FQi+b for i=0,1,2,3.
 Let Ds(q) = [q1-q0;q2-q0;q3-q0] and Dm(Q) = [Q1-Q0;Q2-Q0;Q3-Q0], we have

     Ds = FDm ==> Fᵢⱼ = DsᵢₗDmₗⱼ⁻¹.

 For p = 1,2,3:

     dDsᵢₗ/dqpₖ = δᵢₚδₖₗ,

 so,

     dF/dqpₖ = δᵢₚδₖₗDmₗⱼ⁻¹ = δᵢₚDmₖⱼ⁻¹,

 so for p = 1,2,3, the force on p is

     fₖ = ∑ₑ PᵢⱼδᵢₚDmₖⱼ⁻¹ * Vₑ = ∑ₑ PₚⱼDmⱼₖ⁻ᵀ * Vₑ.

 The force on p = 0 is the -1 times the sum of forces on p = 1,2,3 by force
 balance.

 Examining the equation, we see that we only need to calculate P as Dm⁻¹ and
 Vₑ are constants.

 Damping force:
 We use Rayleigh damping such that
 fd(q,v) = D*v where D = alpha * M + beta * K(q) and K(q) = d²E/dq².
 Using the fact that dF/dq * Vₑ is constant, we calculate

     Kv = -dfₑ/dq * v = -dfₑ(v) = ∑ₑ dP(dF(v)) * dF/dq * Vₑ,

 where dF(v) = dF/dq * v. For p = 1,2,3

     dFᵢⱼ/dqpₖ vpₖ = δᵢₚδₖₗDmₗⱼ⁻¹ vpₖ= vₚₗDmₗⱼ⁻¹,

 A similar computation for p = 0 reveals

     dF(v) = Ds(v)Dm⁻¹.

 The only unknown left is dP(dF) which is equal to dP/dF * dF. We delegate
 this computation to the constitutive model.

 We need to solve A * dx = b for dx where b = -G(x) calculated with
 CalcResidual(), and

     A = dG/dx =  M - ∂f/∂q * dt² - ∂f/∂v * dt.

 Note that f(q,v) = fₑ(q) + fd(q,v), so
 ∂f/∂q =  dfₑ/dq + ∂fd/∂q and ∂f/∂v = ∂fd/∂v, we tackle these terms one at a
 time:

 dfₑ/dq = -K(q) = -d²E/dq² = -∑ₑ (dF/dq) : dP/dF : dF/dq * Vₑ. There is no
 d²F/dq² term because F is linear in q. In the matrix-free version, we need
 -K*x which can be found given dP(dF) as in the process of calculating
 damping force.

 The term ∂fd/∂q is tricky as it involves a third derivative of the energy.
 We omit it as does in [Sifakis, 2012] which claims that this modification
 does not affect convergence of Newton's method. Therefore, technically we
 are using a modified Newton's method.

 ∂fd/∂v = D = alpha * M + beta * K.

 Assembling A, we get

 A = (1+alpha*dt) * M + (beta * dt + dt²) * K.
 */
template <typename T>
class BackwardEulerObjective {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BackwardEulerObjective)

  BackwardEulerObjective(FemSolver<T>& solver, FemForce<T>& force)
      : fem_solver_(solver), force_(force) {}

  /** Move the position of the vertices to tmp_x = q_hat + dt * dv where q_hat =
   * q + v_n * dt, and updates the quantities that depends on vertex positions.
   * @param dv[in] The candidate change of velocity.
   */
  void Update(const VectorX<T>& dv) {
    const Matrix3X<T>& tmp_x =
        Eigen::Map<const Matrix3X<T>>(dv.data(), 3, dv.size() / 3) *
            fem_solver_.get_dt() +
        fem_solver_.get_q_hat();
    auto& elements = fem_solver_.get_mutable_elements();
    for (auto& e : elements) {
      e.UpdateF(tmp_x);
    }
  }

  /** Evaluate -G(x) = -M*x + f(qⁿ + dt * (vⁿ + x), vⁿ + x) * dt, where f = fe +
   * fd + gravity. */
  void CalcResidual(VectorX<T>* residual) {
    Eigen::Map<Matrix3X<T>> impulse(residual->data(), 3, residual->size() / 3);
    impulse.setZero();
    const VectorX<T>& mass = fem_solver_.get_mass();
    const Matrix3X<T>& dv = fem_solver_.get_dv();
    const T& dt = fem_solver_.get_dt();
    const Vector3<T>& gravity = fem_solver_.get_gravity();
    const auto& v_hat = fem_solver_.get_v() + fem_solver_.get_dv();

    // Add -M*x + gravity * dt
    for (int i = 0; i < mass.size(); ++i) {
      impulse.col(i) -= mass(i) * dv.col(i);
      impulse.col(i) += mass(i) * dt * gravity;
    }
    // Add fe * dt.
    force_.AccumulateScaledElasticForce(dt, &impulse);
    // Add fd * dt.
    force_.AccumulateScaledDampingForce(dt, v_hat, &impulse);
    // Apply boundary condition.
    Project(&impulse);
    *residual = Eigen::Map<VectorX<T>>(impulse.data(), impulse.size());
  }

  /** Return the product of matrix-vector multiplication A*x where A =
   * (1+alpha*dt) M + (beta * dt + dt²) * K. */
  void Multiply(const Eigen::Ref<const Matrix3X<T>>& x,
                EigenPtr<Matrix3X<T>> prod) const {
    DRAKE_DEMAND(prod->cols() == fem_solver_.get_mass().size());
    DRAKE_DEMAND(x.cols() == fem_solver_.get_mass().size());
    const VectorX<T>& mass = fem_solver_.get_mass();
    const T& dt = fem_solver_.get_dt();
    // Get M*x.
    for (int i = 0; i < prod->cols(); ++i) {
      prod->col(i) = mass(i) * x.col(i);
    }
    // Get dt * (alpha * M + beta * K) * x.
    force_.AccumulateScaledDampingForceDifferential(-dt, x, prod);
    // Get  dt² * K * x.
    force_.AccumulateScaledElasticForceDifferential(-dt * dt, x, prod);
    // Apply boundary condition.
    Project(prod);
  }

  /** Build the matrix A = (1+alpha*dt) * M + (beta * dt + dt²) * K. */
  void BuildJacobian() const { /* TODO(xuchenhan-tri): implement me. */
  }

  int get_num_dofs() const { return fem_solver_.get_q().size(); }

  void Project(EigenPtr<Matrix3X<T>> impulse) const {
    const auto& bc = fem_solver_.get_v_bc();
    const auto& vertex_indices = fem_solver_.get_vertex_indices();
    const T& time = fem_solver_.get_time();
    const Matrix3X<T>& initial_position = fem_solver_.get_Q();

    for (const auto& boundary_condition : bc) {
      const auto& vertex_range = vertex_indices[boundary_condition.object_id];
      for (int j = 0; j < static_cast<int>(vertex_range.size()); ++j) {
        boundary_condition.bc(vertex_range[j], time, initial_position, impulse);
      }
    }
  }

 private:
  FemSolver<T>& fem_solver_;
  FemForce<T>& force_;
};
}  // namespace fem
}  // namespace drake
