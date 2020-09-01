#pragma once

#include "drake/common/eigen_types.h"
#include "drake/fem/fem_data.h"
#include "drake/fem/fem_force.h"

namespace drake {
namespace fem {
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
 v*) = fₑ(q*) + fd(q*, v*). We delegate the computation of these forces to
 FemForce.


 We need to solve A * dx = b for dx where b = -G(x) calculated with
 CalcResidual(), and

     A = dG/dx =  M - ∂f/∂q * dt² - ∂f/∂v * dt.

 Note that f(q,v) = fₑ(q) + fd(q,v), so
 ∂f/∂q =  dfₑ/dq + ∂fd/∂q and ∂f/∂v = ∂fd/∂v, we tackle these terms one at a
 time:

 dfₑ/dq = -K(q) = -d²E/dq² = -∑ₑ (dF/dq) : dP/dF : dF/dq * Vₑ. There is no
 d²F/dq² term because F is linear in q.
 // TODO(xuchenhan-tri): Work out the expression for the force derivative when
 the element is no longer linear.

 In the matrix-free version, we need
 -K*x which can be found given dP(dF) as in the process of calculating
 damping force.

 The term ∂fd/∂q is tricky as it involves a third derivative of the energy.
 We omit it as does in [Sifakis, 2012] which claims that this modification
 does not affect convergence of Newton's method. Therefore, technically we
 are using a modified Newton's method.

 ∂fd/∂v = D = alpha * M + beta * K.

 Assembling A, we get

 A = (1+alpha*dt) * M + (beta * dt + dt²) * K.

 We delegate the actual calculation of forming K or calculating K*x to FemForce.

 */
template <typename T>
class BackwardEulerObjective {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BackwardEulerObjective)

  BackwardEulerObjective(FemData<T>* data, FemForce<T>* force)
      : fem_data_(*data), force_(*force) {}

  /** Move the position of the vertices to tmp_x = q_hat + dt * dv where q_hat =
   * q + v_n * dt, and updates the quantities that depends on vertex positions.
   * @param dv[in] The candidate change of velocity.
   */
  void Update(const Eigen::Ref<const VectorX<T>>& dv);

  /** Evaluate -G(x) = -M*x + f(qⁿ + dt * (vⁿ + x), vⁿ + x) * dt, where f = fe +
   * fd + gravity. */
  void CalcResidual(EigenPtr<VectorX<T>> residual);

  /** Return the product of matrix-vector multiplication A*x where A =
   * (1+alpha*dt) M + (beta * dt + dt²) * K. */
  void Multiply(const Eigen::Ref<const Matrix3X<T>>& x,
                EigenPtr<Matrix3X<T>> prod) const;

  /** Build the matrix A = (1+alpha*dt) * M + (beta * dt + dt²) * K. */
  void BuildJacobian(Eigen::SparseMatrix<T>* jacobian) const;

  /** Allocate memory for the input Eigen::SparseMatrix for entries that are
   * non-zero in the stiffness and damping matrices. */
  void SetSparsityPattern(Eigen::SparseMatrix<T>* jacobian) const;

  /** Sets the entries corresponding to vertices under Dirichlet boundary
   * conditions to zero. */
  void Project(EigenPtr<Matrix3X<T>> impulse) const;

  /** Returns 3 * number of vertices. */
  int get_num_dofs() const { return fem_data_.get_q().size(); }

  const VectorX<T>& get_mass() const { return fem_data_.get_mass(); }

  /** Take in a vector of impulse, divides by the mass of each vertex and return
   * the infinity norm of the resulting vector that has the unit of velocities.
   */
  T norm(const Eigen::Ref<const VectorX<T>>& x) const;

 private:
  FemData<T>& fem_data_;
  FemForce<T>& force_;
};
}  // namespace fem
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::BackwardEulerObjective)
