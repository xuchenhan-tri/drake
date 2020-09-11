#pragma once

#include <iostream>
#include <vector>

#include <Eigen/Sparse>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/fem_state.h"
#include "drake/fem/fem_data.h"

namespace drake {
namespace fem {
/**
FemForce calculates the elastic and damping forces from the constitutive models.

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
 */
template <typename T>
class FemForce {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemForce)

  explicit FemForce(const FemData<T>& data, FemState<T>* state)
      : fem_data_(data), fem_state_(*state) {}

  /** Called by BackwardEulerObjective::CalcResidual. Calculates the elastic
  forces, scale them by scale, and then add them to the force vector.  */
  void AccumulateScaledElasticForce(T scale, EigenPtr<Matrix3X<T>> force) const;

  /** Called by BackwardEulerObjective::CalcResidual. Calculates the damping
  forces, scale them by scale, and then add them to the force vector.  */
  void AccumulateScaledDampingForce(T scale,
                                    const Eigen::Ref<const Matrix3X<T>>& v,
                                    EigenPtr<Matrix3X<T>> force) const;

  /** Called by BackwardEulerObjective::Multiply. Calculates the K*dx where K is
    the stiffness matrix, scale the result by scale, and then add it to the
    force_differential vector. */
  void AccumulateScaledElasticForceDifferential(
      T scale, const Eigen::Ref<const Matrix3X<T>>& dx,
      EigenPtr<Matrix3X<T>> force_differential) const;

  /** Called by BackwardEulerObjective::Multiply. Calculates the D*dx where D =
      alpha * M + beta * K is the damping matrix, scale the result by scale, and
      then add it to the force_differential vector. */
  void AccumulateScaledDampingForceDifferential(
      T scale, const Eigen::Ref<const Matrix3X<T>>& dx,
      EigenPtr<Matrix3X<T>> force_differential) const;

  /** Returns the total elastic energy stored in the elements. */
  T CalcElasticEnergy() const;

  /** Called by BackwardEulerObjective::BuildJacobian. Calculates K where K is
    the stiffness matrix, scale it by scale, and then add it to the
  stiffness_matrix. */
  void AccumulateScaledStiffnessMatrix(
      T scale, Eigen::SparseMatrix<T>* stiffness_matrix) const;

  /** Called by BackwardEulerObjective::BuildJacobian. Calculates D where D is
    the stiffness matrix, scale it by scale, and then add it to the
  stiffness_matrix. */
  void AccumulateScaledDampingMatrix(
      T scale, Eigen::SparseMatrix<T>* damping_matrix) const;

  /** Set the nonzero entries contributed by the damping and stiffness matrix.
   */
  void SetSparsityPattern(
      std::vector<Eigen::Triplet<T>>* non_zero_entries) const;

  // TODO(xuchenhan-tri): The following two methods don't really belong to
  // FemForce. Think of a better place to put them.
  /** Performs a contraction between a 4th order tensor A and two vectors u and
     v and returns a matrix B. In Einstein notation, the contraction is:
         Bᵢₐ = uⱼ Aᵢⱼₐᵦ vᵦ.
     The 4th order tensor A of dimension 3*3*3*3 is flattened to a
     9*9 matrix that is organized as following

                      β = 1       β = 2       β = 3
                  -------------------------------------
                  |           |           |           |
        j = 1     |   Aᵢ₁ₐ₁   |   Aᵢ₁ₐ₂   |   Aᵢ₁ₐ₃   |
                  |           |           |           |
                  -------------------------------------
                  |           |           |           |
        j = 2     |   Aᵢ₂ₐ₁   |   Aᵢ₂ₐ₂   |   Aᵢ₂ₐ₃   |
                  |           |           |           |
                  -------------------------------------
                  |           |           |           |
        j = 3     |   Aᵢ₃ₐ₁   |   Aᵢ₃ₐ₂   |   Aᵢ₃ₐ₃   |
                  |           |           |           |
                  -------------------------------------
   */
  static inline Matrix3<T> TensorContraction(
      const Eigen::Ref<const Eigen::Matrix<T, 9, 9>>& A,
      const Eigen::Ref<const Vector3<T>>& u,
      const Eigen::Ref<const Vector3<T>>& v) {
    Matrix3<T> B = Matrix3<T>::Zero();
    for (int j = 0; j < 3; ++j) {
      for (int beta = 0; beta < 3; ++beta) {
        B += A.template block<3, 3>(3 * j, 3 * beta) * u(j) * v(beta);
      }
    }
    return B;
  }

  /** Add a 3x3 matrix into the 3x3 block in the sparse matrix 'matrix' with
   * starting row index 3*m and starting column index 3*n. */
  static inline void AccumulateSparseMatrixBlock(
      const Eigen::Ref<const Matrix3<T>> block, const int m, const int n,
      Eigen::SparseMatrix<T>* matrix) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        matrix->coeffRef(3 * m + i, 3 * n + j) += block(i, j);
      }
    }
  }

 private:
    // Evaluates the elastic energy density cache.
    const std::vector<T>& EvalPsi() const;
  // Evaluates the first Piola stress cache.
  const std::vector<Matrix3<T>>& EvalP() const;
    // Evaluates the first Piola stress derivative cache.
  const std::vector<Eigen::Matrix<T,9,9>>& EvaldPdF() const;

  const FemData<T>& fem_data_;
  FemState<T>& fem_state_;
};
}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::FemForce)
