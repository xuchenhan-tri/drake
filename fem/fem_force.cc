#include "drake/fem/fem_force.h"

#include <iostream>

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {

template <typename T>
void FemForce<T>::AccumulateScaledElasticForce(
    T scale, EigenPtr<Matrix3X<T>> force) const {
  // Gradient of the shape function is constant for linear interpolation.
  Eigen::Matrix<T, 3, 4> grad_shape;
  grad_shape.col(0) = -Vector3<T>::Ones();
  grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();

  for (const FemElement<T>& e : elements_) {
    const Matrix3<T> P = e.get_constitutive_model()->CalcFirstPiola();
    Eigen::Matrix<T, 3, 4> element_force = scale * e.get_element_measure() * P *
                                           e.get_Dm_inv().transpose() *
                                           grad_shape;
    const Vector4<int>& indices = e.get_indices();
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      force->col(indices[i]) -= element_force.col(i);
    }
  }
}

template <typename T>
void FemForce<T>::AccumulateScaledDampingForce(
    T scale, const Eigen::Ref<const Matrix3X<T>>& v,
    EigenPtr<Matrix3X<T>> force) const {
  AccumulateScaledDampingForceDifferential(scale, v, force);
}

/*  The stiffness matrix is the second derivative of the energy w.r.t. the
 * degree of freedoms, i.e.: Kₖₗ = ∑ₑ ∂Fₐᵦ/∂qₗ * ∂Pᵢⱼ/∂Fₐᵦ * ∂Fᵢⱼ/∂qₖ * Vₑ  (1).
 *  The constitutive model is responsible for providing ∂Pᵢⱼ/∂Fₐᵦ.
 *  Notice, for vertex p = 1, 2, 3 in an element:
 *      ∂Fᵢⱼ/∂xₚₖ = δᵢₖ * δₚₘ * Dm⁻¹ₘⱼ = δᵢₖ * Dm⁻¹ₚⱼ
 *  For p = 0,
 *      ∂Fᵢⱼ/∂xₚₖ = δᵢₖ * vⱼ,
 *  where v = Dm⁻ᵀ*(-1,-1,-1)ᵀ.
 *  Combining these two cases, we get
 *      ∂Fᵢⱼ/∂qₚₖ = δᵢₖ * vpⱼ,
 *  where vp is the p-th column of Dm⁻ᵀ * grad_shape.
 *  Plugging this expression into equation (1) shows that the pq-th block in the
 * stiffness matrix is: Kₖₗ = ∑ₑ vpⱼ * ∂Pₖⱼ/∂Fₗᵦ * vqᵦ * Vₑ. Reindexing, we get:
 *      Kᵢₐ = ∑ₑ vpⱼ * ∂Pᵢⱼ/∂Fₐᵦ * vqᵦ * Vₑ.
 */
template <typename T>
void FemForce<T>::AccumulateScaledStiffnessMatrix(
    T scale, Eigen::SparseMatrix<T>* stiffness_matrix) const {
  // Gradient of the shape function is constant for linear interpolation.
  Eigen::Matrix<T, 3, 4> grad_shape;
  grad_shape.col(0) = -Vector3<T>::Ones();
  grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();

  for (const FemElement<T>& e : elements_) {
    const Eigen::Matrix<T, 9, 9> dPdF =
        e.get_constitutive_model()->CalcFirstPiolaDerivative();
    const auto& Dm_inv_transpose = e.get_Dm_inv().transpose();
    const auto& dFdq = Dm_inv_transpose * grad_shape;
    const auto& indices = e.get_indices();
    const T& volume = e.get_element_measure();

    for (int p = 0; p < dFdq.cols(); ++p) {
      for (int q = 0; q < dFdq.cols(); ++q) {
        Matrix3<T> K =
            TensorContraction(dPdF, dFdq.col(p), dFdq.col(q) * volume * scale);
        AccumulateSparseMatrixBlock(K, indices(p), indices(q),
                                    stiffness_matrix);
      }
    }
  }
}

template <typename T>
void FemForce<T>::AccumulateScaledDampingMatrix(
    T scale, Eigen::SparseMatrix<T>* damping_matrix) const {
  // Gradient of the shape function is constant for linear interpolation.
  Eigen::Matrix<T, 3, 4> grad_shape;
  grad_shape.col(0) = -Vector3<T>::Ones();
  grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();

  for (const FemElement<T>& e : elements_) {
    const Eigen::Matrix<T, 9, 9> dPdF =
        e.get_constitutive_model()->CalcFirstPiolaDerivative();
    const auto& Dm_inv_transpose = e.get_Dm_inv().transpose();
    const auto& dFdq = Dm_inv_transpose * grad_shape;
    const auto& indices = e.get_indices();
    const T& volume = e.get_element_measure();
    const T& alpha = e.get_constitutive_model()->get_alpha();
    const T& beta = e.get_constitutive_model()->get_beta();
    const T& density = e.get_density();
    const T fraction = 1.0 / static_cast<T>(indices.size());

    // Add in the contribution from the stiffness terms.
    for (int p = 0; p < dFdq.cols(); ++p) {
      for (int q = 0; q < dFdq.cols(); ++q) {
        Matrix3<T> K = TensorContraction(dPdF, dFdq.col(p),
                                         dFdq.col(q) * volume * scale * beta);
        AccumulateSparseMatrixBlock(K, indices(p), indices(q), damping_matrix);
      }
    }
    // Add in the contribution from the mass terms.
    T local_mass = fraction * volume * density;
    Matrix3<T> M = Matrix3<T>::Identity() * (alpha * local_mass * scale);
    for (int i = 0; i < indices.size(); ++i) {
      AccumulateSparseMatrixBlock(M, indices(i), indices(i), damping_matrix);
    }
  }
}

template class FemForce<double>;
}  // namespace fem
}  // namespace drake
