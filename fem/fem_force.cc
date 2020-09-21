#include "drake/fem/fem_force.h"

#include <iostream>

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {

template <typename T>
void FemForce<T>::AccumulateScaledElasticForce(const FemState<T>& state,
    T scale, EigenPtr<Matrix3X<T>> force) const {
  // Gradient of the shape function is constant for linear interpolation.
  Eigen::Matrix<T, 3, 4> grad_shape;
  grad_shape.col(0) = -Vector3<T>::Ones();
  grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();

  const auto& elements = fem_data_.get_elements();
  const auto& P = evaluator_.EvalP(state);
  int quadrature_offset = 0;
  for (const FemElement<T>& e : elements) {
    Eigen::Matrix<T, 3, 4> element_force =
        scale * e.get_element_measure() * P[quadrature_offset++] *
        e.get_Dm_inv().transpose() * grad_shape;
    const Vector4<int>& indices = e.get_indices();
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      force->col(indices[i]) -= element_force.col(i);
    }
  }
}

template <typename T>
void FemForce<T>::AccumulateScaledDampingForce(const FemState<T>& state,
    T scale,
    EigenPtr<Matrix3X<T>> force) const {
    const auto& v = state.get_v();
  AccumulateScaledDampingForceDifferential(state, scale, v, force);
}

template <typename T>
void FemForce<T>::AccumulateScaledElasticForceDifferential(const FemState<T>& state,
    T scale, const Eigen::Ref<const Matrix3X<T>>& dx,
    EigenPtr<Matrix3X<T>> force_differential) const {
  // Gradient of the shape function is constant for linear interpolation.
  // TODO(xuchenhan-tri): support non-linear elements.
  Eigen::Matrix<T, 3, 4> grad_shape;
  grad_shape.col(0) = -Vector3<T>::Ones();
  grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();

  const auto& elements = fem_data_.get_elements();
  const auto& model_cache = evaluator_.EvalHyperelasticCache(state);
  int quadrature_offset = 0;
  for (const FemElement<T>& e : elements) {
    const Matrix3<T> dF =
        e.CalcShapeMatrix(e.get_indices(), dx) * e.get_Dm_inv();
    const Matrix3<T>& dP =
        e.get_constitutive_model()->CalcFirstPiolaDifferential(
            *model_cache[quadrature_offset++], dF);
    const Eigen::Matrix<T, 3, 4> element_force_differential =
        scale * e.get_element_measure() * dP * e.get_Dm_inv().transpose() *
        grad_shape;
    const Vector4<int>& indices = e.get_indices();
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      force_differential->col(indices[i]) -= element_force_differential.col(i);
    }
  }
}

template <typename T>
void FemForce<T>::AccumulateScaledDampingForceDifferential(const FemState<T>& state,
    T scale, const Eigen::Ref<const Matrix3X<T>>& dx,
    EigenPtr<Matrix3X<T>> force_differential) const {
  Eigen::Matrix<T, 3, 4> grad_shape;
  grad_shape.col(0) = -Vector3<T>::Ones();
  grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();
  const auto& elements = fem_data_.get_elements();
  const auto& model_cache = evaluator_.EvalHyperelasticCache(state);
  int quadrature_offset = 0;
  for (const FemElement<T>& e : elements) {
    const Vector4<int>& indices = e.get_indices();
    const Matrix3<T>& Dm_inv = e.get_Dm_inv();
    const T& volume = e.get_element_measure();
    const T& density = e.get_density();
    const auto* model = e.get_constitutive_model();
    const T fraction = 1.0 / static_cast<T>(indices.size());
    Matrix3<T> dF = e.CalcShapeMatrix(indices, dx) * Dm_inv;
    const Matrix3<T>& dP = model->CalcFirstPiolaDifferential(
                               *model_cache[quadrature_offset++], dF) *
                           model->get_beta();
    Eigen::Matrix<T, 3, 4> negative_element_force_differential =
        scale * volume * dP * Dm_inv.transpose() * grad_shape;
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      // Stiffness terms.
      force_differential->col(indices[i]) -=
          negative_element_force_differential.col(i);
      // Mass terms.
      force_differential->col(indices[i]) -= scale * volume * density *
                                             fraction * model->get_alpha() *
                                             dx.col(indices[i]);
    }
  }
}

template <typename T>
T FemForce<T>::CalcElasticEnergy(const FemState<T>& state) const {
  T elastic_energy = 0;
  const auto& elements = fem_data_.get_elements();
  const auto& psi = evaluator_.EvalPsi(state);
  int quadrature_offset = 0;
  for (const FemElement<T>& e : elements) {
    const auto& volume = e.get_element_measure();
    elastic_energy += volume * psi[quadrature_offset++];
  }
  return elastic_energy;
}

template <typename T>
void FemForce<T>::SetSparsityPattern(
    std::vector<Eigen::Triplet<T>>* non_zero_entries) const {
  // Extend the number of nonzero entries.
  // Each element has 4 vertices, which gives 4x4 nonzero blocks and each
  // block is of size 3x3. There will be redundancies in the allocation which
  // the caller will compress away.
  const auto& elements = fem_data_.get_elements();
  non_zero_entries->reserve(non_zero_entries->size() +
                            elements.size() * 4 * 4 * 3 * 3);
  for (const auto& e : elements) {
    const auto& indices = e.get_indices();
    for (int i = 0; i < indices.size(); ++i) {
      for (int k = 0; k < 3; ++k) {
        int row_index = 3 * i + k;
        for (int j = 0; j < indices.size(); ++j) {
          for (int l = 0; l < 3; ++l) {
            int col_index = 3 * j + l;
            non_zero_entries->emplace_back(row_index, col_index, 0);
          }
        }
      }
    }
  }
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
void FemForce<T>::AccumulateScaledStiffnessMatrix(const FemState<T>& state,
    T scale, Eigen::SparseMatrix<T>* stiffness_matrix) const {
  // Gradient of the shape function is constant for linear interpolation.
  Eigen::Matrix<T, 3, 4> grad_shape;
  grad_shape.col(0) = -Vector3<T>::Ones();
  grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();

  const auto& dPdF = evaluator_.EvaldPdF(state);
  int quadrature_offset = 0;
  const auto& elements = fem_data_.get_elements();
  for (const FemElement<T>& e : elements) {
    const auto& Dm_inv_transpose = e.get_Dm_inv().transpose();
    const auto& dFdq = Dm_inv_transpose * grad_shape;
    const auto& indices = e.get_indices();
    const T& volume = e.get_element_measure();

    for (int p = 0; p < dFdq.cols(); ++p) {
      for (int q = 0; q < dFdq.cols(); ++q) {
        const Matrix3<T> K = TensorContraction(
            dPdF[quadrature_offset], dFdq.col(p), dFdq.col(q) * volume * scale);
        AccumulateSparseMatrixBlock(K, indices(p), indices(q),
                                    stiffness_matrix);
      }
    }
    ++quadrature_offset;
  }
}

template <typename T>
void FemForce<T>::AccumulateScaledDampingMatrix(const FemState<T>& state,
    T scale, Eigen::SparseMatrix<T>* damping_matrix) const {
  // Gradient of the shape function is constant for linear interpolation.
  Eigen::Matrix<T, 3, 4> grad_shape;
  grad_shape.col(0) = -Vector3<T>::Ones();
  grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();

  const auto& dPdF = evaluator_.EvaldPdF(state);
  int quadrature_offset = 0;
  const auto& elements = fem_data_.get_elements();
  for (const FemElement<T>& e : elements) {
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
        const Matrix3<T> K =
            TensorContraction(dPdF[quadrature_offset], dFdq.col(p),
                              dFdq.col(q) * volume * scale * beta);
        AccumulateSparseMatrixBlock(K, indices(p), indices(q), damping_matrix);
      }
    }
    ++quadrature_offset;
    // Add in the contribution from the mass terms.
    const T local_mass = fraction * volume * density;
    const Matrix3<T> M = Matrix3<T>::Identity() * (alpha * local_mass * scale);
    for (int i = 0; i < indices.size(); ++i) {
      AccumulateSparseMatrixBlock(M, indices(i), indices(i), damping_matrix);
    }
  }
}
}  // namespace fem
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::FemForce)
