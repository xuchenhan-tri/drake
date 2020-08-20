#pragma once

#include <iostream>
#include <vector>

#include <Eigen/Sparse>

#include "drake/common/eigen_types.h"
#include "drake/fem/fem_element.h"

namespace drake {
namespace fem {
template <typename T>
class FemForce {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemForce)

  explicit FemForce(const std::vector<FemElement<T>>& elements)
      : elements_(elements) {}

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
      EigenPtr<Matrix3X<T>> force_differential) const {
    // Gradient of the shape function is constant for linear interpolation.
    // TODO(xuchenhan-tri): support non-linear elements.
    Eigen::Matrix<T, 3, 4> grad_shape;
    grad_shape.col(0) = -Vector3<T>::Ones();
    grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();
    for (const FemElement<T>& e : elements_) {
      Matrix3<T> dF = e.CalcShapeMatrix(e.get_indices(), dx) * e.get_Dm_inv();
      const Matrix3<T>& dP =
          e.get_constitutive_model()->CalcFirstPiolaDifferential(dF);
      Eigen::Matrix<T, 3, 4> element_force_differential =
          scale * e.get_element_measure() * dP * e.get_Dm_inv().transpose() *
          grad_shape;
      const Vector4<int>& indices = e.get_indices();
      for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        force_differential->col(indices[i]) -=
            element_force_differential.col(i);
      }
    }
  }

  /** Called by BackwardEulerObjective::Multiply. Calculates the D*dx where D =
alpha * M + beta * K is the damping matrix, scale the result by scale, and
then add it to the force_differential vector. */
  void AccumulateScaledDampingForceDifferential(
      T scale, const Eigen::Ref<const Matrix3X<T>>& dx,
      EigenPtr<Matrix3X<T>> force_differential) const {
    Eigen::Matrix<T, 3, 4> grad_shape;
    grad_shape.col(0) = -Vector3<T>::Ones();
    grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();
    for (const FemElement<T>& e : elements_) {
      const Vector4<int>& indices = e.get_indices();
      const Matrix3<T>& Dm_inv = e.get_Dm_inv();
      const T& volume = e.get_element_measure();
      const T& density = e.get_density();
      const auto* model = e.get_constitutive_model();
      const T fraction = 1.0 / static_cast<T>(indices.size());
      Matrix3<T> dF = e.CalcShapeMatrix(indices, dx) * Dm_inv;
      const Matrix3<T>& dP =
          model->CalcFirstPiolaDifferential(dF) * model->get_beta();
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

  /** Returns the total elastic energy stored in the elements. */
  T CalcElasticEnergy() const {
    T elastic_energy = 0;
    for (const FemElement<T>& e : elements_) {
      const T& volume = e.get_element_measure();
      const auto* model = e.get_constitutive_model();
      T energy_density = model->CalcEnergyDensity();
      elastic_energy += volume * energy_density;
    }
    return elastic_energy;
  }
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

  void SetSparsityPattern(
      std::vector<Eigen::Triplet<T>>* non_zero_entries) const {
    // Extend the number of nonzero entries.
    // Each element has 4 vertices, which gives 4x4 nonzero blocks and each
    // block is of size 3x3. There will be redundancies in the allocation which
    // the caller will compress away.
    non_zero_entries->reserve(non_zero_entries->size() +
                              elements_.size() * 4 * 4 * 3 * 3);
    for (const auto& e : elements_) {
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
  const std::vector<FemElement<T>>& elements_;
};
}  // namespace fem
}  // namespace drake
