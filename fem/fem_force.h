#pragma once

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
        force_differential->col(indices[i]) +=
            scale * volume * density * model->get_alpha() * dx.col(indices[i]);
      }
    }
  }

  /** Called by BackwardEulerObjective::BuildJacobian. Calculates K where K is
    the stiffness matrix, scale it by scale, and then add it to the
  stiffness_matrix. */
  void AccumulateScaledStiffnessEntry(
      T scale, Eigen::SparseMatrix<T>* stiffness_matrix) const;

 private:
  const std::vector<FemElement<T>>& elements_;
};
}  // namespace fem
}  // namespace drake
