#include "drake/fem/fem_force.h"

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {

template <typename T>
void FemForce<T>::AccumulateScaledForce(T scale, EigenPtr<Matrix3X<T>> force) const {
  // Gradient of the shape function is constant for linear interpolation.
  // TODO(xuchenhan-tri): support non-linear elements.
  Eigen::Matrix<T, 3, 4> grad_shape;
  grad_shape.col(0) = -Vector3<T>::Ones();
  grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();

  for (const FemElement<T>& e : elements_) {
    const Matrix3<T>& P = e.constitutive_model()->CalcFirstPiola(e.get_F());
    Eigen::Matrix<T, 3, 4> element_force =
        scale * e.element_measure() * P * e.Dm_inv().transpose() * grad_shape;
    const Vector4<int>& indices = e.indices();
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      force->col(indices[i]) -= element_force.col(i);
    }
  }
}

template <typename T>
void FemForce<T>::AccumulateScaledElasticForceDifferential(
    T scale, const Eigen::Ref<const Matrix3X<T>>& dx, EigenPtr<Matrix3X<T>> force_differential) const {
  // Gradient of the shape function is constant for linear interpolation.
  // TODO(xuchenhan-tri): support non-linear elements.
  Eigen::Matrix<T, 3, 4> grad_shape;
  grad_shape.col(0) = -Vector3<T>::Ones();
  grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();
  for (const FemElement<T>& e : elements_) {
    Matrix3<T> dF = e.CalcShapeMatrix(e.indices(), dx) * e.Dm_inv();
    const Matrix3<T>& dP =
        e.constitutive_model()->CalcFirstPiolaDifferential(dF);
    Eigen::Matrix<T, 3, 4> element_force_differential =
        scale * e.element_measure() * dP * e.Dm_inv().transpose() * grad_shape;
    const Vector4<int>& indices = e.indices();
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      force_differential->col(indices[i]) -= element_force_differential.col(i);
    }
  }
}
template <typename T>
void FemForce<T>::AccumulateScaledStiffnessEntry(
    T scale, Eigen::SparseMatrix<T>* stiffness_matrix) const {
  // TODO(xuchenhan-tri): implement me.
}

}  // namespace fem
}  // namespace drake
