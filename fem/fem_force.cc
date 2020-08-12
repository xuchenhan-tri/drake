#include "drake/fem/fem_force.h"
#include <iostream>
#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {

template <typename T>
void FemForce<T>::AccumulateScaledElasticForce(T scale, EigenPtr<Matrix3X<T>> force) const {
  // Gradient of the shape function is constant for linear interpolation.
    // TODO(xuchenhan-tri): support non-linear elements.
  Eigen::Matrix<T, 3, 4> grad_shape;
  grad_shape.col(0) = -Vector3<T>::Ones();
  grad_shape.template topRightCorner<3, 3>() = Matrix3<T>::Identity();

  for (const FemElement<T>& e : elements_) {
    const Matrix3<T>& P = e.get_constitutive_model()->CalcFirstPiola();
    Eigen::Matrix<T, 3, 4> element_force =
        scale * e.get_element_measure() * P * e.get_Dm_inv().transpose() * grad_shape;
    const Vector4<int>& indices = e.get_indices();
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      force->col(indices[i]) -= element_force.col(i);
    }
  }
}

template <typename T>
void FemForce<T>::AccumulateScaledDampingForce(T scale, const Eigen::Ref<const Matrix3X<T>>& v, EigenPtr<Matrix3X<T>> force) const {
    AccumulateScaledDampingForceDifferential(scale, v, force);
}

template <typename T>
void FemForce<T>::AccumulateScaledStiffnessEntry(
    T scale, Eigen::SparseMatrix<T>* stiffness_matrix) const {
  // TODO(xuchenhan-tri): implement me.
  DRAKE_DEMAND(scale == 0);
    DRAKE_DEMAND(stiffness_matrix == 0);
}

template class FemForce<double>;
}  // namespace fem
}  // namespace drake
