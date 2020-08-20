#pragma once

#include <memory>
#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/fem/constitutive_model.h"

namespace drake {
namespace fem {
template <typename T>
class FemElement {
  // TODO(xuchenhan-tri): delegate some functionality of this class to a Mesh
  // class.
 public:
  FemElement(const Vector4<int>& vertex_indices, const Matrix3X<T>& positions,
             std::unique_ptr<ConstitutiveModel<T>> model, T density)
      : vertex_indices_(vertex_indices),
        constitutive_model_(std::move(model)),
        F_(Matrix3<T>::Identity()),
        density_(density) {
    Matrix3<T> Dm = CalcShapeMatrix(vertex_indices_, positions);
    T unit_simplex_volume = 1.0 / 6.0;
    element_measure_ = Dm.determinant() * unit_simplex_volume;
    // Degenerate tetrahedron is not allowed.
    DRAKE_DEMAND(element_measure_ > 0);
    Eigen::HouseholderQR<Matrix3<T>> qr(Dm);
    Dm_inv_ = qr.solve(Matrix3<T>::Identity());
  }

  /** Calculates the deformation gradient F of each element and update all
   states that depend on F.
   @param[in] q  The positions of the vertices.
   */
  void UpdateF(const Eigen::Ref<const Matrix3X<T>>& q) {
    F_ = CalcShapeMatrix(vertex_indices_, q) * Dm_inv_;
    constitutive_model_->UpdateDeformationBasedState(F_);
  }

  void UpdateTimeNPositionBasedState(const Eigen::Ref<const Matrix3X<T>>& q_n)
  {
      Eigen::Matrix<T, 3, 4> local_qn;
      for (int i = 0; i < 4; ++i) {
          local_qn.col(i) = q_n.col(vertex_indices_[i]);
      }
      constitutive_model_->UpdateTimeNPositionBasedState(local_qn);
  }

  /** Given index = [i₀, i₁, i₂, i₃], calculates the shape matrix from linear
   * interpolation [x_i₁-x_i₀; x_i₂-x_i₀; x_i₃-x_i₀]. */
  static inline Matrix3<T> CalcShapeMatrix(
      const Vector4<int>& index, const Eigen::Ref<const Matrix3X<T>>& q) {
    Matrix3<T> shape_matrix;
    for (int i = 0; i < 3; ++i) {
      shape_matrix.col(i) = q.col(index(i + 1)) - q.col(index(0));
    }
    return shape_matrix;
  }

  T get_element_measure() const { return element_measure_; }

  T get_density() const { return density_; }

  const Vector4<int>& get_indices() const { return vertex_indices_; }

  const Matrix3<T>& get_Dm_inv() const { return Dm_inv_; }

  const Matrix3<T>& get_F() const { return F_; }

  const ConstitutiveModel<T>* get_constitutive_model() const {
    return constitutive_model_.get();
  }

 private:
  Vector4<int> vertex_indices_;
  std::unique_ptr<ConstitutiveModel<T>> constitutive_model_;
  Matrix3<T> F_;
  Matrix3<T> Dm_inv_;
  T element_measure_;
  T density_;
};

}  // namespace fem
}  // namespace drake
