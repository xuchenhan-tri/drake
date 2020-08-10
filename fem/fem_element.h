#pragma once

#include <memory>

#include "drake/common/eigen_types.h"
#include "drake/fem/constitutive_model.h"

namespace drake {
namespace fem {
template <typename T>
class FemElement {
  // TODO(xuchenhan-tri): delegate some functionality of this class to a Mesh
  // class.
 public:
    FemElement(const Vector4<int>& vertex_indices, const Matrix3X<T>& positions, std::unique_ptr<ConstitutiveModel<T>> model)
    : vertex_indices_(vertex_indices),
    constitutive_model_(std::move(model)),
    F_(Matrix3<T>::Identity()) {
        Matrix3<T> Dm = CalcShapeMatrix(vertex_indices_, positions);
        T unit_simplex_volume = 1.0 / 6.0;
        element_measure_ = Dm.determinant() * unit_simplex_volume;
        Dm_inv_ = Dm.solve(Matrix3<T>::Identity());
    }
  /** Calculates the deformation gradient F of each element and update all
   states that depend on F.
   @param[in] q  The positions of the vertices.
   */
  void UpdateF(const Matrix3X<T>& q);

  /** Given index = [i₀, i₁, i₂, i₃], calculates the shape matrix from linear
   * interpolation [x_i₁-x_i₀; x_i₂-x_i₀; x_i₃-x_i₀]. */
  Matrix3<T> CalcShapeMatrix(const Vector4<int>& index, const Matrix3X<T>& q) const
  {
      Matrix3<T> shape_matrix;
      for (int i = 0; i < 3; ++i)
      {
          shape_matrix.col(i) = q.col(index(i+1)) - q.col(0);
      }
      return shape_matrix;
  }

  T element_measure() const { return element_measure_; }

  const Vector4<int>& indices() const { return vertex_indices_; }

  const Matrix3<T>& Dm_inv() const { return Dm_inv_; }

  const ConstitutiveModel<T>* constitutive_model() const { return constitutive_model_.get(); }

 private:
  Vector4<int> vertex_indices_;
  std::unique_ptr<ConstitutiveModel<T>> constitutive_model_;
  Matrix3<T> F_;
  Matrix3<T> Dm_inv_;
  T element_measure_;
};

}  // namespace fem
}  // namespace drake
