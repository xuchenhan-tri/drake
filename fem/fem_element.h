#pragma once

#include <memory>
#include <utility>
#include <array>

#include "drake/common/eigen_types.h"
#include "drake/common/default_scalars.h"
#include "drake/fem/hyperelastic_constitutive_model.h"

namespace drake {
namespace fem {
template <typename T>
class FemElement {
  // TODO(xuchenhan-tri): delegate some functionality of this class to a Mesh
  // class.
  // TODO(xuchenhan-tri): We only support linear tetrahedral elements for the
  // moment. Consider supporting other element shapes as well as higher-order
  // elements. class.
 public:
  FemElement(const Vector4<int>& vertex_indices, const Matrix3X<T>& positions,
             std::unique_ptr<HyperelasticConstitutiveModel<T>> model, T density, int vertex_offset)
      : vertex_indices_(vertex_indices + Vector4<int>::Ones() * vertex_offset),
        constitutive_model_(std::move(model)),
        density_(density) {
    Matrix3<T> Dm = CalcShapeMatrix(vertex_indices, positions);
    T unit_simplex_volume = 1.0 / 6.0;
    element_measure_ = Dm.determinant() * unit_simplex_volume;
    // Degenerate tetrahedron in the initial configuration is not allowed.
    DRAKE_DEMAND(element_measure_ > 0);
    Eigen::HouseholderQR<Matrix3<T>> qr(Dm);
    Dm_inv_ = qr.solve(Matrix3<T>::Identity());
  }

  // Allow move constructor and assignment, but disable copy constructor and assignment.
  FemElement(const FemElement&) = delete;
  FemElement(FemElement&&) = default;
  FemElement& operator=(const FemElement&) = delete;
  FemElement& operator=(FemElement&&) = default;

  /** Calculates the deformation gradient F of each quadrature point in the element.
   @param[in] q  The positions of the vertices.
   */
  Matrix3<T> CalcF(const Eigen::Ref<const Matrix3X<T>>& q) const {
      // Linear elements has only one quadrature point
    Matrix3<T> local_F;
    local_F = CalcShapeMatrix(vertex_indices_, q) * Dm_inv_;
    return local_F;
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

  const HyperelasticConstitutiveModel<T>* get_constitutive_model() const {
    return constitutive_model_.get();
  }


 private:
  Vector4<int> vertex_indices_;
  std::unique_ptr<HyperelasticConstitutiveModel<T>> constitutive_model_;
  Matrix3<T> Dm_inv_;
  T element_measure_;
  T density_;
};

}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::drake::fem::FemElement)
