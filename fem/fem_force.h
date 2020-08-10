#pragma once

#include <Eigen/Sparse>
#include "drake/common/eigen_types.h"
#include "drake/fem/fem_element.h"

namespace drake {
namespace fem {
template <typename T>
class FemForce {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemForce)

  FemForce(const std::vector<FemElement<T>>& elements) : elements_(elements) {}

  /** Called by BackwardEulerObjective::CalcResidual. Calculates the elastic and
    damping forces, scale them by scale, and then add them to the force vector.
   */
  void AccumulateScaledForce(T scale, EigenPtr<Matrix3X<T>> force) const;

  /** Called by BackwardEulerObjective::Multiply. Calculates the K*dx where K is
    the stiffness matrix, scale the result by scale, and then add it to the
    force_differential vector. */
  void AccumulateScaledElasticForceDifferential(T scale, const Eigen::Ref<const Matrix3X<T>>& dx,
                                         EigenPtr<Matrix3X<T>> force_differential) const
    {
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
    /** Called by BackwardEulerObjective::Multiply. Calculates the D*dx where D is
      the damping matrix, scale the result by scale, and then add it to the
      force_differential vector. */
    void AccumulateScaledDampingForceDifferential(T scale, const Eigen::Ref<const Matrix3X<T>>& dx,
                                           EigenPtr<Matrix3X<T>> force_differential) const
    {
        AccumulateScaledElasticForceDifferential(scale, dx, force_differential);
    }

  /** Called by BackwardEulerObjective::BuildJacobian. Calculates K where K is
    the stiffness matrix, scale it by scale, and then add it to
   * stiffness_matrx. */
  void AccumulateScaledStiffnessEntry(
      T scale, Eigen::SparseMatrix<T>* stiffness_matrix) const;

 private:
  const std::vector<FemElement<T>>& elements_;
};
}  // namespace fem
}  // namespace drake