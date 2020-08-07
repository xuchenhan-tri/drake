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
                                         EigenPtr<Matrix3X<T>> force_differential) const;
    /** Called by BackwardEulerObjective::Multiply. Calculates the D*dx where D is
      the damping matrix, scale the result by scale, and then add it to the
      force_differential vector. */
    void AccumulateScaledDampingForceDifferential(T scale, const Eigen::Ref<const Matrix3X<T>>& dx,
                                           EigenPtr<Matrix3X<T>> force_differential) const;

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