#pragma once

#include <memory>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {
template <typename T>
class ConstitutiveModel {
 public:
  /** Updates the states of the constitutive model that depends on F.
     @param[in] F  The deformation gradient of the element that the model lives
     on.
     @param[in] q  The positions of the vertices belonging to the element that
     the model lives on .
  */
  virtual ~ConstitutiveModel() = 0;

  virtual void UpdateState(const Matrix3<T>& F,
                           const Eigen::Matrix<T, 3, 4>& q) = 0;

  /** Calculates the First Piola stress. */
  virtual Matrix3<T> CalcP() const = 0;

  /** Calculates the First Piola stress Differential dP(dF). */
  virtual Matrix3<T> CalcdP(const Matrix3<T> dF) const = 0;

  /** Calculates the First Piola stress derivative dP(dF). */
  virtual Eigen::Matrix<T, 9, 9> CalcdPdF() const = 0;
};

}  // namespace fem
}  // namespace drake
