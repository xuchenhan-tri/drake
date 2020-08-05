#pragma once

#include <memory>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/constitutive_model.h"

namespace drake {
namespace fem {
/** Implements the constitutive model described in [Müller, 2004]. */
template <typename T>
class CorotatedLinearElasticity : public ConstitutiveModel<T> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(CorotatedLinearElasticity)

  CorotatedLinearElasticity(T E, T nu) : E_(E), nu_(nu) {
    R_.setIdentity();
    F_.setIdentity();
    strain_.setIdentity();
    trace_strain_ = 3.0;
  }
  virtual void UpdateState(const Matrix3<T>& F,
                           const Eigen::Matrix<T, 3, 4>& q) override;

  virtual Matrix3<T> CalcP() const override;

  virtual Matrix3<T> CalcdP(const Matrix3<T> dF) const override;

  virtual Eigen::Matrix<T, 9, 9> CalcdPdF(const Matrix3<T>& F) const override;

 private:
  Matrix3<T> R_;             // Corotation matrix.
  Matrix3<T> F_;             // Deformation gradient.
  Matrix3<T> strain_;        // Corotated Linear strain.
  Matrix3<T> trace_strain_;  // trace of strain_.
  T E_;                      // Young's modulus.
  T nu_;                     // Poisson ratio.
};
}  // namespace fem
}  // namespace drake
