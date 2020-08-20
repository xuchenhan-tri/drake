#pragma once

#include <memory>

#include "drake/common/eigen_types.h"
#include "drake/fem/constitutive_model.h"

namespace drake {
namespace fem {
/** Implements the constitutive model described in [Müller, 2004]. */
template <typename T>
class CorotatedLinearElasticity : public ConstitutiveModel<T> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(CorotatedLinearElasticity)

  CorotatedLinearElasticity(T E, T nu, T alpha, T beta,
                            const Eigen::Matrix<T, 3, 4>& vertex_positions)
      : ConstitutiveModel<T>(alpha, beta),
        E_(E),
        nu_(nu),
        R_(Matrix3<T>::Identity()),
        F_(Matrix3<T>::Identity()),
        strain_(Matrix3<T>::Identity()),
        trace_strain_(0.0) {
    SetLameParameters(E_, nu_);
    Matrix4<T> P;
    P.template topLeftCorner<3, 4>() = vertex_positions;
    P.template bottomRows<1>() = Vector4<T>::Ones();
    Eigen::HouseholderQR<Matrix4<T>> qr(P);
    inv_P_ = qr.solve(Matrix4<T>::Identity());
  }

  virtual ~CorotatedLinearElasticity() {}

  void UpdateDeformationBasedState(const Eigen::Ref<const Matrix3<T>>& F) override;

  void UpdateTimeNPositionBasedState(const Eigen::Ref<const Eigen::Matrix<T, 3, 4>>& q) override;

  T CalcEnergyDensity() const override;

  Matrix3<T> CalcFirstPiola() const override;

  Matrix3<T> CalcFirstPiolaDifferential(
      const Eigen::Ref<const Matrix3<T>>& dF) const override;

  Eigen::Matrix<T, 9, 9> CalcFirstPiolaDerivative() const override;

  void SetLameParameters(T youngs_modulus, T poisson_ratio) {
    mu_ = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
    lambda_ = youngs_modulus * poisson_ratio /
              ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
  }

 private:
  T E_;                // Young's modulus.
  T nu_;               // Poisson ratio.
  T mu_;               // Shear modulus.
  T lambda_;           // Lame's first parameter.
  Matrix3<T> R_;       // Corotation matrix.
  Matrix3<T> F_;       // Deformation gradient.
  Matrix3<T> strain_;  // Corotated Linear strain.
  T trace_strain_;     // trace of strain_.
  Matrix4<T>
      inv_P_;  // Initial Homogeneous position matrix. See [Müller, 2004].
};
}  // namespace fem
}  // namespace drake
