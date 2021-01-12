#pragma once

#include <iostream>
#include <memory>

#include "drake/common/eigen_types.h"
#include "drake/common/unused.h"
#include "drake/fem/hyperelastic_constitutive_model.h"

namespace drake {
namespace fem {
template <typename T>
class NeoHookean final : public HyperelasticConstitutiveModel<T> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(NeoHookean)

  NeoHookean(T E, T nu, T alpha, T beta)
      : HyperelasticConstitutiveModel<T>(E, nu, alpha, beta),
        F_(Matrix3<T>::Identity()),
        FinvT_(Matrix3<T>::Identity()),
        J_(1.0),
        log_J_(1.0) {}

  virtual ~NeoHookean() {}

 protected:
  void DoUpdateDeformationBasedState(
      const Eigen::Ref<const Matrix3<T>>& F) override {
    using namespace Eigen;
    F_ = F;
    J_ = F_.determinant();
    DRAKE_DEMAND(J_ > std::numeric_limits<T>::epsilon());
    log_J_ = std::log(J_);
    Matrix3<T> JFinvT;
    ComputeCofactor(F_, &JFinvT);
    FinvT_ = (1.0 / J_) * JFinvT;
  }

  void DoUpdatePositionBasedState(
      const Eigen::Ref<const Eigen::Matrix<T, 3, 4>>& q) override {
    unused(q);
  }

  T DoCalcEnergyDensity() const override {
    return 0;
  }

  Matrix3<T> DoCalcFirstPiola() const override {
    Matrix3<T> tau;
    T scale = this->get_lambda() * log_J_ - this->get_mu();
    tau = this->get_mu() * F_ * F_.transpose() + scale * Matrix3<T>::Identity();
    return tau * F_.transpose();
  }

  Matrix3<T> DoCalcFirstPiolaDifferential(
      const Eigen::Ref<const Matrix3<T>>& dF) const override {
    Matrix3<T> dP = Matrix3<T>::Zero();
    AddScaledCofactorMatrixDifferential(F_, dF, 1.0, &dP);
    T scale = this->get_lambda() * log_J_ - this->get_mu();
    dP = this->get_mu() * dF +
         (this->get_lambda() - scale) * FinvT_.cwiseProduct(dF).sum() * FinvT_ +
         scale / J_ * dP;
    return dP;
  }

  Eigen::Matrix<T, 9, 9> DoCalcFirstPiolaDerivative() const override {
    DRAKE_DEMAND(false);
    return Eigen::Matrix<T, 9, 9>::Zero();
  }

 private:
  inline void ComputeCofactor(const Matrix3<T>& F, Matrix3<T>* A) const {
    (*A)(0, 0) = F(1, 1) * F(2, 2) - F(1, 2) * F(2, 1);
    (*A)(0, 1) = F(1, 2) * F(2, 0) - F(1, 0) * F(2, 2);
    (*A)(0, 2) = F(1, 0) * F(2, 1) - F(1, 1) * F(2, 0);
    (*A)(1, 0) = F(0, 2) * F(2, 1) - F(0, 1) * F(2, 2);
    (*A)(1, 1) = F(0, 0) * F(2, 2) - F(0, 2) * F(2, 0);
    (*A)(1, 2) = F(0, 1) * F(2, 0) - F(0, 0) * F(2, 1);
    (*A)(2, 0) = F(0, 1) * F(1, 2) - F(0, 2) * F(1, 1);
    (*A)(2, 1) = F(0, 2) * F(1, 0) - F(0, 0) * F(1, 2);
    (*A)(2, 2) = F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0);
  }

  void AddScaledRotationDifferential(const Matrix3<T>& R, const Matrix3<T>& S,
                                     const Matrix3<T>& dF, T scale,
                                     Matrix3<T>* M) const {
    Matrix3<T> S_hat = -S;
    S_hat.diagonal().array() += S.trace();
    T b = S_hat.determinant();
    Matrix3<T> A = R.transpose() * dF;
    (*M) += (scale / b) * R * S_hat * (A - A.transpose()) * S_hat;
  }

  void AddScaledCofactorMatrixDifferential(const Matrix3<T>& F,
                                           const Matrix3<T>& dF, T scale,
                                           Matrix3<T>* M) const {
    (*M)(0, 0) += scale * (dF(1, 1) * F(2, 2) + F(1, 1) * dF(2, 2) -
                           dF(2, 1) * F(1, 2) - F(2, 1) * dF(1, 2));
    (*M)(1, 0) += scale * (dF(2, 1) * F(0, 2) + F(2, 1) * dF(0, 2) -
                           dF(0, 1) * F(2, 2) - F(0, 1) * dF(2, 2));
    (*M)(2, 0) += scale * (dF(0, 1) * F(1, 2) + F(0, 1) * dF(1, 2) -
                           dF(1, 1) * F(0, 2) - F(1, 1) * dF(0, 2));
    (*M)(0, 1) += scale * (dF(2, 0) * F(1, 2) + F(2, 0) * dF(1, 2) -
                           dF(1, 0) * F(2, 2) - F(1, 0) * dF(2, 2));
    (*M)(1, 1) += scale * (dF(0, 0) * F(2, 2) + F(0, 0) * dF(2, 2) -
                           dF(2, 0) * F(0, 2) - F(2, 0) * dF(0, 2));
    (*M)(2, 1) += scale * (dF(1, 0) * F(0, 2) + F(1, 0) * dF(0, 2) -
                           dF(0, 0) * F(1, 2) - F(0, 0) * dF(1, 2));
    (*M)(0, 2) += scale * (dF(1, 0) * F(2, 1) + F(1, 0) * dF(2, 1) -
                           dF(2, 0) * F(1, 1) - F(2, 0) * dF(1, 1));
    (*M)(1, 2) += scale * (dF(2, 0) * F(0, 1) + F(2, 0) * dF(0, 1) -
                           dF(0, 0) * F(2, 1) - F(0, 0) * dF(2, 1));
    (*M)(2, 2) += scale * (dF(0, 0) * F(1, 1) + F(0, 0) * dF(1, 1) -
                           dF(1, 0) * F(0, 1) - F(1, 0) * dF(0, 1));
  }

  Matrix3<T> F_;      // Deformation gradient.
  Matrix3<T> FinvT_;  // Corotation matrix.
  T J_;
  T log_J_;
};
}  // namespace fem
}  // namespace drake
