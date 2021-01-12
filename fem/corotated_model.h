#pragma once

#include <memory>
#include <iostream>

#include "drake/common/eigen_types.h"
#include "drake/common/unused.h"
#include "drake/fem/hyperelastic_constitutive_model.h"
#include "drake/fem/ImplicitQRSVD.h"

namespace drake {
namespace fem {
template <typename T>
class CorotatedElasticity final : public HyperelasticConstitutiveModel<T> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(CorotatedElasticity)

  CorotatedElasticity(T E, T nu, T alpha, T beta)
      : HyperelasticConstitutiveModel<T>(E, nu, alpha, beta),
        R_(Matrix3<T>::Identity()),
        F_(Matrix3<T>::Identity()),
        S_(Matrix3<T>::Identity()),
        JFinvT_(Matrix3<T>::Identity()),
        J_(0.0) {}

  virtual ~CorotatedElasticity() {}

 protected:
  void DoUpdateDeformationBasedState(
      const Eigen::Ref<const Matrix3<T>>& F) override {
    using namespace Eigen;
    F_ = F;
    J_ = F_.determinant();
    ComputeCofactor(F_, &JFinvT_);
    JIXIE::polarDecomposition(F_, R_, S_);
  }

  void DoUpdatePositionBasedState(
      const Eigen::Ref<const Eigen::Matrix<T, 3, 4>>& q) override {
    unused(q);
  }

  T DoCalcEnergyDensity() const override {
    T Jm1 = J_ - 1;
    return this->get_mu() * (F_ - R_).squaredNorm() +
           T(0.5) * this->get_lambda() * Jm1 * Jm1;
  }

  Matrix3<T> DoCalcFirstPiola() const override {
    Matrix3<T> P;
    P = T(2.0) * this->get_mu() * (F_ - R_) +
        this->get_lambda() * (J_ - T(1.0)) * JFinvT_;
    return P;
  }

  Matrix3<T> DoCalcFirstPiolaDifferential(
      const Eigen::Ref<const Matrix3<T>>& dF) const override {
    Matrix3<T> dP;
    dP = this->get_lambda() * JFinvT_.cwiseProduct(dF).sum() * JFinvT_;
    dP += 2 * this->get_mu() * dF;
    AddScaledRotationDifferential(R_, S_, dF, -2.0 * this->get_mu(), &dP);
    AddScaledCofactorMatrixDifferential(F_, dF,
                                        this->get_lambda() * (J_ - T(1)), &dP);
    return dP;
  }

  Eigen::Matrix<T, 9, 9> DoCalcFirstPiolaDerivative() const override{
    DRAKE_DEMAND(false);
    return Eigen::Matrix<T,9,9>::Zero();
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

  Matrix3<T> R_;       // Corotation matrix.
  Matrix3<T> F_;       // Deformation gradient.
  Matrix3<T> S_;       // Corotation matrix.
  Matrix3<T> JFinvT_;  // Corotation matrix.
  T J_;
};
}  // namespace fem
}  // namespace drake
