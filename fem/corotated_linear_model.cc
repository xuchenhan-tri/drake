#include "drake/fem/corotated_linear_model.h"

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {
template <typename T>
void CorotatedLinearElasticity<T>::DoUpdateDeformationBasedState(
    const Eigen::Ref<const Matrix3<T>>& F) {
  F_ = F;
  Matrix3<T> RtF = R_.transpose() * F_;
  strain_ = 0.5 * (RtF + RtF.transpose()) - Matrix3<T>::Identity();
  trace_strain_ = strain_.trace();
}

template <typename T>
void CorotatedLinearElasticity<T>::DoUpdatePositionBasedState(
    const Eigen::Ref<const Eigen::Matrix<T, 3, 4>>& q) {
  Matrix4<T> Q;
  Q.template topLeftCorner<3, 4>() = q;
  Q.template bottomRows<1>() = Vector4<T>::Ones();
  R_ = (Q * inv_P_).template topLeftCorner<3, 3>();
  Eigen::JacobiSVD<Matrix3<T>> svd(R_,
                                   Eigen::ComputeFullU | Eigen::ComputeFullV);
  R_ = svd.matrixU() * svd.matrixV().transpose();
}

/* Energy density ϕ = μεᵢⱼεᵢⱼ + 0.5λtr(ε)², where ε is the corotated strain
 measure 0.5*(RᵀF + FᵀR) - I.
        */
template <typename T>
T CorotatedLinearElasticity<T>::DoCalcEnergyDensity() const {
  return this->get_mu() * strain_.squaredNorm() +
         0.5 * this->get_lambda() * trace_strain_ * trace_strain_;
}

/* First Piola stress P is the derivative of energy density ϕ with respect
 to deformation gradient F. In Einstein notation, we have
     εᵢⱼ = 0.5 RₖᵢFₖⱼ + 0.5 ₖᵢFₖᵢRₖⱼ - δᵢⱼ.
 Differentiating w.r.t. F gives:
     Pₐᵦ = 2μ * εᵢⱼ * ∂εᵢⱼ/∂Fₐᵦ + λ * εⱼⱼ ∂εᵢᵢ/∂Fₐᵦ.
 Differentiate ε w.r.t. F gives:
     ∂εᵢⱼ/∂Fₐᵦ = 0.5 Rₐᵢ δⱼᵦ  + 0.5 δᵢᵦ Rₐⱼ,
 plug it into the expression for P results in:
     2μ * Rₐᵢ*εᵢᵦ + λ * εⱼⱼ * Rₐᵦ,
 which simplifies to
     2μRε + λtr(ε)R.
*/
template <typename T>
Matrix3<T> CorotatedLinearElasticity<T>::DoCalcFirstPiola() const {
  Matrix3<T> P = 2.0 * this->get_mu() * R_ * strain_ +
                 this->get_lambda() * trace_strain_ * R_;
  return P;
}

template <typename T>
Matrix3<T> CorotatedLinearElasticity<T>::DoCalcFirstPiolaDifferential(
    const Eigen::Ref<const Matrix3<T>>& dF) const {
  const T mu = this->get_mu();
  const T lambda = this->get_lambda();
  Matrix3<T> dP = mu * dF + mu * R_ * dF.transpose() * R_ +
                  lambda * (R_.array() * dF.array()).sum() * R_;
  return dP;
}

/*
 Calculates ∂Pᵢⱼ/∂Fₐᵦ, where
     Pᵢⱼ = 2μ * Rᵢₖ *εₖⱼ + λ * εₖₖ * Rᵢⱼ,
 So,
     ∂Pᵢⱼ/∂Fₐᵦ = 2μ * Rᵢₖ * ∂εₖⱼ/∂Fₐᵦ + λ * ∂εₖₖ/∂Fₐᵦ * Rᵢⱼ,
 We use the calculation result from before that
     ∂εᵢⱼ/∂Fₐᵦ = 0.5 Rₐᵢ δⱼᵦ  + 0.5 δᵢᵦ Rₐⱼ.
 Plugging in, we get
     ∂Pᵢⱼ/∂Fₐᵦ = 2μ * Rᵢₖ * (0.5 Rₐₖ δⱼᵦ  + 0.5 δₖᵦ Rₐⱼ)
                 + λ * (0.5 Rₐₖ δₖᵦ  + 0.5 δₖᵦ Rₐₖ) * Rᵢⱼ,
 which simplifies to:
     ∂Pᵢⱼ/∂Fₐᵦ = μ * (δₐᵢδⱼᵦ + Rᵢᵦ Rₐⱼ) +  λ * Rₐᵦ * Rᵢⱼ.
 Keep in mind that the indices are laid out as following:
                  β = 1       β = 2       β = 3
              -------------------------------------
              |           |           |           |
    j = 1     |   Aᵢ₁ₐ₁   |   Aᵢ₁ₐ₂   |   Aᵢ₁ₐ₃   |
              |           |           |           |
              -------------------------------------
              |           |           |           |
    j = 2     |   Aᵢ₂ₐ₁   |   Aᵢ₂ₐ₂   |   Aᵢ₂ₐ₃   |
              |           |           |           |
              -------------------------------------
              |           |           |           |
    j = 3     |   Aᵢ₃ₐ₁   |   Aᵢ₃ₐ₂   |   Aᵢ₃ₐ₃   |
              |           |           |           |
              -------------------------------------
*/

template <typename T>
Eigen::Matrix<T, 9, 9>
CorotatedLinearElasticity<T>::DoCalcFirstPiolaDerivative() const {
  const T mu = this->get_mu();
  const T lambda = this->get_lambda();
  // Add in μ * δₐᵢδⱼᵦ.
  Eigen::Matrix<T, 9, 9> dPdF = mu * Eigen::Matrix<T, 9, 9>::Identity();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int alpha = 0; alpha < 3; ++alpha) {
        for (int beta = 0; beta < 3; ++beta) {
          // Add in  μ *  Rᵢᵦ Rₐⱼ +   λ * Rₐᵦ * Rᵢⱼ
          dPdF(3 * j + i, 3 * beta + alpha) +=
              mu * R_(i, beta) * R_(alpha, j) +
              lambda * R_(alpha, beta) * R_(i, j);
        }
      }
    }
  }
  return dPdF;
}

template class CorotatedLinearElasticity<double>;
}  // namespace fem
}  // namespace drake
