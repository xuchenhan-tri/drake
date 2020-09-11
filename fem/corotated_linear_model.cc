#include "drake/fem/corotated_linear_model.h"

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {
template <typename T>
void CorotatedLinearElasticity<T>::DoUpdateHyperelasticCache(
    const FemState<T>& fem_state, const int quadrature_id, std::vector<std::unique_ptr<HyperelasticCache<T>>>* cache) const {
  CorotatedLinearCache<T>* corotated_linear_cache =
      static_cast<CorotatedLinearCache<T>*>((*cache)[quadrature_id].get());
  const auto& F = fem_state.get_F()[quadrature_id];
  const auto& F0 = fem_state.get_F0()[quadrature_id];
  corotated_linear_cache->F = F;
  Eigen::JacobiSVD<Matrix3<T>> svd(F0,
                                   Eigen::ComputeFullU | Eigen::ComputeFullV);
  corotated_linear_cache->R = svd.matrixU() * svd.matrixV().transpose();
  Matrix3<T> RtF = corotated_linear_cache->R.transpose() * F;
  corotated_linear_cache->strain =
      0.5 * (RtF + RtF.transpose()) - Matrix3<T>::Identity();
  corotated_linear_cache->trace_strain = corotated_linear_cache->strain.trace();
}

/* Energy density ϕ = μεᵢⱼεᵢⱼ + 0.5λtr(ε)², where ε is the corotated strain
 measure 0.5*(RᵀF + FᵀR) - I. */
template <typename T>
T CorotatedLinearElasticity<T>::DoCalcPsi(
    const HyperelasticCache<T>& cache) const {
  const CorotatedLinearCache<T>& corotated_linear_cache =
      static_cast<const CorotatedLinearCache<T>&>(cache);
  const auto& strain = corotated_linear_cache.strain;
  const auto& trace_strain = corotated_linear_cache.trace_strain;
  return this->get_mu() * strain.squaredNorm() +
         0.5 * this->get_lambda() * trace_strain * trace_strain;
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
Matrix3<T> CorotatedLinearElasticity<T>::DoCalcFirstPiola(
    const HyperelasticCache<T>& cache) const {
  const CorotatedLinearCache<T>& corotated_linear_cache =
      static_cast<const CorotatedLinearCache<T>&>(cache);
  const auto& R = corotated_linear_cache.R;
  const auto& strain = corotated_linear_cache.strain;
  const auto& trace_strain = corotated_linear_cache.trace_strain;
  Matrix3<T> P =
      2.0 * this->get_mu() * R * strain + this->get_lambda() * trace_strain * R;
  return P;
}

template <typename T>
Matrix3<T> CorotatedLinearElasticity<T>::DoCalcFirstPiolaDifferential(
    const HyperelasticCache<T>& cache,
    const Eigen::Ref<const Matrix3<T>>& dF) const {
  const CorotatedLinearCache<T>& corotated_linear_cache =
      static_cast<const CorotatedLinearCache<T>&>(cache);
  const auto& R = corotated_linear_cache.R;
  const T mu = this->get_mu();
  const T lambda = this->get_lambda();
  Matrix3<T> dP = mu * dF + mu * R * dF.transpose() * R +
                  lambda * (R.array() * dF.array()).sum() * R;
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
Eigen::Matrix<T, 9, 9> CorotatedLinearElasticity<T>::DoCalcFirstPiolaDerivative(
    const HyperelasticCache<T>& cache) const {
  const CorotatedLinearCache<T>& corotated_linear_cache =
      static_cast<const CorotatedLinearCache<T>&>(cache);
  const T mu = this->get_mu();
  const T lambda = this->get_lambda();
  const auto& R = corotated_linear_cache.R;
  // Add in μ * δₐᵢδⱼᵦ.
  Eigen::Matrix<T, 9, 9> dPdF = mu * Eigen::Matrix<T, 9, 9>::Identity();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int alpha = 0; alpha < 3; ++alpha) {
        for (int beta = 0; beta < 3; ++beta) {
          // Add in  μ *  Rᵢᵦ Rₐⱼ +   λ * Rₐᵦ * Rᵢⱼ
          dPdF(3 * j + i, 3 * beta + alpha) +=
              mu * R(i, beta) * R(alpha, j) + lambda * R(alpha, beta) * R(i, j);
        }
      }
    }
  }
  return dPdF;
}

template <typename T>
std::unique_ptr<HyperelasticCache<T>>
CorotatedLinearElasticity<T>::DoCreateCache() const {
  return std::make_unique<CorotatedLinearCache<T>>();
}

}  // namespace fem
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::CorotatedLinearElasticity)
