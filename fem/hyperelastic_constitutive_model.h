#pragma once

#include "drake/common/eigen_types.h"
#include "drake/common/nice_type_name.h"
#include "drake/fem/fem_state.h"
#include "drake/fem/hyperelastic_cache.h"

namespace drake {
namespace fem {

/** A constitutive model relates the strain to the stress of the material. It
 governs the material response under deformation. For hyperelastic materials,
 the constitutive relation is defined through the potential energy, which
 increases with non-rigid deformation from the initial state. For viscous or
 plastic behavior, we delegate to another class (TODO (xuchenhan-tri): update
 the comment to be more specific once plasticity is in place.) to modify the
 deformation gradient instead of directly modifying the stress-strain
 relationship.
*/
template <typename T>
class HyperelasticConstitutiveModel {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(HyperelasticConstitutiveModel)

  HyperelasticConstitutiveModel(T E, T nu, T alpha, T beta)
      : E_(E), nu_(nu), alpha_(alpha), beta_(beta) {
    VerifyParameterValidity(E, nu, alpha, beta);
    SetLameParameters(E, nu);
  }

  virtual ~HyperelasticConstitutiveModel() {}

  /** Update the states that depend on the deformation gradient F.
    @param[in] F  The deformation gradient evaluated at where the constitutive
    model lives.
  */
  void UpdateHyperelasticCache(const FemState<T>& fem_state, int quadrature_id, std::vector<std::unique_ptr<HyperelasticCache<T>>>* cache) const {
      DRAKE_DEMAND(quadrature_id < static_cast<int>(cache->size()));
      DRAKE_DEMAND(quadrature_id >= 0);
    DoUpdateHyperelasticCache(fem_state, quadrature_id, cache);
  }

  T get_E() const { return E_; }
  T get_nu() const { return nu_; }
  T get_alpha() const { return alpha_; }
  T get_beta() const { return beta_; }
  T get_mu() const { return mu_; }
  T get_lambda() const { return lambda_; }

  /** Calculates the energy density. */
  T CalcPsi(const HyperelasticCache<T>& cache) const { return DoCalcPsi(cache); }

  /** Calculates the First Piola stress under current states. */
  Matrix3<T> CalcFirstPiola(const HyperelasticCache<T>& cache) const { return DoCalcFirstPiola(cache); }

  /** Calculates the First Piola stress Differential dP(dF) = dP/dF * dF under
   * current states. */
  Matrix3<T> CalcFirstPiolaDifferential(
      const HyperelasticCache<T>& cache, const Eigen::Ref<const Matrix3<T>>& dF) const {
    return DoCalcFirstPiolaDifferential(cache, dF);
  }

  /** Calculates the First Piola stress derivative dP/dF under current states.
     The resulting 4th order tensor in index notation is Aᵢⱼₐᵦ = ∂Pᵢⱼ/∂Fₐᵦ. We
     flatten it out into a matrix, and the indices should be laid out as
     following:
                     β = 0       β = 1       β = 2
                 -------------------------------------
                 |           |           |           |
       j = 0     |   Aᵢ₀ₐ₀   |   Aᵢ₀ₐ₁   |   Aᵢ₀ₐ₂   |
                 |           |           |           |
                 -------------------------------------
                 |           |           |           |
       j = 1     |   Aᵢ₁ₐ₀   |   Aᵢ₁ₐ₁   |   Aᵢ₁ₐ₂   |
                 |           |           |           |
                 -------------------------------------
                 |           |           |           |
       j = 2     |   Aᵢ₂ₐ₀   |   Aᵢ₂ₐ₁   |   Aᵢ₂ₐ₂   |
                 |           |           |           |
                 -------------------------------------

                 where each jβ-th block assumes the standard matrix layout:

                     a = 0     a = 1      a = 2
                 ----------------------------------
                 |                                |
       i = 0     |   A₀ⱼ₀ᵦ      A₀ⱼ₁ᵦ      A₀ⱼ₂ᵦ   |
                 |                                |
                 |                                |
       i = 1     |   A₁ⱼ₀ᵦ      A₁ⱼ₁ᵦ      A₁ⱼ₂ᵦ   |
                 |                                |
                 |                                |
       i = 2     |   A₂ⱼ₀ᵦ      A₂ⱼ₁ᵦ      A₂ⱼ₂ᵦ   |
                 |                                |
                 ----------------------------------
   */
  Eigen::Matrix<T, 9, 9> CalcFirstPiolaDerivative(const HyperelasticCache<T>& cache) const {
    return DoCalcFirstPiolaDerivative(cache);
  }

  /** Create the HyperelasticCache that is used to calculate stress and its derivatives. */
  std::unique_ptr<HyperelasticCache<T>> CreateCache() const {
      return DoCreateCache();
  }
 protected:
  /* Update the states that depend on the deformation gradient F.
    @param[in] F  The deformation gradient evaluated at where the constitutive
    model lives.
  */
  virtual void DoUpdateHyperelasticCache(
      const FemState<T>& fem_state, int quadrature_id, std::vector<std::unique_ptr<HyperelasticCache<T>>>* cache) const = 0;

  /* Updates the states that depend on the positions the control vertices of
    the element that the constitutive model lives on.
    @param[in] q  The positions of the control vertices of the elements where
    the constitutive model lives.
  */

  /* Calculates the energy density. */
  virtual T DoCalcPsi(const HyperelasticCache<T>& cache) const = 0;

  /* Calculates the First Piola stress under current states. */
  virtual Matrix3<T> DoCalcFirstPiola(const HyperelasticCache<T>& cache) const = 0;

  /* Calculates the First Piola stress Differential dP(dF) = dP/dF * dF under
    current states. */
  virtual Matrix3<T> DoCalcFirstPiolaDifferential(
          const HyperelasticCache<T>& cache, const Eigen::Ref<const Matrix3<T>>& dF) const = 0;

  /* Calculates the First Piola stress derivative dP/dF under current states.
   */
  virtual Eigen::Matrix<T, 9, 9> DoCalcFirstPiolaDerivative(const HyperelasticCache<T>& cache) const = 0;

  /* Create the HyperelasticCache that is used to calculate stress and its derivatives. */
  virtual std::unique_ptr<HyperelasticCache<T>> DoCreateCache() const = 0;

 private:
  /* Set the Lamé parameters from Young's modulus and Poisson ratio. It's
     important to keep the Lamé Parameters in sync with Young's modulus and
     Poisson ratio as most computations use Lame parameters. */
  void VerifyParameterValidity(T E, T nu, T alpha, T beta) {
    if (E < 0.0) {
      throw std::logic_error("Young's modulus must be nonnegative.");
    }
    if (nu >= 0.5 || nu <= -1) {
      throw std::logic_error("Poisson ratio must be in (-1, 0.5).");
    }
    if (alpha < 0.0) {
      throw std::logic_error("Mass damping parameter must be nonnegative.");
    }
    if (beta < 0.0) {
      throw std::logic_error(
          "Stiffness damping parameter must be nonnegative.");
    }
  }

  void SetLameParameters(T E, T nu) {
    mu_ = E / (2.0 * (1.0 + nu));
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  }

  // Young's modulus.
  T E_{0};
  // Poisson ratio.
  T nu_{0};
  // Mass damping factor.
  T alpha_{0};
  // Stiffness damping factor.
  T beta_{0};
  // Lamé's second parameter/Shear modulus.
  T mu_{0};
  // Lamé's first parameter.
  T lambda_{0};
};

}  // namespace fem
}  // namespace drake
