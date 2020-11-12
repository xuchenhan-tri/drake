#pragma once

#include <memory>

#include "drake/common/eigen_types.h"
#include "drake/common/default_scalars.h"
#include "drake/fem/hyperelastic_constitutive_model.h"

namespace drake {
namespace fem {
/** Implements the constitutive model described in [Müller, 2004].

  [Müller, 2004] Müller, Matthias, and Markus H. Gross. "Interactive Virtual
  Materials." Graphics interface. Vol. 2004. 2004.
 */

template <typename T>
class CorotatedLinearElasticity final
    : public HyperelasticConstitutiveModel<T> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(CorotatedLinearElasticity)

  CorotatedLinearElasticity(T E, T nu, T alpha, T beta)
      : HyperelasticConstitutiveModel<T>(E, nu, alpha, beta)
  {
  }

  virtual ~CorotatedLinearElasticity() {}

 protected:
  void DoUpdateHyperelasticCache(
      const FemState<T>& fem_state, int quadrature_id, std::vector<std::unique_ptr<HyperelasticCache<T>>>* cache) const override;

  T DoCalcPsi(const HyperelasticCache<T>& cache) const override;

  Matrix3<T> DoCalcFirstPiola(const HyperelasticCache<T>& cache) const override;

  Matrix3<T> DoCalcFirstPiolaDifferential(const HyperelasticCache<T>& cache,
      const Eigen::Ref<const Matrix3<T>>& dF) const override;

  Eigen::Matrix<T, 9, 9> DoCalcFirstPiolaDerivative(const HyperelasticCache<T>& cache) const override;

  std::unique_ptr<HyperelasticCache<T>> DoCreateCache() const override;
};
}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::drake::fem::CorotatedLinearElasticity)
