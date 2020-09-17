#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/fem_data.h"
#include "drake/fem/fem_state.h"

namespace drake {
namespace fem {
template <typename T>
class FemCacheEvaluator {
public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FemCacheEvaluator);
  FemCacheEvaluator(const FemData<T>& data) : data_(data) {}

  const std::vector<Matrix3<T>>& EvalF(const FemState<T>& state) const {
    if (!state.get_cache().F_out_of_date()) {
      return state.get_cache().get_F();
    }
    const auto& elements = data_.get_elements();
    const auto& q = state.get_q();
    auto& F = state.get_mutable_cache().get_mutable_F();
    int quadrature_offset = 0;
    for (const auto& e : elements) {
      F[quadrature_offset++] = e.CalcF(q);
    }
    state.get_mutable_cache().set_F_out_of_date(false);
    return state.get_cache().get_F();
  }

  const std::vector<Matrix3<T>>& EvalF0(const FemState<T>& state) const {
    if (!state.get_cache().F0_out_of_date()) {
      return state.get_cache().get_F0();
    }
    const auto& elements = data_.get_elements();
    const auto& q0 = state.get_q0();
    auto& F0 = state.get_mutable_cache().get_mutable_F0();
    int quadrature_offset = 0;
    for (const auto& e : elements) {
      F0[quadrature_offset++] = e.CalcF(q0);
    }
    state.get_mutable_cache().set_F0_out_of_date(false);
    return state.get_cache().get_F0();
  }

  const std::vector<std::unique_ptr<HyperelasticCache<T>>>&
  EvalHyperelasticCache(const FemState<T>& state) const {
    if (!state.get_cache().hyperelastic_cache_out_of_date()) {
      return state.get_cache().get_hyperelastic_cache();
    }
    if (state.get_cache().F_out_of_date()) {
      EvalF(state);
    }
    if (state.get_cache().F0_out_of_date()) {
      EvalF0(state);
    }
    DRAKE_DEMAND(!state.get_cache().F_out_of_date());
    DRAKE_DEMAND(!state.get_cache().F0_out_of_date());
    const auto& elements = data_.get_elements();
    auto& model_cache = state.get_mutable_cache().get_mutable_hyperelastic_cache();
    DRAKE_DEMAND(state.get_F().size() == model_cache.size());
    int quadrature_offset = 0;
    for (const auto& e : elements) {
      const auto* model = e.get_constitutive_model();
      model->UpdateHyperelasticCache(state, quadrature_offset++, &model_cache);
    }
    state.get_mutable_cache().set_hyperelastic_cache_out_of_date(false);
    return state.get_cache().get_hyperelastic_cache();
  }

  // Evaluates the elastic energy density cache.
  const std::vector<T>& EvalPsi(const FemState<T>& state) const {
    if (!state.get_cache().psi_out_of_date()) {
      return state.get_cache().get_psi();
    }
    if (state.get_cache().hyperelastic_cache_out_of_date()) {
      EvalHyperelasticCache(state);
    }
    DRAKE_DEMAND(!state.get_cache().hyperelastic_cache_out_of_date());
    const auto& model_cache = state.get_cache().get_hyperelastic_cache();
    const auto& elements = data_.get_elements();
    auto& psi = state.get_mutable_cache().get_mutable_psi();
    int quadrature_offset = 0;
    for (const FemElement<T>& e : elements) {
      psi[quadrature_offset] =
          e.get_constitutive_model()->CalcPsi(*model_cache[quadrature_offset]);
      ++quadrature_offset;
    }
    state.get_mutable_cache().set_psi_out_of_date(false);
    return state.get_cache().get_psi();
  }

  // Evaluates the first Piola stress cache.
  const std::vector<Matrix3<T>>& EvalP(const FemState<T>& state) const {
    if (!state.get_cache().P_out_of_date()) {
      return state.get_cache().get_P();
    }
    if (state.get_cache().hyperelastic_cache_out_of_date()) {
      EvalHyperelasticCache(state);
    }
    DRAKE_DEMAND(!state.get_cache().hyperelastic_cache_out_of_date());
    const auto& model_cache = state.get_cache().get_hyperelastic_cache();
    const auto& elements = data_.get_elements();
    auto& P = state.get_mutable_cache().get_mutable_P();
    int quadrature_offset = 0;
    for (const FemElement<T>& e : elements) {
      P[quadrature_offset] = e.get_constitutive_model()->CalcFirstPiola(
          *model_cache[quadrature_offset]);
      ++quadrature_offset;
    }
    state.get_mutable_cache().set_P_out_of_date(false);
    return state.get_cache().get_P();
  }

  // Evaluates the first Piola stress derivative cache.
  const std::vector<Eigen::Matrix<T, 9, 9>>& EvaldPdF(
      const FemState<T>& state) const {
    if (!state.get_cache().dPdF_out_of_date()) {
      return state.get_cache().get_dPdF();
    }
    if (state.get_cache().hyperelastic_cache_out_of_date()) {
      EvalHyperelasticCache(state);
    }
    DRAKE_DEMAND(!state.get_cache().hyperelastic_cache_out_of_date());
    const auto& model_cache = state.get_cache().get_hyperelastic_cache();
    const auto& elements = data_.get_elements();
    auto& dPdF = state.get_mutable_cache().get_mutable_dPdF();
    int quadrature_offset = 0;
    for (const FemElement<T>& e : elements) {
      dPdF[quadrature_offset] =
          e.get_constitutive_model()->CalcFirstPiolaDerivative(
              *model_cache[quadrature_offset]);
      ++quadrature_offset;
    }
    state.get_mutable_cache().set_dPdF_out_of_date(false);
    return state.get_cache().get_dPdF();
  }

 private:
  const FemData<T>& data_;
};
}  // namespace fem
}  // namespace drake
