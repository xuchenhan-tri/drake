#pragma once

#include "drake/multibody/fem/fem_model.h"
#include "drake/multibody/fem/fem_state.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace multibody {
namespace fem {

/* FemDataManager stores FEM states and manages per-element cache entry values
 that depend on the state. The FEM states can be set with SetFemState() and
 retrieved through EvalFemState(). The per-element data can be retrieved via
 EvalElementData(). */
template <typename T>
class FemDataManager {
 public:
  /* Creates an FemDataManager that manages the data consumed by the given
   `fem_model`. */
  FemDataManager(const FemModel<T>* fem_model) : fem_model_(fem_model) {
    DRAKE_DEMAND(fem_model_ != nullptr);
    AllocateFemData();
  }

  /* Updates the FEM state. */
  void SetFemState(const FemState<T>& fem_state) {
    const int num_dofs = fem_state.num_dofs();
    DRAKE_DEMAND(fem_model_ != nullptr);
    DRAKE_DEMAND(num_dofs == fem_model_->num_dofs());
    context_->SetAbstractState<FemState<T>>(fem_state_index_, fem_state);
  }

  const FemState<T>& EvalFemState() const {
    return context_->get_abstract_state<FemState<T>>(fem_state_index_);
  }

  const ElementData<T>& EvalElementData() const {
    return system_.get_cache_entry(element_data_index)
        .template Eval<FemData<T>>(*context_);
  }

 private:
  /* Declares states and cache entries for FEM data in the owning system. */
  void AllocateFemData() {
    fem_state_index_ = system_.DeclareAbstractState(
        Value<FemState<T>>(fem_model_->MakeFemState()));
    /* FEM element data. */
    std::unique_ptr<ElementData<T>> model_element_data =
        fem_model_->MakeElementData();
    const auto& element_data_cache_entry = system->DeclareCacheEntry(
        "FEM state dependent element data",
        systems::ValueProducer(*model_element_data,
                               &FemDataManager<T>::CalcElementData),
        {system_.abstract_state_ticket(fem_state_index_)});
    element_data_index_ = element_data_cache_entry.index();
    context_ = std::move(system_.CreateDefaultContext());
  }

  void CalcElementData(const systems::Context<T>& context,
                       ElementData<T>* element_data) const {
    DRAKE_DEMAND(element_data != nullptr);
    DRAKE_DEMAND(fem_model_ != nullptr);
    fem_model_->CalcElementData(EvalFemState(), element_data);
  }

  // TODO(xuchenhan-tri): These quantities need not be per FemDataManager.
  //  It's enough to have one system per FemModel.
  systems::LeafSystem<T> system_;
  systems::AbstractStateIndex fem_state_index_;
  systems::CacheIndex element_data_index_;

  std::unique_ptr<systems::Context<T>> context_{nullptr};
};

}  // namespace fem
}  // namespace multibody
}  // namespace drake
