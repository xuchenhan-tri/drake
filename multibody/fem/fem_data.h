#pragma once

#include "drake/multibody/fem/element_data.h"
#include "drake/multibody/fem/fem_state.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace multibody {
namespace fem {

template <typename T>
class FemModel;

/* Contains system-side information required to construct a FemData.
 stored in a system. Typically, one would declare system resources via
 FemModel::AllocateFemData() and hold onto the returned FemDataInfo to construct
 FemData. */
template <typename T>
struct FemDataInfo {
  const systems::LeafSystem<T>* system;
  const FemModel<T>* model;
  const systems::DiscreteStateIndex fem_state_index;
  const systems::CacheIndex element_data_index;
};

/* FemData provides access to private workspace FEM state and per-element
 state-dependent cache entry values stored in the context provided at
 construction. */
template <typename T>
class FemData {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemData);

  /* Creates an FemData described by the given information. */
  FemData(const FemDataInfo& data_info) : info_(data_info) {
    context_ = info_.system.CreateDefaultContext();
  }

  /* Returns the FEM state in `this` FemData. */
  const FemState<T>& GetFemState() const {
    context_->get_discrete_state(info_.fem_state_index);
  }

  /* Returns the mutable FEM state in `this` FemData. */
  FemState<T>& GetFemState() const {
    context_->get_mutable_discrete_state(info_.fem_state_index);
  }

  /* Returns per element data in `this` FemData. */
  const ElementData<T>& EvalElementData() const {
    return info_.system->get_cache_entry(info_.element_data_index)
        .template Eval<ElementData<T>>(*context_);
  }

  /* Sugar to get/set a part of the FEM state (q, v, or a). */
  const VectorX<T>& GetPositions() const {
    return GetFemState().GetPositions();
  }

  const VectorX<T>& GetVelocities() const {
    return GetFemState().GetVelocities();
  }

  const VectorX<T>& GetAccelerations() const {
    return GetFemState().GetAccelerations();
  }

  void SetPositions(const VectorX<T>& q) {
    GetMutableFemState().SetPositions(q);
  }

  void SetVelocities(const VectorX<T>& v) {
    GetMutableFemState().SetVelocities(v);
  }

  void SetAccelerations(const VectorX<T>& a) {
    GetMutableFemState().SetAccelerations(a);
  }

  int num_dofs() const { return GetFemState().num_dofs(); }

  /* (Internal) Return the model that creates and consumes the state and cache
   entries of `this` FemData. */
  const FemModel<T>* model() const { return info_.model; }

 private:
  FemDataInfo<T> info_;
  std::unique_ptr<systems::Context<T>> context_{nullptr};
};

}  // namespace fem
}  // namespace multibody
}  // namespace drake
