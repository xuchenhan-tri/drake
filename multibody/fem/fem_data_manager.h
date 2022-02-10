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

/* Contains system-side information to construct an FemDataManager. Can be
 stored in a system. Typically, one would allocate for system resources via
 FemModel::AllocateFemData() and hold onto the returned FemDataInfo to gain
 access to the allocated data in a specific context through FemDataManager (see
 below). */
template <typename T>
struct FemDataInfo {
  const systems::LeafSystem<T>* system;
  const FemModel<T>* model;
  const int num_dofs;
  const systems::CacheIndex scratch_state_index;
  const systems::CacheIndex element_data_index;
};

// TODO(xuchenhan-tri): Since the allocation is happening out side of this class
// now, this class should not be called a "manager" anymore. Perhaps just
// "FemData"?
/* FemDataManager provides access to scratch FEM state and per-element
 state-dependent cache entry values stored in the context provided at
 construction. It's lifespan should be short and restricted within a
 CalcFoo(const Context<T>&, Foo*) function. In particular, it shouldn't be
 stored in context or system itself. */
template <typename T>
class FemDataManager {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemDataManager);

  /* Creates an FemDataManager that provides access to the scratch state and
   cache entry stored in the given `context`. */
  FemDataManager(const FemDataInfo<T>& data_info,
                 const systems::Context<T>* context)
      : info_(data_info), context_(context) {
    /* Initialize the scratch state cache entry as up-to-date. */
    const auto& cache_entry =
        info_.system->get_cache_entry(info_.scratch_state_index);
    systems::CacheEntryValue& value =
        cache_entry.get_mutable_cache_entry_value(*context_);
    value.mark_up_to_date();
    // TODO(xuchenhan-tri): Verify input.
    // TODO(xuchenhan-tri): Don't construct two copies of FemDataManager with
    // same data_info and context.
  }

  /* Returns the FEM state managed by `this` manager. */
  const FemState<T>& GetFemState() const {
    const auto& cache_entry =
        info_.system->get_cache_entry(info_.scratch_state_index);
    return cache_entry.template GetKnownUpToDate<FemState<T>>(*context_);
  }

  /* (Advanced) Returns a mutable version of FEM state managed by `this`
   manager. Calling this method invalidates downstream cache entries once and
   only once. For example,
     // Get the mutable state.
     FemState<T>& state = fem_data_manager.GetMutableFemState();
     // Modify the state.
     state.SetPositions(q1);
     // The downstream cache is correctly evaluated.
     const ElementData<T>& correct_data = EvalElementData();
     // Modify the state again (dangerous!)
     state.SetPositions(q2);
     // The downstream cache is not correctly evaluated.
     const ElementData<T>& wrong_data = EvalElementData();
  */
  FemState<T>& GetMutableFemState() const {
    DRAKE_DEMAND(context_ != nullptr);
    const auto& cache_entry =
        info_.system->get_cache_entry(info_.scratch_state_index);
    systems::CacheEntryValue& value =
        cache_entry.get_mutable_cache_entry_value(*context_);
    value.mark_out_of_date();
    auto& fem_state = value.GetMutableValueOrThrow<FemState<T>>();

    /* Invalidate downsteam cache entries. */
    auto* mutable_context = const_cast<systems::Context<T>*>(context_);
    context_->get_tracker(cache_entry.ticket())
        .NoteValueChange(mutable_context->start_new_change_event());

    /* Any new value is treated as the up-to-date value. */
    value.mark_up_to_date();
    return fem_state;
  }

  /* Returns the FEM element data managed by `this` manager. */
  const ElementData<T>& EvalElementData() const {
    DRAKE_DEMAND(context_ != nullptr);
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

  const FemModel<T>* model() const { return info_.model; }

  int num_dofs() const { return info_.num_dofs; }

 private:
  FemDataInfo<T> info_;
  const systems::Context<T>* context_{nullptr};
};

}  // namespace fem
}  // namespace multibody
}  // namespace drake
