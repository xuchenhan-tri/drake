#pragma once

#include "drake/multibody/fem/element_data_impl.h"
#include "drake/multibody/fem/fem_indexes.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace multibody {
namespace fem {

/* Contains system-side information required to construct a FemData.
 stored in a system. Typically, one would declare system resources via
 FemModel::AllocateFemData() and hold onto the returned FemDataInfo to construct
 FemData. */
template <typename T>
struct FemDataInfo {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FemDataInfo)
  FemDataInfo() = default;

  const systems::LeafSystem<T>* system{nullptr};
  ModelIndex model_index;
  systems::DiscreteStateIndex fem_position_index;
  systems::DiscreteStateIndex fem_velocity_index;
  systems::DiscreteStateIndex fem_acceleration_index;
  systems::CacheIndex element_data_index;
};

/* FemData provides access to private workspace FEM state and per-element
 state-dependent cache entry values stored in the context provided at
 construction. */
template <typename T>
class FemData {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemData);

  /* Creates an FemData described by the given information. */
  FemData(const FemDataInfo<T>& data_info);

  /* Returns per element data in `this` FemData.
  @throws std::exception if the value doesn't actually have type V. */
  template <typename DataType>
  const DataType& EvalElementData(ElementIndex element_index) const {
    const auto& element_data =
        info_.system->get_cache_entry(info_.element_data_index)
            .template Eval<internal::ElementDataImpl<DataType>>(*context_);
    return element_data.get_data(element_index);
  }

  /* Sugar to get/set a part of the FEM state (q, v, or a). */
  const VectorX<T>& GetPositions() const;
  const VectorX<T>& GetVelocities() const;
  const VectorX<T>& GetAccelerations() const;

  void SetPositions(const VectorX<T>& q);
  void SetVelocities(const VectorX<T>& v);
  void SetAccelerations(const VectorX<T>& a);

  int num_dofs() const {
    return context_->get_discrete_state(info_.fem_position_index).size();
  }

  /* Return the index of the FEM model that creates and consumes the
   state and cache entries of `this` FemData. */
  ModelIndex model_index() const { return info_.model_index; }

 private:
  FemDataInfo<T> info_;
  std::unique_ptr<systems::Context<T>> context_{nullptr};
};

}  // namespace fem
}  // namespace multibody
}  // namespace drake
