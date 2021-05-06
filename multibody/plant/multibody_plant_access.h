#pragma once

#include <set>
#include <string>
#include <utility>

#include "drake/common/drake_assert.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/system.h"

namespace drake {
namespace multibody {
namespace internal {
template <typename T>
class ContactComputationManagerImpl;

// This class is used to grant access to a selected collection of
// MultibodyPlant's private members and/or methods to ContactComputationManager.
template <typename T>
class MultibodyPlantContactComputationManagerAttorney {
 private:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(
      MultibodyPlantContactComputationManagerAttorney);
  MultibodyPlantContactComputationManagerAttorney() = delete;

  friend class ContactComputationManagerImpl<T>;

  static inline const MultibodyTree<T>& internal_tree(
      const MultibodyPlant<T>& plant) {
    return plant.internal_tree();
  }

  static inline systems::DiscreteStateIndex DeclareDiscreteState(
      MultibodyPlant<T>* plant, const VectorX<T>& model_value) {
    return plant->DeclareDiscreteState(model_value);
  }

  static inline systems::LeafOutputPort<T>& DeclareAbstractOutputPort(
      MultibodyPlant<T>* plant, std::string name,
      typename systems::LeafOutputPort<T>::AllocCallback alloc_function,
      typename systems::LeafOutputPort<T>::CalcCallback calc_function,
      std::set<systems::DependencyTicket> prerequisites_of_calc = {
          systems::System<T>::all_sources_ticket()}) {
    return plant->DeclareAbstractOutputPort(
        std::move(name), std::move(alloc_function), std::move(calc_function),
        std::move(prerequisites_of_calc));
  }
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake
