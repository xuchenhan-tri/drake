#pragma once

#include <set>
#include <string>
#include <utility>

#include "drake/multibody/plant/contact_computation_manager.h"
#include "drake/multibody/plant/multibody_plant_access.h"
#include "drake/systems/framework/leaf_system.h"
namespace drake {
namespace multibody {
namespace internal {
// This class exposes a selection of private/protected MbP methods so that
// derived classes can invoke these methods as if they are MbP. It achieves this
// by forwarding these calls to the MbP attorney. This class also helps breaking
// circular dependency by allowing ContactComputationManager to be pure virtual
// and not having to depend on the attorney for MbP. All concrete
// ContactComputationManager should derive from this class instead of
// ContactComputationManager.
// @tparam_default_scalar
template <typename T>
class ContactComputationManagerImpl : public ContactComputationManager<T> {
 public:
  explicit ContactComputationManagerImpl(MultibodyPlant<T>* plant)
      : plant_(plant) {
    DRAKE_DEMAND(plant_ != nullptr);
  }

  // TODO(xuchenhan-tri): Make the desctructor pure virtual so that the class is
  // abstract.
  ~ContactComputationManagerImpl() = default;

 protected:
  const MultibodyTree<T>& internal_tree() const {
    return MultibodyPlantContactComputationManagerAttorney<T>::internal_tree(
        *plant_);
  }

  systems::DiscreteStateIndex DeclareDiscreteState(
      const VectorX<T>& model_value) const {
    return MultibodyPlantContactComputationManagerAttorney<
        T>::DeclareDiscreteState(plant_, model_value);
  }

  systems::LeafOutputPort<T>& DeclareAbstractOutputPort(
      std::string name,
      typename systems::LeafOutputPort<T>::AllocCallback alloc_function,
      typename systems::LeafOutputPort<T>::CalcCallback calc_function,
      std::set<systems::DependencyTicket> prerequisites_of_calc = {
          systems::System<T>::all_sources_ticket()}) const {
    return MultibodyPlantContactComputationManagerAttorney<
        T>::DeclareAbstractOutputPort(plant_, std::move(name),
                                      std::move(alloc_function),
                                      std::move(calc_function),
                                      std::move(prerequisites_of_calc));
  }

  const systems::OutputPort<T>& get_output_port(
      systems::OutputPortIndex output_port_index) const {
    return plant_->get_output_port(output_port_index);
  }

 private:
  MultibodyPlant<T>* plant_;
};
}  // namespace internal
}  // namespace multibody
}  // namespace drake
