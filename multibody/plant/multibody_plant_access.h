#pragma once

#include <set>
#include <string>
#include <utility>

#include "drake/common/drake_assert.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace multibody {
namespace internal {
// This class is used to grant access to a selected collection of
// MultibodyPlant's private members and/or methods to ContactComputationManager.
// All it does is that it forwards the invoked method to the associated
// MultibodyPlant.
template <typename T>
class MultibodyPlantAccess {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MultibodyPlantAccess);

  explicit MultibodyPlantAccess(MultibodyPlant<T>* plant) : plant_(plant) {
    DRAKE_DEMAND(plant_ != nullptr);
  }

  const MultibodyTree<T>& internal_tree() { return plant_->internal_tree(); }

  systems::DiscreteStateIndex DeclareDiscreteState(
      const VectorX<T>& model_value) {
    return plant_->DeclareDiscreteState(model_value);
  }

  systems::LeafOutputPort<T>& DeclareAbstractOutputPort(
      std::string name,
      typename systems::LeafOutputPort<T>::AllocCallback alloc_function,
      typename systems::LeafOutputPort<T>::CalcCallback calc_function,
      std::set<systems::DependencyTicket> prerequisites_of_calc = {
          systems::System<T>::all_sources_ticket()}) {
    return plant_->DeclareAbstractOutputPort(
        std::move(name), std::move(alloc_function), std::move(calc_function),
        std::move(prerequisites_of_calc));
  }

  const systems::OutputPort<T>& get_output_port(
      systems::OutputPortIndex output_port_index) const {
    return plant_->get_output_port(output_port_index);
  }

  const contact_solvers::internal::ContactSolverResults<T>&
  EvalContactSolverResults(const systems::Context<T>& context) const {
    return plant_->EvalContactSolverResults(context);
  }

 private:
  MultibodyPlant<T>* plant_{nullptr};
};
}  // namespace internal
}  // namespace multibody
}  // namespace drake
