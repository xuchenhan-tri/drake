#include <gtest/gtest.h>

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {

using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::Simulator;

namespace multibody {
namespace {

GTEST_TEST(ContactResultsTest, MyTest) {
  MultibodyPlantConfig plant_config;
  const double kDt = 0.01;
  plant_config.time_step = kDt;
  plant_config.discrete_contact_solver = "sap";
  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  plant.Finalize();

  class ContactResultsComputer : public systems::LeafSystem<double> {
   public:
    explicit ContactResultsComputer(double period) {
      this->DeclareAbstractInputPort("contact results",
                                     Value<ContactResults<double>>{});
      this->DeclarePeriodicPublishEvent(period, 0.0,
                                        &ContactResultsComputer::Publish);
    }

   private:
    systems::EventStatus Publish(const Context<double>& context) const {
      std::cout << "Computer publishes at time " << context.get_time()
                << std::endl;
      get_input_port().Eval<ContactResults<double>>(context);
      return systems::EventStatus::Succeeded();
    }
  };

  const auto& computer = *builder.AddSystem<ContactResultsComputer>(kDt);
  builder.Connect(plant.get_contact_results_output_port(),
                  computer.get_input_port());
  auto diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);
  simulator.Initialize();
  simulator.AdvanceTo(2 * kDt);
  // Force a failure to dump debug messages to console.
  EXPECT_TRUE(false);
}

}  // namespace
}  // namespace multibody
}  // namespace drake
