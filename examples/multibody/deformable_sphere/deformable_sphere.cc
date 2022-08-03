#include <memory>

#include <gflags/gflags.h>

#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
namespace examples {
namespace deformable_sphere {
namespace {

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time (usually between 0 and 1). "
              "This is documented in Simulator::set_target_realtime_rate().");
DEFINE_double(simulation_time, 3.0, "Simulation duration in seconds");
DEFINE_double(time_step, 1.0E-3,
              "If time_step > 0, the fixed-time step period (in seconds) of "
              "discrete updates for the plant (modeled as a discrete system). "
              "If time_step = 0, the plant is modeled as a continuous system "
              "and no contact forces are displayed.  time_step must be >= 0.");

using math::RigidTransformd;
using multibody::internal::DeformableModel;

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;

  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_solver = "sap";
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlant(plant_config, &builder);

  multibody::fem::DeformableBodyConfig<double> deformable_config;
  deformable_config.set_youngs_modulus(1e6);
  deformable_config.set_poissons_ratio(0.45);
  auto deformable_model = std::make_unique<DeformableModel<double>>(&plant);
  constexpr double kRezHint = 0.025;
  deformable_model->RegisterDeformableBody(
      std::make_unique<geometry::GeometryInstance>(
          RigidTransformd::Identity(), std::make_unique<geometry::Sphere>(0.1),
          "sphere"),
      deformable_config, kRezHint);
  const DeformableModel<double>* deformable_model_ptr = deformable_model.get();
  plant.AddPhysicalModel(std::move(deformable_model));
  plant.Finalize();
  builder.Connect(
      deformable_model_ptr->get_vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));

  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace deformable_sphere
}  // namespace examples
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("deformable sphere");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::examples::deformable_sphere::do_main();
}
