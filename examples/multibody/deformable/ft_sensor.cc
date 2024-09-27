#include <iostream>
#include <fstream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"

DEFINE_double(simulation_time, 40.0,
              "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 0.0, "Desired real time rate.");
DEFINE_double(discrete_time_step, 1e-2,
              "Discrete time step for the system [s].");

using drake::geometry::Box;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::ProximityProperties;
using drake::geometry::Rgba;
using drake::geometry::Sphere;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::math::RotationMatrixd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::BodyIndex;
using drake::multibody::CoulombFriction;
using drake::multibody::Joint;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::PackageMap;
using drake::multibody::Parser;
using drake::multibody::PrismaticJoint;
using drake::multibody::SpatialForce;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::EventStatus;
using Eigen::Vector3d;
using Eigen::Vector4d;

namespace drake {
namespace examples {
namespace multibody {
namespace ft_sensor {
namespace {

class SimpleController : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SimpleController);

  SimpleController(double period, double amplitude)
      : period_(period),
        amplitude_(amplitude),
        slope_(4.0 * amplitude / period_) {
    this->DeclareVectorOutputPort("desired state", BasicVector<double>(2),
                                  &SimpleController::CalcDesiredState);
  }

 private:
  double period_{};
  double amplitude_{};
  double slope_{};

  void CalcDesiredState(const systems::Context<double>& context,
                      systems::BasicVector<double>* output) const {
  double t = context.get_time();
  t -= std::floor(t / period_) * period_;  // Wrap time to [0, period_)

  double position;
  double velocity;

  if (t < period_ / 4.0) {
    // First phase: increasing velocity linearly (rising slope)
    velocity = slope_ * t;  // Linear velocity
    position = 0.5 * slope_ * t * t;  // Quadratic position (integral of velocity)
  } else if (t < period_ * 3.0 / 4.0) {
    // Second phase: decreasing velocity linearly (falling slope)
    const double v_max = slope_ * (period_ / 4.0);  // Maximum velocity reached at t = period_ / 4
    const double t_rel = t - period_ / 4.0;  // Relative time in this phase
    velocity = -slope_ * t_rel + v_max;  // Linear velocity falling from v_max to 0
    const double p_mid = 0.5 * slope_ * std::pow(period_ / 4.0, 2);  // Position at the end of first phase
    position = -0.5 * slope_ * std::pow(t_rel, 2) + v_max * t_rel + p_mid;  // Quadratic position
  } else {
    // Third phase: increasing velocity linearly again (negative slope)
    const double v_min = -slope_ * (period_ / 4.0);  // Minimum velocity reached at t = period_ * 3/4
    const double t_rel = t - period_ * 3.0 / 4.0;  // Relative time in this phase
    velocity = slope_ * t_rel + v_min;  // Linear velocity increasing from v_min
    const double p_end = 0.5 * slope_ * std::pow(period_ / 4.0, 2);  // Position at the end of second phase
    position = 0.5 * slope_ * std::pow(t_rel, 2) + v_min * t_rel + p_end;  // Quadratic position
  }

  velocity = 0.0;
  output->get_mutable_value() << position, velocity;
}

};

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  DRAKE_DEMAND(FLAGS_discrete_time_step > 0.0);
  plant_config.time_step = FLAGS_discrete_time_step;
  plant_config.discrete_contact_approximation = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Parse the gripper model (without the bubbles). */
  Parser parser(&plant, &scene_graph);

  const auto model_instance = parser.AddModelsFromUrl(
      "package://drake/examples/multibody/deformable/models/robot.sdf")[0];

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  const double period = 10.0;
  const double amplitude = 10.0;
  const auto& control = *builder.AddSystem<SimpleController>(period, amplitude);
  builder.Connect(control.get_output_port(),
                  plant.get_desired_state_input_port(model_instance));

  geometry::DrakeVisualizerParams params;
  params.role = geometry::Role::kIllustration;
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, nullptr,
                                           params);

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  std::cout << "start" << std::endl;
  double time = 0.0;
  std::ofstream output_file("ft_data.txt", std::ios::app);

  simulator.set_monitor(
      [&plant, &time, &output_file](const Context<double>& root_context) {
        const Context<double>& plant_context =
            plant.GetMyContextFromRoot(root_context);
        time = root_context.get_time();
        const auto& reaction_forces =
            plant.get_reaction_forces_output_port()
                .Eval<std::vector<SpatialForce<double>>>(plant_context);
        DRAKE_DEMAND(reaction_forces.size() == 2);
        const Joint<double>& joint = plant.GetJointByName("weld");
        const SpatialForce<double>& ft = reaction_forces.at(joint.index());
        const Vector3d force = ft.translational();
        const Vector3d torque = ft.rotational();
        output_file << fmt::format(
            "time: {}, force: {}, {}, {}, torque: {}, {}, {}\n",
            root_context.get_time(), force.x(), force.y(), force.z(),
            torque.x(), torque.y(), torque.z());

        return EventStatus::ReachedTermination(&plant, "");
      });
  while (time < FLAGS_simulation_time) {
    simulator.AdvanceTo(FLAGS_simulation_time);
  }
  output_file.close();
  std::cout << "finished" << std::endl;

  return 0;
}

}  // namespace
}  // namespace ft_sensor
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::ft_sensor::do_main();
}
