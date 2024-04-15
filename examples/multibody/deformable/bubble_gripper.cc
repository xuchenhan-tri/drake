#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/examples/multibody/deformable/parallel_gripper_controller.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/render_gl/factory.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/sensors/camera_config.h"
#include "drake/systems/sensors/camera_config_functions.h"

DEFINE_double(simulation_time, 15.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(E, 5e2, "Young's modulus for the teddy bear [Pa].");
DEFINE_double(discrete_time_step, 1e-2,
              "Discrete time step for the system [s].");

using drake::examples::deformable::ParallelGripperController;
using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::PerceptionProperties;
using drake::geometry::ProximityProperties;
using drake::geometry::RenderEngineGlParams;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::math::RotationMatrixd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::PackageMap;
using drake::multibody::Parser;
using drake::multibody::PrismaticJoint;
using drake::multibody::fem::DeformableBodyConfig;
using drake::schema::Transform;
using drake::systems::Context;
using drake::systems::sensors::ApplyCameraConfig;
using drake::systems::sensors::CameraConfig;
using Eigen::Vector3d;
using Eigen::Vector4d;

namespace drake {
namespace examples {
namespace multibody {
namespace bubble_gripper {
namespace {

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  DRAKE_DEMAND(FLAGS_discrete_time_step > 0.0);
  plant_config.time_step = FLAGS_discrete_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_approximation = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Minimum required proximity properties for rigid bodies to interact with
   deformable bodies.
   1. A valid Coulomb friction coefficient, and
   2. A resolution hint. (Rigid bodies need to be tessellated so that collision
   queries can be performed against deformable geometries.) */
  ProximityProperties rigid_proximity_props;
  const CoulombFriction<double> surface_friction(1.0, 1.0);
  const double resolution_hint = 0.01;
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  rigid_proximity_props.AddProperty(geometry::internal::kHydroGroup,
                                    geometry::internal::kRezHint,
                                    resolution_hint);

  /* Set up a ground. */
  Box ground{1, 1, 1};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -0.505});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                  "ground_collision", rigid_proximity_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.95, 0.80, 0.65, 0.9));
  /* Avoid rendering the ground as it clutters the background.
   Currently, all visual geometries added through MultibodyPlant are
   automatically assigned perception properties. When that automatic assignment
   is no longer done, we can remove this and simply not assign a perception
   property. */
  illustration_props.AddProperty("renderer", "accepting",
                                 std::set<std::string>{"nothing"});
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground,
                               "ground_visual", illustration_props);

  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);
  DeformableModel<double>* deformable_model = owned_deformable_model.get();

  /* Minimally required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  ProximityProperties deformable_proximity_props;
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);

  /* Add in a deformable manipuland. */
  DeformableBodyConfig<double> teddy_config;
  teddy_config.set_youngs_modulus(FLAGS_E);              // [Pa]
  teddy_config.set_poissons_ratio(0.45);                 // unitless
  teddy_config.set_mass_density(1000);                   // [kg/m³]
  teddy_config.set_stiffness_damping_coefficient(0.05);  // [1/s]
  const std::string teddy_vtk = FindResourceOrThrow(
      "drake/examples/multibody/deformable/models/teddy.vtk");
  auto teddy_mesh = std::make_unique<Mesh>(teddy_vtk, /* scale */ 0.15);
  auto teddy_instance = std::make_unique<GeometryInstance>(
      RigidTransformd(math::RollPitchYawd(M_PI / 2.0, 0, -M_PI / 2.0),
                      Vector3d(-0.17, 0, 0)),
      std::move(teddy_mesh), "teddy");
  teddy_instance->set_proximity_properties(deformable_proximity_props);
  deformable_model->RegisterDeformableBody(std::move(teddy_instance),
                                           teddy_config, 1.0);
  plant.AddPhysicalModel(std::move(owned_deformable_model));

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* It's essential to connect the vertex position port in DeformableModel to
   the source configuration port in SceneGraph when deformable bodies are
   present in the plant. */
  builder.Connect(
      deformable_model->vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));

  /* Add a visualizer that emits LCM messages for visualization. */
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
  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace bubble_gripper
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase the following features in deformable "
      "body simulation in Drake:\n"
      "  1. frictional contact resolution among deformable bodies;\n"
      "  2. deformable geometry rendering;\n"
      "  3. fixed constraints between rigid bodies and deformable bodies.\n"
      "Note that this example only runs on Linux systems.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::bubble_gripper::do_main();
}
