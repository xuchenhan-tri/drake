#include <fstream>
#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/sine.h"

DEFINE_double(simulation_time, 10.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 0.0, "Desired real time rate.");
DEFINE_double(time_step, 1.0e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e6, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.49, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1000, "Mass density of the deformable body [kg/m³].");
DEFINE_double(beta, 0.1,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(radius, 0.008, "The radius of the rigid cylinder. [m].");
DEFINE_double(amplitude, 3.0,
              "The amplitude of the harmonic oscillations carried out by the "
              "rigid object. [N].");
DEFINE_double(shift, 7.0,
              "The baseline force carried out by the rigid object. [N].");
DEFINE_double(frequency, 1.0, "The frequency of the harmonic force. [1/s].");
DEFINE_double(x, -0.03, "The initial position of the rigid object [m].");
DEFINE_double(z, 0.09, "The initial position of the rigid object [m].");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::Cylinder;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::PrismaticJoint;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::UnitInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace multibody {
namespace finray {
namespace {

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Set up a ground. */
  Box ground{0.2, 0.2, 0.2};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -0.1});
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.35, 0.35, 0.35, 0.95));
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground,
                               "ground_visual", illustration_props);

  const double cylinder_radius = FLAGS_radius;
  const double cylinder_height = 0.03;
  const double cylinder_mass = 1;
  const Eigen::Vector3d cylinder_com(0, 0, 0);
  const UnitInertia<double> cylinder_unit_inertia =
      UnitInertia<double>::SolidCylinder(cylinder_radius, cylinder_height);

  const RigidBody<double>& cylinder_body = plant.AddRigidBody(
      "rigid cylinder", SpatialInertia<double>(cylinder_mass, cylinder_com,
                                               cylinder_unit_inertia));
  Cylinder cylinder(cylinder_radius, cylinder_height);
  /* Minimum required proximity properties for rigid bodies to interact with
   deformable bodies.
   1. A valid Coulomb friction coefficient, and
   2. A resolution hint. (Rigid bodies need to be tesselated so that collision
   queries can be performed against deformable geometries.) */
  ProximityProperties rigid_proximity_props;
  /* Set the friction coefficient close to that of rubber against rubber. */
  const CoulombFriction<double> surface_friction(1.0, 1.0);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  rigid_proximity_props.AddProperty(geometry::internal::kHydroGroup,
                                    geometry::internal::kRezHint, 0.006);
  plant.RegisterCollisionGeometry(cylinder_body, RigidTransformd::Identity(),
                                  cylinder, "cylinder_collision",
                                  std::move(rigid_proximity_props));
  IllustrationProperties cylinder_illustration_props;
  cylinder_illustration_props.AddProperty("phong", "diffuse",
                                          Eigen::Vector4d(0.6, 0.8, 0.4, 0.8));
  plant.RegisterVisualGeometry(cylinder_body, RigidTransformd::Identity(),
                               cylinder, "cylinder_visual",
                               std::move(cylinder_illustration_props));

  const auto& translate_x_joint =
      plant.AddJoint<drake::multibody::PrismaticJoint>(
          "translate_x_joint", plant.world_body(),
          RigidTransformd(RollPitchYawd(M_PI / 2.0, 0, 0),
                          Vector3d(FLAGS_x, 0, FLAGS_z)),
          cylinder_body, std::nullopt, Eigen::Vector3d::UnitX());
  plant.AddJointActuator("translate_x_joint", translate_x_joint);

  /* Set up a deformable finray gripper. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);
  DeformableModel<double>* deformable_model = owned_deformable_model.get();

  DeformableBodyConfig<double> finray_config;
  finray_config.set_youngs_modulus(FLAGS_E);
  finray_config.set_poissons_ratio(FLAGS_nu);
  finray_config.set_mass_density(FLAGS_density);
  finray_config.set_stiffness_damping_coefficient(FLAGS_beta);

  const std::string finray_vtk =
      FindResourceOrThrow("drake/examples/multibody/finray/finray_fine.vtk");
  auto finray_mesh = std::make_unique<Mesh>(finray_vtk, 1.0);
  const RigidTransformd X_WB(Eigen::Vector3d{0, 0, 0.06});
  auto finray_instance = std::make_unique<GeometryInstance>(
      X_WB, std::move(finray_mesh), "finray");
  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  ProximityProperties deformable_proximity_props;
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);
  finray_instance->set_proximity_properties(deformable_proximity_props);
  IllustrationProperties finray_illustration_props;
  finray_illustration_props.AddProperty("phong", "diffuse",
                                        Vector4d(0.53, 0.42, 0.34, 1.0));
  finray_instance->set_illustration_properties(finray_illustration_props);

  auto body_id = deformable_model->RegisterDeformableBody(
      std::move(finray_instance), finray_config, 1.0);
  deformable_model->SetWallBoundaryCondition(
      body_id, Vector3d(0.015, 0, 0.04596), Vector3d(-1, 0, 1));
  Box cage(0.065, 0.03, 0.065);
  const RigidTransformd X_WC(RollPitchYawd(0, M_PI / 4, 0),
                             Eigen::Vector3d{0.015, 0.0, 0.0});
  plant.RegisterVisualGeometry(plant.world_body(), X_WC, cage, "cage visual",
                               illustration_props);

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
  //   auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  //   auto& meshcat_visualizer =
  //   drake::geometry::MeshcatVisualizerd::AddToBuilder(
  //       &builder, scene_graph, meshcat);

  const drake::Vector1d amplitude(FLAGS_amplitude);
  const drake::Vector1d frequency(FLAGS_frequency);
  const drake::Vector1d phase(-M_PI / 2);
  const auto& harmonic_force = *builder.AddSystem<drake::systems::Sine<double>>(
      amplitude, frequency, phase);
  const auto& constant_force =
      *builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
          drake::Vector1d(FLAGS_shift));
  const auto& adder = *builder.AddSystem<drake::systems::Adder<double>>(2, 1);
  builder.Connect(harmonic_force.get_output_port(0), adder.get_input_port(0));
  builder.Connect(constant_force.get_output_port(), adder.get_input_port(1));
  builder.Connect(adder.get_output_port(0), plant.get_actuation_input_port());

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
}  // namespace finray
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase deformable body simulations in Drake. "
      "A simple parallel gripper grasps a deformable box on the ground, lifts "
      "it up, and then drops it back on the ground. "
      "Launch meldis before running this example. "
      "Refer to README for instructions on meldis as well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::finray::do_main();
}
