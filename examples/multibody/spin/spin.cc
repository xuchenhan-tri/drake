#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 1.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
/* We vary the time step size in the set [1e-3, 1e-2, 1e-1, 1] to observe the
 effect of large time step in rotational deformation. */
DEFINE_double(time_step, 1.0,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 5e5, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1e3, "Mass density of the deformable body [kg/m³].");
DEFINE_double(beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_bool(nonlinear, false,
              "Whether or not to use nonlinear corotated model.");
using drake::geometry::AddContactMaterial;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableModel;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using Eigen::Vector3d;
using Eigen::Vector4d;

namespace drake {
namespace examples {
namespace multibody {
namespace deformable_box {

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Set up a deformable body. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);

  DeformableBodyConfig<double> deformable_config;
  deformable_config.set_youngs_modulus(FLAGS_E);
  deformable_config.set_poissons_ratio(FLAGS_nu);
  deformable_config.set_mass_density(FLAGS_density);
  deformable_config.set_stiffness_damping_coefficient(FLAGS_beta);

  const std::string box_vtk =
      FindResourceOrThrow("drake/examples/multibody/spin/box.vtk");
  auto box_mesh = std::make_unique<Mesh>(box_vtk, 1.0);
  auto box_mesh2 = std::make_unique<Mesh>(box_vtk, 1.0);
  const RigidTransformd X_WB(RollPitchYawd(0, 0, 0), Vector3d(0, 0, 0));
  const RigidTransformd X_WB2(RollPitchYawd(0, 0, 0), Vector3d(0, 0.4, 0));
  ProximityProperties deformable_proximity_props;
  const CoulombFriction<double> surface_friction(0.0, 0.0);
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);

  auto box_instance =
      std::make_unique<GeometryInstance>(X_WB, std::move(box_mesh), "box");
  auto box_instance2 =
      std::make_unique<GeometryInstance>(X_WB2, std::move(box_mesh2), "box2");
  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  box_instance->set_proximity_properties(deformable_proximity_props);
  box_instance2->set_proximity_properties(deformable_proximity_props);

  auto id = owned_deformable_model->RegisterDeformableBody(
      std::move(box_instance), deformable_config, 1.0);
  deformable_config.set_material_model(
      drake::multibody::fem::MaterialModel ::kCorotated);
  auto id2 = owned_deformable_model->RegisterDeformableBody(
      std::move(box_instance2), deformable_config, 1.0);
  owned_deformable_model->SetWallBoundaryCondition(
      id, Vector3<double>(-0.499, 0, 0), Vector3<double>(1, 0, 0));
  owned_deformable_model->SetWallBoundaryCondition(
      id2, Vector3<double>(-0.499, 0, 0), Vector3<double>(1, 0, 0));

  const DeformableModel<double>* deformable_model =
      owned_deformable_model.get();
  plant.AddPhysicalModel(std::move(owned_deformable_model));

  drake::geometry::Box wall{1, 2, 1};
  const RigidTransformd X_WG(Eigen::Vector3d{-1, 0, 0});
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.7, 0.5, 0.4, 0.8));
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, wall, "ground_visual",
                               std::move(illustration_props));

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* It's essential to connect the vertex position port in DeformableModel to
   the source configuration port in SceneGraph when deformable bodies are
   present in the plant. */
  builder.Connect(
      deformable_model->vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

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

}  // namespace deformable_box
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
  return drake::examples::multibody::deformable_box::do_main();
}
