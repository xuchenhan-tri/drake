#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/examples/multibody/deformable_torus/parallel_gripper_controller.h"
#include "drake/examples/multibody/deformable_torus/point_source_force_field.h"
#include "drake/examples/multibody/deformable_torus/suction_cup_controller.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/render_gl/factory.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
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
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/sensors/image_to_lcm_image_array_t.h"
#include "drake/systems/sensors/pixel_types.h"
#include "drake/systems/sensors/rgbd_sensor.h"

DEFINE_double(simulation_time, 12.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 3e4, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1e3,
              "Mass density of the deformable body [kg/m³]. We observe that "
              "density above 2400 kg/m³ makes the torus too heavy to be picked "
              "up by the suction gripper.");
DEFINE_double(beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_string(gripper, "parallel",
              "Type of gripper used to pick up the deformable torus. Options "
              "are: 'parallel' and 'suction'.");

using drake::examples::deformable_torus::ParallelGripperController;
using drake::examples::deformable_torus::PointSourceForceField;
using drake::examples::deformable_torus::SuctionCupController;
using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::Capsule;
using drake::geometry::Ellipsoid;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::PerceptionProperties;
using drake::geometry::ProximityProperties;
using drake::geometry::RenderEngineGlParams;
using drake::geometry::render::ColorRenderCamera;
using drake::geometry::render::DepthRenderCamera;
using drake::geometry::render::RenderLabel;
using drake::math::RigidTransformd;
using drake::math::RotationMatrixd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::PrismaticJoint;
using drake::multibody::SpatialInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::sensors::PixelType;
using drake::systems::sensors::RgbdSensor;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace {

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP. */
  plant_config.discrete_contact_approximation = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  const std::string render_name("renderer");
  scene_graph.AddRenderer(render_name,
                          MakeRenderEngineGl(RenderEngineGlParams()));
  /* Minimum required proximity properties for rigid bodies to interact with
   deformable bodies.
   1. A valid Coulomb friction coefficient, and
   2. A resolution hint. (Rigid bodies need to be tessellated so that collision
   queries can be performed against deformable geometries.) The value dictates
   how fine the mesh used to represent the rigid collision geometry is. */
  ProximityProperties rigid_proximity_props;
  /* Set the friction coefficient close to that of rubber against rubber. */
  const CoulombFriction<double> surface_friction(1.15, 1.15);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  rigid_proximity_props.AddProperty(geometry::internal::kHydroGroup,
                                    geometry::internal::kRezHint, 0.01);
  /* Set up a ground. */
  Box ground{4, 4, 4};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -2});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                  "ground_collision", rigid_proximity_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.7, 0.5, 0.4, 0.8));
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground,
                               "ground_visual", std::move(illustration_props));

  /* Set up a deformable torus. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);

  DeformableBodyConfig<double> deformable_config;
  deformable_config.set_youngs_modulus(FLAGS_E);
  deformable_config.set_poissons_ratio(FLAGS_nu);
  deformable_config.set_mass_density(FLAGS_density);
  deformable_config.set_stiffness_damping_coefficient(FLAGS_beta);

  const std::string textured_torus_obj = FindResourceOrThrow(
      "drake/examples/multibody/deformable_torus/textured_torus.obj");
  const std::string torus_vtk = FindResourceOrThrow(
      "drake/examples/multibody/deformable_torus/torus.vtk");
  /* Load the geometry and scale it down to 65% (to showcase the scaling
   capability and to make the torus suitable for grasping by the gripper). */
  const double scale = 0.65;

  auto torus_mesh = std::make_unique<Mesh>(torus_vtk, scale);
  auto torus_render_mesh = std::make_unique<Mesh>(textured_torus_obj);
  /* Minor diameter of the torus inferred from the vtk file. */
  const double kL = 0.09 * scale;
  /* Set the initial pose of the torus such that its bottom face is touching the
   ground. */
  const RigidTransformd X_WT(Vector3<double>(0.0, 0.0, kL / 2.0 + 0.5));
  auto torus_instance = std::make_unique<GeometryInstance>(
      X_WT, std::move(torus_mesh), "deformable_torus");
  auto visual_torus_instance = std::make_unique<GeometryInstance>(
      X_WT, std::move(torus_render_mesh), "deformable_torus_visual");

  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  ProximityProperties deformable_proximity_props;
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);
  torus_instance->set_proximity_properties(deformable_proximity_props);

  PerceptionProperties perception_properties;
  perception_properties.AddProperty("phong", "diffuse",
                                    Vector4d{1.0, 1.0, 1.0, 1.0});
  perception_properties.AddProperty("label", "id", RenderLabel(42));
  visual_torus_instance->set_perception_properties(perception_properties);

  /* Registration of all deformable geometries ostensibly requires a resolution
   hint parameter that dictates how the shape is tessellated. In the case of a
   `Mesh` shape, the resolution hint is unused because the shape is already
   tessellated. */
  // TODO(xuchenhan-tri): Though unused, we still asserts the resolution hint is
  // positive. Remove the requirement of a resolution hint for meshed shapes.
  const double unused_resolution_hint = 1.0;
  owned_deformable_model->RegisterDeformableBody(
      std::move(torus_instance), deformable_config, unused_resolution_hint,
      std::move(visual_torus_instance));
  const DeformableModel<double>* deformable_model =
      owned_deformable_model.get();
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
  drake::lcm::DrakeLcm lcm;
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, &lcm);

  // Create the camera.
  const ColorRenderCamera color_camera{
      {render_name, {1280, 960, M_PI_4}, {0.1, 2.0}, {}}, false};
  const DepthRenderCamera depth_camera{color_camera.core(), {0.1, 2.0}};
  // We need to position and orient the camera. We have the camera body frame
  // B (see rgbd_sensor.h) and the camera frame C (see camera_info.h).
  // By default X_BC = I in the RgbdSensor. So, to aim the camera, Cz = Bz
  // should point from the camera position to the origin. By points *down* the
  // image, so we need to align it in the -Wz direction. So,  we compute the
  // basis using camera Y-ish in the By ≈ -Wz direction to compute Bx, and
  // then use Bx an and Bz to compute By.
  const Vector3d p_WB(0.3, -1, 0.25);
  // Set rotation looking at the origin.
  const Vector3d Bz_W = -p_WB.normalized();
  const Vector3d Bx_W = -Vector3d::UnitZ().cross(Bz_W).normalized();
  const Vector3d By_W = Bz_W.cross(Bx_W).normalized();
  const RotationMatrixd R_WB =
      RotationMatrixd::MakeFromOrthonormalColumns(Bx_W, By_W, Bz_W);
  const RigidTransformd X_WB(R_WB, p_WB);

  auto camera = builder.AddSystem<RgbdSensor>(scene_graph.world_frame_id(),
                                              X_WB, color_camera, depth_camera);
  builder.Connect(scene_graph.get_query_output_port(),
                  camera->query_object_input_port());
  // Broadcast the images to Meldis (available after #18862 is finished).
  auto image_to_lcm_image_array =
      builder.template AddSystem<systems::sensors::ImageToLcmImageArrayT>();
  image_to_lcm_image_array->set_name("converter");

  systems::lcm::LcmPublisherSystem* image_array_lcm_publisher =
      builder.template AddSystem(
          systems::lcm::LcmPublisherSystem::Make<lcmt_image_array>(
              "DRAKE_RGBD_CAMERA_IMAGES", &lcm, 0.1 /* publish period */));
  image_array_lcm_publisher->set_name("publisher");

  builder.Connect(image_to_lcm_image_array->image_array_t_msg_output_port(),
                  image_array_lcm_publisher->get_input_port());
  {
    const auto& port =
        image_to_lcm_image_array->DeclareImageInputPort<PixelType::kRgba8U>(
            "color");
    builder.Connect(camera->color_image_output_port(), port);
  }

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
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase deformable body simulations in Drake. "
      "A parallel (or suction) gripper grasps a deformable torus on the "
      "ground, lifts it up, and then drops it back on the ground. "
      "Launch meldis before running this example. "
      "Refer to README for instructions on meldis as well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
