#include <math.h>

#include <fstream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/meshcat_point_cloud_visualizer.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/meshcat_visualizer_params.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/unit_inertia.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/visualization/visualization_config.h"
#include "drake/visualization/visualization_config_functions.h"

DEFINE_double(simulation_time, 3.0, "Desired duration of the simulation [s].");
DEFINE_double(time_step, 1e-3,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(
    substep, 1e-4,
    "Discrete time step for the substepping scheme [s]. Must be positive.");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::math::RigidTransformd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableModel;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::PrismaticJoint;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::gmpm::MpmConfigParams;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace {

std::vector<Vector3d> MakeInitialPositions(Vector3d min_corner,
                                           Vector3d max_corner,
                                           int num_per_dim) {
  std::vector<Vector3d> positions;
  Vector3d delta = (max_corner - min_corner) / (num_per_dim - 1);
  for (int i = 0; i < num_per_dim; ++i) {
    for (int j = 0; j < num_per_dim; ++j) {
      for (int k = 0; k < num_per_dim; ++k) {
        positions.push_back(min_corner + Vector3d(i, j, k).cwiseProduct(delta));
      }
    }
  }
  return positions;
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;

  plant_config.discrete_contact_approximation = "lagged";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  ProximityProperties box_proximity_props;
  const CoulombFriction<double> surface_friction(0.8, 0.8);
  AddContactMaterial(FLAGS_damping, {}, surface_friction, &box_proximity_props);
  const double mpm_box_width = 0.1;

  const double rigid_box_side_x = 1.0 / 60.0;
  const double rigid_box_side_y = 0.14;
  const double rigid_box_side_z = 0.1;
  const double rigid_box_density = 1000;
  const double mpm_shift = 0.5;
  const RigidTransformd X_WB(Eigen::Vector3d{mpm_shift, mpm_shift, mpm_shift});
  ModelInstanceIndex left_box_model_instance =
      plant.AddModelInstance("left_box_instance");
  const SpatialInertia<double> box_spatial =
      SpatialInertia<double>::SolidBoxWithDensity(
          rigid_box_density, rigid_box_side_x, rigid_box_side_y,
          rigid_box_side_z);
  const RigidBody<double>& left_box =
      plant.AddRigidBody("left_box", left_box_model_instance, box_spatial);
  const auto& left_prismatic_joint_x = plant.AddJoint<PrismaticJoint>(
      "left_translate_x_joint", plant.world_body(), X_WB, left_box,
      std::nullopt, Vector3d::UnitX());
  plant.GetMutableJointByName<PrismaticJoint>("left_translate_x_joint")
      .set_default_translation(-(0.5 + 0.5 / 6.0 + 0.0 / FLAGS_ppc) *
                               box_width);
  const auto left_actuator_x_index =
      plant.AddJointActuator("left x actuator", left_prismatic_joint_x).index();
  unused(left_actuator_x_index);

  // box controlled on the right
  ModelInstanceIndex right_box_model_instance =
      plant.AddModelInstance("right_box_instance");
  const RigidBody<double>& right_box =
      plant.AddRigidBody("right_box", right_box_model_instance, box_spatial);
  const auto& right_prismatic_joint_x = plant.AddJoint<PrismaticJoint>(
      "right_translate_x_joint", plant.world_body(), X_WB, right_box,
      std::nullopt, Vector3d::UnitX());
  plant.GetMutableJointByName<PrismaticJoint>("right_translate_x_joint")
      .set_default_translation((0.5 + 0.5 / 6.0) * box_width);
  const auto right_actuator_x_index =
      plant.AddJointActuator("right x actuator", right_prismatic_joint_x)
          .index();
  unused(right_actuator_x_index);

  Box rigid_box(rigid_box_side_x, rigid_box_side_y, rigid_box_side_z);
  const Vector4<double> grey(0.5, 0.5, 0.5, 1.0);
  plant.RegisterVisualGeometry(left_box, RigidTransformd::Identity(), rigid_box,
                               "LeftCubeV", grey);
  plant.RegisterCollisionGeometry(left_box, RigidTransformd::Identity(),
                                  rigid_box, "LeftCube", box_proximity_props);
  plant.RegisterVisualGeometry(right_box, RigidTransformd::Identity(),
                               rigid_box, "RightCubeV", grey);
  plant.RegisterCollisionGeometry(right_box, RigidTransformd::Identity(),
                                  rigid_box, "RightCube", box_proximity_props);

  // mpm stuff
  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();

  const int num_per_dim = 12;
  const std::vector<Eigen::Vector3d> inital_pos = MakeInitialPositions(
      {mpm_shift - box_width * 0.5, mpm_shift - box_width * 0.5,
       mpm_shift - box_width * 0.5},
      {mpm_shift + box_width * 0.5, mpm_shift + box_width * 0.5,
       mpm_shift + box_width * 0.5},
      num_per_dim);
  const int num_particles_per_box = inital_pos.size();
  const std::vector<Eigen::Vector3d> inital_vel(inital_pos.size(),
                                                Vector3d{0, 0, 0});
  const double particle_vol =
      box_width * box_width * box_width / num_particles_per_box;
  deformable_model.RegisterMpmParticle(inital_pos, inital_vel, particle_vol);

  MpmConfigParams mpm_config;
  mpm_config.substep_dt = FLAGS_substep;
  mpm_config.write_files = true;
  mpm_config.contact_stiffness = 1e3;
  mpm_config.contact_damping = 10;
  mpm_config.contact_friction_mu = 0.8;
  mpm_config.exact_line_search = true;
  deformable_model.SetMpmConfig(std::move(mpm_config));

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams visualize_params;
  visualize_params.show_mpm =
      geometry::DrakeVisualizerParams::ShowMpmOpt::kParticleMpm;
  auto& visualizer = geometry::DrakeVisualizerd::AddToBuilder(
      &builder, scene_graph, nullptr, visualize_params);

  // NOTE (changyu): MPM shortcut port shuould be explicit connected for
  // visualization.
  builder.Connect(
      plant.get_output_port(plant.deformable_model().mpm_output_port_index()),
      visualizer.mpm_input_port());

  // meshcat viz
  auto meshcat = std::make_shared<geometry::Meshcat>();
  if (FLAGS_write_files) {
    auto meshcat_params = drake::geometry::MeshcatVisualizerParams();
    meshcat_params.show_mpm =
        drake::geometry::MeshcatVisualizerParams::ShowMpmOpt::kParticleMpm;
    auto& meshcat_visualizer =
        drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
            &builder, scene_graph, meshcat, meshcat_params);
    visualization::ApplyVisualizationConfig(
        visualization::VisualizationConfig{
            .default_proximity_color = geometry::Rgba{1, 0, 0, 0.25},
            .enable_alpha_sliders = true,
        },
        &builder, nullptr, nullptr, nullptr, meshcat);

    builder.Connect(
        plant.get_output_port(plant.deformable_model().mpm_output_port_index()),
        meshcat_visualizer.mpm_input_port());
  }

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  auto& mutable_context = simulator.get_mutable_context();
  auto& plant_context = plant.GetMyMutableContextFromRoot(&mutable_context);
  const VectorXd external_normal_force = VectorXd::Ones(1) * 10.0;
  plant.get_actuation_input_port(right_box_model_instance)
      .FixValue(&plant_context, -external_normal_force);
  plant.get_actuation_input_port(left_box_model_instance)
      .FixValue(&plant_context, external_normal_force);
  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);

  if (FLAGS_write_files) {
    meshcat->StartRecording();
    simulator.AdvanceTo(FLAGS_simulation_time);
    meshcat->StopRecording();
    meshcat->PublishRecording();
    std::ofstream htmlFile("/home/xuchenhan/drake/hold.html");
    htmlFile << meshcat->StaticHtml();
    htmlFile.close();
  } else {
    simulator.AdvanceTo(FLAGS_simulation_time);
  }

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