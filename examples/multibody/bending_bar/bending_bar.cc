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

DEFINE_double(simulation_time, 30.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 0.0, "Desired real time rate.");
DEFINE_double(time_step, 1.0e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(bar_E, 1e5, "Young's modulus of the deformable body [Pa].");
DEFINE_double(bar_nu, 0.45,
              "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(bar_density, 1000,
              "Mass density of the deformable body [kg/m³].");
DEFINE_double(bar_beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(t0, 3.0, "Time to start pressing [s].");
DEFINE_double(tn, 20.0, "Time to stop pressing [s].");
DEFINE_double(tl, 25.0, "Time to release the bar [s].");
DEFINE_double(k, 100.0, "Slope of force [N/s].");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::Cylinder;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::geometry::Sphere;
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
using drake::multibody::SpatialInertia;
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
namespace bending_bar {
namespace {

/* A Leaf system that uses outputs a force signal to linear in time. */
class ForceControl : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ForceControl);

  /* Constructs a ForceControl system with the given parameters. The output
   force is k*(t-t0) + f0 and is capped at k(tn-t0)+f0. */
  ForceControl(double t0, double tn, double tl, double k)
      : t0_(t0), tn_(tn), tl_(tl), k_(k) {
    this->DeclareVectorOutputPort("gripper force", BasicVector<double>(1),
                                  &ForceControl::SetAppliedForce);
  }

 private:
  void SetAppliedForce(const Context<double>& context,
                       BasicVector<double>* output) const {
    double force = f0_;
    const double t = context.get_time();
    if (t > t0_) {
      force = k_ * (t - t0_) + f0_;
    } 
    if (t > tn_) {
      force = k_ * (tn_ - t0_) + f0_;
    } 
    if (t > tl_) {
      force = -100;
    }
    output->get_mutable_value() << force;
  }

  double t0_{0.0};
  double tn_{0.0};
  double tl_{0.0};
  double f0_{-9.81};
  double k_{20};
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

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
                                    geometry::internal::kRezHint, 0.005);

  const auto& collision_body = plant.AddRigidBody(
      "collision_body", SpatialInertia<double>::SolidCylinderWithMass(
                            1.0, 0.05, 0.3, Vector3d(0, 1, 0)));
  const auto& collision_joint = plant.AddJoint<PrismaticJoint>(
      "collision_joint", plant.world_body(), {}, collision_body,
      RigidTransformd(Vector3d(-0.5, -0.025, 0)), -Vector3d::UnitZ());
  plant.AddJointActuator("collision_joint_actuator", collision_joint);

  /* Set up collision and visualization geometries. */
  RigidTransformd X_WG(math::RollPitchYaw<double>(1.57, 0, 0),
                       Vector3d::Zero());
  Cylinder cylinder{0.05, 0.3};
  plant.RegisterCollisionGeometry(collision_body, X_WG, cylinder, "collision",
                                  rigid_proximity_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.2, 0.9, 0.2, 0.8));
  plant.RegisterVisualGeometry(collision_body, X_WG, cylinder, "visual",
                               illustration_props);

  /* Set up a deformable bar. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);
  DeformableModel<double>* deformable_model = owned_deformable_model.get();

  DeformableBodyConfig<double> bar_config;
  bar_config.set_youngs_modulus(FLAGS_bar_E);
  bar_config.set_poissons_ratio(FLAGS_bar_nu);
  bar_config.set_mass_density(FLAGS_bar_density);
  bar_config.set_stiffness_damping_coefficient(FLAGS_bar_beta);

  const std::string bar_vtk =
      FindResourceOrThrow("drake/examples/multibody/bending_bar/bar.vtk");
  auto bar_mesh = std::make_unique<Mesh>(bar_vtk, 0.1);
  const RigidTransformd X_WB = RigidTransformd::Identity();
  auto bar_instance =
      std::make_unique<GeometryInstance>(X_WB, std::move(bar_mesh), "bar");
  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  ProximityProperties deformable_proximity_props;
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);
  bar_instance->set_proximity_properties(deformable_proximity_props);

  auto bar_id = deformable_model->RegisterDeformableBody(
      std::move(bar_instance), bar_config, 0.1);
  deformable_model->SetWallBoundaryCondition(bar_id, Vector3d(0.001, 0, 0),
                                             Vector3d(1, 0, 0));
  plant.AddPhysicalModel(std::move(owned_deformable_model));

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  const auto& control =
      *builder.AddSystem<ForceControl>(FLAGS_t0, FLAGS_tn, FLAGS_tl, FLAGS_k);
  builder.Connect(control.get_output_port(), plant.get_actuation_input_port());

  /* It's essential to connect the vertex position port in DeformableModel to
   the source configuration port in SceneGraph when deformable bodies are
   present in the plant. */
  builder.Connect(
      deformable_model->vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  params.default_color = drake::geometry::Rgba(0.53, 0.42, 0.34, 1.0);
  params.role = geometry::Role::kIllustration;
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, nullptr,
                                           params);

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  /* Set initial conditions for the gripper. */
  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());
  collision_joint.set_translation(&plant_context, -0.5);

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  simulator.AdvanceTo(FLAGS_simulation_time);
  return 0;
}

}  // namespace
}  // namespace bending_bar
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase deformable body simulations in Drake.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::bending_bar::do_main();
}
