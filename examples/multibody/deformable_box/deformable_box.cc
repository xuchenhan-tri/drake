#include <memory>

#include <gflags/gflags.h>
#include <iostream>
#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
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
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/sine.h"

DEFINE_double(simulation_time, 10.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 5.0e-3,
              "Discrete time step for the system [s]. Must be "
              "positive.");
DEFINE_double(E, 1e4, "Young's modulus of the deformable objects [Pa].");
DEFINE_double(nu, 0.4, "Poisson ratio of the deformable objects, unitless.");
DEFINE_double(density, 1e3, "Mass density of the deformable objects [kg/m³].");
DEFINE_double(
    mass_damping, 0.001,
    "Mass damping coefficient [1/s]. The damping ratio contributed by this "
    "coefficient is inversely proportional to the frequency of the motion. "
    "Note that mass damping damps out rigid body motion and thus this "
    "coefficient should be kept small.");
DEFINE_double(
    stiffness_damping, 0.002,
    "Stiffness damping coefficient [s]. The damping ratio contributed by "
    "this coefficient is proportional to the frequency of the motion.");
DEFINE_double(min_gripper_force, 2,
              "The minimum force in the harmonic oscillation carried out by "
              "the gripper [N]. Must be positive.");
DEFINE_double(max_gripper_force, 17,
              "The maximum force in the harmonic oscillation carried out by "
              "the gripper [N]. Must be greater than `min_gripper_force`.");
DEFINE_double(grip_frequency, 2.0,
              "The frequency of the harmonic oscillation forces carried out "
              "by the gripper [Hz].");

namespace drake {
namespace multibody {
namespace examples {
namespace deformable_box {
namespace {

using math::RigidTransformd;
using multibody::internal::DeformableModel;

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;

  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_solver = "sap";
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlant(plant_config, &builder);

  /* Proximity properties for all bodies. */
  const CoulombFriction<double> surface_friction(1.0, 1.0);
  geometry::ProximityProperties proximity_props;

  geometry::AddContactMaterial({}, 1.0, surface_friction, &proximity_props);
  proximity_props.AddProperty(geometry::internal::kHydroGroup, geometry::internal::kRezHint, 1.0);
  geometry::Box ground{4, 4, 4};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -2});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                  "ground_collision", proximity_props);

  geometry::IllustrationProperties illus_prop;
  illus_prop.AddProperty("phong", "diffuse", Eigen::Vector4d(0.7, 0.5, 0.4, 0.5));
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground, "ground_visual",
                               std::move(illus_prop));

//   /* Set up a simple gripper. */
//   Parser parser(&plant);
//   std::string full_name =
//       FindResourceOrThrow("drake/examples/simple_gripper/simple_gripper.sdf");
//   parser.AddModelFromFile(full_name);
//   /* Add collision geometries. */
//   const math::RigidTransformd X_BG = math::RigidTransformd::Identity();
//   const Body<double>& left_finger = plant.GetBodyByName("left_finger");
//   const Body<double>& right_finger = plant.GetBodyByName("right_finger");
//   /* The size of the finger is set to match the visual geometries in
//    examples/simple_gripper/simple_gripper.sdf. */
//   plant.RegisterCollisionGeometry(left_finger, X_BG,
//                                   geometry::Box(0.007, 0.081, 0.028),
//                                   "left_finger_collision", proximity_props);
//   plant.RegisterCollisionGeometry(right_finger, X_BG,
//                                   geometry::Box(0.007, 0.081, 0.028),
//                                   "left_finger_collision", proximity_props);

  /* Add a deformable body. */
  multibody::fem::DeformableBodyConfig<double> deformable_config;
  deformable_config.set_youngs_modulus(FLAGS_E);
  deformable_config.set_poissons_ratio(FLAGS_nu);
  deformable_config.set_mass_density(FLAGS_density);
  deformable_config.set_mass_damping_coefficient(FLAGS_mass_damping);
  deformable_config.set_stiffness_damping_coefficient(FLAGS_stiffness_damping);

  auto deformable_model = std::make_unique<DeformableModel<double>>(&plant);
  constexpr double kRezHint = 0.02;
  /* Initial pose of the box. */
//   const math::RigidTransform<double> X_WB(Vector3<double>(0.03, 0.013, 0.06));
  const math::RigidTransform<double> X_WB(Vector3<double>(0.0, 0.0, 0.06));
  /* Side length of the deformable box. */
//   constexpr double kL = 0.06;
  std::string box_vtk =
      FindResourceOrThrow("drake/examples/multibody/deformable_box/box_coarse.vtk");
  auto box_instance = std::make_unique<geometry::GeometryInstance>(
      X_WB, std::make_unique<geometry::VolumeMeshShape>(box_vtk), "box");
  box_instance->set_proximity_properties(proximity_props);

  deformable_model->RegisterDeformableBody(std::move(box_instance),
                                           deformable_config, kRezHint);
  const DeformableModel<double>* deformable_model_ptr = deformable_model.get();
  plant.AddPhysicalModel(std::move(deformable_model));
  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  builder.Connect(
      deformable_model_ptr->get_vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));

  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

//   /* We use a force-controlled gripper to
//   1. compensate for gravity of the entire system and verify that the force
//      required to hold the system in place in the z-direction matches
//      expectation, and
//   2. "squeeze" the deformable box in the y-direction to show grasping with
//      friction as well as deformation of deformable objects under external
//      forces. */

//   /* The total mass of the system =
//      The mass of the gripper + the mass of the deformable box. */
//   const Body<double>& gripper_body = plant.GetBodyByName("body");
//   const double kGripperMass = gripper_body.default_mass() +
//                               left_finger.default_mass() +
//                               right_finger.default_mass();
//   const double kTotalMass =
//       kGripperMass + kL * kL * kL * FLAGS_density;  // [kg]
//   const double g = 9.81;                            // [m/s²]
//   /* The force magnitude in the positive z direction to compensate for gravity
//    of the system. */
//   const double kConstantZForce = kTotalMass * g;  // [N]

//   DRAKE_DEMAND(FLAGS_max_gripper_force > FLAGS_min_gripper_force);
//   DRAKE_DEMAND(FLAGS_min_gripper_force > 0);
//   const double kAmplitude =
//       (FLAGS_max_gripper_force - FLAGS_min_gripper_force) / 2.0;
//   const double kOffset = kAmplitude + FLAGS_min_gripper_force;
//   /* Here we are use the same Sine source to:
//     1. Generate a horizontal harmonic forcing of the finger with the prescribed
//        phase, amplitude and frequency.
//     2. Impose a constant vertical force to hold up the gripper. */
//   const Vector2<double> amplitudes(0, kAmplitude);
//   const Vector2<double> frequencies(0.0, FLAGS_grip_frequency);
//   const Vector2<double> phases(0, 3 * M_PI_2);  // Start with the minimum force.
//   const auto& harmonic_force = *builder.AddSystem<systems::Sine<double>>(
//       amplitudes, frequencies, phases);
//   const auto& constant_force =
//       *builder.AddSystem<systems::ConstantVectorSource<double>>(
//           Vector2<double>(kConstantZForce, kOffset));
//   const auto& adder = *builder.AddSystem<systems::Adder<double>>(2, 2);
//   /* Add up the constant force and the harmonic force source and supply the sum
//    to the actuation port. */
//   builder.Connect(harmonic_force.get_output_port(0), adder.get_input_port(0));
//   builder.Connect(constant_force.get_output_port(), adder.get_input_port(1));
//   builder.Connect(adder.get_output_port(0), plant.get_actuation_input_port());

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();

//   /* Set initial conditions for the gripper. */
//   auto& plant_context =
//       diagram->GetMutableSubsystemContext(plant, diagram_context.get());
//   const PrismaticJoint<double>& finger_slider =
//       plant.GetJointByName<PrismaticJoint>("finger_sliding_joint");
//   /* Set the initial position of the gripper to be of the same width as the
//    deformable box. */
//   finger_slider.set_translation(&plant_context, -kL);
//   finger_slider.set_translation_rate(&plant_context, 0);
//   const PrismaticJoint<double>& translate_joint =
//       plant.GetJointByName<PrismaticJoint>("translate_joint");
//   translate_joint.set_translation(&plant_context, 0.0);
//   translate_joint.set_translation_rate(&plant_context, 0.0);

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace deformable_box
}  // namespace examples
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("deformable box");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::examples::deformable_box::do_main();
}
