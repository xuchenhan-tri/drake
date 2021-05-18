/**
A demo to showcase deformable solver.

To run the demo. First ensure that you have the visualizer and the demo itself
built:

```
bazel build //tools:drake_visualizer
bazel build //multibody/fixed_fem/dev:run_simple_gripper
```

Then, in one terminal, launch the visualizer
```
bazel-bin/tools/drake_visualizer
```

In another terminal, launch the demo
```
bazel-bin/multibody/fixed_fem/dev/run_simple_gripper
``` */
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/contact_solvers/pgs_solver.h"
#include "drake/multibody/fixed_fem/dev/deformable_body_config.h"
#include "drake/multibody/fixed_fem/dev/deformable_visualizer.h"
#include "drake/multibody/fixed_fem/dev/mesh_utilities.h"
#include "drake/multibody/fixed_fem/dev/softsim_system.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/sine.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_vector_source.h"

DEFINE_double(simulation_time, 10.0,
              "How many seconds to simulate the system.");
DEFINE_double(dx, 0.02,
              "Distance between consecutive vertices in the tet mesh, with "
              "unit m. Must be positive and smaller than or equal to 0.01 to "
              "provide enough resolution for reasonable accuracy. The smaller "
              "this number is, the higher the resolution of the mesh will be.");
DEFINE_double(E, 1e5,
              "Young's modulus of the deformable objects, with unit Pa");
DEFINE_double(nu, 0.4, "Poisson ratio of the deformable objects, unitless");
DEFINE_double(density, 1e4,
              "Mass density of the deformable objects, with unit kg/m³");
DEFINE_double(alpha, 0.001,
              "Mass damping coefficient. The damping ratio contributed by this "
              "coefficient is inversely proportional to the frequency of the "
              "motion. Note that mass damping damps out rigid body "
              "motion and thus this coefficient should be kept small. ");
DEFINE_double(
    beta, 0.002,
    "Stiffness damping coefficient. The damping ratio contributed by this "
    "coefficient is proportional to the frequency of the motion.");

DEFINE_double(grip_width, 0.065,
              "The initial distance between the gripper fingers. [m].");
// Gripping force.
DEFINE_double(gripper_force, 40,
              "The force to be applied by the gripper. [N]. "
              "A value of 0 indicates a fixed grip width as set with option "
              "grip_width.");

// Parameters for squeezing the deformable.
DEFINE_double(amplitude, 30,
              "The amplitude of the harmonic oscillations "
              "carried out by the gripper. [m].");
DEFINE_double(frequency, 2.0,
              "The frequency of the harmonic oscillations "
              "carried out by the gripper. [Hz].");
namespace drake {
namespace multibody {
namespace fixed_fem {

int DoMain() {
  systems::DiagramBuilder<double> builder;
  auto* scene_graph = builder.AddSystem<geometry::SceneGraph<double>>();
  const double dt = 1e-3;
  auto* plant = builder.AddSystem<multibody::MultibodyPlant<double>>(dt);
  plant->RegisterAsSourceForSceneGraph(scene_graph);
  plant->set_deformable_solver(std::make_unique<SoftsimSystem<double>>(plant));
  SoftsimSystem<double>& deformable_solver =
      static_cast<SoftsimSystem<double>&>(plant->mutable_deformable_solver());

  // Size of the deformable block.
  const double L = 0.06;
  const geometry::Box box(L, L, L);

  /* Set up the deformable block. */
  const math::RigidTransform<double> p_WB(Vector3<double>(0.03, 0.01, 0.06));
  DeformableBodyConfig<double> nonlinear_bar_config;
  nonlinear_bar_config.set_youngs_modulus(FLAGS_E);
  nonlinear_bar_config.set_poisson_ratio(FLAGS_nu);
  nonlinear_bar_config.set_mass_damping_coefficient(FLAGS_alpha);
  nonlinear_bar_config.set_stiffness_damping_coefficient(FLAGS_beta);
  nonlinear_bar_config.set_mass_density(FLAGS_density);
  nonlinear_bar_config.set_material_model(MaterialModel::kCorotated);
  const geometry::VolumeMesh<double> nonlinear_bar_geometry =
      MakeDiamondCubicBoxVolumeMesh<double>(box, FLAGS_dx, p_WB);
  const CoulombFriction<double> surface_friction(1.0, 1.0);
  geometry::ProximityProperties corotated_proximity_props;
  // Use default stiffness and dissipation.
  geometry::AddContactMaterial({}, {}, {}, surface_friction,
                               &corotated_proximity_props);
  // Use the Corotated constitutive model.
  deformable_solver.RegisterDeformableBody(nonlinear_bar_geometry, "Corotated",
                                           nonlinear_bar_config,
                                           corotated_proximity_props);

  Parser parser(plant);
  std::string full_name =
      FindResourceOrThrow("drake/examples/simple_gripper/simple_gripper.sdf");
  parser.AddModelFromFile(full_name);

  plant->set_contact_solver(
      std::make_unique<contact_solvers::internal::PgsSolver<double>>());
  plant->Finalize();

  // The mass of the gripper in simple_gripper.sdf + the mass of the deformable
  // block.
  const double mass = 1.0890 + L * L * L * FLAGS_density;  // kg.
  const double g = 9.81;
  const double f0 = mass * g;  // Force amplitude, Newton.

  // Notice we are using the same Sine source to:
  //   1. Generate a harmonic forcing of the finger with the prescribed phase,
  //      amplitude and frequency.
  //   2. Impose a constant force to hold up the gripper. There will be some
  //      drift over time but we don't care.
  const Vector2<double> amplitudes(0, FLAGS_amplitude);
  const Vector2<double> frequencies(0.0, FLAGS_frequency);
  const Vector2<double> phases(0, 3*M_PI_2);
  const auto& harmonic_force = *builder.AddSystem<systems::Sine<double>>(
      amplitudes, frequencies, phases);
  const auto& constant_force =
      *builder.AddSystem<systems::ConstantVectorSource<double>>(
          Vector2<double>(f0, FLAGS_gripper_force));
  const auto& adder = *builder.AddSystem<systems::Adder<double>>(2, 2);
  builder.Connect(harmonic_force.get_output_port(0), adder.get_input_port(0));
  builder.Connect(constant_force.get_output_port(), adder.get_input_port(1));

  builder.Connect(adder.get_output_port(0),
                  plant->get_actuation_input_port());

  auto& visualizer = *builder.AddSystem<DeformableVisualizer>(
      1.0 / 250.0, deformable_solver.names(), deformable_solver.meshes());
  builder.Connect(plant->get_deformable_vertex_positions_output_port(),
                  visualizer.get_input_port());

  DRAKE_DEMAND(!!plant->get_source_id());
  builder.Connect(scene_graph->get_query_output_port(),
                  plant->get_geometry_query_input_port());
  builder.Connect(
      plant->get_geometry_poses_output_port(),
      scene_graph->get_source_pose_port(plant->get_source_id().value()));
  geometry::DrakeVisualizerd::AddToBuilder(&builder, *scene_graph);

  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  auto& plant_context =
      diagram->GetMutableSubsystemContext(*plant, context.get());
  // Get joints so that we can set initial conditions.
  const PrismaticJoint<double>& finger_slider =
      plant->GetJointByName<PrismaticJoint>("finger_sliding_joint");
  // Set initial position of the left finger.
  finger_slider.set_translation(&plant_context, -FLAGS_grip_width);
  // Set the initial height of the gripper and its initial velocity.
  const PrismaticJoint<double>& translate_joint =
      plant->GetJointByName<PrismaticJoint>("translate_joint");
  translate_joint.set_translation(&plant_context, 0.0);
  translate_joint.set_translation_rate(&plant_context, 0.0);

  auto simulator =
      systems::MakeSimulatorFromGflags(*diagram, std::move(context));
  simulator->AdvanceTo(FLAGS_simulation_time);
  return 0;
}
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Demonstration of contact solver working with deformable objects.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::fixed_fem::DoMain();
}
