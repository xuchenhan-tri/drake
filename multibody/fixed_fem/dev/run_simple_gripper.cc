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
 #include "drake/common/profiler.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/contact_solvers/pgs_solver.h"
#include "drake/multibody/fixed_fem/dev/deformable_body_config.h"
#include "drake/multibody/fixed_fem/dev/deformable_model.h"
#include "drake/multibody/fixed_fem/dev/deformable_rigid_manager.h"
#include "drake/multibody/fixed_fem/dev/deformable_visualizer.h"
#include "drake/multibody/fixed_fem/dev/mesh_utilities.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/sine.h"

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

DEFINE_double(grip_width, 0.06,
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
  const double dt = 5e-3;
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlantSceneGraph(&builder, dt);

  // Size of the deformable block.
  const double L = 0.06;
  const geometry::Box box(L, L, L);

  /* Set up the deformable block. */
  const math::RigidTransform<double> p_WB(Vector3<double>(0.03, 0.013, 0.06));
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
  auto deformable_model = std::make_unique<DeformableModel<double>>(&plant);
  deformable_model->RegisterDeformableBody(nonlinear_bar_geometry, "Corotated",
                                           nonlinear_bar_config,
                                           corotated_proximity_props);
  const DeformableModel<double>* deformable_model_ptr = deformable_model.get();
  plant.AddPhysicalModel(std::move(deformable_model));

  Parser parser(&plant);
  std::string full_name =
      FindResourceOrThrow("drake/examples/simple_gripper/simple_gripper.sdf");
  parser.AddModelFromFile(full_name);

  plant.Finalize();
  auto pgs_solver =
      std::make_unique<contact_solvers::internal::PgsSolver<double>>();
  auto deformable_rigid_manager =
      std::make_unique<DeformableRigidManager<double>>();
  const DeformableRigidManager<double>* deformable_rigid_manager_ptr =
      deformable_rigid_manager.get();
  deformable_rigid_manager->SetContactSolver(std::move(pgs_solver));
  plant.SetDiscreteUpdateManager(std::move(deformable_rigid_manager));
  deformable_rigid_manager_ptr->RegisterCollisionObjects(scene_graph);

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
  const Vector2<double> phases(0, 3 * M_PI_2);
  const auto& harmonic_force = *builder.AddSystem<systems::Sine<double>>(
      amplitudes, frequencies, phases);
  const auto& constant_force =
      *builder.AddSystem<systems::ConstantVectorSource<double>>(
          Vector2<double>(f0, FLAGS_gripper_force));
  const auto& adder = *builder.AddSystem<systems::Adder<double>>(2, 2);
  builder.Connect(harmonic_force.get_output_port(0), adder.get_input_port(0));
  builder.Connect(constant_force.get_output_port(), adder.get_input_port(1));

  builder.Connect(adder.get_output_port(0), plant.get_actuation_input_port());

  auto& visualizer = *builder.AddSystem<DeformableVisualizer>(
      1.0 / 250.0, deformable_model_ptr->names(),
      deformable_model_ptr->reference_configuration_meshes());
  builder.Connect(deformable_model_ptr->get_vertex_positions_output_port(),
                  visualizer.get_input_port());

  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, context.get());
  // Get joints so that we can set initial conditions.
  const PrismaticJoint<double>& finger_slider =
      plant.GetJointByName<PrismaticJoint>("finger_sliding_joint");
  // Set initial position of the left finger.
  finger_slider.set_translation(&plant_context, -FLAGS_grip_width);
  // Set the initial height of the gripper and its initial velocity.
  const PrismaticJoint<double>& translate_joint =
      plant.GetJointByName<PrismaticJoint>("translate_joint");
  translate_joint.set_translation(&plant_context, 0.0);
  translate_joint.set_translation_rate(&plant_context, 0.0);

  auto simulator =
      systems::MakeSimulatorFromGflags(*diagram, std::move(context));
  simulator->AdvanceTo(FLAGS_simulation_time);
  std::cout << TableOfAverages() << "\n";
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
