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
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/demultiplexer.h"

DEFINE_double(simulation_time, 8.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e4, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1e3, "Mass density of the deformable body [kg/m³].");
DEFINE_double(beta, 0.1,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(target_speed, 5, "The target speed of the rotor. [rad/s].");
DEFINE_double(start, 5, "Start time of the control signal [s].");
DEFINE_double(end, 20, "End time of the control signal [s].");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::Cylinder;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::geometry::Shape;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::RevoluteJoint;
using drake::multibody::SpatialInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::Demultiplexer;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace {

class Square final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Square)

  // @param[in] target_speeds the square wave target_speed. (unitless)
  // @param[in] start the start time of the on wave. (seconds)
  // @param[in] end the end time of the on wave. (seconds)
  Square(double target_speed, double start, double end)
      : target_speed_(target_speed), start_(start), end_(end) {
    DRAKE_THROW_UNLESS(start > 0);
    DRAKE_THROW_UNLESS(target_speed > 0);
    DRAKE_THROW_UNLESS(end > start);
    this->DeclareVectorOutputPort("Square Wave Output", 1,
                                  &Square::CalcValueOutput);
    this->DeclareVectorInputPort("Rotor State", BasicVector<double>(1));
  }

 private:
  void CalcValueOutput(const Context<double>& context,
                       BasicVector<double>* output) const {
    Eigen::VectorBlock<VectorX<double>> output_block =
        output->get_mutable_value();
    const double measured_speed =
        EvalVectorInput(context, GetInputPort("Rotor State").get_index())
            ->get_value()[0];
    double desired_speed = 0;
    const double time = context.get_time();
    const double duration = end_ - start_;
    if (time >= start_ && time < start_ + 0.5 * duration) {
      desired_speed = (time - start_) / (0.5 * duration) * target_speed_;
    } else if (time > start_ + 0.5 * duration && time < end_) {
      desired_speed = (end_ - time) / (0.5 * duration) * target_speed_;
    }
    const double k = 10;
    output_block[0] = k * (desired_speed - measured_speed);
  }

  const double target_speed_{};
  const double start_{};
  const double end_{};
};

void RegisterAndFixDeformableBodyTo(const Body<double>& rigid_body,
                                    const Shape& rigid_shape,
                                    const RigidTransformd& X_WD, int i,
                                    DeformableModel<double>* model) {
  DeformableBodyConfig<double> deformable_config;
  deformable_config.set_youngs_modulus(FLAGS_E);
  deformable_config.set_poissons_ratio(FLAGS_nu);
  deformable_config.set_mass_density(FLAGS_density);
  deformable_config.set_stiffness_damping_coefficient(FLAGS_beta);

  const std::string bar_vtk =
      FindResourceOrThrow("drake/examples/multibody/blades/bar_186.vtk");
  /* Load the geometry and scale it down to 10%. */
  const double scale = 0.1;
  auto bar_mesh = std::make_unique<Mesh>(bar_vtk, scale);
  /* Set the initial pose of the torus such that its bottom face is touching
   the ground. */
  auto bar_instance = std::make_unique<GeometryInstance>(
      X_WD, std::move(bar_mesh), "deformable_bar" + std::to_string(i));

  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  ProximityProperties deformable_proximity_props;
  const CoulombFriction<double> surface_friction(1.0, 1.0);
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);
  bar_instance->set_proximity_properties(deformable_proximity_props);

  /* Registration of all deformable geometries ostensibly requires a
   resolution hint parameter that dictates how the shape is tessellated. In
   the case of a `Mesh` shape, the resolution hint is unused because the shape
   is already tessellated. */
  // TODO(xuchenhan-tri): Though unused, we still asserts the resolution hint
  // is positive. Remove the requirement of a resolution hint for meshed
  // shapes.
  const double unused_resolution_hint = 1.0;
  DeformableBodyId body_id = model->RegisterDeformableBody(
      std::move(bar_instance), deformable_config, unused_resolution_hint);
  model->AddFixedConstraint(body_id, rigid_body, X_WD, rigid_shape,
                            RigidTransformd::Identity());
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  const Body<double>& rotor =
      plant.AddRigidBody("rotor", SpatialInertia<double>::SolidCylinderWithMass(
                                      1.0, 0.1, 0.2, Vector3d::UnitZ()));
  const auto& spin_joint = plant.AddJoint<RevoluteJoint>(
      "spin", plant.world_body(), {}, rotor, {}, Vector3d::UnitZ());
  plant.AddJointActuator("spin_actuator", spin_joint);

  /* Set up a visual geometry. */
  Cylinder rotor_geometry{0.1, 0.2};
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.7, 0.5, 0.4, 0.8));
  plant.RegisterVisualGeometry(rotor, RigidTransformd::Identity(),
                               rotor_geometry, "rotor_visual",
                               std::move(illustration_props));

  /* Set up a deformable blade. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);
  DeformableModel<double>* deformable_model = owned_deformable_model.get();
  for (int i = 0; i < 4; ++i) {
    const RigidTransformd X_RD(RollPitchYawd(M_PI_2, 0, 0),
                               Vector3<double>(0.0, 0.05, 0.0));
    const RigidTransformd X_WR(RollPitchYawd(0, 0, i * M_PI_2),
                               Vector3<double>::Zero());
    RegisterAndFixDeformableBodyTo(rotor, rotor_geometry, X_WR * X_RD, i,
                                   deformable_model);
  }

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
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  const auto& demux = *builder.AddSystem<Demultiplexer>(2);
  builder.Connect(plant.get_state_output_port(), demux.get_input_port());

  const auto& square_force =
      *builder.AddSystem<Square>(FLAGS_target_speed, FLAGS_start, FLAGS_end);
  builder.Connect(demux.get_output_port(1), square_force.get_input_port());
  builder.Connect(square_force.get_output_port(),
                  plant.get_actuation_input_port());

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
  gflags::SetUsageMessage("Stress stiffening.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
