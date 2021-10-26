#include <fstream>

#include <fmt/format.h>
#include <gflags/gflags.h>

#include "drake/geometry/drake_visualizer.h"
#include "drake/multibody/contact_solvers/unconstrained_primal_solver.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/tree/planar_joint.h"
#include "drake/multibody/tree/rigid_body.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/multibody/plant/compliant_contact_computation_manager.h"
#include "drake/math/wrap_to.h"

namespace drake {

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

using drake::multibody::CompliantContactComputationManager;
using drake::multibody::contact_solvers::internal::UnconstrainedPrimalSolver;
using drake::multibody::contact_solvers::internal::
    UnconstrainedPrimalSolverIterationMetrics;
using drake::multibody::contact_solvers::internal::
    UnconstrainedPrimalSolverParameters;
using drake::multibody::contact_solvers::internal::
    UnconstrainedPrimalSolverStats;
using math::RigidTransformd;
using math::RotationMatrixd;
using multibody::AddMultibodyPlantSceneGraph;
using multibody::ContactResults;
using multibody::CoulombFriction;
using multibody::MultibodyPlant;
using multibody::PlanarJoint;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::SpatialVelocity;
using multibody::UnitInertia;
using systems::Context;
using systems::DiagramBuilder;
using systems::Simulator;

namespace examples {
namespace multibody {
namespace rimless_wheel {
namespace {

DEFINE_double(simulation_time, 4.0, "Duration of the simulation in seconds.");
DEFINE_double(mbt_dt, 0.02,
              "Discrete time step. Defaults to 0.0 for a continuous system.");
DEFINE_bool(visualize, true, "Whether to visualize or not.");

// Model parameters.
DEFINE_double(mass, 1.0, "Mass in kg.");
DEFINE_double(spoke_length, 1.0, "Spoke length in m.");
DEFINE_double(spoke_radius, 0.02, "Spoke radius in m.");
DEFINE_int32(num_legs, 6, "Number of legs.");
DEFINE_double(stiffness, 1.0e5, "Point contact stiffness in N/m.");
DEFINE_double(tau_dissipation, 0.02,
              "Linear dissipation time scale in seconds.");
DEFINE_double(friction, 1.0, "Friction coefficient.");
DEFINE_double(slope, 5.0, "Slope in degrees.");
DEFINE_double(vy0, 5.0, "Initial x velocity in m/s.");
DEFINE_double(z0, -1.0, "Initial z position. If negative not used.");

// Solver.
DEFINE_string(solver, "primal",
              "Underlying solver. 'tamsi', 'primal', 'geodesic'");
DEFINE_bool(use_sdf_query, false, "Use SDF instead of penetration query.");
DEFINE_int32(time_method, 2,
             "1: Explicit Euler, 2: Symplectic Euler, 3: Implicit Euler, 4: "
             "Midpoint rule.");
DEFINE_double(contact_theta, 1.0, "Theta method parameter for contact.");

const Vector4<double> red(1.0, 0.0, 0.0, 1.0);
const Vector4<double> orange(1.0, 0.55, 0.0, 1.0);
const Vector4<double> purple(204.0 / 255, 0.0, 204.0 / 255, 1.0);
const Vector4<double> green(0, 153.0 / 255, 0, 1.0);
const Vector4<double> cyan(51 / 255, 1.0, 1.0, 1.0);
const Vector4<double> pink(1.0, 204.0 / 255, 204.0 / 255, 1.0);

void BuildRimlessWheelModel(MultibodyPlant<double>* plant) {
  const double mass = FLAGS_mass;
  const double length = FLAGS_spoke_length;
  const double spoke_radius = FLAGS_spoke_radius;

  // Visual parameters.
  const double center_mass_radius = 0.1;
  const int num_legs = FLAGS_num_legs;
  const double alpha = 2 * M_PI / num_legs;  // angle between legs.

  const Vector3d p_BoBcm_B = Vector3d::Zero();
  // Center mass.
  //const UnitInertia<double> G_BBcm_B = UnitInertia<double>::SolidCylinder(
    //  center_mass_radius, center_mass_radius, Vector3d::UnitX());

  const UnitInertia<double> G_BBcm_B = UnitInertia<double>::SolidSphere(1e-3);
  const SpatialInertia<double> M_BBcm_B(mass, p_BoBcm_B, G_BBcm_B);

  // Create a rigid body B with the mass properties of a uniform solid block.
  const RigidBody<double>& body = plant->AddRigidBody("body", M_BBcm_B);

  // Center body visual. Place cylinder canonical z axis on the body's x axis.
  const RigidTransformd X_BC(RotationMatrixd::MakeYRotation(M_PI_2));
  plant->RegisterVisualGeometry(
      body, X_BC,
      geometry::Cylinder(center_mass_radius,
                         center_mass_radius /* 2D example, not relevant */),
      "body_visual", orange);

  // Collision properties.
  geometry::ProximityProperties props;
  props.AddProperty(geometry::internal::kMaterialGroup,
                    geometry::internal::kPointStiffness, FLAGS_stiffness);
  props.AddProperty(geometry::internal::kMaterialGroup, "dissipation_time_constant",
                    FLAGS_tau_dissipation);
  props.AddProperty(geometry::internal::kMaterialGroup,
                    geometry::internal::kFriction,
                    CoulombFriction<double>(FLAGS_friction, FLAGS_friction));

  // Add legs collision and visual.
  for (int leg = 0; leg < num_legs; ++leg) {
    // Collision. For the collision model we only need to specify spheres at the
    // end of each spoke.
    const RotationMatrixd R_BS = RotationMatrixd::MakeXRotation(leg * alpha);
    const Vector3d p_BS = R_BS * Vector3d(0, 0, -length);
    const RigidTransformd X_BS(p_BS);
    const std::string collision_name = "leg_collision_" + std::to_string(leg);
    plant->RegisterCollisionGeometry(body, X_BS, geometry::Sphere(spoke_radius),
                                     collision_name, props);

    // Leg visual.
    const std::string leg_spoke_name = "leg_spoke_" + std::to_string(leg);
    const Vector3d p_BSpoke = R_BS * Vector3d(0, 0, -length / 2.0);
    const RigidTransformd X_BSpoke(R_BS, p_BSpoke);
    plant->RegisterVisualGeometry(body, X_BSpoke,
                                  geometry::Cylinder(spoke_radius, length),
                                  leg_spoke_name, red);
    const std::string leg_tip_name = "leg_tip_" + std::to_string(leg);
    plant->RegisterVisualGeometry(body, X_BS, geometry::Sphere(spoke_radius),
                                  leg_tip_name, red);
  }

  // Make 2D.
  const auto Ry = math::RotationMatrixd::MakeYRotation(-M_PI / 2.0);
  const auto Xy = math::RigidTransformd(Ry, Vector3d::Zero());
  const Vector3d damping = Vector3d::Zero();
  plant->AddJoint<PlanarJoint>("planar", plant->world_body(), Xy, body, Xy,
                               damping);
}

void SetRimlessWheelInitialConditions(const MultibodyPlant<double>& plant,
                                      Context<double>* context) {
  const double length = FLAGS_spoke_length;
  const double spoke_radius = FLAGS_spoke_radius;
  //const auto& body = plant.GetRigidBodyByName("body");
  const double z0 = FLAGS_z0 > 0 ? FLAGS_z0 : length + spoke_radius;
  const RigidTransformd X_WB(Vector3d(0, 0, z0));
  //plant.SetFreeBodyPose(context, body, X_WB);
  const auto& planar = plant.GetJointByName<PlanarJoint>("planar");
  //const double gamma = FLAGS_slope * M_PI / 180.0;
  //const double alpha = M_PI / FLAGS_num_legs;
  planar.set_pose(context, Vector2d(z0, 0.0), 0);

  // Initial angular velocity to enforce rolling.
  const double w0 = FLAGS_vy0 / (length + spoke_radius);

  //const SpatialVelocity<double> V_WB(Vector3d(w0, 0, 0),
  //                                   Vector3d(0, FLAGS_vy0, 0));
  //plant.SetFreeBodySpatialVelocity(context, body, V_WB);
  planar.set_translational_velocity(context, Vector2d(0.0, FLAGS_vy0));
  planar.set_angular_velocity(context, w0);
}

void AddGround(MultibodyPlant<double>* plant) {  
  // Collision properties.
  const double ground_stiffness = 1.0e40;
  const double ground_dissipation = 0.0;
  geometry::ProximityProperties props;
  props.AddProperty(geometry::internal::kMaterialGroup,
                    geometry::internal::kPointStiffness, ground_stiffness);
  props.AddProperty(geometry::internal::kMaterialGroup, "dissipation_time_constant",
                    ground_dissipation);
  props.AddProperty(geometry::internal::kMaterialGroup,
                    geometry::internal::kFriction,
                    CoulombFriction<double>(FLAGS_friction, FLAGS_friction));

  // Pose of the ground at given slope.
  const double gamma = -FLAGS_slope * M_PI / 180.0;
  const RigidTransformd X_WG(RotationMatrixd::MakeXRotation(gamma));
  plant->RegisterCollisionGeometry(plant->world_body(), X_WG,
                                   geometry::HalfSpace(), "ground_collision",
                                   props);
  plant->RegisterVisualGeometry(plant->world_body(), X_WG,
                                geometry::HalfSpace(), "ground_visual", green);
}

int do_main() {
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      AddMultibodyPlantSceneGraph(&builder, FLAGS_mbt_dt);
  BuildRimlessWheelModel(&plant);
  AddGround(&plant);
  plant.Finalize();

  // Swap discrete contact solver.
  UnconstrainedPrimalSolver<double>* primal_solver{nullptr};
  CompliantContactComputationManager<double>* manager{nullptr};
  if (FLAGS_solver == "primal") {
    manager = &plant.set_discrete_update_manager(
        std::make_unique<CompliantContactComputationManager<double>>(
            std::make_unique<UnconstrainedPrimalSolver<double>>()));
    primal_solver =
        &manager->mutable_contact_solver<UnconstrainedPrimalSolver>();
    UnconstrainedPrimalSolverParameters params;
    params.abs_tolerance = 1.0e-6;
    params.rel_tolerance = 1.0e-5;
    params.Rt_factor = 1.0e-3;
    params.max_iterations = 300;
    params.ls_alpha_max = 1.0;
    params.verbosity_level = 1;
    params.theta = FLAGS_contact_theta;
    params.log_stats = true;
    primal_solver->set_parameters(params);
  }

  double theta_q{1.0}, theta_qv{0.0}, theta_v{1.0};
  switch (FLAGS_time_method) {
    case 1:  // Explicit Euler
      theta_q = 0;
      theta_qv = 0;
      theta_v = 0;
      break;
    case 2:  // Symplectic Euler
      theta_q = 1;
      theta_qv = 0;
      theta_v = 1;
      break;
    case 3:  // Implicit Euler
      theta_q = 1;
      theta_qv = 1;
      theta_v = 1;
      break;
    case 4:  // Midpoint rule
      theta_q = 0.5;
      theta_qv = 0.5;
      theta_v = 0.5;
      break;
    default:
      throw std::runtime_error("Invalide time_method option");
  }

  manager->set_theta_q(theta_q);    // how v is computed in equation of q.
  manager->set_theta_qv(theta_qv);  // how q is computed in equation of v.
  manager->set_theta_v(theta_v);    // how v is computed in equation of v.
  plant.set_use_sdf_query(FLAGS_use_sdf_query);

  // Add visualization.
  if (FLAGS_visualize) {
    geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);
    ConnectContactResultsToDrakeVisualizer(&builder, plant);
  }

  // Done creating the full model diagram.
  auto diagram = builder.Build();

  // Create context and set initial conditions.
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(diagram_context.get());
  SetRimlessWheelInitialConditions(plant, &plant_context);

  // Simulate.
  std::unique_ptr<Simulator<double>> simulator =
      MakeSimulatorFromGflags(*diagram, std::move(diagram_context));

  std::ofstream sol_file("sol.dat");
  const auto& planar = plant.GetJointByName<PlanarJoint>("planar");
  sol_file << fmt::format("time x y th th_wrap vx vy thdot nc pe ke E slip1 slip2\n");
  simulator->set_monitor([&](const systems::Context<double>& root_context) {
    const systems::Context<double>& ctxt =
        plant.GetMyContextFromRoot(root_context);
    const ContactResults<double>& contact_results =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            ctxt);
    const int nc = contact_results.num_point_pair_contacts();

    double slip1 = -1;  // invalid value.
    if (contact_results.num_point_pair_contacts() >= 1) {
      const auto& info = contact_results.point_pair_contact_info(0);
      slip1 = info.slip_speed();
    }

    double slip2 = -1;  // invalid value.
    if (contact_results.num_point_pair_contacts() == 2) {
      const auto& info = contact_results.point_pair_contact_info(1);
      slip2 = info.slip_speed();
    }

    const double ke = plant.CalcKineticEnergy(ctxt);
    const double pe = plant.CalcPotentialEnergy(ctxt);

    const Vector2d p_WB = planar.get_translation(ctxt);
    const double theta = planar.get_rotation(ctxt);
    const Vector2d v_WB = planar.get_translational_velocity(ctxt);
    const double theta_dot = planar.get_angular_velocity(ctxt);

    const double gamma = FLAGS_slope * M_PI / 180.0;
    const double alpha = M_PI / FLAGS_num_legs;
    const double theta_wrap = math::wrap_to(theta, -(alpha-gamma), alpha+gamma);

    // time, ke, vt, vn, phi_plus, phi_minus.
    sol_file << fmt::format("{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
                            ctxt.get_time(), p_WB.x(), p_WB.y(), theta,
                            theta_wrap, v_WB.x(), v_WB.y(), theta_dot, nc, ke,
                            pe, ke + pe, slip1, slip2);
    return systems::EventStatus::Succeeded();
  });

  simulator->AdvanceTo(FLAGS_simulation_time);

  // Print some useful statistics.
  PrintSimulatorStatistics(*simulator);

  // Log solver stats.
  if (primal_solver) {
    primal_solver->LogIterationsHistory("log.dat");
  }

  return 0;
}

}  // namespace
}  // namespace rimless_wheel
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("MultibodyPlant model of a rimless wheel.");

  // Default parameters when using continuous integration.
  FLAGS_simulator_accuracy = 1e-4;
  FLAGS_simulator_max_time_step = 1e-1;
  FLAGS_simulator_integration_scheme = "implicit_euler";

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::rimless_wheel::do_main();
}
