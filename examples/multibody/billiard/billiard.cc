#include <fstream>

#include <fmt/format.h>
#include <gflags/gflags.h>

#include "drake/geometry/drake_visualizer.h"
#include "drake/math/wrap_to.h"
#include "drake/multibody/contact_solvers/unconstrained_primal_solver.h"
#include "drake/multibody/plant/compliant_contact_computation_manager.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/tree/planar_joint.h"
#include "drake/multibody/tree/rigid_body.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/multibody/tree/linear_spring_damper.h"
#include "drake/multibody/tree/one_dimensional_spring_damper.h"

namespace drake {

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

using drake::multibody::world_index;
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
using multibody::LinearSpringDamper;
using multibody::OneDimensionalSpringDamper;
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

DEFINE_double(simulation_time, 5.0, "Duration of the simulation in seconds.");
DEFINE_double(mbt_dt, 0.02,
              "Discrete time step. Defaults to 0.0 for a continuous system.");
DEFINE_bool(visualize, true, "Whether to visualize or not.");
DEFINE_bool(visualize_forces, true, "Whether to visualize forces or not.");

// Model parameters.
// From Wikipedia: According to World Pool-Billiard Association equipment
// specifications, the weight may be from 51⁄2 to 6.0 oz (160–170 g) with a
// diameter of 21⁄4 in (57 mm), plus or minus 0.005 in (0.127 mm).
DEFINE_double(mass1, 0.5, "Mass ball 1 in kg.");
DEFINE_double(radius, 0.05, "Billard ball radius in m.");
DEFINE_double(stiffness, 1.0e4, "Point contact stiffness in N/m.");
DEFINE_double(tau_dissipation, 0.02,
              "Linear dissipation time scale in seconds.");
DEFINE_double(friction, 1.0, "Billiard balls friction coefficient.");
DEFINE_double(ground_friction, 1.0, "Friction coefficient.");
DEFINE_double(slope, 0.0, "Slope in degrees.");
DEFINE_double(y0, 0.1, "Initial x in m.");
DEFINE_double(vy0, 0.0, "Initial x velocity in m/s.");
DEFINE_double(spring_stiffness, 100.0,
              "Attach a spring to ball 1 with this stiffness. If zero, no "
              "spring is attached.");

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

void BuildModel(MultibodyPlant<double>* plant) {  
  const double radius = FLAGS_radius;

  const Vector3d p_BoBcm_B = Vector3d::Zero();
  const UnitInertia<double> G_BBcm_B = UnitInertia<double>::SolidSphere(radius);
  const SpatialInertia<double> M1_BBcm_B(FLAGS_mass1, p_BoBcm_B, G_BBcm_B);

  // Create a rigid body B with the mass properties of a uniform solid block.
  const RigidBody<double>& body = plant->AddRigidBody("body", M1_BBcm_B);

  // Add linear srping.
  if (FLAGS_spring_stiffness != 0) {
    plant->AddForceElement<OneDimensionalSpringDamper>(
        plant->world_body(), body, FLAGS_spring_stiffness, 0.0);

    // LinearSpringDamper(const Body<T>& bodyA, const Vector3<double>& p_AP,
    //                 const Body<T>& bodyB, const Vector3<double>& p_BQ,
    //               double free_length, double stiffness, double damping);
  }

  // Visual
  const RigidTransformd X_BC = RigidTransformd::Identity();
  plant->RegisterVisualGeometry(body, X_BC, geometry::Sphere(radius),
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

  plant->RegisterCollisionGeometry(body, X_BC, geometry::Sphere(radius),
                                   "body_collision", props);

  // Make 2D.
  const auto Ry = math::RotationMatrixd::MakeYRotation(-M_PI / 2.0);
  const auto Xy = math::RigidTransformd(Ry, Vector3d::Zero());
  const Vector3d damping = Vector3d::Zero();
  plant->AddJoint<PlanarJoint>("planar", plant->world_body(), Xy, body, Xy,
                               damping);
}

void SetInitialConditions(const MultibodyPlant<double>& plant,
                          Context<double>* context) {  
  // Estimate steady state penetration.
  const double th = FLAGS_slope * M_PI / 180.0;
  const double g_vert = 9.81 * cos(th);
  const double phi1 = FLAGS_mass1 * g_vert / FLAGS_stiffness;
  const double z1 = FLAGS_radius - phi1;  
  const Vector2d p_WB1(z1, FLAGS_y0);
  const auto& planar = plant.GetJointByName<PlanarJoint>("planar");
  planar.set_pose(context, p_WB1, 0);

  // Initial angular velocity to enforce rolling.
  const double w1 = FLAGS_vy0 / z1;
  const Vector2d v_WB1(0.0, FLAGS_vy0);
  planar.set_translational_velocity(context, v_WB1);
  planar.set_angular_velocity(context, w1);  
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
  props.AddProperty(
      geometry::internal::kMaterialGroup, geometry::internal::kFriction,
      CoulombFriction<double>(FLAGS_ground_friction, FLAGS_ground_friction));

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
  BuildModel(&plant);
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
    //params.tau_slip = 1.0e-7;
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
    if (FLAGS_visualize_forces) {
      ConnectContactResultsToDrakeVisualizer(&builder, plant);
    }
  }

  // Done creating the full model diagram.
  auto diagram = builder.Build();

  // Create context and set initial conditions.
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(diagram_context.get());
  SetInitialConditions(plant, &plant_context);

  // Simulate.
  std::unique_ptr<Simulator<double>> simulator =
      MakeSimulatorFromGflags(*diagram, std::move(diagram_context));

  std::ofstream sol_file("sol.dat");
  const auto& planar = plant.GetJointByName<PlanarJoint>("planar");
  const auto& body = plant.GetBodyByName("body");
  sol_file << fmt::format(
      "time x1 y1 th1 vx1 vy1 thdot1 pe ke E nc "
      "f1w_y f1w_z slip1 d1\n");
  simulator->set_monitor([&](const systems::Context<double>& root_context) {
    const systems::Context<double>& ctxt =
        plant.GetMyContextFromRoot(root_context);
    const ContactResults<double>& contact_results =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            ctxt);
    const int nc = contact_results.num_point_pair_contacts();
    // Since we use an sdf_threshold = 0.5, we always have three contacts (even
    // if some forces are zero)
    DRAKE_DEMAND(nc <= 1);

    std::vector<Vector3d> fBi_W(3, Vector3d::Zero());
    std::vector<double> slip(3, 0.0);
    std::vector<double> depth(3, 0.0);
    for (int i = 0; i < nc; ++i)    {
      const auto& info = contact_results.point_pair_contact_info(i);

      DRAKE_DEMAND(info.bodyA_index() == body.index() ||
                   info.bodyB_index() == body.index());

      fBi_W[0] = info.contact_force();
      slip[0] = info.slip_speed();
      depth[0] = info.point_pair().depth;
    }

    const double ke = plant.CalcKineticEnergy(ctxt);
    const double pe = plant.CalcPotentialEnergy(ctxt);

    const Vector2d p_WB = planar.get_translation(ctxt);
    const double theta = planar.get_rotation(ctxt);
    const Vector2d v_WB = planar.get_translational_velocity(ctxt);
    const double theta_dot = planar.get_angular_velocity(ctxt);

    // time x y th vx vy thdot pe ke E nc slip
    sol_file << fmt::format(
        "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
        ctxt.get_time(), p_WB.x(), p_WB.y(), theta, v_WB.x(), v_WB.y(),
        theta_dot, pe, ke, ke + pe, nc,
        // Contact forces.
        fBi_W[0].y(), fBi_W[0].z(),
        // Slip velocities.
        slip[0],
        // Penetration depths.
        depth[0]);
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
