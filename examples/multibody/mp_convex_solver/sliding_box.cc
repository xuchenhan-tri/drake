#include <fstream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/contact_solvers/mp_convex_solver.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/nlopt_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(time_step, 1.0E-3,
              "If time_step > 0, the fixed-time step period (in seconds) of "
              "discrete updates for the plant (modeled as a discrete system). "
              "If time_step = 0, the plant is modeled as a continuous system "
              "and no contact forces are displayed.  time_step must be >= 0.");
DEFINE_double(simulation_time, 1.0, "Duration of the simulation in seconds.");
DEFINE_double(force, 0.0, "External force in x, [N]");
DEFINE_double(vx0, 1.0, "Initial horizontal velocity, [m/s]");
DEFINE_bool(use_tamsi, false,
            "If true it uses Drake's TAMSI solver, otherwise MpConvexSolver.");
// Compile with: bazel run --config gurobi ....            
DEFINE_string(solver, "scs", "Underlying solver. 'gurobi', 'scs'");

namespace drake {
namespace multibody {
namespace examples {
namespace mp_convex_solver {
namespace {

using drake::multibody::contact_solvers::internal::MpConvexSolver;
using drake::multibody::contact_solvers::internal::MpConvexSolverParameters;
using drake::multibody::contact_solvers::internal::MpConvexSolverStats;
using Eigen::Vector3d;
using systems::Context;

int do_main() {
  // Build a generic multibody plant.
  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(
      &builder, std::make_unique<MultibodyPlant<double>>(FLAGS_time_step));

  const std::string model_file_name =
      "drake/examples/multibody/mp_convex_solver/sliding_box.sdf";
  const std::string full_name = FindResourceOrThrow(model_file_name);

  Parser parser(&plant);
  parser.AddModelFromFile(full_name);

  const double gravity = 10.0;
  plant.mutable_gravity_field().set_gravity_vector(
      Vector3d(0.0, 0.0, -gravity));

  // We are done defining the model. Finalize.
  plant.Finalize();

  const auto& box = plant.GetBodyByName("box");

  geometry::DrakeVisualizerParams viz_params;
  viz_params.publish_period = FLAGS_time_step;
  geometry::DrakeVisualizer::AddToBuilder(&builder, scene_graph, nullptr,
                                          viz_params);

  // Create full Diagram for with the model.
  auto diagram = builder.Build();

  // Create a context for the diagram and extract the context for the
  // multibody plant model.
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(diagram_context.get());

  // Add external forcing.
  DRAKE_DEMAND(plant.num_actuators() == 1);
  plant.get_actuation_input_port().FixValue(&plant_context,
                                            Vector1d(FLAGS_force));
  // Set initial conditions.
  // N.B. The value of mu = 0.5 must be in sync with the model sliding_box.sdf.
  const double phi0 =
      FLAGS_time_step * 0.5 * std::abs(FLAGS_vx0);  // = dt * mu * ||vt||
  const auto& z_slider = plant.GetJointByName<PrismaticJoint>("z_slider");
  const auto& x_slider = plant.GetJointByName<PrismaticJoint>("x_slider");
  z_slider.set_translation(&plant_context, phi0);
  z_slider.set_translation_rate(&plant_context, 0.0);
  x_slider.set_translation_rate(&plant_context, FLAGS_vx0);

  // Set contact solver.
  MpConvexSolver<double>* solver{nullptr};
  if (!FLAGS_use_tamsi) {
    solver =
        &plant.set_contact_solver(std::make_unique<MpConvexSolver<double>>());
    MpConvexSolverParameters params;
    params.alpha = 1.0;
    params.Rt_factor = 0.01;
    // Opopt: It fails very often.
    // params.solver_id = solvers::IpoptSolver::id();

    // Nlopt: "converges", but analytical ID errors are large.
    // params.solver_id = solvers::NloptSolver::id();

    if (FLAGS_solver == "scs") {
      // ScsSolver: Shows good performance/convergence.
      params.solver_id = solvers::ScsSolver::id();
    } else if (FLAGS_solver == "gurobi") {
      // GurobiSolver.
      // Compile with: bazel run --config gurobi ....
      params.solver_id = solvers::GurobiSolver::id();
    } else {
      throw std::runtime_error("Solver not supported.");
    }
    solver->set_parameters(params);
  }

  // N.B. We create the simulator here by hand instead of using
  // MakeSimulatorFromGflags() so that we can set the monitor "before" calling
  // Simulator::Initialize() (done inside MakeSimulatorFromGflags()). We do this
  // so that the first initialization update, which leads to the very first
  // contact solve, also gets recorded by the monitor.
  auto simulator = std::make_unique<systems::Simulator<double>>(
      *diagram, std::move(diagram_context));
  const systems::SimulatorConfig config{
      FLAGS_simulator_integration_scheme,
      FLAGS_simulator_max_time_step,
      FLAGS_simulator_accuracy,
      FLAGS_simulator_use_error_control,
      FLAGS_simulator_target_realtime_rate,
      FLAGS_simulator_publish_every_time_step};
  systems::ApplySimulatorConfig(simulator.get(), config);

  std::ofstream file("sol.dat");
  simulator->set_monitor([&](const Context<double>& root_context) {
    const systems::Context<double>& ctxt =
        plant.GetMyContextFromRoot(root_context);
    const ContactResults<double>& contact_results =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            ctxt);
    DRAKE_DEMAND(contact_results.num_point_pair_contacts() == 1);
    const PointPairContactInfo<double>& point_pair_info =
        contact_results.point_pair_contact_info(0);
    const double sign =
        point_pair_info.bodyB_index() == box.index() ? 1.0 : -1.0;
    // Force as applied to the box B.
    const Vector3d f_Bc_W = point_pair_info.contact_force() * sign;
    const double vn = point_pair_info.separation_speed();
    const double slip = point_pair_info.slip_speed();

    const MpConvexSolverStats& stats = solver->get_iteration_stats();

    // time, x, xdot, z, zdot, ft, fn, vr, vn, id_rel_err, id_abs_err
    file << fmt::format(
        "{:20.8g} {:20.8g} {:20.8g} {:20.8g} {:20.8g} {:20.8g} {:20.8g} "
        "{:20.8g} {:20.8g} "
        "{:20.8g} {:20.8g}\n",
        ctxt.get_time(), x_slider.get_translation(ctxt),
        x_slider.get_translation_rate(ctxt), z_slider.get_translation(ctxt),
        z_slider.get_translation_rate(ctxt), f_Bc_W(0), f_Bc_W(2), slip, vn,
        stats.iteration_errors.id_rel_error,
        stats.iteration_errors.id_abs_error);
    return systems::EventStatus::Succeeded();
  });

  simulator->Initialize();
  simulator->AdvanceTo(FLAGS_simulation_time);
  file.close();

  // Print some useful statistics.
  PrintSimulatorStatistics(*simulator);

  // Print contact solver stats.
  if (!FLAGS_use_tamsi) {
    const std::vector<MpConvexSolverStats>& stats_hist =
        solver->get_stats_history();
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    fmt::print("{:>18} {:>18} {:>18}  {:>18}\n", "num_contacts", "id_rel_err",
               "id_abs_error", "gamma_norm", "vs_max");
    for (const auto& s : stats_hist) {
      fmt::print("{:d} {:20.8g} {:20.8g} {:20.8g} {:20.8g}\n", s.num_contacts,
                 s.iteration_errors.id_rel_error,
                 s.iteration_errors.id_abs_error, s.iteration_errors.gamma_norm,
                 s.iteration_errors.vs_max);
    }
  }

  return 0;
}
}  // namespace
}  // namespace mp_convex_solver
}  // namespace examples
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "\nSliding box case to test convex formulation of contact.\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::examples::mp_convex_solver::do_main();
}
