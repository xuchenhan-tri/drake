#include <fstream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/contact_solvers/mp_convex_solver.h"
#include "drake/multibody/contact_solvers/unconstrained_primal_solver.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/conex_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/nlopt_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(time_step, 1.0E-3,
              "If time_step > 0, the fixed-time step period (in seconds) of "
              "discrete updates for the plant (modeled as a discrete system). "
              "If time_step = 0, the plant is modeled as a continuous system "
              "and no contact forces are displayed.  time_step must be >= 0.");
DEFINE_double(simulation_time, 0.15, "Duration of the simulation in seconds.");
DEFINE_bool(use_tamsi, false,
            "If true it uses TAMSI, otherwise MpConvexSolver.");
// If using Gurobi, compile with: bazel run --config gurobi ....
DEFINE_string(solver, "conex",
              "Underlying solver. 'scs', 'gurobi', 'conex','primal'");

namespace drake {
namespace multibody {
namespace examples {
namespace mp_convex_solver {
namespace {

using drake::multibody::contact_solvers::internal::MpConvexSolver;
using drake::multibody::contact_solvers::internal::MpConvexSolverParameters;
using drake::multibody::contact_solvers::internal::MpConvexSolverStats;

using drake::multibody::contact_solvers::internal::UnconstrainedPrimalSolver;
using drake::multibody::contact_solvers::internal::
    UnconstrainedPrimalSolverParameters;
using drake::multibody::contact_solvers::internal::
    UnconstrainedPrimalSolverStats;

using Eigen::Vector3d;
using systems::Context;

int do_main() {
  // Build a generic multibody plant.
  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(
      &builder, std::make_unique<MultibodyPlant<double>>(FLAGS_time_step));

  const std::string model_file_name =
      "drake/examples/multibody/mp_convex_solver/conveyor_belt.sdf";
  const std::string full_name = FindResourceOrThrow(model_file_name);

  Parser parser(&plant);
  parser.AddModelFromFile(full_name);

  plant.mutable_gravity_field().set_gravity_vector(Vector3d(0.0, 0.0, -10.0));

  // We are done defining the model. Finalize.
  plant.Finalize();

  // ConnectDrakeVisualizer(&builder, scene_graph);
  geometry::DrakeVisualizerParams viz_params;
  viz_params.publish_period = FLAGS_time_step;
  geometry::DrakeVisualizer::AddToBuilder(&builder, scene_graph, nullptr,
                                          viz_params);

  // Create full Diagram for with the model.
  auto diagram = builder.Build();

  // Create a context for the diagram and extract the context for the
  // multibody model.
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(diagram_context.get());

  // No external forcing.
  DRAKE_DEMAND(plant.num_actuators() == 0);

  const double vt0 = -0.1;  // Conveyor belt velocity.
  const double mu = 0.5;    // Friction. Must be in sync with the model.

  // Set initial conditions.
  const double phi0 = mu * std::abs(vt0) * FLAGS_time_step;
  const auto& z_slider = plant.GetJointByName<PrismaticJoint>("z_slider");
  z_slider.set_translation(&plant_context, phi0);
  z_slider.set_translation_rate(&plant_context, 0.0);

  // Ground horizontal velocity.
  const auto& belt_slider = plant.GetJointByName<PrismaticJoint>("belt_slider");
  belt_slider.set_translation_rate(&plant_context, vt0);
  belt_slider.set_translation(&plant_context, 0.0);

  // Set contact solver.
  MpConvexSolver<double>* solver{nullptr};
  if (!FLAGS_use_tamsi && FLAGS_solver != "primal") {
    solver =
        &plant.set_contact_solver(std::make_unique<MpConvexSolver<double>>());
    MpConvexSolverParameters params;
    params.alpha = 1.0;
    params.Rt_factor = 0.01;
    // Opopt: It fails very often.
    // params.solver_id = solvers::IpoptSolver::id();

    // Nlopt: "converges", but analytical ID errors are large.
    // params.solver_id = solvers::NloptSolver::id();


    if (FLAGS_solver == "conex") {
      // ScsSolver: Shows good performance/convergence.
      params.solver_id = solvers::ConexSolver::id();
    } else {
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
    }
    
    solver->set_parameters(params);
  }

  UnconstrainedPrimalSolver<double>* primal_solver{nullptr};
  if (!FLAGS_use_tamsi && FLAGS_solver == "primal") {
    primal_solver = &plant.set_contact_solver(
        std::make_unique<UnconstrainedPrimalSolver<double>>());
    UnconstrainedPrimalSolverParameters params;
    params.abs_tolerance = 1.0e-6;
    params.rel_tolerance = 1.0e-6;
    params.max_iterations = 10;
    primal_solver->set_parameters(params);
  }

  // Create a simulator and run the simulation.
  std::unique_ptr<systems::Simulator<double>> simulator =
      systems::MakeSimulatorFromGflags(*diagram, std::move(diagram_context));

  const auto& box = plant.GetBodyByName("box");

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
    const Vector3d f_Bc_W = point_pair_info.contact_force() * sign;
    const double vn = point_pair_info.separation_speed();
    const double slip = point_pair_info.slip_speed();

    if (solver) {
      const MpConvexSolverStats& stats = solver->get_iteration_stats();

      // time, z, zdot, ft, fn, vr, vn, id_rel_err, id_abs_err
      file << fmt::format(
          "{:20.8g} {:20.8g} {:20.8g} {:20.8g} {:20.8g} {:20.8g} {:20.8g} "
          "{:20.8g} {:20.8g}\n",
          ctxt.get_time(), z_slider.get_translation(ctxt),
          z_slider.get_translation_rate(ctxt), f_Bc_W(0), f_Bc_W(2), slip, vn,
          stats.iteration_errors.id_rel_error,
          stats.iteration_errors.id_abs_error);
    }

    if (primal_solver) {
      // time, z, zdot, ft, fn, vr, vn
      file << fmt::format(
          "{:20.8g} {:20.8g} {:20.8g} {:20.8g} {:20.8g} {:20.8g} {:20.8g}\n",
          ctxt.get_time(), z_slider.get_translation(ctxt),
          z_slider.get_translation_rate(ctxt), f_Bc_W(0), f_Bc_W(2), slip, vn);
    }

    return systems::EventStatus::Succeeded();
  });

  simulator->AdvanceTo(FLAGS_simulation_time);
  file.close();

  // Print some useful statistics.
  PrintSimulatorStatistics(*simulator);

  // Print contact solver stats.
  if (!FLAGS_use_tamsi) {
    if (solver) {
      const std::vector<MpConvexSolverStats>& stats_hist =
          solver->get_stats_history();
      std::cout << std::string(80, '-') << std::endl;
      std::cout << std::string(80, '-') << std::endl;
      fmt::print("{:>18} {:>18} {:>18}  {:>18}  {:>18}\n", "num_contacts",
                 "id_rel_err", "id_abs_error", "gamma_norm", "solve_time");
      for (const auto& s : stats_hist) {
        fmt::print("{:d} {:20.8g} {:20.8g} {:20.8g} {:20.8g}\n", s.num_contacts,
                   s.iteration_errors.id_rel_error,
                   s.iteration_errors.id_abs_error,
                   s.iteration_errors.gamma_norm, s.solver_time);
      }
    }

    if (primal_solver) {

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
      "\nTest case with a box sitting on top of a running conveyor belt used "
      "to evaluate our convex formulation of contact.\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::examples::mp_convex_solver::do_main();
}
