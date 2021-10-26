#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
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
#include "drake/solvers/conex_solver.h"
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
DEFINE_double(simulation_time, 2.0, "Duration of the simulation in seconds.");
DEFINE_double(friction_coefficient, 0.2, "Friction coefficient.");
DEFINE_double(horizontal_force, 1.0, "External horizontal force, [N].");
DEFINE_bool(use_tamsi, true,
            "If true it uses TAMSI, otherwise MpConvexSolver.");

#include <iostream>
#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;

namespace drake {
namespace multibody {
namespace examples {
namespace push_box {
namespace {

using drake::multibody::contact_solvers::internal::MpConvexSolver;
using drake::multibody::contact_solvers::internal::MpConvexSolverParameters;
using drake::multibody::contact_solvers::internal::MpConvexSolverStats;
using Eigen::Vector3d;

int do_main() {
  // Build a generic multibody plant.
  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(
      &builder, std::make_unique<MultibodyPlant<double>>(FLAGS_time_step));

  const std::string model_file_name =
      "drake/examples/multibody/mp_convex_solver/push_box.sdf";
  const std::string full_name = FindResourceOrThrow(model_file_name);

  Parser parser(&plant);
  parser.AddModelFromFile(full_name);

  plant.mutable_gravity_field().set_gravity_vector(Vector3d(0.0, 0.0, -10.0));

  // We are done defining the model. Finalize.
  plant.Finalize();

  geometry::DrakeVisualizer::AddToBuilder(&builder, scene_graph);
  ConnectContactResultsToDrakeVisualizer(&builder, plant);

  // Create full Diagram for with the model.
  auto diagram = builder.Build();

  // Create a context for the diagram and extract the context for the
  // strandbeest model.
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(diagram_context.get());

  // Add external forcing.
  DRAKE_DEMAND(plant.num_actuators() == 1);
  plant.get_actuation_input_port().FixValue(&plant_context,
                                            Vector1d(FLAGS_horizontal_force));

  // Set initial conditions.
  const double phi0 = -1.0e-4;
  const auto& z_slider = plant.GetJointByName<PrismaticJoint>("z_slider");
  z_slider.set_translation(&plant_context, phi0);

  // Set contact solver.
  MpConvexSolver<double>* solver{nullptr};
  if (!FLAGS_use_tamsi) {
    solver =
        &plant.set_contact_solver(std::make_unique<MpConvexSolver<double>>());
    MpConvexSolverParameters params;
    params.alpha = 0.1;
    params.Rt_factor = 0.01;
    // Opopt: It fails very often.
    // params.solver_id = solvers::IpoptSolver::id();

    // Nlopt: "converges", but analytical ID errors are large.
    // params.solver_id = solvers::NloptSolver::id();

    // ScsSolver: Shows good performance/convergence.
    // params.solver_id = solvers::ScsSolver::id();


    params.solver_id = solvers::ConexSolver::id();

    // GurobiSolver.
    // Compile with: bazel run --config gurobi ....
    //params.solver_id = solvers::GurobiSolver::id();
    solver->set_parameters(params);
  }

  // Create a simulator and run the simulation.
  std::unique_ptr<systems::Simulator<double>> simulator =
      systems::MakeSimulatorFromGflags(*diagram, std::move(diagram_context));

  simulator->AdvanceTo(FLAGS_simulation_time);

  // Print some useful statistics.
  PrintSimulatorStatistics(*simulator);

  PRINT_VAR(z_slider.get_translation(plant_context));

  // Print contact solver stats.
  if (!FLAGS_use_tamsi) {
    const std::vector<MpConvexSolverStats>& stats_hist =
        solver->get_stats_history();
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    fmt::print("{:>18} {:>18} {:>18}  {:>18}\n", "num_contacts", "id_rel_err",
               "id_abs_error", "gamma_norm", "vs_max");
    for (const auto& s : stats_hist) {
      fmt::print("{:d} {:18.6g} {:18.6g} {:18.6g} {:18.6g}\n", s.num_contacts,
                 s.iteration_errors.id_rel_error,
                 s.iteration_errors.id_abs_error,
                 s.iteration_errors.gamma_norm,
                 s.iteration_errors.vs_max);
    }
  }

  return 0;
}
}  // namespace
}  // namespace push_box
}  // namespace examples
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("\nBox pushed by a constant external force.\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::examples::push_box::do_main();
}
