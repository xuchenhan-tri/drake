///
/// @brief  An fem example.
///
#include "drake/fem/fem_solver.h"
#include "drake/fem/eigen_sparse_matrix.h"
#include "drake/fem/eigen_conjugate_gradient_solver.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

#include <gflags/gflags.h>

// #include "drake/geometry/geometry_visualization.h"
// #include "drake/geometry/scene_graph.h"
#include "drake/systems/analysis/simulator_gflags.h"
// #include "drake/systems/framework/diagram.h"
// #include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace fem {

int DoMain() {
  FemSolver<double> fem(0.01);
  const BackwardEulerObjective<double>& objective = fem.get_objective();
  EigenConjugateGradientSolver<double> m(objective);
  // fem.AdvanceOneTimeStep();
  // std::cout << fem.get_dt() << std::endl;
  return 0;
}

}  // namespace fem
}  // namespace drake

int main(int argc, char** argv) {
  gflags::SetUsageMessage("A simple demonstration of pancake");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::fem::DoMain();
}
