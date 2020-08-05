///
/// @brief  An fem example.
///
#include "drake/fem/fem_solver.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

#include <gflags/gflags.h>

#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace fem {

int DoMain() {
  FemSolver<double> fem(0.01);
  std::cout << fem.dt() << std::endl;
  return 0;
}

}  // namespace fem
}  // namespace drake

int main(int argc, char** argv) {
  gflags::SetUsageMessage("A simple demonstration of pancake");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::fem::DoMain();
}
