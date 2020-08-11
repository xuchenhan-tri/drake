///
/// @brief  An fem example.
///
#include "drake/fem/fem_system.h"
#include "drake/fem/obj_writer.h"

#include <cstdlib>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 10.0,
              "How long to simulate the system");
namespace drake {
namespace fem {

int DoMain() {
    systems::DiagramBuilder<double> builder;
    double dt = 0.1;
    auto* fem_system = builder.AddSystem<FemSystem<double>>(dt);
    fem_system->AddRectangularBlock(10,10,3,0.01);
    auto* obj_writer = builder.AddSystem<ObjWriter<double>>(*fem_system);
    builder.Connect(fem_system->get_output_port(0),
                     obj_writer->get_input_port(0));
    auto diagram = builder.Build();
    auto context = diagram->CreateDefaultContext();
    auto simulator =
            systems::MakeSimulatorFromGflags(*diagram, std::move(context));
    simulator->AdvanceTo(FLAGS_simulation_time);
  return 0;
}

}  // namespace fem
}  // namespace drake

int main(int argc, char** argv) {
  gflags::SetUsageMessage("A simple demonstration of pancake");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::fem::DoMain();
}
