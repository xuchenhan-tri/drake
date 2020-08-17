///
/// @brief  An fem example.
///
#include <cstdlib>
#include <memory>
#include <chrono>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/fem/fem_system.h"
#include "drake/fem/obj_writer.h"
#include "drake/fem/deformable_visualizer.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 5, "How long to simulate the system");
namespace drake {
namespace fem {

int DoMain() {
  systems::DiagramBuilder<double> builder;
  const double dt = 0.01;
  auto* fem_system = builder.AddSystem<FemSystem<double>>(dt);
  const double mesh_spacing = 0.005;
  const int nx = 20;
  const int ny = 20;
  const int nz = 2;
  FemConfig config;
  config.density = 1e3;
  config.youngs_modulus = 2e4;
  config.poisson_ratio = 0.4;
  config.mass_damping = 0;
  config.stiffness_damping = 0;
  auto velocity_transform = [](int vertex_index,
                               EigenPtr<Matrix3X<double>> vel) {
    vel->col(vertex_index).setZero();
  };
  auto bc = [](int index, const Matrix3X<double>& initial_pos,
               EigenPtr<Matrix3X<double>> velocity) {
    if (initial_pos.col(index).norm() <= 0.01) {
      velocity->col(index).setZero();
    }
  };
  bool use_vtk = true;
  if (use_vtk) {
    const char* kModelPath = "drake/fem/models/pancake.vtk";
      const std::string vtk = FindResourceOrThrow(kModelPath);
      auto position_transform = []([[maybe_unused]] int vertex_index,
                                   [[maybe_unused]] EigenPtr<Matrix3X<double>> pos) {};
      fem_system->AddObjectFromVtkFile(vtk, config, position_transform,
                                       velocity_transform, bc);

  } else {
    auto position_transform = [nx, ny, nz, mesh_spacing](
                                  int vertex_index,
                                  EigenPtr<Matrix3X<double>> pos) {
      pos->col(vertex_index) -=
          Vector3<double>(static_cast<double>(nx) / 2 * mesh_spacing,
                          static_cast<double>(ny) / 2 * mesh_spacing,
                          static_cast<double>(nz) / 2 * mesh_spacing);
    };
    fem_system->AddRectangularBlock(nx, ny, nz, mesh_spacing, config,
                                    position_transform, velocity_transform, bc);
  }
    auto& visualizer = *builder.AddSystem<DeformableVisualizer>(
            dt, "pancake", fem_system->get_indices());
    builder.Connect(*fem_system, visualizer);
//    auto* obj_writer = builder.AddSystem<ObjWriter<double>>(*fem_system);
//  builder.Connect(fem_system->get_output_port(0),
//                  obj_writer->get_input_port(0));
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  auto simulator =
      systems::MakeSimulatorFromGflags(*diagram, std::move(context));
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    simulator->AdvanceTo(FLAGS_simulation_time);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
  return 0;
}

}  // namespace fem
}  // namespace drake

int main(int argc, char** argv) {
  gflags::SetUsageMessage("A simple demonstration of pancake");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::fem::DoMain();
}
