///
/// @brief  An fem example.
///
#include <chrono>
#include <cstdlib>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/fem/collision_object.h"
#include "drake/fem/deformable_visualizer.h"
#include "drake/fem/fem_system.h"
#include "drake/fem/half_space.h"
#include "drake/fem/obj_writer.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 10, "How long to simulate the system");
namespace drake {
namespace fem {

int DoMain() {
  systems::DiagramBuilder<double> builder;
  const double dt = 1.0 / 600.0;
  auto* fem_system = builder.AddSystem<FemSystem<double>>(dt);

  auto ground_ls = std::make_unique<HalfSpace<double>>(
      Vector3<double>(0, -0.15, 0), Vector3<double>(0, 1, 0));
  auto ground = std::make_unique<CollisionObject<double>>(std::move(ground_ls));
  fem_system->AddCollisionObject(std::move(ground));
  //  auto pusher_ls = std::make_unique<HalfSpace<double>>(
  //      Vector3<double>(0, -0.06, 0), Vector3<double>(0, 1, 0));
  //  auto pusher_update = [](double time, CollisionObject<double>* cb) {
  //      double translation_velocity = 0.1;
  //      Vector3<double> translation(0, time * translation_velocity, 0);
  //      cb->set_translation(translation);
  //  };
  //    auto pusher =
  //    std::make_unique<CollisionObject<double>>(std::move(pusher_ls),
  //    pusher_update);
  //  fem_system->AddCollisionObject(std::move(pusher));

  const double mesh_spacing = 0.005;
  const int nx = 2;
  const int ny = 2;
  const int nz = 2;
  FemConfig config;
  config.density = 1e3;
  config.youngs_modulus = 4e3;
  config.poisson_ratio = 0.45;
  config.mass_damping = 0.0;
  config.stiffness_damping = 0.00;
  //    config.density = 1e3;
  //    config.youngs_modulus = 1e4;
  //    config.poisson_ratio = 0.4;
  //    config.mass_damping = 4;
  //    config.stiffness_damping = 0.0;
  auto velocity_transform = [](int vertex_index,
                               EigenPtr<Matrix3X<double>> vel) {
    vel->col(vertex_index).setZero();
  };
  auto bc = [](int index, const Matrix3X<double>& initial_pos,
               EigenPtr<Matrix3X<double>> velocity) {
    unused(index);
    unused(initial_pos);
    unused(velocity);
    //    if (initial_pos.col(index).norm() <= 0.01) {
    //      velocity->col(index).setZero();
    //    }
  };
  bool use_vtk = true;
  if (use_vtk) {
    const char* kModelPath = "drake/fem/models/pancake.vtk";
    const std::string vtk = FindResourceOrThrow(kModelPath);
    auto position_transform =
        []([[maybe_unused]] int vertex_index,
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
#if 0
  auto& visualizer = *builder.AddSystem<DeformableVisualizer>(
      dt, "pancake", fem_system->get_indices());
  builder.Connect(*fem_system, visualizer);
#else
  auto* obj_writer = builder.AddSystem<ObjWriter<double>>(*fem_system);
  builder.Connect(fem_system->get_output_port(0),
                  obj_writer->get_input_port(0));
  builder.Connect(fem_system->get_output_port(1),
                  obj_writer->get_input_port(1));
#endif
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
