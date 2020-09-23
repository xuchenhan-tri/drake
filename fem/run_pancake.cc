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

DEFINE_double(simulation_time, 10.0, "How long to simulate the system");
DEFINE_bool(use_pancake, false,
            "Whether to simulate a pancake geometry or a rectangular block.");
DEFINE_bool(
    use_drake_visualizer, true,
    "Whether to visualize with Drake visualizer or save the frames to file.");
DEFINE_double(E, 1e4, "Young's modulus of the deformable object.");
DEFINE_double(nu, 0.35, "Poisson ratio of the deformable object");
DEFINE_double(density, 1e3, "Mass density of the deformable object");
DEFINE_double(alpha, 0.2, "Mass damping coefficient.");
DEFINE_double(beta, 0.002, "Stiffness damping coefficient");

namespace drake {
namespace fem {

int DoMain() {
  systems::DiagramBuilder<double> builder;
  const double dt = 1.0 / 250.0;
  auto* fem_system = builder.AddSystem<FemSystem<double>>(dt);
  MaterialConfig config;
  config.density = FLAGS_density;
  config.youngs_modulus = FLAGS_E;
  config.poisson_ratio = FLAGS_nu;
  config.mass_damping = FLAGS_alpha;
  config.stiffness_damping = FLAGS_beta;
  if (FLAGS_use_pancake) {
    const char* kModelPath = "drake/fem/models/pancake.vtk";
    const std::string vtk = FindResourceOrThrow(kModelPath);
    auto position_transform =
        [](int vertex_index,
           EigenPtr<Matrix3X<double>> pos) {
        Vector3<double> q = pos->col(vertex_index);
        double tmp_x = q(1);
        q(1) = -q(2);
        q(2) = tmp_x;
        pos->col(vertex_index) = q;
    };
    fem_system->AddObjectFromVtkFile(vtk, config, position_transform);

    // Make a ground at z = -0.2 to catch the pancake.
    auto ground_ls = std::make_unique<HalfSpace<double>>(
        Vector3<double>(0, 0, -0.2), Vector3<double>(0, 0, 1));
    auto ground =
        std::make_unique<CollisionObject<double>>(std::move(ground_ls));
    fem_system->AddCollisionObject(std::move(ground));
  } else {
      const double mesh_spacing = 0.007;
      const int nx = 9;
      const int ny = 9;
      const int nz = 9;
    // Move the center of the rectangular block to the origin.
    auto position_transform1 = [nx, ny, nz, mesh_spacing](
                                  int vertex_index,
                                  EigenPtr<Matrix3X<double>> pos) {
      pos->col(vertex_index) -=
          Vector3<double>(static_cast<double>(nx) / 2 * mesh_spacing,
                          static_cast<double>(ny) / 2 * mesh_spacing,
                          static_cast<double>(nz) / 2 * mesh_spacing);
    };
    // Fix the nodes whose initial positions are near the center of the block.
    auto bc = [](int index, const Matrix3X<double>& initial_pos) {
      if (initial_pos.col(index).norm() <= 0.01) {
          return true;
      }
      return false;
    };
    // Make a simple rectangular block geometry.
    fem_system->AddRectangularBlock(nx, ny, nz, mesh_spacing, config,
                                    position_transform1, nullptr, bc);
  }
  // Finalize when all objects have been added.
  fem_system->Finalize();
  if (FLAGS_use_drake_visualizer) {
    auto& visualizer = *builder.AddSystem<DeformableVisualizer>(
        dt, "pancake", fem_system->get_mesh_base());
    builder.Connect(*fem_system, visualizer);
  } else {
    auto* obj_writer = builder.AddSystem<ObjWriter<double>>(*fem_system);
    builder.Connect(fem_system->get_output_port(0),
                    obj_writer->get_input_port(0));
  }
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  auto simulator =
      systems::MakeSimulatorFromGflags(*diagram, std::move(context));
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  simulator->AdvanceTo(FLAGS_simulation_time);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Total simulation time: " << elapsed_seconds.count() << "s\n";
  return 0;
}

}  // namespace fem
}  // namespace drake

int main(int argc, char** argv) {
  gflags::SetUsageMessage("A simple demonstration of pancake");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::fem::DoMain();
}
