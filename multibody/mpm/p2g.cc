#include <chrono>
#include <iostream>

#include "transfer.h"

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using Eigen::Matrix3f;
using Eigen::Vector3f;

/* Creates a grid with `num_nodes_per_dim` nodes in each dimension. Samples
 `particles_per_cell` particles around a single grid node.
 @param[in] dx grid spacing (meter).
 @param[out] particles Sampled particle data. */
void SetUp(int num_nodes_per_dim, int particles_per_cell, float dx,
           ParticleData<float>* particles) {
  // Create the particles.
  for (int i = 0; i < num_nodes_per_dim; ++i) {
    for (int j = 0; j < num_nodes_per_dim; ++j) {
      for (int k = 0; k < num_nodes_per_dim; ++k) {
        const Vector3f base_node(dx * i, dx * j, dx * k);
        for (int p = 0; p < particles_per_cell; ++p) {
          const Vector3f x =
              base_node + static_cast<float>(p) * dx /
                              (static_cast<float>(particles_per_cell) + 1.0) *
                              Vector3f::Ones();
          float m = 1.0;
          auto v = Vector3f::Ones();
          auto F = Matrix3f::Identity();
          auto C = Matrix3f::Zero();
          auto P = Matrix3f::Zero();
          BSplineWeights<float> bspline(x, dx);
          particles->particles.emplace_back(Particle<float>(m, x, v, F, C, P, bspline)); 
        }
      }
    }
  }
}

int do_main() {
  /* Total number of particles is 32^3 * 8 = 262k. */
  int num_nodes_per_dim = 32;
  int particles_per_cell = 8;
  const float dx = 0.01;
  const float dt = 0.002;
  ParticleData<float> particles;
  SparseGrid<float> grid(dx);

  SetUp(num_nodes_per_dim, particles_per_cell, dx, &particles);
  Transfer<float> transfer(dt, &grid, &particles);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 300; ++i) {
    transfer.ParticleToGrid(false);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Each time step takes: " << duration.count() / 300.0 * 1000.0
            << " milliseconds" << std::endl;
  return 0;
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::multibody::mpm::internal::do_main();
}