#include <gflags/gflags.h>

#include "drake/multibody/mpm/transfer.h"
#include "drake/tools/performance/fixture_common.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using Eigen::Vector3d;
using Eigen::Vector3f;

/* Sample N^3 points in the box [0, 1] x [0, 1] x [0, 1]*/
std::vector<Vector3<double>> SamplePoints(int N) {
  std::vector<Vector3<double>> points;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        points.push_back(
            Vector3<double>(i / (N - 1.0), j / (N - 1.0), k / (N - 1.0)));
      }
    }
  }
  return points;
}

class TransferBenchmark : public benchmark::Fixture {
 public:
  TransferBenchmark() {
    tools::performance::AddMinMaxStatistics(this);
    this->Unit(benchmark::kMillisecond);
    this->MinWarmUpTime(2.0);
  }

  void SetUp(benchmark::State& state) {  // NOLINT(runtime/references)
    // Number of inputs.
    const int num_particles_per_dim = state.range(0);
    DRAKE_DEMAND(num_particles_per_dim > 0);
    const std::vector<Vector3d>& positions =
        SamplePoints(num_particles_per_dim);
    multibody::fem::DeformableBodyConfig<double> config;
    ParticleData<float> particle_data;
    particle_data.AddParticles(positions, /* total volume */ 1e-3, config);
    particles_.data = particle_data;
    particles_.Sort(grid_);
    grid_.Allocate(particles_.sorter);
  }

 protected:
  double dx_{0.2};
  float dt_{1e-2};
  SparseGrid<float> grid_{dx_};
  Particles<float> particles_;
  Transfer<float> transfer_{dt_, &grid_, &particles_};
};

BENCHMARK_DEFINE_F(TransferBenchmark, ScalarP2G)
(benchmark::State& state) {  // NOLINT
  for (auto _ : state) {
    transfer_.ScalarParticleToGrid(particles_, &grid_);
  }
}
// The Args are { num_nodes_per_dim, particles_per_cell }.
BENCHMARK_REGISTER_F(TransferBenchmark, ScalarP2G)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

BENCHMARK_DEFINE_F(TransferBenchmark, SimdP2G)
(benchmark::State& state) {  // NOLINT
  for (auto _ : state) {
    transfer_.SimdParticleToGrid(particles_, &grid_);
  }
}
// The Args are { num_nodes_per_dim, particles_per_cell }.
BENCHMARK_REGISTER_F(TransferBenchmark, SimdP2G)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
