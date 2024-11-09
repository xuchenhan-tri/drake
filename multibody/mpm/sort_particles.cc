#include "sort_particles.h"

#include <bitset>

#include "ips2ra/ips2ra.hpp"
#include "spgrid.h"

#include "drake/common/drake_assert.h"
#include "drake/multibody/mpm/grid_data.h"
#include "drake/multibody/mpm/math.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T, typename SpGrid>
void SortParticlePositions(const SpGrid& spgrid, std::vector<Vector3<T>>* q_WPs,
                           T dx) {
  DRAKE_DEMAND(q_WPs != nullptr);
  const auto& flags = spgrid.flags();
  const int log2_max_grid_size = flags.log2_max_grid_size;
  const int data_bits = flags.data_bits;

  const int num_particles = q_WPs->size();
  std::vector<uint64_t> particle_sorters(num_particles);
  const int index_bits = 64 - 3 * log2_max_grid_size;
  const int zero_page_bits = 64 - data_bits - 3 * log2_max_grid_size;
  for (int p = 0; p < num_particles; ++p) {
    const Vector3<int> base_node = ComputeBaseNode<T>(q_WPs->at(p) / dx);
    uint64_t base_node_offsets =
        spgrid.CoordinateToOffset(base_node[0], base_node[1], base_node[2]);
    uint64_t data_indices = p;
    particle_sorters[p] = (base_node_offsets << zero_page_bits) + data_indices;
  }
  ips2ra::sort(particle_sorters.begin(), particle_sorters.end());
  std::vector<Vector3<T>> result(num_particles);
  for (int p = 0; p < ssize(particle_sorters); ++p) {
    int data_indices = particle_sorters[p] & ((uint64_t(1) << index_bits) - 1);
    result[p] = q_WPs->at(data_indices);
  }
  *q_WPs = result;
}

// Explicit instantiation for float
template void SortParticlePositions<float, SpGrid<GridData<float>>>(
    const SpGrid<GridData<float>>&, std::vector<Vector3<float>>*, float);

// Explicit instantiation for double
template void SortParticlePositions<double, SpGrid<GridData<double>>>(
    const SpGrid<GridData<double>>&, std::vector<Vector3<double>>*, double);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
