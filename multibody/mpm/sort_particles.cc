#include "sort_particles.h"

#include "ips2ra/ips2ra.hpp"

#include "drake/common/drake_assert.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/mpm/math.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <bitset>

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T, typename SpGrid>
void SortParticles(const SpGrid& spgrid, double dx, ParticleData<T>* particles,
                   Parallelism parallelism) {
  const auto& flags = spgrid.flags();
  const int log2_max_grid_size = flags.log2_max_grid_size;
  const int data_bits = flags.data_bits;
  const int log2_page = flags.log2_page;

  const int num_particles = particles->x.size();
  particles->data_indices.resize(num_particles);
  particles->base_node_offsets.resize(num_particles);
  particles->particle_sorters.resize(num_particles);

  auto& data_indices = particles->data_indices;
  auto& base_node_offsets = particles->base_node_offsets;
  auto& particle_sorters = particles->particle_sorters;
  auto& sentinel_particles = particles->sentinel_particles;
  auto& colored_blocks = particles->colored_blocks;

  /* We sort particles first based on their base node offsets, and if those
   are the same, we sort by their data indices. To do that, we notice that the
   base node offset of the particle looks like

       page bits | block bits | data bits

   with all the data bits being equal to zero. Also, the left most bits of the
   page bits are zero because at most 2^(3*log2_max_grid_size) number of grid
   nodes and that takes up 3*log2_max_grid_size bits. The page bits and block
   bits have 64 - data bits in total, so the left most 64 - data bits - 3 *
   log2_max_grid_size bits are zero. So we left shift the base node offset by
   that amount and now we get the lowest 64 - 3 * log2_max_grid_size bits (which
   we name `index_bits`) to be zero. With log2_max_grid_size == 10, we have 44
   bits to work with, more than enough to store the particle indices. We then
   sort the resulting 64 bit unsigned integers which is enough to achieve the
   sorting objective. */
  const int index_bits = 64 - 3 * log2_max_grid_size;
  const int zero_page_bits = 64 - data_bits - 3 * log2_max_grid_size;
  [[maybe_unused]] const int num_threads = parallelism.num_threads();

#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
  for (int p = 0; p < num_particles; ++p) {
    const auto& particle_x = [&]() -> Vector3<double> {
      if constexpr (std::is_same_v<T, double>) {
        return particles->x[p];
      } else if constexpr (std::is_same_v<T, float>) {
        return particles->x[p].template cast<double>();
      } else {
        return math::DiscardZeroGradient(particles->x[p]);
      }
    }();
    const Vector3<int> base_node = ComputeBaseNode<double>(particle_x / dx);
    base_node_offsets[p] =
        spgrid.CoordinateToOffset(base_node[0], base_node[1], base_node[2]);
    data_indices[p] = p;
    /* Confirm the data bits of the base node offset are all zero. */
    DRAKE_ASSERT((base_node_offsets[p] & ((uint64_t(1) << data_bits) - 1)) ==
                 0);
    /* Confirm the left most bits in the page bits are unused. */
    DRAKE_ASSERT((base_node_offsets[p] &
                  ~((uint64_t(1) << (64 - zero_page_bits)) - 1)) == 0);
    particle_sorters[p] =
        (base_node_offsets[p] << zero_page_bits) + data_indices[p];
  }

#if defined(_OPENMP)
  ips2ra::parallel::sort(particle_sorters.begin(), particle_sorters.end(),
                         ips2ra::Config<>::identity{}, num_threads);
#else
  ips2ra::sort(particle_sorters.begin(), particle_sorters.end());
#endif

  /* Peel off the data indices and the base node offsets from
   particle_sorters. Meanwhile, reorder the data indices and the base node
   offsets based on the sorting results. */
#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
  for (int p = 0; p < ssize(particle_sorters); ++p) {
    data_indices[p] = particle_sorters[p] & ((uint64_t(1) << index_bits) - 1);
    base_node_offsets[p] = (particle_sorters[p] >> index_bits) << data_bits;
  }

  /* Record the sentinel particles and the coloring of the blocks. */
  sentinel_particles.clear();
  for (int b = 0; b < 8; ++b) {
    colored_blocks[b].clear();
  }
  uint64_t previous_page{};
  int block = 0;
  for (int p = 0; p < num_particles; ++p) {
    /* The bits in the offset is ordered as follows:

      page bits | block bits | data bits

     block bits and data bits add up to log2_page bits.
     We right shift to get the page bits. */
    const uint64_t page = base_node_offsets[p] >> log2_page;
    if (p == 0 || previous_page != page) {
      previous_page = page;
      sentinel_particles.push_back(p);
      const int color = spgrid.get_color(page);
      colored_blocks[color].push_back(block++);
    }
  }
  sentinel_particles.push_back(num_particles);
}

template <typename T, typename SpGrid>
void SortParticlePositions(const SpGrid& spgrid,
                           std::vector<Vector3<double>>* q_WPs, T dx) {
  DRAKE_DEMAND(q_WPs != nullptr);
  const auto& flags = spgrid.flags();
  const int log2_max_grid_size = flags.log2_max_grid_size;
  const int data_bits = flags.data_bits;
  const int log2_page = flags.log2_page;

  const int num_particles = q_WPs->size();
  std::vector<uint64_t> particle_sorters(num_particles);
  const int index_bits = 64 - 3 * log2_max_grid_size;
  const int zero_page_bits = 64 - data_bits - 3 * log2_max_grid_size;
  for (int p = 0; p < num_particles; ++p) {
    const Vector3<int> base_node = ComputeBaseNode<double>(q_WPs->at(p) / dx);
    uint64_t base_node_offsets =
        spgrid.CoordinateToOffset(base_node[0], base_node[1], base_node[2]);
    uint64_t data_indices = p;
    particle_sorters[p] = (base_node_offsets << zero_page_bits) + data_indices;
  }
  ips2ra::sort(particle_sorters.begin(), particle_sorters.end());
  std::vector<Vector3<double>> result(num_particles);
  for (int p = 0; p < ssize(particle_sorters); ++p) {
    int data_indices = particle_sorters[p] & ((uint64_t(1) << index_bits) - 1);
    result[p] = q_WPs->at(data_indices);
  }
  *q_WPs = result;
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template void drake::multibody::mpm::internal::SortParticles<double>(
    const drake::multibody::mpm::internal::SpGrid<
        drake::multibody::mpm::internal::GridData<double>>&,
    double, drake::multibody::mpm::internal::ParticleData<double>*,
    drake::Parallelism);

template void drake::multibody::mpm::internal::SortParticles<float>(
    const drake::multibody::mpm::internal::SpGrid<
        drake::multibody::mpm::internal::GridData<float>>&,
    double, drake::multibody::mpm::internal::ParticleData<float>*,
    drake::Parallelism);

template void drake::multibody::mpm::internal::SortParticles<drake::AutoDiffXd>(
    const drake::multibody::mpm::internal::SpGrid<
        drake::multibody::mpm::internal::GridData<double>>&,
    double, drake::multibody::mpm::internal::ParticleData<drake::AutoDiffXd>*,
    drake::Parallelism);