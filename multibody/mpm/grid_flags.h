#pragma once

#include "drake/common/drake_throw.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

struct GridFlags {
  int log2_page{12};
  int log2_max_grid_size{10};
  int data_bits{-1};
  int num_nodes_in_block_x{-1};
  int num_nodes_in_block_y{-1};
  int num_nodes_in_block_z{-1};

  void Validate() const {
    DRAKE_THROW_UNLESS(log2_page > 0);
    DRAKE_THROW_UNLESS(log2_max_grid_size > 0);
    DRAKE_THROW_UNLESS(data_bits > 0);
    DRAKE_THROW_UNLESS(num_nodes_in_block_x > 0);
    DRAKE_THROW_UNLESS(num_nodes_in_block_y > 0);
    DRAKE_THROW_UNLESS(num_nodes_in_block_z > 0);
  }
};

/* Sorts particles based on their positions in the background grid. */
void SortParticles(const GridFlags& flags, ParticleData<T>* particles) {
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
   page bits are zero because at most 2^(3*kLog2MaxGridSize) number of grid
   nodes and that takes up 3*kLog2MaxGridSize bits. The page bits and block
   bits have 64 - data bits in total, so the left most 64 - data bits - 3 *
   kLog2MaxGridSize bits are zero. So we left shift the base node offset by
   that amount and now we get the lowest 64 - 3 * kLog2MaxGridSize bits (which
   we name `kIndexBits`) to be zero. With kLog2MaxGridSize == 10, we have 44
   bits to work with, more than enough to store the particle indices. We then
   sort the resulting 64 bit unsigned integers which is enough to achieve the
   sorting objective. */
  constexpr int kIndexBits = 64 - 3 * kLog2MaxGridSize;
  constexpr int kZeroPageBits = 64 - kDataBits - 3 * kLog2MaxGridSize;
  [[maybe_unused]] const int num_threads = parallelism_.num_threads();

#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
  for (int p = 0; p < num_particles; ++p) {
    const Vector3<int> base_node = ComputeBaseNode<T>(particles->x[p] / dx_);
    base_node_offsets[p] =
        CoordinateToOffset(base_node[0], base_node[1], base_node[2]);
    data_indices[p] = p;
    /* Confirm the data bits of the base node offset are all zero. */
    DRAKE_ASSERT((base_node_offsets[p] & ((uint64_t(1) << kDataBits) - 1)) ==
                 0);
    /* Confirm the left most bits in the page bits are unused. */
    DRAKE_ASSERT((base_node_offsets[p] &
                  ~((uint64_t(1) << (64 - kZeroPageBits)) - 1)) == 0);
    particle_sorters[p] =
        (base_node_offsets[p] << kZeroPageBits) + data_indices[p];
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
    data_indices[p] = particle_sorters[p] & ((uint64_t(1) << kIndexBits) - 1);
    base_node_offsets[p] = (particle_sorters[p] >> kIndexBits) << kDataBits;
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

     block bits and data bits add up to kLog2Page bits.
     We right shift to get the page bits. */
    const uint64_t page = base_node_offsets[p] >> kLog2Page;
    if (p == 0 || previous_page != page) {
      previous_page = page;
      sentinel_particles.push_back(p);
      const int color = get_color(page);
      colored_blocks[color].push_back(block++);
    }
  }
  sentinel_particles.push_back(num_particles);
}

}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake