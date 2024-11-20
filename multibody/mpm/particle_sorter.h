#pragma once

#include <bitset>
#include <vector>

#include "ips2ra/ips2ra.hpp"

#include "drake/common/drake_assert.h"
#include "drake/common/eigen_types.h"
#include "drake/common/parallelism.h"
#include "drake/common/unused.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/mpm/math.h"
#include "drake/multibody/mpm/particle_data.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* Helper data structure that assists sorting particle data based on their
 positions within the grid. */
class ParticleSorter {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ParticleSorter);
  ParticleSorter() = default;

  /* Using the order defined by `this` sorter, iterate through all particles and
   the grid data relevant to the particle. Potentially write to those grid
   nodes too upon the request of the `write_to_grid` flag. */
  template <typename Func, typename Grid, typename T>
  void Iterate(Grid* grid, ParticleData<T>* particles, bool write_to_grid,
               Func&& func) const {
    const int num_blocks = grid->num_blocks();
    DRAKE_DEMAND(ssize(sentinel_particles_) == num_blocks + 1);
    decltype(grid->GetPadNodes(std::declval<typename Grid::NodeType>())) grid_x;
    decltype(grid->GetPadData(std::declval<uint64_t>())) grid_data;
    bool need_new_pad = true;
    for (int b = 0; b < num_blocks; ++b) {
      const int particle_start = sentinel_particles_[b];
      const int particle_end = sentinel_particles_[b + 1];
      for (int p = particle_start; p < particle_end; ++p) {
        int data_index = data_indices_[p];
        if (need_new_pad) {
          grid_data = grid->GetPadData(base_node_offsets_[p]);
          grid_x = grid->GetPadNodes(particles->x[data_index]);
        }
        std::forward<Func>(func)(grid_x, &grid_data, particles, data_index);
        need_new_pad = (p + 1 == particle_end) ||
                       (base_node_offsets_[p] != base_node_offsets_[p + 1]);
        if (write_to_grid && need_new_pad) {
          grid->SetPadData(base_node_offsets_[p], grid_data);
        }
      }
    }
  }

  template <typename Func, typename Grid, typename T>
  void IterateOneParticlePerPad(const Grid& grid,
                                const ParticleData<T>& particles,
                                Func&& func) const {
    const int num_blocks = grid.num_blocks();
    DRAKE_DEMAND(ssize(sentinel_particles_) == num_blocks + 1);
    decltype(grid.GetPadNodes(std::declval<typename Grid::NodeType>())) grid_x;
    decltype(grid.GetPadData(std::declval<uint64_t>())) grid_data;
    for (int b = 0; b < num_blocks; ++b) {
      const int particle_start = sentinel_particles_[b];
      const int particle_end = sentinel_particles_[b + 1];
      for (int p = particle_start; p < particle_end; ++p) {
        int data_index = data_indices_[p];
        grid_data = grid.GetPadData(base_node_offsets_[p]);
        grid_x = grid.GetPadNodes(particles.x[data_index]);
        std::forward<Func>(func)(grid_x, grid_data, particles, data_index);
        while (p + 1 != particle_end &&
               base_node_offsets_[p] == base_node_offsets_[p + 1]) {
          ++p;
        }
      }
    }
  }

  template <typename Func, typename Grid, typename T>
  void IterateParallelSimd(Grid* grid, ParticleData<T>* particles,
                           bool write_to_grid, Parallelism parallelism,
                           Func&& func) const {
    const int num_blocks = grid->num_blocks();
    const int lanes = SimdScalar<T>::lanes();
    DRAKE_DEMAND(ssize(sentinel_particles_) == num_blocks + 1);
    decltype(grid->GetPadNodes(std::declval<typename Grid::NodeType>())) grid_x;
    decltype(grid->GetPadData(std::declval<uint64_t>())) grid_data;
    std::vector<int> indices;
    indices.reserve(lanes);
    bool need_new_pad = true;

    for (int c = 0; c < 8; ++c) {
      const std::vector<int>& blocks = colored_blocks_[c];
      [[maybe_unused]] const int num_threads = parallelism.num_threads();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
      for (int b : blocks) {
        const int particle_start = sentinel_particles_[b];
        const int particle_end = sentinel_particles_[b + 1];
        int p = particle_start;
        while (p < particle_end) {
          int next_p = p + 1;
          while (next_p < particle_end &&
                 base_node_offsets_[next_p] == base_node_offsets_[p] &&
                 next_p - p < lanes) {
            ++next_p;
          }
          int data_index = data_indices_[p];
          if (need_new_pad) {
            grid_data = grid->GetPadData(base_node_offsets_[p]);
            grid_x = grid->GetPadNodes(particles->x[data_index]);
          }
          indices.clear();
          for (int i = p; i < next_p; ++i) {
            indices.push_back(data_indices_[i]);
          }
          std::forward<Func>(func)(grid_x, &grid_data, particles, indices);
          need_new_pad = (next_p == particle_end) ||
                         (base_node_offsets_[next_p] != base_node_offsets_[p]);
          if (need_new_pad) {
            grid->SetPadData(base_node_offsets_[p], grid_data);
          }
          p = next_p;
        }
      }
    }
  }

  template <typename T, typename SpGrid>
  void Sort(const SpGrid& spgrid, double dx,
            const std::vector<Vector3<T>>& particle_positions,
            Parallelism parallelism = false) {
    const auto& flags = spgrid.flags();
    const int log2_max_grid_size = flags.log2_max_grid_size;
    const int data_bits = flags.data_bits;
    const int log2_page = flags.log2_page;

    const int num_particles = particle_positions.size();
    data_indices_.resize(num_particles);
    base_node_offsets_.resize(num_particles);
    particle_sorters_.resize(num_particles);

    /* We sort particles first based on their base node offsets, and if those
     are the same, we sort by their data indices. To do that, we notice that the
     base node offset of the particle looks like

         page bits | block bits | data bits

     with all the data bits being equal to zero. Also, the left most bits of the
     page bits are zero because at most 2^(3*log2_max_grid_size) number of grid
     nodes and that takes up 3*log2_max_grid_size bits. The page bits and block
     bits have 64 - data bits in total, so the left most 64 - data bits - 3 *
     log2_max_grid_size bits are zero. So we left shift the base node offset by
     that amount and now we get the lowest 64 - 3 * log2_max_grid_size bits
     (which we name `index_bits`) to be zero. With log2_max_grid_size == 10, we
     have 44 bits to work with, more than enough to store the particle indices.
     We then sort the resulting 64 bit unsigned integers which is enough to
     achieve the sorting objective. */
    const int index_bits = 64 - 3 * log2_max_grid_size;
    const int zero_page_bits = 64 - data_bits - 3 * log2_max_grid_size;
    [[maybe_unused]] const int num_threads = parallelism.num_threads();

#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
    for (int p = 0; p < num_particles; ++p) {
      const auto& particle_x = [&]() -> Vector3<double> {
        if constexpr (std::is_same_v<T, double>) {
          return particle_positions[p];
        } else if constexpr (std::is_same_v<T, float>) {
          return particle_positions[p].template cast<double>();
        } else {
          return math::DiscardZeroGradient(particle_positions[p]);
        }
      }();
      const Vector3<int> base_node = ComputeBaseNode<double>(particle_x / dx);
      base_node_offsets_[p] =
          spgrid.CoordinateToOffset(base_node[0], base_node[1], base_node[2]);
      data_indices_[p] = p;
      /* Confirm the data bits of the base node offset are all zero. */
      DRAKE_ASSERT((base_node_offsets_[p] & ((uint64_t(1) << data_bits) - 1)) ==
                   0);
      /* Confirm the left most bits in the page bits are unused. */
      DRAKE_ASSERT((base_node_offsets_[p] &
                    ~((uint64_t(1) << (64 - zero_page_bits)) - 1)) == 0);
      particle_sorters_[p] =
          (base_node_offsets_[p] << zero_page_bits) + data_indices_[p];
    }

#if defined(_OPENMP)
    ips2ra::parallel::sort(particle_sorters_.begin(), particle_sorters_.end(),
                           ips2ra::Config<>::identity{}, num_threads);
#else
    ips2ra::sort(particle_sorters_.begin(), particle_sorters_.end());
#endif

    /* Peel off the data indices and the base node offsets from
     particle_sorters_. Meanwhile, reorder the data indices and the base node
     offsets based on the sorting results. */
#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
    for (int p = 0; p < ssize(particle_sorters_); ++p) {
      data_indices_[p] =
          particle_sorters_[p] & ((uint64_t(1) << index_bits) - 1);
      base_node_offsets_[p] = (particle_sorters_[p] >> index_bits) << data_bits;
    }

    /* Record the sentinel particles and the coloring of the blocks. */
    sentinel_particles_.clear();
    for (int b = 0; b < 8; ++b) {
      colored_blocks_[b].clear();
    }
    uint64_t previous_page{};
    int block = 0;
    for (int p = 0; p < num_particles; ++p) {
      /* The bits in the offset is ordered as follows:

        page bits | block bits | data bits

       block bits and data bits add up to log2_page bits.
       We right shift to get the page bits. */
      const uint64_t page = base_node_offsets_[p] >> log2_page;
      if (p == 0 || previous_page != page) {
        previous_page = page;
        sentinel_particles_.push_back(p);
        const int color = spgrid.get_color(page);
        colored_blocks_[color].push_back(block++);
      }
    }
    sentinel_particles_.push_back(num_particles);
  }

  /* Returns the offsets of each active block. */
  std::vector<uint64_t> GetBlockOffsets() const {
    std::vector<uint64_t> offsets(ssize(sentinel_particles_) - 1);
    for (int i = 0; i < ssize(sentinel_particles_) - 1; ++i) {
      offsets[i] = base_node_offsets_[sentinel_particles_[i]];
    }
    return offsets;
  }

  const std::vector<int>& sentinel_particles() const {
    return sentinel_particles_;
  }
  const std::vector<int>& data_indices() const {
    return data_indices_;
  }
  const std::vector<uint64_t>& base_node_offsets() const {
    return base_node_offsets_;
  }
  const std::array<std::vector<int>, 8>& colored_blocks() const {
    return colored_blocks_;
  }

 private:
  /* All but last entry store indices of particles marking the boundary of a new
   block. The last entry stores the number of particles. */
  std::vector<int> sentinel_particles_;
  /* The order in which the particle data should be accessed when used in tandem
   with a grid. That is, particle_data[particle_indices()[p]] gives the particle
   data for the p-th particle. */
  std::vector<int> data_indices_;
  /* Returns the base node offset of the associated grid for each particle. */
  std::vector<uint64_t> base_node_offsets_;
  /* Helper data to sort the particles according to their base nodes. */
  std::vector<uint64_t> particle_sorters_;
  /* We color SPGrid blocks so that writing to different blocks with the same
  color is guaranteed to be free of write hazards. This function returns the
  block indices for each color associated with this particle data. */
  std::array<std::vector<int>, 8> colored_blocks_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
