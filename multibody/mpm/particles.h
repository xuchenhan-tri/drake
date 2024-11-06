#pragma once

#include "particle_data.h"
#include "particle_sorter.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
struct Particles {
  ParticleData<T> data;
  ParticleSorter sorter;

  template <typename Grid>
  void Sort(const Grid& grid) {
    sorter.Sort(grid.spgrid(), grid.dx(), data.x);
  }
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake