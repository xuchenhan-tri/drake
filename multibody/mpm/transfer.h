#pragma once

#include "drake/multibody/mpm/grid_data.h"
#include "drake/multibody/mpm/particle_data.h"
#include "drake/multibody/mpm/particle_sorter.h"
#include "drake/multibody/mpm/sparse_grid.h"
#include "drake/multibody/mpm/transfer_kernels.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
struct Particles {
  void Sort(const SparseGrid<T>& grid) {
    sorter.Sort(grid.spgrid(), grid.dx(), data.x());
  }
  ParticleData<T> data;
  ParticleSorter sorter;
};

template <typename T>
class Transfer {
 public:
  Transfer(T dt, SparseGrid<T>* grid, Particles<T>* particles);

  void ScalarParticleToGrid(const Particles<T>& particles, SparseGrid<T>* grid);

  void SimdParticleToGrid(const Particles<T>& particles, SparseGrid<T>* grid);

 private:
  void ScalarP2G(const Pad<Vector3<T>>& grid_x,
                 const ParticleData<T>& particle_data, int data_index,
                 Pad<GridData<T>>* grid_data);

  T dt_{};
  T D_inverse_{};
  T D_inverse_dt_{};
  SparseGrid<T>* grid_{};
  Particles<T>* particles_{};
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
