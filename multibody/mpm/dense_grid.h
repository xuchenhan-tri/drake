#pragma once

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* @tparam float, double, or AutoDiffXd. */
template <typename T>
class DenseGrid {
 public:
  DRAKE_DEFAULT_COPY_MOVE_AND_ASSIGN(DenseGrid);

  static constexpr int kDim = 3;

  /* Since the memory for dense blocks are pre-allocated, calling this function
   doesn't allocate any memories. The reason for the name is to keep consistency
   with the SparseGrid class.

   Resets all grid data to zero. Throws if any particle is outside the
   pre-allocated grid.

   As a side effect, this function also orders the particles, based on the
   "offset" of their base nodes. In the process, it builds `sentinel_particles`
   and `data_indices`. */
  void Allocate(ParticleData<T>* particles) {
    SortParticles(particles);
  }

 private:
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake