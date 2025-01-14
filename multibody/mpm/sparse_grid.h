#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/multibody/mpm/grid_data.h"
#include "drake/multibody/mpm/mass_and_momentum.h"
#include "drake/multibody/mpm/particle_sorter.h"
#include "drake/multibody/mpm/spgrid.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* SparseGrid is a 3D grid that is sparsely populated with GridData implemented
 with a Sparse Paged Grid (SpGrid) data structure (see
 multibody::mpm::internal::SpGrid).

 Recall that a "block" in SpGrid is a continuous block of memory of the size of
 a page (4kB). For T == float, GridData has 32 byte, which means a block in
 SparseGrid<float> consists of 4 x 4 x 8 grid nodes. For T == double, GridData
 has 64 byte, which means a block in SparseGrid<double> consists of 4 x 4 x 4
 grid.

 Also recall that a grid node is defined as the "base node" of a particle if
 it's the closest grid node to the particle. A grid node is "in the support" of
 a particle if it's part of a Pad (3x3x3 subgrid) whose center grid node is the
 base node of the particle. A grid node is said to be "active" if it has
 non-zero mass. That happens if and only if the grid node is in the support of a
 particle. Blocks in the grid are allocated lazily, i.e. only when they may
 contain active grid nodes.

 @tparam float or double. */
template <typename T>
class SparseGrid {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SparseGrid);

  /* Constructs a SparseGrid with grid spacing `dx` in meters. */
  explicit SparseGrid(double dx);

  std::unique_ptr<SparseGrid<T>> Clone() const {
    auto result = std::make_unique<SparseGrid<T>>(dx_);
    result->spgrid_.SetFrom(this->spgrid_);
    return result;
  }

  /* Allocates memory for the grid nodes that are in the support of particles.
   All allocated grid data is zeroed out.
   @param[in] particles  A ParticleSorter that has called `Sort()` on this
                         SparseGrid's SPGrid and dx and the particle positions.
  */
  void Allocate(const ParticleSorter& particles);

  /* Grid spacing in meters. */
  T dx() const { return dx_; }

  /* Given the position of a particle in the world frame, returns the world
   frame positions of the grid node in its support. */
  Pad<Vector3<T>> GetPadNodes(const Vector3<T>& q_WP) const;

  /* Given the SpGrid offset of a grid node, returns the grid data in the pad
   with the given node at the center.
   @pre All nodes in the requested pad are active. */
  Pad<GridData<T>> GetPadData(uint64_t center_node_offset) const {
    return spgrid_.GetPadData(center_node_offset);
  }

  /* Given the offset of a grid node, writes the grid data in the pad with the
   given node at the center.
   @pre All nodes in the requested pad are active. */
  void SetPadData(uint64_t center_node_offset,
                  const Pad<GridData<T>>& pad_data) {
    spgrid_.SetPadData(center_node_offset, pad_data);
  }

  /* Sets the grid state with the given callback function that maps the world
   space coordinate of the grid node to the data at that node.
   @pre The callback function only assigns grid data to active grid nodes.
   @note Testing only. */
  void SetGridData(
      const std::function<GridData<T>(const Vector3<int>&)>& callback);

  /* Returns grid data from all active grid nodes as a pair of world space
   coordinate and grid data of the node.
   @note Testing only. */
  std::vector<std::pair<Vector3<int>, GridData<T>>> GetGridData() const;

  /* Computes the mass, linear momentum, and angular momentum (about world
   origin) on the grid.
   @note This function assumes that the grid data stores the mass and velocity
   (instead of mass and momemtum) of grid nodes.
   @note Testing only. */
  MassAndMomentum<T> ComputeTotalMassAndMomentum() const;

  /* Returns the SpGrid underlying this SparseGrid. */
  const SpGrid<GridData<T>>& spgrid() const { return spgrid_; }

  /* Returns the SpGrid underlying this SparseGrid. */
  SpGrid<GridData<T>>& mutable_spgrid() { return spgrid_; }

 private:
  /* Grid spacing (in meters). */
  double dx_{};
  SpGrid<GridData<T>> spgrid_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
