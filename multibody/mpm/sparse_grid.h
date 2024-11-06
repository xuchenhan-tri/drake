#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "spgrid.h"

#include "drake/common/eigen_types.h"
#include "drake/common/parallelism.h"
#include "drake/multibody/mpm/grid_data.h"
#include "particles.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* SparseGrid is a 3D grid that is sparsely populated with GridData implemented
 with a Sparse Paged Grid (SPGrid) data structure. Memory is allocated and used
 in chunks of pages (4KB in size).

 We define a few concepts related to the grid here:

 - A "block" is a subgrid with continuous memory of the size of a page.
 For 16 byte GridData (float), a block consists of 4 x 8 x 8 grid nodes. For 32
 byte GridData (double), a block consists of 4 x 4 x 8 grid nodes.

 - The "base node" of a particle is the center node if the 3x3x3 subgrid that
 affected by the particle.

 - A grid node is "active" if it has non-zero mass. That happens if and only if
 the grid node is in the support of a particle.

 - A block is only allocated if it contains active grid nodes.

 The 3D coordinate for each grid node is mapped to a unique 1D index (64 bit
 integer address) by SPGrid where the data along with the coordinate information
 is stored, and we call that 1D index the "offset" of the node following
 nomenclature from SPGrid.

 This class is used in close conjunction with ParticleData to transfer data
 between particles and the grid. A typical workflow is as follows:

 ```
   SparseGrid grid(dx);

   // `particles` is a ParticleData carrying the particle physical attributes.
   ...

   // Before interacting with the grid, always initialize the grid with the
   // particle positions.
   grid.Allocate(particles.x);

   // grid data is now ready to be used.
   grid.Foo();
   ...

   // If the particle data is modified, the grid must be updated before the next
   // interaction.
   particles.x[0] += Vector3d(0.1, 0.1, 0.1);
   grid.Allocate(particles.x);
   grid.Foo();
   ...
 ```

 @tparam float or double. */
template <typename T>
class SparseGrid {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SparseGrid);
  
  using NodeType = Vector3<T>;

  /* Constructs a SparseGrid with grid spacing `dx` in meters. */
  explicit SparseGrid(double dx, Parallelism parallelism = false);

  std::unique_ptr<SparseGrid<T>> Clone() const {
    auto result = std::make_unique<SparseGrid<T>>(dx_, parallelism_);
    result->spgrid_.SetFrom(this->spgrid_);
    result->num_active_nodes_ = this->num_active_nodes_;
    return result;
  }

  /* Allocates memory for the grid pages affected by particles and initialize
   all grid data to zero.

   As a side effect, this function also orders the particles based on the
   "offset" of their base nodes. In the process, it builds `sentinel_particles`
   and `data_indices`. */
  void Allocate(const ParticleSorter& particles);

  /* Grid spacing in meters. */
  T dx() const { return dx_; }

  /* The number of blocks that contain grid nodes which serve as base nodes for
   at least one particle. */
  int num_blocks() const { return spgrid_.num_blocks(); }

  /* Given the position of a particle in the world frame, returns the world
   frame positions of the grid node in its support. */
  Pad<Vector3<T>> GetPadNodes(const Vector3<T>& q_WP) const;

  /* Given the offset of a grid node, returns the grid data in the pad with the
   given node at the center.
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

  /* For each active grid node, divide by the grid node mass to convert the
   momentum to velocity and then increment the velocity by dv.
   Also apply boundary conditions along the way.
   @pre the grid data stores momentum of the node, not velocity. */
  void ExplicitVelocityUpdate(const Vector3<T>& dv);

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

  /* Assigning consecutive indices to all active nodes [0, num_active_nodes()).
   All non-active grid nodes (those with zero mass) gets index -1. */
  void SetNodeIndices();

  /* Returns the number of active grid nodes as computed by last call to
   SetNodeIndices(). */
  int num_active_nodes() const { return num_active_nodes_; }

  /* Converts 3D coordinates to 1D indices (offset). Testing only. */
  SpGrid<GridData<T>>::Offset CoordinateToOffset(int x, int y, int z) const {
    return spgrid_.CoordinateToOffset(x, y, z);
  }

  /* Converts 1D indices (offset) to 3D coordinates. Testing only. */
  Vector3<int> OffsetToCoordinate(SpGrid<GridData<T>>::Offset offset) const {
    return spgrid_.OffsetToCoordinate(offset);
  }

  /* Returns the SpGrid underlying this SparseGrid. */
  const SpGrid<GridData<T>>& spgrid() const { return spgrid_; }

  template <typename Func>
  void IterateGrid(Func&& func) {
    spgrid_.IterateGrid(std::forward<Func>(func));
  }

 private:
  /* Grid spacing (in meters). */
  double dx_{};
  Parallelism parallelism_;
  SpGrid<GridData<T>> spgrid_;
  /* Number of grid nodes with non-zero mass (i.e. those that are affected by at
   least one particle).*/
  int num_active_nodes_{0};
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
