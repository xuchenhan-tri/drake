#pragma once

#include "sparse_grid.h"

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/grid_data.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* This class is used for unit testing only. It uses an std::map<Vector3d,
 GridState<T>> to store the grid data.
 @tparam double or AutoDiffXd. */

template <typename T>
class MockSparseGrid {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MockSparseGrid);
  explicit MockSparseGrid(double dx) : sparse_grid_(dx) {}

  double dx() const { return sparse_grid_.dx(); }

  int num_blocks() const { return sparse_grid_.num_blocks(); }

  void Allocate(ParticleData<T>* particles) {
    // sparse_grid_.Allocate(particles);
  }

  void SortParticles(ParticleData<T>* particles) {
    // sparse_grid_.SortParticles(particles);
  }

  Pad<Vector3<double>> GetPadNodes(const Vector3<double>& q_WP) const {
    return sparse_grid_.GetPadNodes(q_WP);
  }

  /* Given the offset of a grid node, returns the grid data in the pad with the
   given node at the center.
   @pre All nodes in the requested pad are active. */
  Pad<GridData<T>> GetPadData(uint64_t center_node_offset) const;

  void SetPadData(uint64_t center_node_offset,
                  const Pad<GridData<T>>& pad_data);

 private:
  SparseGrid<double> sparse_grid_;

  struct Vector3Comparator {
    bool operator()(const Vector3<int>& lhs, const Vector3<int>& rhs) const {
      // Lexicographical comparison
      if (lhs.x() != rhs.x()) return lhs.x() < rhs.x();
      if (lhs.y() != rhs.y()) return lhs.y() < rhs.y();
      return lhs.z() < rhs.z();
    }
  };
  std::map<Vector3<int>, GridData<T>, Vector3Comparator> grid_data_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake