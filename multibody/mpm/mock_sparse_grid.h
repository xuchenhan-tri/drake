#pragma once

#include <map>

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

  explicit MockSparseGrid(double dx) : dx_(dx) {}

  std::unique_ptr<MockSparseGrid<T>> Clone() const {
    auto clone = std::make_unique<MockSparseGrid<T>>(dx_);
    clone->spgrid_.SetFrom(this->spgrid_);
    clone->grid_data_ = this->grid_data_;
    return clone;
  }

  void Allocate(ParticleData<T>* particles) { grid_data_.clear(); }

  double dx() const { return dx_; }

  int num_blocks() const { return spgrid_.num_blocks(); }

  Pad<Vector3<double>> GetPadNodes(const Vector3<double>& q_WP) const {
    Pad<Vector3<double>> result;
    const Vector3<int> base_node = ComputeBaseNode<double>(q_WP / dx_);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const Vector3<int> shift(i - 1, j - 1, k - 1);
          result[i][j][k] = dx_ * (base_node + shift).cast<double>();
        }
      }
    }
    return result;
  }

  Pad<GridData<T>> GetPadData(uint64_t center_node_offset) const;

  void SetPadData(uint64_t center_node_offset,
                  const Pad<GridData<T>>& pad_data);

  void ExplicitVelocityUpdate(const Vector3<T>& dv);

  void SetGridData(
      const std::function<GridData<T>(const Vector3<int>&)>& callback);

  std::vector<std::pair<Vector3<int>, GridData<T>>> GetGridData() const;

  MassAndMomentum<T> ComputeTotalMassAndMomentum() const;

  void SetNodeIndices();

  const SpGrid<GridData<double>>& spgrid() const { return spgrid_; }

 private:
  double dx_{};
  SpGrid<GridData<double>> spgrid_;

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