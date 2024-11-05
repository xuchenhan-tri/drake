#include "mock_sparse_grid.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
Pad<GridData<T>> MockSparseGrid<T>::GetPadData(
    uint64_t center_node_offset) const {
  const Vector3<int> center_node_coordinate =
      sparse_grid_.OffsetToCoordinate(center_node_offset);
  Pad<GridData<T>> pad_data;
  for (int i = 0; i < 3; ++i) {
    const int x = center_node_coordinate.x() + i - 1;
    for (int j = 0; j < 3; ++j) {
      const int y = center_node_coordinate.y() + j - 1;
      for (int k = 0; k < 3; ++k) {
        const int z = center_node_coordinate.z() + k - 1;
        const Vector3<int> node_coordinate(x, y, z);
        const auto it = grid_data_.find(node_coordinate);
        DRAKE_DEMAND(it != grid_data_.end());
        pad_data[i][j][k] = it->second;
      }
    }
  }
  return pad_data;
}

template <typename T>
void MockSparseGrid<T>::SetPadData(uint64_t center_node_offset,
                                   const Pad<GridData<T>>& pad_data) {
  const Vector3<int> center_node_coordinate =
      sparse_grid_.OffsetToCoordinate(center_node_offset);
  for (int i = 0; i < 3; ++i) {
    const int x = center_node_coordinate.x() + i - 1;
    for (int j = 0; j < 3; ++j) {
      const int y = center_node_coordinate.y() + j - 1;
      for (int k = 0; k < 3; ++k) {
        const int z = center_node_coordinate.z() + k - 1;
        const Vector3<int> node_coordinate(x, y, z);
        grid_data_[node_coordinate] = pad_data[i][j][k];
      }
    }
  }
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::internal::MockSparseGrid<double>;
template class drake::multibody::mpm::internal::MockSparseGrid<
    drake::AutoDiffXd>;