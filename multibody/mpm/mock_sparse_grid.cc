#include "mock_sparse_grid.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
Pad<GridData<T>> MockSparseGrid<T>::GetPadData(
    uint64_t center_node_offset) const {
  const Vector3<int> center_node_coordinate =
      spgrid_.OffsetToCoordinate(center_node_offset);
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
      spgrid_.OffsetToCoordinate(center_node_offset);
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

template <typename T>
void MockSparseGrid<T>::ExplicitVelocityUpdate(const Vector3<T>& dv) {
  for (auto& [_, data] : grid_data_) {
    if (data.m > 0.0) {
      data.v /= data.m;
      data.v += dv;
    }
  }
}

template <typename T>
void MockSparseGrid<T>::SetGridData(
    const std::function<GridData<T>(const Vector3<int>&)>& callback) {
  for (auto& [node, data] : grid_data_) {
    data = callback(node);
  }
}

template <typename T>
std::vector<std::pair<Vector3<int>, GridData<T>>>
MockSparseGrid<T>::GetGridData() const {
  std::vector<std::pair<Vector3<int>, GridData<T>>> result;
  for (const auto& [node, data] : grid_data_) {
    result.emplace_back(node, data);
  }
  return result;
}

template <typename T>
MassAndMomentum<T> MockSparseGrid<T>::ComputeTotalMassAndMomentum() const {
  MassAndMomentum<T> result;
  for (const auto& [node, data] : grid_data_) {
    if (data.m > 0.0) {
      const Vector3<double> xi = dx_ * node.template cast<double>();
      result.mass += data.m;
      result.linear_momentum += data.m * data.v;
      result.angular_momentum += data.m * xi.cross(data.v);
    }
  }
  return result;
}

template <typename T>
void MockSparseGrid<T>::SetNodeIndices() {
  int node_index = 0;
  for (auto& [node, data] : grid_data_) {
    if (data.m > 0.0) {
      data.index = node_index++;
    } else {
      data.index = -1;
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