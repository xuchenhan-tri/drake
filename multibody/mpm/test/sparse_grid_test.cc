#include "drake/multibody/mpm/sparse_grid.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using Eigen::Vector3d;
using Eigen::Vector3f;
using Eigen::Vector3i;

GTEST_TEST(SparseGridTest, Allocate) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  EXPECT_EQ(grid.dx(), 0.01);

  const Vector3d q_WP = Vector3d(1.001, 0.001, 0.001);
  const std::vector<Vector3d> q_WPs = {q_WP};
  ParticleSorter sorter;
  sorter.Sort(grid.spgrid(), grid.dx(), q_WPs);
  grid.Allocate(sorter);

  /* Verify grid data is all zeroed out. */
  const std::vector<std::pair<Vector3i, GridData<double>>> grid_data =
      grid.GetGridData();
  for (const auto& [node, data] : grid_data) {
    EXPECT_EQ(data.m, 0.0);
    EXPECT_EQ(data.v, Vector3d::Zero());
  }

  int num_active_nodes = 0;
  auto count_active_nodes = [&num_active_nodes](uint64_t,
                                                const GridData<double>&) {
    ++num_active_nodes;
  };
  grid.spgrid().IterateConstGridWithOffset(count_active_nodes);
  /* We allocate the block that contains the particle and its 27 immediate
   neighbors to ensure that all grid nodes in support of the particle are
   allocated. */
  EXPECT_EQ(num_active_nodes, 4 * 4 * 4 * 27);
}

GTEST_TEST(SparseGrid, Clone) {
  /* Set up a grid with grid nodes in [0, 2] x [0, 2] x [0, 2] all active. */
  const double dx = 0.5;
  SparseGrid<double> grid(dx);
  std::vector<Vector3d> q_WPs;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        q_WPs.emplace_back(Vector3d(i * dx, j * dx, k * dx));
      }
    }
  }
  ParticleSorter sorter;
  sorter.Sort(grid.spgrid(), grid.dx(), q_WPs);
  grid.Allocate(sorter);

  /* Set an arbitrary grid data. */
  auto set_data = [](const Vector3i& coordinate) {
    GridData<double> result;
    result.m = coordinate[0] + coordinate[1] + coordinate[2];
    result.v = Vector3d(coordinate[0], coordinate[1], coordinate[2]);
    return result;
  };
  grid.SetGridData(set_data);

  /* Clone the grid. */
  auto cloned_grid = grid.Clone();

  /* Verify that the cloned grid has the same grid data. */
  const std::vector<std::pair<Vector3i, GridData<double>>> grid_data =
      grid.GetGridData();
  const std::vector<std::pair<Vector3i, GridData<double>>> cloned_grid_data =
      cloned_grid->GetGridData();
  ASSERT_EQ(grid_data.size(), cloned_grid_data.size());
  for (size_t i = 0; i < grid_data.size(); ++i) {
    const auto& [node, data] = grid_data[i];
    const auto& [cloned_node, cloned_data] = cloned_grid_data[i];
    EXPECT_EQ(node, cloned_node);
    EXPECT_EQ(data, cloned_data);
  }
}

GTEST_TEST(SparseGridTest, GetPadNodes) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  const Vector3d q_WP = Vector3d(0.001, 0.001, 0.001);
  /* Base node is (0, 0, 0), so we should get the 27 immediate neighbors of the
   (0, 0, 0) as the pad nodes. */
  const Pad<Vector3d> pad_nodes = grid.GetPadNodes(q_WP);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        const Vector3d node =
            Vector3d((i - 1) * dx, (j - 1) * dx, (k - 1) * dx);
        EXPECT_EQ(pad_nodes[i][j][k], node);
      }
    }
  }
}

GTEST_TEST(SparseGridTest, PadData) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  /* Base node is (2, 3, 0). */
  const Vector3d q_WP = Vector3d(0.021, 0.031, -0.001);
  std::vector<Vector3d> q_WPs = {q_WP};

  ParticleSorter sorter;
  sorter.Sort(grid.spgrid(), grid.dx(), q_WPs);
  grid.Allocate(sorter);

  const std::vector<uint64_t>& base_node_offsets = sorter.base_node_offsets();
  ASSERT_EQ(base_node_offsets.size(), 1);
  const uint64_t base_node_offset = base_node_offsets[0];

  Pad<GridData<double>> arbitrary_data;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        GridData<double> foo;
        foo.m = i + j + k;
        foo.v = Vector3d(i, j, k);
        arbitrary_data[i][j][k] = foo;
      }
    }
  }

  grid.SetPadData(base_node_offset, arbitrary_data);
  const Pad<GridData<double>> pad_data = grid.GetPadData(base_node_offset);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        EXPECT_EQ(pad_data, arbitrary_data);
      }
    }
  }

  /* Now get the pad centered at (1, 2, -1). It should overlap with the pad
   centered at (2, 3, 0). The non-overlapping portion should be zeroed out
   (during Allocate()).  */
  const Pad<GridData<double>> pad_data2 =
      grid.GetPadData(grid.spgrid().CoordinateToOffset(1, 2, -1));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        if (i == 0 || j == 0 || k == 0) {
          EXPECT_EQ(pad_data2[i][j][k].m, 0.0);
          EXPECT_EQ(pad_data2[i][j][k].v, Vector3d::Zero());
        } else {
          EXPECT_EQ(pad_data2[i][j][k], arbitrary_data[i - 1][j - 1][k - 1]);
        }
      }
    }
  }
}

GTEST_TEST(SparseGridTest, ComputeTotalMassAndMomentum) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  const Vector3d q_WP = Vector3d(0.001, 0.001, 0.001);
  std::vector<Vector3d> q_WPs = {q_WP};

  ParticleSorter sorter;
  sorter.Sort(grid.spgrid(), grid.dx(), q_WPs);
  grid.Allocate(sorter);

  const double mass = 1.2;
  const Vector3d velocity = Vector3d(1, 2, 3);
  /* World frame position of the node with non-zero mass. */
  const Vector3d q_WN = Vector3d(dx, dx, dx);
  /* Set grid data so that the grid node (1, 1, 1) has velocity (1, 2, 3) and
   all other grid nodes have zero velocity. */
  auto set_grid_data = [mass, velocity](const Vector3i& node) {
    GridData<double> result;
    if (node[0] == 1 && node[1] == 1 && node[2] == 1) {
      result.m = mass;
      result.v = velocity;
    } else {
      result.set_zero();
    }
    return result;
  };

  grid.SetGridData(set_grid_data);

  const MassAndMomentum<double> computed = grid.ComputeTotalMassAndMomentum();
  EXPECT_EQ(computed.mass, mass);
  EXPECT_TRUE(CompareMatrices(computed.linear_momentum, mass * velocity,
                              4.0 * std::numeric_limits<double>::epsilon()));
  EXPECT_TRUE(CompareMatrices(computed.angular_momentum,
                              mass * q_WN.cross(velocity),
                              4.0 * std::numeric_limits<double>::epsilon()));
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
