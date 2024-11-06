#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <SPGrid_Allocator.h>
#include <SPGrid_Array.h>
#include <SPGrid_Mask.h>
#include <SPGrid_Page_Map.h>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/grid_data.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* A subset of SPGrid flags and configs we care about. */
struct SpGridFlags {
  int log2_page{12};            // 4KB page size.
  int log2_max_grid_size{10};   // Largest grid size is 1024 x 1024 x 1024.
  int data_bits{6};             // Number of bits to represent GridData.
  int num_nodes_in_block_x{4};  // Number of nodes in a block in x direction.
  int num_nodes_in_block_y{4};  // Number of nodes in a block in y direction.
  int num_nodes_in_block_z{4};  // Number of nodes in a block in z direction.
};

// TODO(xuchenhan-tri): Add documentation on SpGrid.
template <typename GridData>
class SpGrid {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SpGrid);

  static constexpr int kDim = 3;
  static constexpr int kLog2Page = 12;  // 4KB page size.
  /* The each block is assigned a unique color to facilitate lock-free particle
   to grid transfer. */
  static constexpr int kNumColors = 1 << kDim;
  using Allocator = SPGrid::SPGrid_Allocator<GridData, kDim, kLog2Page>;
  /* PageMap keeps track of which blocks in the SPGrid are allocated. */
  using PageMap = SPGrid::SPGrid_Page_Map<kLog2Page>;
  /* Mask helps convert from offset (1D index) to 3D index and vice versa. */
  using Mask = typename Allocator::template Array_mask<GridData>;
  /* Array type for GridData. */
  using Array = typename Allocator::template Array_type<GridData>;
  using ConstArray = typename Allocator::template Array_type<const GridData>;
  using Offset = uint64_t;

  SpGrid()
      : allocator_(kMaxGridSize, kMaxGridSize, kMaxGridSize),
        helper_blocks_(allocator_),
        blocks_(allocator_) {
    /* Compute the cell offset strides. */
    for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
          cell_offset_strides_[i + 1][j + 1][k + 1] =
              Mask::Linear_Offset(i, j, k);
        }
      }
    }
    /* Compute the block offset strides. */
    for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
          block_offset_strides_[i + 1][j + 1][k + 1] =
              Mask::Linear_Offset(i * kNumNodesInBlockX, j * kNumNodesInBlockY,
                                  k * kNumNodesInBlockZ);
        }
      }
    }
  }

  /* Makes `this` an exact copy of the `other` SpGrid. */
  void SetFrom(const SpGrid& other) {
    /* Copy over the page maps. */
    helper_blocks_.Clear();
    auto [block_offsets, num_blocks] = other.helper_blocks_.Get_Blocks();
    for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
      helper_blocks_.Set_Page(block_offsets[b]);
    }
    helper_blocks_.Update_Block_Offsets();

    blocks_.Clear();
    std::tie(block_offsets, num_blocks) = other.blocks_.Get_Blocks();
    for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
      blocks_.Set_Page(block_offsets[b]);
    }
    blocks_.Update_Block_Offsets();
    /* Copy over the data. */
    IterateGridWithOffset([&](uint64_t offset, GridData* node_data) {
      *node_data = other.get_data(offset);
    });
  }

  /* Allocate memory for the grid data for all grid nodes that are in the
   one-ring of nodes with the given offsets. */
  void Allocate(const std::vector<Offset>& offsets) {
    helper_blocks_.Clear();
    for (const Offset& offset : offsets) {
      helper_blocks_.Set_Page(offset);
    }
    helper_blocks_.Update_Block_Offsets();

    auto [block_offsets, num_blocks] = helper_blocks_.Get_Blocks();
    /* Touch all neighboring blocks of each block in `helper_blocks_` to
     ensure all grid nodes in the one ring of the given offsets have memory
     allocated. */
    for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
      const uint64_t current_offset = block_offsets[b];
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            const uint64_t neighbor_block_offset = Mask::Packed_Add(
                current_offset, block_offset_strides_[i][j][k]);
            blocks_.Set_Page(neighbor_block_offset);
          }
        }
      }
    }
    blocks_.Update_Block_Offsets();
    IterateGrid([](GridData* node_data) {
      node_data->set_zero();
    });
  }

  /* Returns the offset (1D index) of a grid node given its 3D grid
   coordinates in world space. */
  Offset CoordinateToOffset(int x, int y, int z) const {
    const uint64_t world_space_offset = Mask::Linear_Offset(x, y, z);
    return Mask::Packed_Add(world_space_offset, origin_offset_);
  }

  /* Returns the 3D grid coordinates in world space given the offset (1D
   index) of a grid node.
   @note This function is not particularly efficient. Do not use it in
   computationally intensive inner loops. Instead, prefer accessing grid data
   directly using the offsets. */
  Vector3<int> OffsetToCoordinate(uint64_t offset) const {
    const std::array<int, 3> reference_space_coordinate =
        Mask::LinearToCoord(offset);
    const std::array<int, 3> reference_space_origin =
        Mask::LinearToCoord(origin_offset_);
    return Vector3<int>(
        reference_space_coordinate[0] - reference_space_origin[0],
        reference_space_coordinate[1] - reference_space_origin[1],
        reference_space_coordinate[2] - reference_space_origin[2]);
  }

  /* The number of blocks that contain grid nodes which serve as base nodes
   for at least one particle. */
  int num_blocks() const { return helper_blocks_.Get_Blocks().second; }

  /* Given the offset of a grid node, returns the grid data in the pad with
   the given node at the center.
   @pre All nodes in the requested pad are active. */
  Pad<GridData> GetPadData(uint64_t center_node_offset) const {
    Pad<GridData> result;
    ConstArray data = allocator_.Get_Const_Array();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const uint64_t offset = Mask::Packed_Add(
              center_node_offset, cell_offset_strides_[i][j][k]);
          result[i][j][k] = data(offset);
        }
      }
    }
    return result;
  }

  /* Given the offset of a grid node, writes the grid data in the pad with the
   given node at the center.
   @pre All nodes in the requested pad are active. */
  void SetPadData(uint64_t center_node_offset, const Pad<GridData>& pad_data) {
    Array grid_data = allocator_.Get_Array();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const uint64_t offset = Mask::Packed_Add(
              center_node_offset, cell_offset_strides_[i][j][k]);
          grid_data(offset) = pad_data[i][j][k];
        }
      }
    }
  }

  // TODO(xuchenhan-tri): Consider adding a parallel version of IterateGrid.
  /* Func is a function that can be casted to std::function<void(GridData*)>.
   */
  template <typename Func>
  void IterateGrid(Func&& func) {
    const uint64_t data_size = 1 << kDataBits;
    auto [block_offsets, num_blocks] = blocks_.Get_Blocks();
    Array grid_data = allocator_.Get_Array();
    for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
      const uint64_t block_offset = block_offsets[b];
      uint64_t node_offset = block_offset;
      /* The coordinate of the origin of this block. */
      for (int i = 0; i < kNumNodesInBlockX; ++i) {
        for (int j = 0; j < kNumNodesInBlockY; ++j) {
          for (int k = 0; k < kNumNodesInBlockZ; ++k) {
            GridData& node_data = grid_data(node_offset);
            std::forward<Func>(func)(&node_data);
            node_offset += data_size;
          }
        }
      }
    }
  }

  /* Func is a function that can be casted to
      std::function<void(uint64_t, GridData*)>. */
  template <typename Func>
  void IterateGridWithOffset(Func&& func) {
    const uint64_t data_size = 1 << kDataBits;
    auto [block_offsets, num_blocks] = blocks_.Get_Blocks();
    Array grid_data = allocator_.Get_Array();
    for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
      const uint64_t block_offset = block_offsets[b];
      uint64_t node_offset = block_offset;
      /* The coordinate of the origin of this block. */
      for (int i = 0; i < kNumNodesInBlockX; ++i) {
        for (int j = 0; j < kNumNodesInBlockY; ++j) {
          for (int k = 0; k < kNumNodesInBlockZ; ++k) {
            GridData& node_data = grid_data(node_offset);
            std::forward<Func>(func)(node_offset, &node_data);
            node_offset += data_size;
          }
        }
      }
    }
  }

  /* Func is a function that can be casted to
      std::function<void(uint64_t, const GridData&)>. */
  template <typename Func>
  void IterateConstGridWithOffset(Func&& func) const {
    const uint64_t data_size = 1 << kDataBits;
    auto [block_offsets, num_blocks] = blocks_.Get_Blocks();
    ConstArray grid_data = allocator_.Get_Const_Array();
    for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
      const uint64_t block_offset = block_offsets[b];
      uint64_t node_offset = block_offset;
      /* The coordinate of the origin of this block. */
      for (int i = 0; i < kNumNodesInBlockX; ++i) {
        for (int j = 0; j < kNumNodesInBlockY; ++j) {
          for (int k = 0; k < kNumNodesInBlockZ; ++k) {
            const GridData& node_data = grid_data(node_offset);
            std::forward<Func>(func)(node_offset, node_data);
            node_offset += data_size;
          }
        }
      }
    }
  }

  /* Returns the flags associated with `this` SpGrid. */
  SpGridFlags flags() const {
    return SpGridFlags{.log2_page = kLog2Page,
                       .log2_max_grid_size = kLog2MaxGridSize,
                       .data_bits = kDataBits,
                       .num_nodes_in_block_x = kNumNodesInBlockX,
                       .num_nodes_in_block_y = kNumNodesInBlockY,
                       .num_nodes_in_block_z = kNumNodesInBlockZ};
  }

  /* Returns the color of the block given the page bits of the node offset.
   According to [Setaluri et al. 2014], the blocks are arranged in a 3D space
   using Z-order curve. Blocks with 8 blocks apart are guaranteed to be
   non-adjacent. */
  static int get_color(uint64_t page) {
    int color = (page & (kNumColors - 1));
    DRAKE_ASSERT(color >= 0 && color < 8);
    return color;
  }

 private:
  /* Get the data at the given grid offset.
   @pre Given `offset` is valid. */
  const GridData& get_data(uint64_t offset) const {
    return allocator_.Get_Const_Array()(offset);
  }

  /* The maximum grid size along a single dimension. That is even
   though the grid is sparsely populated, the maximum grid size
   that can ever be allocated is kMaxGridSize^3. With 1cm grid dx, that
   corresponds to more than 10 meters in each dimension, which should be
   enough for most manipulation simulations. */
  static constexpr int kLog2MaxGridSize = 10;
  static constexpr int kMaxGridSize = 1 << kLog2MaxGridSize;

  static constexpr int kDataBits = Mask::data_bits;
  static constexpr int kNumNodesInBlockX = 1 << Mask::block_xbits;
  static constexpr int kNumNodesInBlockY = 1 << Mask::block_ybits;
  static constexpr int kNumNodesInBlockZ = 1 << Mask::block_zbits;

  // TODO(xuchenhan-tri): Allow moving the maximumly allowed grid around the
  // center of the objects so that the grid can be accommodated to the objects
  // that are translating.
  /* 3D coordinates in SPGrid starts at (i, j, k) = (0, 0, 0) with i, j, k
   always non-negative. We want the grid to center around (0, 0, 0) in world
   space so we shift the origin by
   (kMaxGridSize/2, kMaxGridSize/2, kMaxGridSize/2)*/
  const uint64_t origin_offset_{Mask::Linear_Offset(
      kMaxGridSize / 2, kMaxGridSize / 2, kMaxGridSize / 2)};

  /* SPGrid allocator. */
  Allocator allocator_;
  /* Blocks containing grid nodes that are base nodes for at least one
   particle. */
  PageMap helper_blocks_;
  /* Blocks containing all active grid nodes. These are the blocks that are
   actually allocated. */
  PageMap blocks_;

  /* Stores the difference in linear offset from a given grid node to the grid
   node exactly one block away. For example, let `a` be
   `block_offset_strides_[0][1][2]` and `b` be the linear offset of a grid
   node `n`. Then, `a + b` gives the linear offset of the grid node `m` in the
   block
   (-1, 0, 1) relative to the block containing `n`. Both `m` and `n` reside at
   the same relative position within their respective blocks. */
  std::array<std::array<std::array<uint64_t, 3>, 3>, 3> block_offset_strides_;
  /* Similar to block_offset_strides_, but instead of providing strides for
   nodes a "block" away, provides the strides for nodes a "cell" away (i.e.
   immediate grid neighbors). */
  std::array<std::array<std::array<uint64_t, 3>, 3>, 3> cell_offset_strides_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
