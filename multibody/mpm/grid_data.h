#pragma once

#include <array>

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* A class representing an index or a flag for a grid node in preparation of
  indexing the grid, with support for specific states.

  This class is a lightweight wrapper around an integer type (`int32_t` or
  `int64_t`) used to differentiate between active indices, inactive states, and
  special flags. A GridNodeIndex can be in exactly one of the following states:
  1. Active index: A non-negative integer representing the index of a grid.
  2. Generic inactive state (the default state).
  3. The participating state: A special inactive state used to mark grid nodes
     to be processed seaparately when activated.

  A GridNodeIndex can transition freely between any two states, except that it
  cannot transition from the active index state to the participating state.

  @tparam T The integer type for the index. Must be `int32_t` or `int64_t`. */
template <typename T>
class GridNodeIndex {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(GridNodeIndex);

  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>,
                "T must be int32_t or int64_t.");

  /* Default constructor initializes the index to the inactive state. */
  constexpr GridNodeIndex() = default;

  /* Constructor for an active index. */
  explicit constexpr GridNodeIndex(T index) : value_(index) {}

  /* Sets the index to the given value, which must be non-negative. Turns `this`
  into active state. */
  void set_value(T index) {
    DRAKE_ASSERT(index >= 0);
    value_ = index;
  }

  /* Returns true if the index is active. */
  bool is_index() const { return value_ >= 0; }

  /* Returns true iff the index is in generic inactive state. */
  bool is_inactive() const { return value_ == kInactive; }

  /* Returns the index value.
   @pre is_index() == true; */
  T value() const {
    DRAKE_ASSERT(is_index());
    return value_;
  }

  /* Sets `this` to the generic inactive state. */
  void reset() { value_ = kInactive; }
  /* Sets `this` to the participating state. */
  void set_participating() {
    DRAKE_ASSERT(!is_index());
    value_ = kParticipating;
  }
  /* Returns true iff `this` is in the participating state. */
  bool is_participating() const { return value_ == kParticipating; }

 private:
  template <typename U>
  friend bool operator==(const GridNodeIndex<U>& a, const GridNodeIndex<U>& b);

  static constexpr T kInactive{-1};
  static constexpr T kParticipating{-2};
  T value_{kInactive};
};

template <typename T>
bool operator==(const GridNodeIndex<T>& a, const GridNodeIndex<T>& b) {
  return a.value_ == b.value_;
}

/* GridData stores data at a single a grid node of SparseGrid.

 The Vector3<T> entry contains the velocity of the node (sometimes used
 temporarily to store the momentum of the node), and the scalar entry is mass of
 the node.

 The size of GridData is required to be a power of 2 to work with SPGrid.
 With T = float, GridData is 4 * 8 = 32 byte.
 With T = double, GridData is 8 * 8 = 64 byte.

 @tparam T double or float. */
template <typename T>
struct GridData {
  void set_zero() {
    v.setZero();
    m = 0.0;
    index.reset();
  }

  bool operator==(const GridData<T>& other) const = default;

  Vector3<T> v{Vector3<T>::Zero()};
  T m{0.0};
  typename std::conditional<std::is_same<T, float>::value,
                            GridNodeIndex<int32_t>,
                            GridNodeIndex<int64_t>>::type index;
  Vector3<T> scratch{Vector3<T>::Zero()};
};

/* A Pad is a 3x3x3 subgrid around a particle.

 We use quadratic B-spline kernel to compute the interaction weight between
 particles and a grid node, and the support of a single particle is 3x3x3 grid
 nodes. The Pad is a 3x3x3 grid that stores the grid data of the 3x3x3 neighbors
 of a grid node that is affected by a particle (or a group of particles that
 share the same support). */
template <typename T>
using Pad = std::array<std::array<std::array<T, 3>, 3>, 3>;

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
