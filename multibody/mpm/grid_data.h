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
     for deferred processing. The participating state is intended as a marker
     that must only be set from the inactive state (not from an active index)
     to maintain consistent workflow logic.

  Transitions between states are as follows:
  - Any state can become inactive.
  - Any inactive state can become participating.
  - Any inactive state can become active (with a non-negative index).
  - Active cannot directly become participating (must go inactive first).

  A GridNodeIndex object is guaranteed to have size equal to its template
  parameter T.

  @tparam T The integer type for the index. Must be `int32_t` or `int64_t`. */
template <typename T>
class GridNodeIndex {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(GridNodeIndex);

  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>,
                "T must be int32_t or int64_t.");

  /* Default constructor initializes the index to the inactive state. */
  constexpr GridNodeIndex() = default;

  /* Constructor for an active index.
     @pre index >= 0 */
  explicit constexpr GridNodeIndex(T index) {
    DRAKE_ASSERT(index >= 0);
    value_ = index;
  }

  /* Sets the index to the given value, which must be non-negative, thereby
     making `this` active.
     @pre index >= 0 */
  void set_value(T index) {
    DRAKE_ASSERT(index >= 0);
    value_ = index;
  }

  /* Returns true if the index is active (i.e., a non-negative integer). */
  constexpr bool is_index() const { return value_ >= 0; }

  /* Returns true iff the index is in the generic inactive state. */
  constexpr bool is_inactive() const { return value_ == kInactive; }

  /* Returns true iff `this` is in the participating state. */
  constexpr bool is_participating() const { return value_ == kParticipating; }

  /* Returns the index value.
     @pre is_index() == true; */
  constexpr T value() const {
    DRAKE_ASSERT(is_index());
    return value_;
  }

  /* Sets `this` to the generic inactive state. */
  void reset() { value_ = kInactive; }

  /* Sets `this` to the participating state.
     @pre !is_index() (i.e., must currently be inactive) */
  void set_participating() {
    DRAKE_ASSERT(!is_index());
    value_ = kParticipating;
  }

 private:
  enum : T { kInactive = -1, kParticipating = -2 };

  template <typename U>
  friend bool operator==(const GridNodeIndex<U>& a, const GridNodeIndex<U>& b);

  template <typename U>
  friend bool operator!=(const GridNodeIndex<U>& a, const GridNodeIndex<U>& b);

  T value_{kInactive};
};

/* Equality operator. Two GridNodeIndex objects are equal if and only if their
   internal values are the same. */
template <typename U>
inline bool operator==(const GridNodeIndex<U>& a, const GridNodeIndex<U>& b) {
  return a.value_ == b.value_;
}

/* Inequality operator. */
template <typename U>
inline bool operator!=(const GridNodeIndex<U>& a, const GridNodeIndex<U>& b) {
  return !(a == b);
}

/* GridData stores data at a single grid node of SparseGrid.

 The Vector3<T> entry contains the velocity of the node (sometimes used
 temporarily to store the momentum of the node), and the scalar entry is the
 mass of the node.

 It's important to be conscious of the size of GridData since the MPM algorithm
 is usually memory-bound. We carefully pack GridData to be a power of 2 to work
 with SPGrid, which automatically packs the data to the next power of 2.

 @tparam T double or float. */
template <typename T>
struct GridData {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "T must be float or double.");

  /* Resets `this` GridData to its default state. */
  void set_zero() {
    v.setZero();
    m = 0.0;
    index.reset();
  }

  /* Default equality operator to compare all members. */
  bool operator==(const GridData<T>& other) const = default;

  Vector3<T> v{Vector3<T>::Zero()};
  T m{0.0};
  Vector3<T> scratch{Vector3<T>::Zero()};
  typename std::conditional<std::is_same_v<T, float>, GridNodeIndex<int32_t>,
                            GridNodeIndex<int64_t>>::type index;
};

/* With T = float, GridData is expected to be 32 bytes. With T = double,
 GridData is expected to be 64 bytes. We enforce these sizes at compile time
 with static_assert, so that if future changes to this code, compiler alignment,
 or Eigen alignment rules cause a size shift, it will be caught early. */
static_assert(sizeof(GridData<float>) == 32,
              "Unexpected size for GridData<float>.");
static_assert(sizeof(GridData<double>) == 64,
              "Unexpected size for GridData<double>.");

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
