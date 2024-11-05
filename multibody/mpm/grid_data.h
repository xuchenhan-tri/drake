#pragma once

#include <array>

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

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
    index = -1;
  }

  bool operator==(const GridData<T>& other) const = default;

  Vector3<T> v{Vector3<T>::Zero()};
  T m{0.0};
  typename std::conditional<std::is_same<T, float>::value, int32_t,
                            int64_t>::type index{-1};
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
