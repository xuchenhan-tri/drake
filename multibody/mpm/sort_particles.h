#pragma once
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* Sort the given particle positions in place first according to their base
 node offsets and then according to their indices in `q_WPs`.
@pre q_WPs != nullptr.
@pre q_WPs->size() < 2^31. */
template <typename T, typename SpGrid>
void SortParticlePositions(const SpGrid& spgrid,
                           std::vector<Vector3<T>>* q_WPs, T dx);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake