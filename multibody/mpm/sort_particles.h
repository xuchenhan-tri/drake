#pragma once

#include "particles.h"
#include "spgrid.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

// TODO(xuchenhan-tri): Remove dependency on SPGrid and depend on its flags
// only.
/* Sorts particles based on their positions in the background grid. */
template <typename T, typename SpGrid>
void SortParticles(const SpGrid& spgrid, double dx, ParticleData<T>* particles,
                   Parallelism parallelism = false);

/* Sort the given particle positions in place first according to their base
 node offsets and then according to their indices in `q_WPs`.
@pre q_WPs != nullptr.
@pre q_WPs->size() < 2^31. */
template <typename T, typename SpGrid>
void SortParticlePositions(std::vector<Vector3<double>>* q_WPs, T dx);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake