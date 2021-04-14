#pragma once
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace multibody {
template <typename T>
class MultibodyPlant;

namespace fixed_fem {
/** A pure virtual softsim utility class. */
template <typename T>
class SoftsimBase {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SoftsimBase)

  explicit SoftsimBase(multibody::MultibodyPlant<T>* mbp) : mbp_(mbp) {
    DRAKE_DEMAND(mbp_ != nullptr);
  }

  virtual ~SoftsimBase() = default;

  /* Registers a collision object used for computing rigid-deformable contact
   information given a collision geometry in the MultibodyPlant associated with
   this SoftsimBase.
   @param geometry_id   The GeometryId of the collision geometry.
   @param shape         The shape of the collision geometry.
   @param properties    The proximity properties of the collision geometry.
   @throws std::exception if `geometry_id` is not registered in the associated
   Multibodyplant or if `geometry_id` is already registered in this SoftsimBase.
  */
  virtual void RegisterCollisionObject(
      geometry::GeometryId geometry_id, const geometry::Shape& shape,
      geometry::ProximityProperties properties) = 0;

 private:
  MultibodyPlant<T>* mbp_;
};
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
