#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "drake/multibody/plant/multibody_plant.h"
namespace drake {
namespace multibody {
namespace fixed_fem {
/** A pure virtual softsim utility class. */
template <typename T>
class SoftsimBase {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SoftsimBase)

  explicit SoftsimBase(MultibodyPlant<T>* mbp) : mbp_(mbp) {
    DRAKE_DEMAND(mbp_ != nullptr);
  }

  virtual ~SoftsimBase() = 0;

  /* Registers a collision object used for computing rigid-deformable contact
   information given a collision geometry in the MultibodyPlant associated with
   this SoftsimBase.
   @param geometry_id   The GeometryId of the collision geometry.
   @param shape         The shape of the collision geometry.
   @param properties    The proximity properties of the collision geometry.
   @throws std::exception if `geometry_id` is not registered in the associated
   Multibodyplant or if `geometry_id` is already registered in this SoftsimBase.
  */
  void RegisterCollisionObject(geometry::GeometryId geometry_id,
                               const geometry::Shape& shape,
                               geometry::ProximityProperties properties) final;

 private:
  MultibodyPlant<T>* mbp_;
};
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
