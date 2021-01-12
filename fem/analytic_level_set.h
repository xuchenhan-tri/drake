#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {
template <typename T>
class AnalyticLevelSet {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(AnalyticLevelSet)
  AnalyticLevelSet() {}
  virtual ~AnalyticLevelSet() {}

  /** Evaluates the outward normal of the collision object at a point x in
   * material space. */
  Vector3<T> Normal(const Vector3<T>& x) const { return DoNormal(x); }

  /** Evaluates the signed distance to the surface of the collision object at a
   * point x in material space. */
  T SignedDistance(const Vector3<T>& x) const { return DoSignedDistance(x); }

  bool is_inside(const Vector3<T>& x) const {
    return SignedDistance(x) <= T(0);
  }

  /** Returns the signed distance to the surface of the collision object at a
   * point x in material space and fills the normal to the collision object in
   * material space at x. */
  T Query(const Vector3<T>& x, Vector3<T>* normal) const {
    *normal = Normal(x);
    return SignedDistance(x);
  }

 protected:
  virtual Vector3<T> DoNormal(const Vector3<T>& x) const = 0;

  virtual T DoSignedDistance(const Vector3<T>& x) const = 0;
};
}  // namespace fem
}  // namespace drake
