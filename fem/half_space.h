#pragma once

#include <vector>
#include <limits>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/unused.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/analytic_level_set.h"

namespace drake {
namespace fem {
template <typename T>
class HalfSpace final : public AnalyticLevelSet<T> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(HalfSpace)

  HalfSpace(const Vector3<T>& origin, const Vector3<T>& normal)
      : origin_(origin), normal_(normal) {
    DRAKE_DEMAND(normal_.norm() > std::numeric_limits<T>::epsilon());
    normal_.normalize();
  }

  virtual ~HalfSpace() {}

 protected:
  virtual Vector3<T> DoNormal(const Vector3<T>& x) const {
      unused(x);
      return normal_;
  }

  virtual T DoSignedDistance(const Vector3<T>& x) const {
    return normal_.dot(x - origin_);
  }

 private:
  Vector3<T> origin_;
  Vector3<T> normal_;
};
}  // namespace fem
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::HalfSpace)

