#pragma once

#include <memory>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

Eigen::Vector3d AnalyticalProjection(const Eigen::Vector3d& y, double Rt,
                                     double Rn, double mu,
                                     double epsilon = 1.0e-7) {
  using std::sqrt;
  const auto yt = y.head<2>();
  const double yn = y(2);
  const double yt_squared = yt.squaredNorm();
  const double yr = sqrt(yt_squared);
  const double Rt_over_Rn = Rt / Rn;

  if (yr <= mu * yn) {
    // Within the friction cone.
    return y;
  } else if (-mu * Rt_over_Rn * yr < yn) {
    // Outside the cone "Region II".
    // We need to "project".
    const double eps2 = epsilon * epsilon;
    const double yr_soft = sqrt(yt_squared + eps2);
    const Eigen::Vector2d t = yt / yr_soft;
    const double denom = 1.0 + mu * mu * Rt_over_Rn;
    const double gn = (yn + mu * Rt_over_Rn * yr) / denom;
    const double gr = mu * gn;
    return Eigen::Vector3d(gr * t(0), gr * t(1), gn);
  } else {
    // within the polar cone.
    return Eigen::Vector3d::Zero();
  }
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
