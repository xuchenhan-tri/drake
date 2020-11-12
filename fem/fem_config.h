#pragma once

namespace drake {
namespace fem {
// TODO(xuchenhan-tri): Move initialization to a constructor that throws for invalid configurations.
struct MaterialConfig {
  double density{0};
  double youngs_modulus{0};
  double poisson_ratio{-2};
  double mass_damping{-1};
  double stiffness_damping{-1};
  bool is_valid() const {
    return density > 0 && youngs_modulus > 0 && poisson_ratio > -1 &&
           poisson_ratio < 0.5 && mass_damping >= 0 && stiffness_damping >= 0;
  }
};
}  // namespace fem
}  // namespace drake
