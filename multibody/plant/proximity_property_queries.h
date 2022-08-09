#pragma once

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph_inspector.h"

namespace drake {
namespace multibody {
namespace internal {

// Returns the point contact stiffness stored in group
// geometry::internal::kMaterialGroup with property
// geometry::internal::kPointStiffness for the specified geometry.
// If the stiffness property is absent, it returns the supplied default value.
// @pre id is a valid GeometryId in the inspector.
template <typename T>
T GetPointContactStiffness(geometry::GeometryId id,
                           const geometry::SceneGraphInspector<T>& inspector,
                           double default_value) {
  const geometry::ProximityProperties* prop =
      inspector.GetProximityProperties(id);
  DRAKE_DEMAND(prop != nullptr);
  // N.B. Here we rely on the resolution of #13289 and #5454 to get properties
  // with the proper scalar type T. This will not work on scalar converted
  // models until those issues are resolved.
  return prop->template GetPropertyOrDefault<T>(
      geometry::internal::kMaterialGroup, geometry::internal::kPointStiffness,
      default_value);
}

// Returns the dissipation time constant stored in group
// geometry::internal::kMaterialGroup with property
// "dissipation_time_constant".
// If the property is absent, it returns the supplied default value.
// @throws std::exceptionif the .
// @pre id is a valid GeometryId in the inspector.
template <typename T>
T GetDissipationTimeConstant(geometry::GeometryId id,
                             const geometry::SceneGraphInspector<T>& inspector,
                             double default_value,
                             const std::string& body_name) {
  const geometry::ProximityProperties* prop =
      inspector.GetProximityProperties(id);
  DRAKE_DEMAND(prop != nullptr);

  auto provide_context_string =
      [&inspector,
       &body_name](geometry::GeometryId geometry_id) -> std::string {
    return fmt::format("For geometry {} on body {}.",
                       inspector.GetName(geometry_id), body_name);
  };

  // N.B. Here we rely on the resolution of #13289 and #5454 to get properties
  // with the proper scalar type T. This will not work on scalar converted
  // models until those issues are resolved.
  const T relaxation_time = prop->template GetPropertyOrDefault<double>(
      geometry::internal::kMaterialGroup, "relaxation_time", default_value);
  if (relaxation_time < 0.0) {
    const std::string message = fmt::format(
        "Relaxation time must be non-negative and relaxation_time "
        "= {} was provided. {}",
        relaxation_time, provide_context_string(id));
    throw std::runtime_error(message);
  }
  return relaxation_time;
}

// Returns the coefficient of dynamic friction stored by SceneGraph.
// @pre id is a valid GeometryId in the inspector.
template <typename T>
double GetCoulombFriction(geometry::GeometryId id,
                          const geometry::SceneGraphInspector<T>& inspector) {
  const geometry::ProximityProperties* prop =
      inspector.GetProximityProperties(id);
  DRAKE_DEMAND(prop != nullptr);
  DRAKE_THROW_UNLESS(prop->HasProperty(geometry::internal::kMaterialGroup,
                                       geometry::internal::kFriction));
  return prop
      ->GetProperty<CoulombFriction<double>>(geometry::internal::kMaterialGroup,
                                             geometry::internal::kFriction)
      .dynamic_friction();
}

// Utility to combine stiffnesses k1 and k2 according to the rule:
//   k  = k₁⋅k₂/(k₁+k₂)
// In other words, the combined compliance (the inverse of stiffness) is the
// sum of the individual compliances.
template <typename T>
T CombineStiffnesses(const T& k1, const T& k2) {
  // Simple utility to detect 0 / 0. As it is used in this method, denom
  // can only be zero if num is also zero, so we'll simply return zero.
  auto safe_divide = [](const T& num, const T& denom) {
    return denom == 0.0 ? 0.0 : num / denom;
  };
  return safe_divide(k1 * k2, k1 + k2);
}

// Utility to combine linear dissipation time constants. Consider two
// spring-dampers with stiffnesses k₁ and k₂, and dissipation timescales τ₁
// and τ₂, respectively. When these spring-dampers are connected in series,
// they result in an equivalent spring-damper with stiffness k  =
// k₁⋅k₂/(k₁+k₂) and dissipation τ = τ₁ + τ₂.
// This method returns tau1 + tau2.
template <typename T>
T CombineDissipationTimeConstant(const T& tau1, const T& tau2) {
  return tau1 + tau2;
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake
