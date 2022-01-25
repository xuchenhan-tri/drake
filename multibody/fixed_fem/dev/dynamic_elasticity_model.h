#pragma once

#include <array>
#include <memory>
#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fixed_fem/dev/damping_model.h"
#include "drake/multibody/fixed_fem/dev/elasticity_model.h"

namespace drake {
namespace multibody {
namespace fem {
/** The FEM model for dynamic 3D elasticity problems. Implements the interface
 in FemModel. It is assumed that elements are only added to, but never deleted
 from, the model.
 @tparam Element    The type of DynamicElasticityElement used in this
 %DynamicElasticityModel, must be an instantiation of DynamicElasticityElement.
 */
template <class Element>
class DynamicElasticityModel : public ElasticityModel<Element> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DynamicElasticityModel);

  using T = typename Element::Traits::T;

  // TODO(xuchenhan-tri): Currently the time stepping scheme is hard coded to
  //  gamma = 1.0 and beta = 0.5. Consider letting the user configure the time
  //  stepping scheme.
  /** Creates a new %DynamicElasticityModel with the given discrete time step.
   */
  explicit DynamicElasticityModel(double dt)
      : ElasticityModel<Element>(
            std::make_unique<internal::AccelerationNewmarkScheme<T>>(dt, 1.0,
                                                                     0.5)) {}

  ~DynamicElasticityModel() = default;


 private:
};
}  // namespace fem
}  // namespace multibody
}  // namespace drake
