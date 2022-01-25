#pragma once

#include <array>

#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/multibody/fixed_fem/dev/elasticity_element.h"

namespace drake {
namespace multibody {
namespace fem {
/** Traits class for FEM dynamic elasticity. */
template <class IsoparametricElementType, class QuadratureType,
          class ConstitutiveModelType>
struct DynamicElasticityElementTraits
    : public ElasticityElementTraits<IsoparametricElementType, QuadratureType,
                                     ConstitutiveModelType> {
  /* The dynamic elasticity problem forms a second order ODE. */
  static constexpr int kOdeOrder = 2;
};

/** The FEM element class for dynamic 3D elasticity problems that model
 inertia effects and damping forces. Implements the CRTP base class,
 ElasticityElement. See ElasticityElement for documentation on the template
 paramenters. */
template <class IsoparametricElementType, class QuadratureType,
          class ConstitutiveModelType>
class DynamicElasticityElement final
    : public ElasticityElement<
          IsoparametricElementType, QuadratureType, ConstitutiveModelType,
          DynamicElasticityElement<IsoparametricElementType, QuadratureType,
                                   ConstitutiveModelType>,
          DynamicElasticityElementTraits<IsoparametricElementType,
                                         QuadratureType,
                                         ConstitutiveModelType>> {
 public:
  /** Assignment and copy constructions are prohibited. Move constructor is
   allowed so that DynamicElasticityElement can be stored in `std::vector`. */
  DynamicElasticityElement(const DynamicElasticityElement&) = delete;
  DynamicElasticityElement(DynamicElasticityElement&&) = default;
  const DynamicElasticityElement& operator=(const DynamicElasticityElement&) =
      delete;
  DynamicElasticityElement&& operator=(const DynamicElasticityElement&&) =
      delete;

  using Traits =
      DynamicElasticityElementTraits<IsoparametricElementType, QuadratureType,
                                     ConstitutiveModelType>;
  using T = typename Traits::T;

  /** Constructs a new FEM dynamic elasticity element.
   @pre density > 0. */
  DynamicElasticityElement(
      ElementIndex element_index,
      const std::array<NodeIndex, Traits::kNumNodes>& node_indices,
      const ConstitutiveModelType& constitutive_model,
      const Eigen::Ref<const Eigen::Matrix<T, Traits::kSolutionDimension,
                                           Traits::kNumNodes>>&
          reference_positions,
      const T& density, const Vector<T, Traits::kSpatialDimension>& gravity,
      : ElasticityElementType(element_index, node_indices, constitutive_model,
                              reference_positions, density, gravity),
        damping_model_(damping_model) {}

 private:
  /* Type alias for convenience and readability. */
  using ElementType =
      DynamicElasticityElement<IsoparametricElementType, QuadratureType,
                               ConstitutiveModelType>;
  using ElasticityElementType =
      ElasticityElement<IsoparametricElementType, QuadratureType,
                        ConstitutiveModelType, ElementType, Traits>;
  using FemElementType = FemElement<ElementType, Traits>;
  /* Friend the base class so that the interface in the CRTP base class can
   access the private implementations of this class. */
  friend FemElementType;
  friend class DynamicElasticityElementTest;

  DampingModel<T> damping_model_;
};
}  // namespace fem
}  // namespace multibody
}  // namespace drake
