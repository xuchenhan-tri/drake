#pragma once

#include "drake/multibody/fixed_fem/dev/fem_model.h"
#include "drake/multibody/fixed_fem/dev/test/dummy_element.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace test {

/* A dummy FemModel with a single DummyElement for testing purpose. */
class DummyModel final : public FemModel<DummyElement> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DummyModel);

  using Traits = DummyElement::Traits;
  using T = Traits::T;
  using ConstitutiveModel = typename Traits::ConstitutiveModel;
  static constexpr int kNumDofs = Traits::num_dofs;
  const T kYoungsModulus = 1e7;
  const T kPoissonRatio = 0.49;
  const T kMassDamping = 0.001;
  const T kStiffnessDamping = 0.02;

  /* Creates a dummy FEM model with a single element. */
  DummyModel() {
    const ElementIndex element_index(0);
    const std::array node_indices = {NodeIndex(0), NodeIndex(1)};
    const ConstitutiveModel constitutive_model(1e7, 0.49);
    const DampingModel<T> damping_model(kMassDamping, kStiffnessDamping);
    this->AddElement(element_index, node_indices, constitutive_model,
                     damping_model);
    increment_num_nodes(2);
  }

 private:
  /* Creates an all-zero FEM state for the dummy model. */
  FemState<DummyElement> DoMakeFemState() const final {
    return FemState<DummyElement>(VectorX<T>::Zero(kNumDofs),
                                  VectorX<T>::Zero(kNumDofs),
                                  VectorX<T>::Zero(kNumDofs));
  }
};

}  // namespace test
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake