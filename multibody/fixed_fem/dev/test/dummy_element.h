#pragma once

#include <array>

#include "drake/multibody/fixed_fem/dev/damping_model.h"
#include "drake/multibody/fixed_fem/dev/element_cache_entry.h"
#include "drake/multibody/fixed_fem/dev/fem_element.h"
#include "drake/multibody/fixed_fem/dev/fem_state.h"
#include "drake/multibody/fixed_fem/dev/linear_constitutive_model.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace test {

/* The traits for the DummyElement. In this case, all of the traits are unique
 values so we can detect that each value is used in the expected context. */
struct DummyElementTraits {
  using T = double;
  struct Data {
    double value{0};
  };
  static constexpr int num_quadrature_points = 1;
  static constexpr int num_nodes = 2;
  static constexpr int num_natural_dimension = 3;
  static constexpr int kSpatialDimension = 4;
  static constexpr int num_dofs = 5;
  using ConstitutiveModel = LinearConstitutiveModel<T, num_quadrature_points>;
};

/* A simple FemElement implementation. The calculation methods are implemented
 as returning a fixed value (which can independently be accessed by calling
 the corresponding dummy_* method -- e.g., CalcResidual() should return the
 value in dummy_residual(). */
class DummyElement final : public FemElement<DummyElement, DummyElementTraits> {
 public:
  using Base = FemElement<DummyElement, DummyElementTraits>;
  using Traits = DummyElementTraits;
  using ConstitutiveModel = typename Traits::ConstitutiveModel;
  using T = typename Base::T;

  DummyElement(ElementIndex element_index,
               const std::array<NodeIndex, Traits::num_nodes>& node_indices,
               const ConstitutiveModel& constitutive_model,
               const DampingModel<T>& damping_model)
      : Base(element_index, node_indices, constitutive_model, damping_model) {}

  /* Provides a fixed return value for CalcResidual(). */
  static Vector<T, Traits::num_dofs> dummy_residual() {
    return Vector<T, Traits::num_dofs>::Constant(1.23456);
  }

  /* Provides a fixed return value for CalcStiffnessMatrix(). */
  static Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>
  dummy_stiffness_matrix() {
    return Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>::Constant(1.23);
  }

  /* Provides a fixed return value for CalcDampingMatrix(). */
  static Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>
  dummy_damping_matrix() {
    return Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>::Constant(4.56);
  }

  /* Provides a fixed return value for CalcMassMatrix(). */
  static Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>
  dummy_mass_matrix() {
    return Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>::Constant(7.89);
  }

  /* Provides a fixed value for the `Data` for `ComputeData()`. */
  static typename Traits::Data dummy_data() { return {1.732}; }

 private:
  /* Friend the base class so that the interface in the CRTP base class can
   access the private implementations of this class. */
  friend Base;

  /* Implements FemElement::ComputeData(). Returns a dummy data if `state` is
    empty. Otherwise return the sum of the last entries in each state. */
  typename Traits::Data DoComputeData(
      const FemState<DummyElement>& state) const {
    const int num_dofs = state.num_dofs();
    if (num_dofs == 0) {
      return dummy_data();
    }
    typename Traits::Data data;
    data.value = state.GetPositions()(num_dofs - 1);
    data.value += state.GetVelocities()(num_dofs - 1);
    data.value += state.GetAccelerations()(num_dofs - 1);
    return data;
  }

  /* Implements FemElement::CalcResidual(). */
  void DoCalcResidual(const FemState<DummyElement>&,
                      EigenPtr<Vector<T, Traits::num_dofs>> residual) const {
    *residual = dummy_residual();
  }

  /* Implements FemElement::AddScaledStiffnessMatrix(). */
  void DoAddScaledStiffnessMatrix(
      const FemState<DummyElement>&, const T& scale,
      EigenPtr<Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>> K) const {
    *K += scale * dummy_stiffness_matrix();
  }

  /* Implements FemElement::AddScaledDampingMatrix(). */
  void DoAddScaledDampingMatrix(
      const FemState<DummyElement>&, const T& scale,
      EigenPtr<Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>> D) const {
    *D += scale * dummy_damping_matrix();
  }

  /* Implements FemElement::AddScaledMassMatrix(). */
  void DoAddScaledMassMatrix(
      const FemState<DummyElement>&, const T& scale,
      EigenPtr<Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>> M) const {
    *M += scale * dummy_mass_matrix();
  }
};

}  // namespace test
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
