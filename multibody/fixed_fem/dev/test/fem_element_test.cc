#include "drake/multibody/fixed_fem/dev/fem_element.h"

#include <gtest/gtest.h>

#include "drake/multibody/fixed_fem/dev/test/dummy_element.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace test {
namespace {

using T = DummyElementTraits::T;
const ElementIndex kZeroIndex = ElementIndex(0);
const std::array<NodeIndex, DummyElementTraits::num_nodes> kNodeIndices = {
    {NodeIndex(0), NodeIndex(1)}};
const DummyElementTraits::ConstitutiveModel kConstitutiveModel(5e4, 0.4);
const DampingModel<T> kDampingModel(0.01, 0.02);

/* An minimal FemElement to test FemElement::CalcFoo() methods. */
/*
class CalcFooElement final
    : public FemElement<CalcFooElement, DummyElementTraits> {
 public:
  using Base = FemElement<CalcFooElement, DummyElementTraits>;
  CalcFooElement(ElementIndex element_index,
                 const std::array<NodeIndex, Traits::num_nodes>& node_indices,
                 double value)
      : Base(element_index, node_indices), value_(value) {}

  Vector<T, Traits::num_dofs> expected_residual() const {
    return Vector<T, Traits::num_dofs>::Constant(value_);
  }

  Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs> expected_matrix() const {
    return Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>::Constant(
        value_);
  }

 private:
  friend Base;

  void DoCalcResidual(const FemState<CalcFooElement>& state,
                      EigenPtr<Vector<T, Traits::num_dofs>> residual) const {
    for (int i = 0; i < Traits::num_dofs; ++i) {
      if ((*residual)(i) != 0) {
        throw std::runtime_error("Input vector non-zero!");
      }
      (*residual)(i) = value_;
    }
  }

  void DoCalcStiffnessMatrix(
      const FemState<CalcFooElement>& state,
      EigenPtr<Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>> K) const {
    VerifyInputIsZeroAndOverwriteWithConstant(K);
  }

  void DoCalcDampingMatrix(
      const FemState<CalcFooElement>& state,
      EigenPtr<Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>> D) const {
    VerifyInputIsZeroAndOverwriteWithConstant(D);
  }

  void DoCalcMassMatrix(
      const FemState<CalcFooElement>& state,
      EigenPtr<Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>> M) const {
    *M = expected_matrix();
  }

  void VerifyInputIsZeroAndOverwriteWithConstant(
      EigenPtr<Eigen::Matrix<T, Traits::num_dofs, Traits::num_dofs>> matrix)
      const {
    for (int i = 0; i < Traits::num_dofs; ++i) {
      for (int j = 0; j < Traits::num_dofs; ++j) {
        if ((*matrix)(i, j) != 0) {
          throw std::runtime_error("Input vector non-zero!");
        }
        (*matrix)(i, j) = value_;
      }
    }
  }

  double value_;
};
*/

class FemElementTest : public ::testing::Test {
 protected:
  /* Default values for the state. */
  static VectorX<double> q() { return Vector3<double>(0.1, 0.2, 0.3); }
  static VectorX<double> v() { return Vector3<double>(0.3, 0.4, 0.5); }
  static VectorX<double> a() { return Vector3<double>(0.6, 0.7, 0.8); }

  /* FemElement under test. */
  DummyElement element_{kZeroIndex, kNodeIndices, kConstitutiveModel,
                        kDampingModel};
  FemState<DummyElement> state_{q(), v(), a()};
};

TEST_F(FemElementTest, Constructor) {
  EXPECT_EQ(element_.node_indices(), kNodeIndices);
  EXPECT_EQ(element_.element_index(), kZeroIndex);
}

/* The following tests confirm that CalcResidual(), AddScaledStiffnessMatrix,
 AddScaledDampingMatrix(), DoAddScaledMassMatrix(), correctly invoke their
 DoCalc and DoAdd counterparts. We confirm this with a custom subclass of
 FemElement whose implementation returns/adds a specific value. */
TEST_F(FemElementTest, Residual) {
  Vector<T, DummyElementTraits::num_dofs> residual;
  element_.CalcResidual(state_, &residual);
  EXPECT_EQ(residual, element_.dummy_residual());
}

TEST_F(FemElementTest, StiffnessMatrix) {
  Eigen::Matrix<T, DummyElementTraits::num_dofs, DummyElementTraits::num_dofs>
      K;
  K.setZero();
  const T scale = 3.14;
  element_.AddScaledStiffnessMatrix(state_, scale, &K);
  EXPECT_EQ(K, scale * element_.dummy_stiffness_matrix());
}

TEST_F(FemElementTest, DampingMatrix) {
  Eigen::Matrix<T, DummyElementTraits::num_dofs, DummyElementTraits::num_dofs>
      D;
  D.setZero();
  const T scale = 3.14;
  element_.AddScaledDampingMatrix(state_, scale, &D);
  EXPECT_EQ(D, scale * element_.dummy_damping_matrix());
}

/* Test that CalcMassMatrix() is calling the expected DoCalcMassMatrix(). */
TEST_F(FemElementTest, MassMatrix) {
  Eigen::Matrix<T, DummyElementTraits::num_dofs, DummyElementTraits::num_dofs>
      M;
  M.setZero();
  const T scale = 3.14;
  element_.AddScaledMassMatrix(state_, scale, &M);
  EXPECT_EQ(M, scale * element_.dummy_mass_matrix());
}

}  // namespace
}  // namespace test
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
