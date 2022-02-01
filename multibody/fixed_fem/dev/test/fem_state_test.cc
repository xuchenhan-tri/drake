#include "drake/multibody/fixed_fem/dev/fem_state.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/multibody/fixed_fem/dev/element_cache_entry.h"
#include "drake/multibody/fixed_fem/dev/test/dummy_element.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

using test::DummyElement;
using test::DummyElementTraits;
using T = DummyElementTraits::T;
constexpr int kNumDofs = 3;
constexpr int kNumElements = 2;
const std::vector<ElementIndex> kElementIndices{ElementIndex(0),
                                                ElementIndex(1)};
const std::array<NodeIndex, DummyElementTraits::num_nodes> kNodeIndices = {
    {NodeIndex(0), NodeIndex(1)}};
const DummyElementTraits::ConstitutiveModel kConstitutiveModel(5e4, 0.4);
const DampingModel<T> kDampingModel(0.01, 0.02);

using Eigen::VectorXd;

class FemStateImplTest : public ::testing::Test {
 protected:
  void SetUp() {
    for (ElementIndex element_id : kElementIndices) {
      elements_.emplace_back(element_id, kNodeIndices, kConstitutiveModel,
                             kDampingModel);
    }
    state_.MakeElementData(elements_);
  }

  /* Default values for the state. */
  static VectorX<double> q() { return Vector3<double>(0.1, 0.2, 0.3); }
  static VectorX<double> v() { return Vector3<double>(0.3, 0.4, 0.5); }
  static VectorX<double> a() { return Vector3<double>(0.6, 0.7, 0.8); }

  const ElementCacheEntry<DummyElement::Traits::Data>& cache_entry(
      int i) const {
    return state_.element_cache_[i];
  }

  /* For each cache entry,
   1) Verify it is stale initially,
   2) Verify calling element_data() provides the correct result, and
   3) Verify it is not stale after the call to element_data(). */
  void VerifyCacheEntries() const {
    for (int i = 0; i < kNumElements; ++i) {
      const auto& element_cache_entry = cache_entry(i);
      EXPECT_TRUE(element_cache_entry.is_stale());
      /* For DummyElement, the element data value is set to be the sum of the
       last entries of all the states. */
      EXPECT_EQ(
          (state_.GetPositions().tail(1) + state_.GetVelocities().tail(1) +
           state_.GetAccelerations().tail(1))(0),
          state_.element_data(elements_[i]).value);
      EXPECT_FALSE(element_cache_entry.is_stale());
    }
  }

  /* FemStateImpl under test. */
  FemStateImpl<DummyElement> state_{q(), v(), a()};
  std::vector<DummyElement> elements_;
};

namespace {

/* Verify setters and getters are working properly. */
TEST_F(FemStateImplTest, GetStates) {
  EXPECT_EQ(state_.num_dofs(), kNumDofs);
  EXPECT_EQ(state_.GetPositions(), q());
  EXPECT_EQ(state_.GetVelocities(), v());
  EXPECT_EQ(state_.GetAccelerations(), a());
}

TEST_F(FemStateImplTest, SetStates) {
  state_.SetPositions(-1.23 * q());
  state_.SetVelocities(3.14 * v());
  state_.SetAccelerations(-1.29 * a());
  EXPECT_EQ(state_.GetPositions(), -1.23 * q());
  EXPECT_EQ(state_.GetVelocities(), 3.14 * v());
  EXPECT_EQ(state_.GetAccelerations(), -1.29 * a());
  /* Setting values with incompatible sizes should throw. */
  EXPECT_THROW(state_.SetPositions(VectorXd::Constant(1, 1.0)), std::exception);
  EXPECT_THROW(state_.SetVelocities(VectorXd::Constant(1, 1.0)),
               std::exception);
  EXPECT_THROW(state_.SetAccelerations(VectorXd::Constant(1, 1.0)),
               std::exception);
}

/* Test that element_data() retrieves the updated data. */
TEST_F(FemStateImplTest, ElementData) {
  EXPECT_EQ(state_.element_cache_size(), kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(state_.GetPositions()(kNumDofs - 1) +
                  state_.GetVelocities()(kNumDofs - 1) +
                  state_.GetAccelerations()(kNumDofs - 1),
              state_.element_data(elements_[i]).value);
  }
}

TEST_F(FemStateImplTest, MakeElementData) {
  std::vector<DummyElement> invalid_ordered_elements;
  invalid_ordered_elements.emplace_back(ElementIndex(1), kNodeIndices,
                                        kConstitutiveModel, kDampingModel);
  DRAKE_EXPECT_THROWS_MESSAGE(
      state_.MakeElementData(invalid_ordered_elements),
      "Input element entry at 0 has index 1 instead of 0. The entry with index "
      "i must be stored at position i.");
}

/* Tests that element data cache is invalidated when the state changes and that
 the request for the cached data triggers appropriate recalculations. */
TEST_F(FemStateImplTest, ElementCache) {
  /* Verify that cache entries are initially invalid and becomes valid after the
   request for data triggers computation. */
  VerifyCacheEntries();

  /* Verify that state setters thrash the cache entries and the cached
   quantities are correctly recomputed. */
  state_.SetPositions(2 * q());
  VerifyCacheEntries();
  state_.SetVelocities(2 * v());
  VerifyCacheEntries();
  state_.SetAccelerations(2 * a());
  VerifyCacheEntries();
}

}  // namespace
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
