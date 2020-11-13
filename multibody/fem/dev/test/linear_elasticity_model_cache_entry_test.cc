#include "drake/multibody/fem/dev/linear_elasticity_model_cache_entry.h"

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace fem {
namespace {
const ElementIndex kElementIndex(3);
constexpr int kNumQuads = 1;

class LinearElasticityCacheEntryTest : public ::testing::Test {
 protected:
  void SetUp() {
    linear_elasticity_cache_entry_.UpdateCacheEntry(
        {MakeDeformationGradient()});
  }
  LinearElasticityModelCacheEntry<double> linear_elasticity_cache_entry_{
      kElementIndex, kNumQuads};

  // Make an arbitrary deformation gradient.
  Matrix3<double> MakeDeformationGradient() {
    Matrix3<double> F;
    F << 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0;
    return F;
  }
};

TEST_F(LinearElasticityCacheEntryTest,
       LinearElasticityCacheEntryInitialization) {
  EXPECT_EQ(linear_elasticity_cache_entry_.element_index(), kElementIndex);
  EXPECT_EQ(linear_elasticity_cache_entry_.num_quadrature_points(), kNumQuads);
  EXPECT_EQ(linear_elasticity_cache_entry_.deformation_gradient().size(),
            kNumQuads);
  EXPECT_EQ(linear_elasticity_cache_entry_.strain().size(), kNumQuads);
  EXPECT_EQ(linear_elasticity_cache_entry_.trace_strain().size(), kNumQuads);
}

TEST_F(LinearElasticityCacheEntryTest, UpdateCacheEntry) {
  const Matrix3<double> F = MakeDeformationGradient();
  const Matrix3<double> strain =
      0.5 * (F + F.transpose()) - Matrix3<double>::Identity();
  const double trace_strain = strain.trace();
  EXPECT_EQ(linear_elasticity_cache_entry_.deformation_gradient()[0], F);
  EXPECT_EQ(linear_elasticity_cache_entry_.strain()[0], strain);
  EXPECT_EQ(linear_elasticity_cache_entry_.trace_strain()[0], trace_strain);
}

TEST_F(LinearElasticityCacheEntryTest, Clone) {
  const std::unique_ptr<DeformationGradientCacheEntry<double>> clone =
      linear_elasticity_cache_entry_.Clone();
  // Test that the Clone() method returns the correct concrete type.
  const auto* linear_elasticity_cache_entry_clone =
      dynamic_cast<const LinearElasticityModelCacheEntry<double>*>(clone.get());
  EXPECT_TRUE(linear_elasticity_cache_entry_clone != nullptr);
  // Test that the Clone() method returns an identical copy.
  EXPECT_EQ(linear_elasticity_cache_entry_clone->deformation_gradient(),
            linear_elasticity_cache_entry_.deformation_gradient());
  EXPECT_EQ(linear_elasticity_cache_entry_clone->strain(),
            linear_elasticity_cache_entry_.strain());
  EXPECT_EQ(linear_elasticity_cache_entry_clone->trace_strain(),
            linear_elasticity_cache_entry_.trace_strain());
  EXPECT_EQ(linear_elasticity_cache_entry_clone->num_quadrature_points(),
            linear_elasticity_cache_entry_.num_quadrature_points());
  EXPECT_EQ(linear_elasticity_cache_entry_clone->element_index(),
            linear_elasticity_cache_entry_.element_index());
}
}  // namespace
}  // namespace fem
}  // namespace multibody
}  // namespace drake
