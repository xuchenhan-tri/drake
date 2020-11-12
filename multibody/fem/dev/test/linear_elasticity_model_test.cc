#include "drake/multibody/fem/dev/linear_elasticity_model.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace multibody {
namespace fem {
namespace {
const ElementIndex kDummyElementIndex(0);
constexpr int kNumQuads = 1;

// Creates a vector of arbitrary deformation gradients.
std::vector<Matrix3<AutoDiffXd>> MakeDeformationGradients() {
  // Create random AutoDiffXd deformation.
  Matrix3<double> F;
  F << 0.18, 0.63, 0.54, 0.13, 0.92, 0.17, 0.03, 0.86, 0.85;
  const std::vector<Matrix3<double>> Fs(kNumQuads, F);
  std::vector<Matrix3<AutoDiffXd>> Fs_autodiff(kNumQuads);
  const Eigen::Matrix<double, 9, Eigen::Dynamic> gradient =
      MatrixX<double>::Identity(9, 9);
  for (int i = 0; i < kNumQuads; ++i) {
    const auto F_autodiff_flat = math::initializeAutoDiffGivenGradientMatrix(
        Eigen::Map<const Eigen::Matrix<double, 9, 1>>(Fs[i].data(), 9),
        gradient);
    Fs_autodiff[i] =
        Eigen::Map<const Matrix3<AutoDiffXd>>(F_autodiff_flat.data(), 3, 3);
  }
  return Fs_autodiff;
}

GTEST_TEST(LinearElasticityTest, Parameters) {
  const LinearElasticityModel<double> model(100.0, 0.25);
  const double mu = 40.0;
  const double lambda = 40.0;
  EXPECT_EQ(model.youngs_modulus(), 100.0);
  EXPECT_EQ(model.poisson_ratio(), 0.25);
  EXPECT_EQ(model.shear_modulus(), mu);
  EXPECT_EQ(model.lame_first_parameter(), lambda);
}

GTEST_TEST(LinearElasticityTest, InvalidYoungsModulus) {
  DRAKE_EXPECT_THROWS_MESSAGE(LinearElasticityModel<double>(-1.0, 0.25),
                              std::logic_error,
                              "Young's modulus must be nonnegative.");
}

GTEST_TEST(LinearElasticityTest, InvalidPoissonRatioAtUpperLimit) {
  DRAKE_EXPECT_THROWS_MESSAGE(LinearElasticityModel<double>(100.0, 0.5),
                              std::logic_error,
                              "Poisson ratio must be in \\(-1, 0.5\\).");
}

GTEST_TEST(LinearElasticityTest, InvalidPoissonRatioOverUpperLimit) {
  DRAKE_EXPECT_THROWS_MESSAGE(LinearElasticityModel<double>(100.0, 0.6),
                              std::logic_error,
                              "Poisson ratio must be in \\(-1, 0.5\\).");
}

GTEST_TEST(LinearElasticityTest, InvalidPoissonRatioAtLower) {
  DRAKE_EXPECT_THROWS_MESSAGE(LinearElasticityModel<double>(100.0, -1.0),
                              std::logic_error,
                              "Poisson ratio must be in \\(-1, 0.5\\).");
}

GTEST_TEST(LinearElasticityTest, InvalidPoissonRatioBelowLowerLimit) {
  DRAKE_EXPECT_THROWS_MESSAGE(LinearElasticityModel<double>(100.0, -1.1),
                              std::logic_error,
                              "Poisson ratio must be in \\(-1, 0.5\\).");
}

GTEST_TEST(LinearElasticityTest, UndeformedState) {
  const LinearElasticityModel<double> model(100.0, 0.25);
  LinearElasticityModelCacheEntry<double> cache_entry(kDummyElementIndex,
                                                      kNumQuads);
  const std::vector<Matrix3<double>> F(kNumQuads, Matrix3<double>::Identity());
  cache_entry.UpdateCacheEntry(F);
  // In undeformed state, the energy density should be zero.
  const std::vector<double> analytic_energy_density(kNumQuads, 0);
  // In undeformaed state, the stress should be zero.
  const std::vector<Matrix3<double>> analytic_stress(kNumQuads,
                                                     Matrix3<double>::Zero());
  EXPECT_EQ(model.CalcElasticEnergyDensity(cache_entry),
            analytic_energy_density);
  EXPECT_EQ(model.CalcFirstPiolaStress(cache_entry), analytic_stress);
}

// TODO(xuchenhan-tri): This test applies to all ConstitutiveModels. Consider
// moving it to ConstitutiveModelTest.
GTEST_TEST(LinearElasticityTest, PIsDerivativeOfPsi) {
  const LinearElasticityModel<AutoDiffXd> model_autodiff(100.0, 0.3);
  LinearElasticityModelCacheEntry<AutoDiffXd> cache_autodiff(kDummyElementIndex,
                                                             kNumQuads);
  std::vector<Matrix3<AutoDiffXd>> Fs = MakeDeformationGradients();
  // P should be derivative of Psi.
  cache_autodiff.UpdateCacheEntry(Fs);
  const auto energy = model_autodiff.CalcElasticEnergyDensity(cache_autodiff);
  const auto P = model_autodiff.CalcFirstPiolaStress(cache_autodiff);
  for (int i = 0; i < kNumQuads; ++i) {
    EXPECT_TRUE(CompareMatrices(
        Eigen::Map<const Matrix3<double>>(energy[i].derivatives().data(), 3, 3),
        P[i]));
  }
}

// TODO(xuchenhan-tri): This test applies to all ConstitutiveModels. Consider
// moving it to ConstitutiveModelTest.
GTEST_TEST(LinearElasticityTest, dPdFIsDerivativeOfP) {
  const LinearElasticityModel<AutoDiffXd> model_autodiff(100.0, 0.3);
  LinearElasticityModelCacheEntry<AutoDiffXd> cache_autodiff(kDummyElementIndex,
                                                             kNumQuads);
  std::vector<Matrix3<AutoDiffXd>> Fs = MakeDeformationGradients();
  cache_autodiff.UpdateCacheEntry(Fs);
  const std::vector<Matrix3<AutoDiffXd>> P =
      model_autodiff.CalcFirstPiolaStress(cache_autodiff);
  const std::vector<Eigen::Matrix<AutoDiffXd, 9, 9>> dPdF =
      model_autodiff.CalcFirstPiolaStressDerivative(cache_autodiff);
  for (int q = 0; q < kNumQuads; ++q) {
    for (int i = 0; i < kSpaceDimension; ++i) {
      for (int j = 0; j < kSpaceDimension; ++j) {
        Matrix3<double> dPijdF;
        for (int k = 0; k < kSpaceDimension; ++k) {
          for (int l = 0; l < kSpaceDimension; ++l) {
            dPijdF(k, l) = dPdF[q](3 * j + i, 3 * l + k).value();
          }
        }
        EXPECT_TRUE(CompareMatrices(Eigen::Map<const Matrix3<double>>(
                                        P[q](i, j).derivatives().data(), 3, 3),
                                    dPijdF));
      }
    }
  }
}
}  // namespace
}  // namespace fem
}  // namespace multibody
}  // namespace drake
