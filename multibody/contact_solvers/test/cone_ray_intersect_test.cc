#include "drake/multibody/contact_solvers/cone_ray_intersect.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using math::RotationMatrixd;

// An arbitrary value defining a cone.
constexpr double kMu = 0.5;
constexpr double deg2rad = M_PI / 180;

class RayConeIntersectionTestData {
 public:
  struct VariantParameters {
    double theta_deg;       // rotation about z in degrees.
    double scale;           // Multiplies all dimensional data by this.
    double soft_tolerance;  // Soft cone's tolerance.
  };

  RayConeIntersectionTestData(double mu, const Vector3d& P, const Vector3d& U,
                              const std::vector<double>& expected_intersections,
                              const VariantParameters& variant)
      : mu_(mu),
        P_(P),
        U_(U),
        expected_intersections_(expected_intersections),
        variant_(variant) {
    const auto Rz = RotationMatrixd::MakeZRotation(variant.theta_deg * deg2rad);
    P_ = Rz * P * variant.scale;
    U_ = Rz * U * variant.scale;
  }

  // Google Test uses this operator to report the test data in the log file
  // when a test fails.
  friend std::ostream& operator<<(std::ostream& os,
                                  const RayConeIntersectionTestData& data) {
    return os << "{\n"
              << "  mu: " << data.mu() << std::endl
              << "  P: " << data.P().transpose() << std::endl
              << "  U: " << data.U().transpose() << std::endl
              << "  Expected number of intersections: "
              << data.expected_intersections().size() << std::endl
              << "  Expected intersections: "
              << Eigen::Map<const VectorXd>(
                     data.expected_intersections().data(),
                     data.expected_intersections().size())
                     .transpose()
              << std::endl
              << " Variant parameters:\n"
              << "   theta [deg]: " << data.theta_deg() << std::endl
              << "   scale: " << data.scale() << std::endl
              << "   soft tolerance: " << data.soft_tolerance() << std::endl
              << "}" << std::flush;
  }

  double mu() const { return mu_; }
  const Vector3d P() const { return P_; }
  const Vector3d U() const { return U_; }
  const std::vector<double>& expected_intersections() const {
    return expected_intersections_;
  }
  double theta_deg() const { return variant_.theta_deg; }
  double scale() const { return variant_.scale; }
  double soft_tolerance() const { return variant_.soft_tolerance; }

 private:
  // Friction coefficient defines the cone.
  double mu_;

  // Ray is defined as X(alpha) = P + alpha * U.
  // U does not need to be normalized (for our use case, it's usually not).
  Vector3d P_;
  Vector3d U_;

  // Expected set of values in alpha at which X(alpha) intersects the cone.
  std::vector<double> expected_intersections_;

  // Variant parameters.
  VariantParameters variant_;
};

// Generate arbitrary angles in degrees.
std::vector<double> GenArbitraryAngles() {
  std::vector<double> angles;
  angles.push_back(0.0);
  // An arbitrary angle, one on each quadrant.
  angles.push_back(30);
  angles.push_back(120);
  angles.push_back(225);
  angles.push_back(300);
  return angles;
}

// Generate arbitrary scaling values.
std::vector<double> GenArbitraryScales() {
  std::vector<double> scales;
  scales.push_back(0.05);
  scales.push_back(1.0);
  scales.push_back(20.0);

  // We test with vectors scaled nearly to zero to stress the robustness of the
  // implementation to round-off errors.
  // CalcRayConeIntersection() considers data with magnitude of machine epsilon
  // or smaller to be zero, and therefore reports no intersections. With 1.0e-15
  // we still get intersetions.
  scales.push_back(1.0e-15);
  return scales;
}

std::vector<double> GenArbitrarySoftTolerances() {
  std::vector<double> tolerances;
  // With zero tolerance the cone matches the exact geometric cone.
  tolerances.push_back(0.0);
  // We perform a test here we a tolerance value closer to what our solvers
  // would use in practice.
  tolerances.push_back(1.0e-7);
  return tolerances;
}

std::vector<RayConeIntersectionTestData> GenVariants(
    std::vector<RayConeIntersectionTestData> (*gen)(
        const RayConeIntersectionTestData::VariantParameters&)) {
  std::vector<RayConeIntersectionTestData> all_variants;
  // N.B. We are exploring a three-dimensional set of parameters here. Be
  // mindful when choosing parameters for each axis so that we avoid an
  // explosion in the number of test cases.
  for (double theta : GenArbitraryAngles()) {
    for (double scale : GenArbitraryScales()) {
      for (double tolerance : GenArbitrarySoftTolerances()) {
        auto data = gen({theta, scale, tolerance});
        all_variants.insert(all_variants.end(), data.begin(), data.end());
      }
    }
  }
  return all_variants;
}

// This method generates test data in a canonical frame in which analytical math
// is simpler to perform. Results are invariants under rotations about the z
// axis. To test this invariance, we allow to generate the same data but rotated
// theta radians about the z axis.
std::vector<RayConeIntersectionTestData> GenRaysWithEmptyIntersection(
    const RayConeIntersectionTestData::VariantParameters& variant) {
  // An arbitrary value defining the cone.
  const double mu = 0.5;
  std::vector<RayConeIntersectionTestData> data;

  // Ray is completely outside the cone.
  data.push_back(
      {mu, Vector3d(1.0, 0.0, 1.0), Vector3d(0.0, 1.0, 0.0), {}, variant});
  // Point P  = 0, at the origin, and U is inside the cone.
  // We expect no intersections since alpha=0 is not reported as such.
  data.push_back(
      {mu, Vector3d::Zero(), Vector3d(mu / 2.0, 0.0, 1.0), {}, variant});

  // Ray intersects the mirror cone at two points. However, since they are in
  // the -z axis, the are not reported.
  data.push_back(
      {mu, Vector3d(-1.0, 0, -1.5), Vector3d(1.0, 0.0, 0.0), {}, variant});

  // Starts with P outside the cone and U is parallel to the wall. However the
  // ray intersects the mirror, not the positive cone.
  data.push_back(
      {mu, Vector3d(-0.5, 0.0, -2.0), Vector3d(mu, 0.0, 1.0), {}, variant});

  return data;
}

std::vector<RayConeIntersectionTestData> GenRaysWithEmptyIntersection() {
  return GenVariants(GenRaysWithEmptyIntersection);
}

// These are very interesting cases for which the number of intersections can
// change simply by changing the soft tolerance. See notes on each case for
// details.
std::vector<RayConeIntersectionTestData>
GenCasesWhereSoftParameterChangesNumberOfIntersections(
    const RayConeIntersectionTestData::VariantParameters& variant) {
  // An arbitrary value defining the cone.
  const double mu = 0.5;
  std::vector<RayConeIntersectionTestData> data;

  // P = 0 and U is not inside the cone. For the exact geometric cone, when the
  // soft tolerance is zero, this will result in not intersections. However, the
  // small curve of the soft cone at g = 0 will lead to an intersection.

  // Approximate value of intersection for the soft cone case.
  const double soft_alpha = variant.soft_tolerance / variant.scale;
  const std::vector<double> intersections =
      variant.soft_tolerance == 0 ? std::vector<double>()
                                  : std::vector<double>({soft_alpha});
  data.push_back({mu, Vector3d::Zero(), Vector3d(2.0 * mu, 0.0, 1.0),
                  intersections, variant});

  return data;
}

std::vector<RayConeIntersectionTestData>
GenCasesWhereSoftParameterChangesNumberOfIntersections() {
  return GenVariants(GenCasesWhereSoftParameterChangesNumberOfIntersections);
}

// Generates test data for cases in which the point P is inside the cone and
// therefore the ray crosses towards the outside once.
// N.B. This generates data for when U IS NOT parallel to the cone.
std::vector<RayConeIntersectionTestData>
GenRaysInsideConeWithSingleIntersection(
    const RayConeIntersectionTestData::VariantParameters& variant) {
  // An arbitrary value defining the cone.
  const double mu = 0.5;
  std::vector<RayConeIntersectionTestData> data;
  data.push_back(
      {mu, Vector3d(0, 0, 1.0), Vector3d(1.0, 0.0, 0.0), {kMu}, variant});
  data.push_back(
      {mu, Vector3d(0.3, 0.0, 1.0), Vector3d(1.0, 0.0, 0.0), {0.2}, variant});

  // In this case P belongs to the mirror image of the cone. Then the ray goes
  // outside the mirror image and goes into the positive cone. We want to test
  // that only the intersection with the positive cone is reported.
  data.push_back(
      {mu, Vector3d(0.25, 0.0, -1.0), Vector3d(0.0, 0.0, 1.0), {1.5}, variant});

  // Starts with P outside the cone and U is parallel to the wall. Therefore
  // there should be a single intersection.
  data.push_back(
      {mu, Vector3d(-0.5, 0.0, -0.5), Vector3d(mu, 0.0, 1.0), {0.75}, variant});

  return data;
}

std::vector<RayConeIntersectionTestData>
GenRaysInsideConeWithSingleIntersection() {
  return GenVariants(GenRaysInsideConeWithSingleIntersection);
}

// Generates cases with two intersections. This corresponds to rays starting
// with P outside the cone and that past through it. The first intersection
// corresponds to the ray going into the cone and the second to the ray coming
// back outside.
// TODO: add rotation around z. These tests are invariant with rotation about
// the z axis. These applis to all the GenFooData() methods.
// TODO: add U scaling. If we multiply U by t, then alpha is divided by t.
std::vector<RayConeIntersectionTestData> GenRaysWithTwoIntersections(
    const RayConeIntersectionTestData::VariantParameters& variant) {
  // An arbitrary value defining the cone.
  const double mu = 0.5;
  std::vector<RayConeIntersectionTestData> data;
  data.push_back({mu,
                  Vector3d(-1.0, 0, 1.0),
                  Vector3d(1.0, 0.0, 0.0),
                  {0.5, 1.5},
                  variant});
  return data;
}

std::vector<RayConeIntersectionTestData> GenRaysWithTwoIntersections() {
  return GenVariants(GenRaysWithTwoIntersections);
}

// This test fixture takes data generated by GenDistanceTestData*(),
// GenDistTestData*(), and GenDistTestTransform*() above.
struct RayConeIntersectionTest
    : public testing::TestWithParam<RayConeIntersectionTestData> {
  // Tolerance used to verify the precision in the computation of intersection
  // values.
  static constexpr double kTolerance = 1e-14;

  // TODO: consider having a specific constructor to do some pre-prosecing if
  // needed.
};

// This fixture verifies results for the non-smooth cone when the soft norm
// parameter is ε = 0. We test this case since it's simple to obtain hand
// written expected solutions.
TEST_P(RayConeIntersectionTest, VerifyExpectedResults) {
  const auto& data = GetParam();

  // Absolute tolerance used in numerical comparisons when the soft tolerance is
  // zero, i.e. the cone matches the exact geometric cone.
  const double min_abs_tolerance = 2.0 *
                                   std::numeric_limits<double>::epsilon() *
                                   std::max(1.0, data.scale());

  // When using soft tolerance, hand written results are more difficult to
  // obtain and thus we just use the same values used for the exact cones. These
  // values are expected to differ by an amount proportional to the ratio
  // soft_tolerance/scale. This is because we can write:
  //   ‖x‖ₛ = sqrt(‖x‖²+ε²) = ‖x‖ sqrt(1+(ε/‖x‖)²) ≈ ‖x‖ (1 + 0.5 ε/‖x‖)
  const double abs_tolerance =
      std::max(min_abs_tolerance, 2 * data.soft_tolerance() / data.scale());

  Vector2d intersections2d;
  const int num_intersections = CalcRayConeIntersection(
      data.mu(), data.soft_tolerance(), data.P(), data.U(), &intersections2d);

  // When the cone is "soft", cases that before had no intersections might now
  // intersect with the "curved" tip of the soft cone. This happens when we
  // scale the data to values in the order or smaller than the soft tolerance.
  // Since we don't have available hand computed solutions for this cases, we
  // simply skip the numeric comparison of intersection values. We still however
  // perform the test bellow fo s(V(alpha)) = 0 to verify we indeed found a true
  // intersection.
  if (data.scale() > data.soft_tolerance()) {
    EXPECT_EQ(num_intersections, data.expected_intersections().size());
    Eigen::Map<const VectorXd> vec_intersections(intersections2d.data(),
                                                 num_intersections);
    Eigen::Map<const VectorXd> vec_expected_intersections(
        data.expected_intersections().data(),
        data.expected_intersections().size());
    EXPECT_TRUE(CompareMatrices(vec_intersections, vec_expected_intersections,
                                abs_tolerance))
        << " intersections = " << vec_intersections.transpose() << std::endl
        << " expected      = " << vec_expected_intersections.transpose();
  }

  // Verify the "cone distance" is zero.
  for (int i = 0; i < num_intersections; ++i) {
    const double alpha = intersections2d[i];
    const Vector3d V = data.P() + alpha * data.U();
    const double s = SoftConeDistance(data.mu(), data.soft_tolerance(), V);
    EXPECT_NEAR(s, 0.0, min_abs_tolerance);
  }
}

// To debug a specific test, you can use Bazel flag --test_filter and
// --test_output.  For example, you can use the command:
// ```
//   bazel run //multibody/contact_solvers:cone_ray_intersect_test
//   --test_filter=RaysWithTwoIntersections/RayConeIntersectionTest.VerifyExpectedResults/0
//   --test_output=all
// ```
// to run the first case from the test data generated by
// GenRaysWithTwoIntersections() with the function
// TEST_P(RayConeIntersectionTest, VerifyExpectedResults).
// RaysWithTwoIntersections
INSTANTIATE_TEST_SUITE_P(RaysWithEmptyIntersection, RayConeIntersectionTest,
                         testing::ValuesIn(GenRaysWithEmptyIntersection()));
INSTANTIATE_TEST_SUITE_P(
    RaysInsideConeWithSingleIntersection, RayConeIntersectionTest,
    testing::ValuesIn(GenRaysInsideConeWithSingleIntersection()));
INSTANTIATE_TEST_SUITE_P(RaysWithTwoIntersections, RayConeIntersectionTest,
                         testing::ValuesIn(GenRaysWithTwoIntersections()));
INSTANTIATE_TEST_SUITE_P(
    CasesWhereSoftParameterChangesNumberOfIntersections,
    RayConeIntersectionTest,
    testing::ValuesIn(
        GenCasesWhereSoftParameterChangesNumberOfIntersections()));

}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
