#include "drake/multibody/contact_solvers/rtsafe.h"

#include <optional>

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

using Eigen::Vector3d;

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

typedef std::function<double(double, std::optional<double*>)> Function;

struct RootFindingTestData {
  // Google Test uses this operator to report the test data in the log file
  // when a test fails.
  friend std::ostream& operator<<(std::ostream& os,
                                  const RootFindingTestData& data) {
    return os << "{\n"
              << "  Function: " << data.description << std::endl
              << "  [a , b ] = [" << data.a << ", " << data.b << "]"
              << std::endl
              << "  [fa, fb] = [" << data.function(data.a, std::nullopt) << ", "
              << data.function(data.b, std::nullopt) << "]" << std::endl
              << "  Root: " << data.root << std::endl
              << "  Tolerance: " << data.tolerance << std::endl
              << "}" << std::flush;
  }

  // Text description of the function, typically in the format y(x) = a * x + b.
  std::string description;
  // The function to find the root within interval [a, b].
  // We evaluate it as f = function(x, &dfdx) to evaluate function f and
  // derivative dfdx at x.
  // dfdx can be nullopt or {} if the derivative is not needed.
  Function function;
  double a, b;       // Interval used for root finding.
  double root;       // The true root in [a, b].
  double tolerance;  // Convergence tolerance, absolute.
};

std::vector<RootFindingTestData> GenerateTestCases() {
  std::vector<RootFindingTestData> cases;
  cases.push_back({"y = 1.5 * x + 3.0",
                   [](double x, std::optional<double*> dfdx) {
                     double f = 1.5 * x + 3.0;
                     if (dfdx) *dfdx.value() = 1.5;
                     return f;
                   },
                   -4.0, 3.0, -2.0, kEpsilon});

  // This function has two roots. We push them as two separate cases with two
  // different search intervals.
  cases.push_back({"y = (x - 1.5) * (x + 2.0) = x² + 0.5 x − 3",
                   [](double x, std::optional<double*> dfdx) {
                     double f = (x - 1.5) * (x + 2.0);
                     if (dfdx) *dfdx.value() = 2.0 * x + 0.5;
                     return f;
                   },
                   .a = -1.0, .b = 2.0, .root = 1.5, .tolerance = kEpsilon});
  cases.push_back({"y = (x - 1.5) * (x + 2.0) = x² + 0.5 x − 3",
                   [](double x, std::optional<double*> dfdx) {
                     double f = (x - 1.5) * (x + 2.0);
                     if (dfdx) *dfdx.value() = 2.0 * x + 0.5;
                     return f;
                   },
                   .a = -4.0, .b = 1.0, .root = -2.0, .tolerance = kEpsilon});

  cases.push_back({"y = arctan(x)",
                   [](double x, std::optional<double*> dfdx) {
                     double f = std::atan(x);
                     if (dfdx) *dfdx.value() = 1.0 / (1.0 + x * x);
                     return f;
                   },
                   .a = -1.0, .b = 10.0, .root = 0.0, .tolerance = kEpsilon});

  // For y = x³ − 2x + 2 with initial guess x0 = 0, Newton-Raphson will enter
  // into an infinity cycle without convergence.
  // Since rtsafe uses x0 = (a+b)/2, we choose [a, b] = [-3, 3] so that x0=0 to
  // force the case where Newton-Raphson has problems.
  cases.push_back({"y = x³ − 2x + 2",
                   [](double x, std::optional<double*> dfdx) {
                     const double x2 = x * x;
                     const double f = x * (x2 - 2) + 2;
                     if (dfdx) *dfdx.value() = 3 * x2 - 2;
                     return f;
                   },
                   .a = -3.0, .b = 3.0,
                   .root = -1.769292354238631415240409464335,
                   .tolerance = kEpsilon});

  // This is a difficult case since y' = ∞ at the root x = 0.
  cases.push_back({"y = sign(x) * abs(x)^(1/2)",
                   [](double x, std::optional<double*> dfdx) {
                     const double s = x >= 0 ? 1.0 : -1.0;
                     const double sqrt_x = std::sqrt(std::abs(x));
                     const double f = s * sqrt_x;
                     if (dfdx) *dfdx.value() = 0.5 / sqrt_x;
                     return f;
                   },
                   .a = -1.0, .b = 4.0, .root = 0.0, .tolerance = kEpsilon});

  // For y = x - tan(x) Newton-Raphson will diverge if the guess is outside
  // [4.3, 4.7].
  cases.push_back({"y = x - tan(x)",
                   [](double x, std::optional<double*> dfdx) {
                     const double tan_x = std::tan(x);
                     const double f = x - tan_x;
                     if (dfdx) *dfdx.value() = -tan_x * tan_x;
                     return f;
                   },
                   .a = 2.0, .b = 4.7,
                   .root = 4.4934094579090641753078809272803,
                   .tolerance = kEpsilon});

  // Newton-Raphson struggles with this case since the derivative also goes to
  // zero at the (triple) root x = 1.5.
  cases.push_back({"y = (x-1.5)³",
                   [](double x, std::optional<double*> dfdx) {
                     const double arg = x - 1.5;
                     const double f = arg * arg * arg;
                     if (dfdx) *dfdx.value() = 3.0 * arg * arg;
                     return f;
                   },
                   .a = 0.0, .b = 2.0, .root = 1.5, .tolerance = kEpsilon});

  // Discontinuous derivative. A linear function plus a discontinuity will cause
  // the traditional Newton-Raphson to fall into an infinite cycle without
  // converging to the solution. rtsafe switches to bisection when the bracket
  // starts getting close to the solution.
  cases.push_back({"y = x + H(x+0.3)",
                   [](double x, std::optional<double*> dfdx) {
                     const double f = x + ((x > -0.3) ? 1.0 : 0.0);
                     if (dfdx) *dfdx.value() = 1.0;
                     return f;
                   },
                   .a = -1.0, .b = 1.5, .root = -0.3, .tolerance = kEpsilon});
  return cases;
}

// Test parametrized on different root finding cases.
// To see debug information printed out by RtSafe, run with:
//   bazel run -c dbg multibody/contact_solvers:newton_safe_test --
//   --spdlog_level debug
struct RootFindingTest : public testing::TestWithParam<RootFindingTestData> {};

TEST_P(RootFindingTest, VerifyExpectedResults) {
  const RootFindingTestData& data = GetParam();
  // We printout the human-readable description so that when running the tests
  // we see more than Test/0, Test/1, etc.
  std::cout << data << std::endl;
  int num_iters;
  const double x =
      RtSafe<double>(data.function, data.a, data.b, data.tolerance, &num_iters);
  EXPECT_NEAR(x, data.root, data.tolerance);
}

// To debug a specific test, you can use Bazel flag --test_filter and
// --test_output.  For example, you can use the command:
// ```
//   bazel run //multibody/contact_solvers:newton_safe_test
//   --test_filter=AllCases/RootFindingTest.VerifyExpectedResults/0
//   --test_output=all
// ```
// to run the first case from the test data generated by
// GenRaysWithTwoIntersections() with the function
// TEST_P(RayConeIntersectionTest, VerifyExpectedResults).
// RaysWithTwoIntersections
INSTANTIATE_TEST_SUITE_P(AllCases, RootFindingTest,
                         testing::ValuesIn(GenerateTestCases()));

}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
