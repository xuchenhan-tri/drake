#include "drake/common/autodiff.h"

#include <algorithm>

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace fem {
namespace {

GTEST_TEST(AutodiffTest, StdAbs) {
  using std::abs;
  using std::max;
  const AutoDiffXd a{1}, b{2};
  const AutoDiffXd c = max(abs(a), abs(b));
}

}  // namespace
}  // namespace fem
}  // namespace multibody
}  // namespace drake
