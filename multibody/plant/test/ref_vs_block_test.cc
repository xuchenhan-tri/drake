#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace internal {
namespace {

using Eigen::Ref;
using Eigen::Vector3d;
using Eigen::VectorBlock;
using Eigen::VectorXd;

/* A few notes:
 1. Ref and Block both derive from Map.
 2. Block is guaranteed to have the DirectAccessBit (https://eigen.tuxfamily.org/dox/group__flags.html#gabf1e9d0516a933445a4c307ad8f14915),
    but neither is guaranteed to have the LvalueBit (https://eigen.tuxfamily.org/dox/group__flags.html#gae2c323957f20dfdc6cb8f44428eaec1a).
 3. Block gets the pointer to the underlying data: https://gitlab.com/libeigen/eigen/-/blob/3.3.4/Eigen/src/Core/Block.h#L347 
 4. Ref does too: https://gitlab.com/libeigen/eigen/-/blob/3.3.4/Eigen/src/Core/Ref.h#L106
*/

Ref<const VectorXd> get_ref(const Ref<const VectorXd>& v) { return v.head(1); }

VectorBlock<const Ref<const VectorXd>> get_block(const Ref<const VectorXd>& v) {
  return v.head(1);
}

// These pass.
GTEST_TEST(RefVsBlock, Ref) {
  const Vector3d v(1, 2, 3);
  const Ref<const VectorXd> r = get_ref(v.tail(2));
  EXPECT_EQ(r(0), 2);
}
GTEST_TEST(RefVsBlock, Block) {
  const Vector3d v(1, 2, 3);
  const VectorBlock<const Ref<const VectorXd>> b = get_block(v.tail(2));
  EXPECT_EQ(b(0), 2);
}

// These fail (as expected) because the temp v1 + v2 is gone.
GTEST_TEST(RefVsBlock, RefIntoTemp) {
  const Vector3d v1(1, 2, 3);
  const Vector3d v2(1, 2, 3);
  const Ref<const VectorXd> r = get_ref(v1 + v2);
  EXPECT_EQ(r(0), 2);
}
GTEST_TEST(RefVsBlock, BlockIntoTemp) {
  const Vector3d v1(1, 2, 3);
  const Vector3d v2(1, 2, 3);
  const VectorBlock<const Ref<const VectorXd>> b = get_block(v1 + v2);
  EXPECT_EQ(b(0), 2);
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
