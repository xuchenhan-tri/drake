#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace fem {

template <typename T>
class FemState {
  FemState(VectorX<T>&& q, VectorX<T>&& v, VectorX<T>&& a)
      : q_(std::move(q)), v_(std::move(v)), a_(std::move(a)) {}

  const VectorX<T>& q() const { return q_; }
  const VectorX<T>& v() const { return q_; }
  const VectorX<T>& a() const { return q_; }

 private:
  VectorX<T> q_;
  VectorX<T> v_;
  VectorX<T> a_;
};

}  // namespace fem
}  // namespace multibody
}  // namespace drake
