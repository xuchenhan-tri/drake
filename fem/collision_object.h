#pragma once

#include <functional>
#include <memory>

#include "drake/common/eigen_types.h"
#include "drake/fem/analytic_level_set.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace fem {
template <typename T>
class CollisionObject {
 public:
  // Allow move constructor, but disable copy constructor and assignments.
  CollisionObject(const CollisionObject&) = delete;
  CollisionObject(CollisionObject&&) = default;
  CollisionObject& operator=(const CollisionObject&) = delete;
  CollisionObject& operator=(CollisionObject&&) = delete;

  explicit CollisionObject(std::unique_ptr<AnalyticLevelSet<T>> ls)
      : ls_(std::move(ls)), update_(nullptr) {
    transform_.SetIdentity();
    inverse_transform_.SetIdentity();
  }

  CollisionObject(std::unique_ptr<AnalyticLevelSet<T>> ls,
                  std::function<void(T, CollisionObject*)> update)
      : ls_(std::move(ls)), update_(update) {
    transform_.SetIdentity();
    inverse_transform_.SetIdentity();
  }

  /** RotationType is a placeholder for drake::math::RotationMatrix<T>,
   * drake::math::RollPitchYaw<T>, and Eigen::Quaternion<T>. */
  template <typename RotationType>
  void set_rotation(const RotationType& R) {
    transform_.set_rotation(R);
  }

  void set_translation(const Vector3<T>& p) { transform_.set_translation(p); }

  /** Returns the signed distance to the surface of the collision object at a
   * point x in the world space and fills the normal to the collision object in
   * worlds space at x. */
  T Query(const Vector3<T>& x, Vector3<T>* normal) const {
    // Transform the point to material coordinate.
    const Vector3<T> X = inverse_transform_ * x;
    T signed_distance = ls_->Query(X, normal);
    // Transform the normal to world space.
    *normal = transform_.rotation() * (*normal);
    return signed_distance;
  }

  void Update(T time) {
    if (update_) {
      update_(time, this);
      inverse_transform_ = transform_.inverse();
    }
  }

  T get_friction_coeff() const {return friction_coeff_; }

  void set_friction_coeff(T mu) { friction_coeff_ = mu; }

 private:
  std::unique_ptr<AnalyticLevelSet<T>> ls_;
  drake::math::RigidTransform<T> transform_;
  drake::math::RigidTransform<T> inverse_transform_;
  T friction_coeff_{0};
  std::function<void(T, CollisionObject*)> update_;
};
}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::drake::fem::CollisionObject)
