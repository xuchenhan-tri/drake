#pragma once

#include <memory>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/default_scalars.h"
#include "drake/fem/collision_object.h"
#include "drake/math/orthonormal_basis.h"

namespace drake {
namespace fem {
template <typename T>
class ContactJacobian {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ContactJacobian)
  ContactJacobian(
      const Matrix3X<T>& q,
      const std::vector<std::unique_ptr<CollisionObject<T>>>& collision_objects)
      : q_(q), collision_objects_(collision_objects) {}

  // Returns the contact jacobian. More specifically,
  // if nc is the number of vertices in contact,
  // vc ∈ ℝ³ˣⁿᶜ is the vector that concatenates the 3D contact velocities of all
  // nc contact points and, v is the vector that concatenates the 3D velocities
  // of all nv vertices in the mesh, then the contact Jacobian Jc is defined
  // such that vc = Jc⋅v. Jc is of size 3nc x 3nv.
  void QueryContact(Eigen::SparseMatrix<T>* jacobian,
                    VectorX<T>* penetration_depth) {
    CalcContact();
    DRAKE_DEMAND(3 * nc_ == jacobian_.rows());
    DRAKE_DEMAND(nc_ == normals_.cols());
    DRAKE_DEMAND(nc_ == penetration_depth_.size());
    *jacobian = jacobian_;
    *penetration_depth = penetration_depth_;
  }

  const Eigen::SparseMatrix<T>& get_jacobian() const { return jacobian_; }

  const Matrix3X<T>& get_normals() const { return normals_; }

  const VectorX<T>& get_penetration_depth() const {
    return penetration_depth_;
  };

 private:
  void CalcContact();

  const Matrix3X<T>& q_;
  const std::vector<std::unique_ptr<CollisionObject<T>>>& collision_objects_;
  // Number of contact points.
  int nc_;
  Eigen::SparseMatrix<T> jacobian_;
  Matrix3X<T> normals_;
  VectorX<T> penetration_depth_;
};
}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::drake::fem::ContactJacobian)
