#pragma once

#include <memory>
#include <vector>

#include "drake/common/eigen_types.h"
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
  void CalcContact() {
    const int nv = q_.cols();
    // We know nc <= nv * collision_objects_size(), so we allocate enough memory
    // here.
    std::vector<Vector3<T>> normals;
    std::vector<T> penetration_depth;
    std::vector<Eigen::Triplet<T>> triplets;
    normals.reserve(nv * collision_objects_.size());
    penetration_depth.reserve(nv * collision_objects_.size());
    // The number of non-zeros in the jacobian is 3 * nc * nv.
    triplets.reserve(3 * nv * collision_objects_.size() * nv);
    nc_ = 0;
    for (int i = 0; i < nv; ++i) {
      for (int j = 0; j < static_cast<int>(collision_objects_.size()); ++j) {
        Vector3<T> normal;
        T signed_distance = collision_objects_[j]->Query(q_.col(i), &normal);
        if (signed_distance <= 0.0) {
          normals.push_back(normal);
          penetration_depth.push_back(signed_distance);
          Matrix3<T> R_LW = drake::math::ComputeBasisFromAxis(2, normal).transpose();
          for (int k = 0; k < 3; ++k){
              for (int l = 0; l < 3; ++l) {
                  triplets.emplace_back(3 * nc_ + k, 3 * i + l, R_LW(k,l));
              }
          }
          ++nc_;
        }
      }
    }
    jacobian_.resize(3*nc_, 3*nv);
    jacobian_.setFromTriplets(triplets.begin(), triplets.end());
    jacobian_.makeCompressed();
    normals_.resize(3, nc_);
    penetration_depth_.resize(penetration_depth.size());
    for (int i = 0; i < nc_; ++i) {
      penetration_depth_(i) = penetration_depth[i];
      normals_.col(i) = normals[i];
    }
  }

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
