#include "drake/fem/contact_jacobian.h"

namespace drake {
namespace fem {
template <typename T>
void ContactJacobian<T>::CalcContact() {
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
        Matrix3<T> R_LW =
            drake::math::ComputeBasisFromAxis(2, normal).transpose();
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            triplets.emplace_back(3 * nc_ + k, 3 * i + l, R_LW(k, l));
          }
        }
        ++nc_;
      }
    }
  }
  jacobian_.resize(3 * nc_, 3 * nv);
  jacobian_.setFromTriplets(triplets.begin(), triplets.end());
  jacobian_.makeCompressed();
  normals_.resize(3, nc_);
  penetration_depth_.resize(penetration_depth.size());
  for (int i = 0; i < nc_; ++i) {
    penetration_depth_(i) = penetration_depth[i];
    normals_.col(i) = normals[i];
  }
}

}  // namespace fem
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::drake::fem::ContactJacobian)
