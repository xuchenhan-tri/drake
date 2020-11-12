#pragma once

#include <memory>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/collision_object.h"
#include "drake/math/orthonormal_basis.h"

namespace drake {
namespace fem {
template <typename T>
static void QueryContact(
    const Matrix3X<T>& q,
    const std::vector<std::unique_ptr<CollisionObject<T>>>& collision_objects,
    Eigen::SparseMatrix<T>* Jc, VectorX<T>* phi0, std::vector<Vector3<T>>* contact_normals = nullptr) {
  const int nv = q.cols();
  // We know nc <= nv * collision_objects_size(), so we allocate enough memory
  // here.
  // Reserve for the maximum number of contact points.
  std::vector<T> penetration;
  penetration.reserve(nv * collision_objects.size());
  if (contact_normals) {
      contact_normals->clear();
      contact_normals->reserve(nv * collision_objects.size());
  }
  // The number of non-zeros in the Jc is 3 * nc * nv.
  std::vector<Eigen::Triplet<T>> triplets;
  triplets.reserve(3 * nv * collision_objects.size() * nv);
  int nc = 0;
  for (int i = 0; i < nv; ++i) {
    for (int j = 0; j < static_cast<int>(collision_objects.size()); ++j) {
      Vector3<T> normal;
      T signed_distance = collision_objects[j]->Query(q.col(i), &normal);
      if (signed_distance <= 0.0) {
        penetration.push_back(signed_distance);
        if (contact_normals){
            contact_normals->push_back(normal);
        }
        Matrix3<T> R_WC =
            drake::math::ComputeBasisFromAxis(2, normal).transpose();
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            triplets.emplace_back(3 * nc + k, 3 * i + l, R_WC(k, l));
          }
        }
        ++nc;
      }
    }
  }
  Jc->resize(3 * nc, 3 * nv);
  Jc->setFromTriplets(triplets.begin(), triplets.end());
  Jc->makeCompressed();
  (*phi0) = Eigen::Map<VectorX<T>>(penetration.data(), penetration.size());
}
}  // namespace fem
}  // namespace drake
