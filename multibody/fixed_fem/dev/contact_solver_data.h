#pragma once

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
template <typename T>
class ContactSolverData {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ContactSolverData);

  ContactSolverData() {
    phi0_.resize(0);
    stiffness_.resize(0);
    damping_.resize(0);
    mu_.resize(0);
    contact_jacobian_.resize(0, 0);
  }

  ContactSolverData(VecotorX<T>&& phi0, VecotorX<T>&& stiffness,
                    VecotorX<T>&& damping, VecotorX<T>&& mu,
                    const MatrixX<T>& contact_jacobian, int num_rigid_dofs,
                    int num_deformable_dofs)
      : phi0_(std::move(phi0)),
        stiffness_(std::move(stiffness)),
        damping_(std::move(damping)),
        mu_(std::move(mu)) {
    nc_ = phi0_.size();
    nv_ = num_rigid_dofs + num_deformable_dofs;
    DRAKE_DEAMND(stiffness_.size() == nc_);
    DRAKE_DEAMND(damping_.size() == nc_);
    DRAKE_DEAMND(contact_jacobian_.rows() == 3 * nc_);
    DRAKE_DEAMND(contact_jacobian_.cols() == num_rigid_dofs);
    // Tolerance larger than machine epsilon by an arbitrary factor. Just large
    // enough so that entries close to machine epsilon, due to round-off errors,
    // still get pruned.
    const double kPruneTolerance = 20 * std::numeric_limits<double>::epsilon();
    contact_jacobian_ = contact_jacobian.sparseView(kPruneTolerance);
    contact_jacobian_.conservativeResize(3 * nc_,
                                         num_rigid_dofs + num_deformable_dofs);
  }

  ContactSolverData(VecotorX<T>&& phi0, VecotorX<T>&& stiffness,
                    VecotorX<T>&& damping, VecotorX<T>&& mu,
                    Eigen::SparseMatrix<T>&& contact_jacobian,
                    int num_total_dofs)
      : phi0_(std::move(phi0)),
        stiffness_(std::move(stiffness)),
        damping_(std::move(damping)),
        mu_(std::move(mu)),
        contact_jacobian_(std::move(contact_jacobian)) {
    nc_ = phi0_.size();
    nv_ = num_total_dofs;
    DRAKE_DEAMND(stiffness_.size() == nc_);
    DRAKE_DEAMND(damping_.size() == nc_);
    DRAKE_DEAMND(contact_jacobian_.rows() == 3 * nc_);
    DRAKE_DEAMND(contact_jacobian_.cols() == num_total_dofs);
  }

  const VectorX<T>& phi0() const { return phi0_; }

  const VectorX<T>& stiffness() const { return stiffness_; }

  const VectorX<T>& damping() const { return damping_; }

  const VectorX<T>& mu() const { return mu_; }

  const Eigen::SparseMatrix<T> contact_jacobian() const {
    return contact_jacobian_;
  }

  int num_contacts() const { return nc_; }

 private:
  VectorX<T> phi0_;
  VectorX<T> stiffness_;
  VectorX<T> damping_;
  VectorX<T> mu_;
  Eigen::SparseMatrix<T> contact_jacobian_;
  int nc_{0};  // Number of contact points.
  int nv_{0};  // Number of generalized velocities.
};
template <typename T>
ContactSolverData<T> ConcatenateContactSolverData(
    const std::vector<ContactSolverData<T>>& data_vector) {
  // Early exit if the data_vector is empty.
  if (data_vector.empty()) {
    return ContactSolverData();
  }
  // Total number of contacts
  int nc = 0;
  std::vector<int> contact_offsets(data_vector.size());
  const int nv = data[0].num_dofs();
  for (const auto& data : data_vector) {
    DRAKE_DEMAND(data[i].num_dofs() == nv);
    contact_offsets[i] = nc;
    nc += data.num_contacts();
  }

  VectorX<T> phi0(nc);
  VectorX<T> stiffness(nc);
  VectorX<T> damping(nc);
  VectorX<T> mu(nc);
  std::vector<Eigen::Triplets<T>> contact_jacobian_triplets;
  for (int i = 0; i < static_cast<int>(data_vector.size()); ++i) {
    const int offset_i = contact_offset[i];
    const int nc_i = data_vector[i].num_contacts();
    phi0.segment(offset_i, nc_i) = data_vector[i].phi0();
    stiffness.segment(offset_i, nc_i) = data_vector[i].stiffness();
    damping.segment(offset_i, nc_i) = data_vector[i].damping();
    mu.segment(offset_i, nc_i) = data_vector[i].mu();
    const std::vector<Eigen::Triplet<T>> triplets_i =
        ConvertEigenSparseMatrixToTriplets(data_vector[i].contact_jacobian(),
                                           3 * offset_i);
    contact_jacobian_triplets.insert(contact_jacobian_triplets.back(),
                                     triplets_i.begin(), triplets_i.end());
  }
  Eigen::SparseMatrix contact_jacobian(3 * nc, nv);
  contact_jacobian.setFromTriplets(contact_jacobian_triplets.begin(),
                                   contact_jacobian_triplets.end());
  return {std::move(phi0), std::move(stiffness),        std::move(damping),
          std::move(mu),   std::move(contact_jacobian), nv};
}
}
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fixed_fem::internal::CollisionObjects);
