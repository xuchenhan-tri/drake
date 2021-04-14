#pragma once

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
template <typename T>
class ContactSolverData {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ContactSolverData);

  ContactSolverData(VecotorX<T>&& phi0, VecotorX<T>&& stiffness,
                    VecotorX<T>&& damping, VecotorX<T>&& mu,
                    const MatrixX<T>& contact_jacobian, int num_rigid_dofs,
                    int num_deformable_dofs)
      : phi0_(std::move(phi0)),
        stiffness_(std::move(stiffness)),
        damping_(std::move(damping)),
        mu_(std::move(mu)) {
    nc_ = phi0_.size();
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
                    int num_rigid_dofs, int num_deformable_dofs)
      : phi0_(std::move(phi0)),
        stiffness_(std::move(stiffness)),
        damping_(std::move(damping)),
        mu_(std::move(mu)),
        contact_jacobian_(std::move(contact_jacobian)) {
    nc_ = phi0_.size();
    DRAKE_DEAMND(stiffness_.size() == nc_);
    DRAKE_DEAMND(damping_.size() == nc_);
    DRAKE_DEAMND(contact_jacobian_.rows() == 3 * nc_);
    DRAKE_DEAMND(contact_jacobian_.cols() ==
                 num_rigid_dofs + num_deformable_dofs);
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
  int nc_;  // Number of contact points.
};
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fixed_fem::internal::CollisionObjects);
