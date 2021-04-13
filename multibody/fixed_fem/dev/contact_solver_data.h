#pragma once
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fixed_fem/dev/matrix_utilities.h"
namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {

template <typename T>
class PointContactDataStorage {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PointContactDataStorage);

  PointContactDataStorage(int nv) : nv_(nv) {}

  void AppendData(VectorX<T>&& phi0, VectorX<T>&& stiffness,
                  VectorX<T>&& damping, VectorX<T>&& mu, const MatrixX<T>& Jc) {
    const int num_new_contacts = phi0.size();
    DRAKE_DEMAND(stiffness.size() == num_new_contacts);
    DRAKE_DEMAND(damping.size() == num_new_contacts);
    DRAKE_DEMAND(mu.size() == num_new_contacts);
    // Tolerance larger than machine epsilon by an arbitrary factor. Just large
    // enough so that entries close to machine epsilon, due to round-off errors,
    // still get pruned.
    const double kPruneTolerance = 20 * std::numeric_limits<double>::epsilon();
    const Eigen::SparseMatrix<T> Jc_sparse = Jc.sparseView(kPruneTolerance);
    std::vector<Eigen::Triplet<T>> Jc_triplets =
        ConvertEigenSparseMatrixToTripletsWithOffsets(Jc_sparse, 0, 0);
    AppendDataHelper(phi0.data(), stiffness.data(), damping.data(), mu.data(),
                     num_new_contacts, Jc_triplets);
  }

  void AppendData(std::vector<T>&& phi0, std::vector<T>&& stiffness,
                  std::vector<T>&& damping, std::vector<T>&& mu,
                  const std::vector<Eigen::Triplet<T>>& Jc_triplets) {
    const int num_new_contacts = phi0.size();
    DRAKE_DEMAND(static_cast<int>(stiffness.size()) == num_new_contacts);
    DRAKE_DEMAND(static_cast<int>(damping.size()) == num_new_contacts);
    DRAKE_DEMAND(static_cast<int>(mu.size()) == num_new_contacts);
    AppendDataHelper(&phi0[0], &stiffness[0], &damping[0], &mu[0],
                     num_new_contacts, Jc_triplets);
  }

  const std::vector<T>& phi0() const { return phi0_; }
  const std::vector<T>& stiffness() const { return stiffness_; }
  const std::vector<T>& damping() const { return damping_; }
  const std::vector<T>& mu() const { return mu_; }
  const std::vector<Eigen::Triplet<T>> Jc_triplets() const {
    return Jc_triplets_;
  }

  int num_contacts() const { return nc_; }
  int num_dofs() const { return nv_; }

 private:
  void AppendDataHelper(T* phi0, T* stiffness, T* damping, T* mu,
                        int num_new_contacts,
                        const std::vector<Eigen::Triplet<T>>& Jc_triplets) {
    phi0_.insert(phi0_.end(), std::make_move_iterator(phi0),
                 std::make_move_iterator(phi0 + num_new_contacts));
    stiffness_.insert(stiffness_.end(), std::make_move_iterator(stiffness),
                      std::make_move_iterator(stiffness + num_new_contacts));
    damping_.insert(damping_.end(), std::make_move_iterator(damping),
                    std::make_move_iterator(damping + num_new_contacts));
    mu_.insert(mu_.end(), std::make_move_iterator(mu),
               std::make_move_iterator(mu + num_new_contacts));
    for (const Eigen::Triplet<T>& t : Jc_triplets) {
      DRAKE_DEMAND(t.row() <= 3 * num_new_contacts);
      DRAKE_DEMAND(t.col() <= nv_);
      Jc_triplets_.emplace_back(t.row() + 3 * nc_, t.col(), t.value());
    }
  }
  std::vector<T> phi0_;
  std::vector<T> stiffness_;
  std::vector<T> damping_;
  std::vector<T> mu_;
  std::vector<Eigen::Triplet<T>> Jc_triplets_;
  int nc_{0};  // Number of contact points.
  int nv_{0};  // Number of generalized velocities.
};

}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
