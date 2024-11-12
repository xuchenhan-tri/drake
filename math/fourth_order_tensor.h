#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace math {

/* This class provides functionalities related to 4th order tensors of dimension
 3*3*3*3. The tensor is represented using a a 9*9 matrix that is organized as
 following

                  l = 1       l = 2       l = 3
              -------------------------------------
              |           |           |           |
    j = 1     |   Aᵢ₁ₖ₁   |   Aᵢ₁ₖ₂   |   Aᵢ₁ₖ₃   |
              |           |           |           |
              -------------------------------------
              |           |           |           |
    j = 2     |   Aᵢ₂ₖ₁   |   Aᵢ₂ₖ₂   |   Aᵢ₂ₖ₃   |
              |           |           |           |
              -------------------------------------
              |           |           |           |
    j = 3     |   Aᵢ₃ₖ₁   |   Aᵢ₃ₖ₂   |   Aᵢ₃ₖ₃   |
              |           |           |           |
              -------------------------------------
 Namely the ik-th entry in the jl-th block corresponds to the value Aᵢⱼₖₗ.
 @tparam float, double, AutoDiffXd. */
template <typename T>
class FourthOrderTensor {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FourthOrderTensor)
  using MatrixType = Eigen::Matrix<T, 9, 9>;

  /* Constructs a 4th-order tensor represented by the given matrix using the
   convention layed out in the class documentation. */
  explicit FourthOrderTensor(const MatrixType& data) : data_(data) {}

  /* Constructs a zero 4th-order tensor. */
  FourthOrderTensor() = default;

  /* Performs contraction between this 4th order tensor A
   and two vectors u and v and outputs 2nd order tensor B. In Einstein notation,
   the contraction being done is Bᵢₖ = uⱼ Aᵢⱼₖₗ vₗ. */
  void ContractWithVectors(const Eigen::Ref<const Vector3<T>>& u,
                           const Eigen::Ref<const Vector3<T>>& v,
                           EigenPtr<Matrix3<T>> B) const {
    B->setZero();
    for (int l = 0; l < 3; ++l) {
      for (int j = 0; j < 3; ++j) {
        *B += data.template block<3, 3>(3 * j, 3 * l) * u(j) * v(l);
      }
    }
  }

  /* Returns this fourth order tensor encoded as a matrix. */
  const MatrixType& data() const { return data_; }

 private:
  MatrixType data_{MatrixType::Zero()};
};

}  // namespace math
}  // namespace drake
