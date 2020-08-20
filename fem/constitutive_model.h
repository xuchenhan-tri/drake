#pragma once

#include "drake/common/eigen_types.h"
#include "drake/common/unused.h"

namespace drake {
namespace fem {
template <typename T>
class ConstitutiveModel {
 public:
  /** Updates the states of the constitutive model that depends on F.
     @param[in] F  The deformation gradient of the element that the model lives
     on.
     @param[in] q  The positions of the vertices belonging to the element that
     the model lives on .
  */
  ConstitutiveModel(T alpha, T beta) : alpha_(alpha), beta_(beta) {}
  virtual ~ConstitutiveModel() {}

  virtual void UpdateDeformationBasedState(
      const Eigen::Ref<const Matrix3<T>>& F) = 0;

  virtual void UpdateTimeNPositionBasedState(
      const Eigen::Ref<const Eigen::Matrix<T, 3, 4>>& q) {
    unused(q);
    throw std::runtime_error(
        "UpdateTimeNPositionBasedState must provide an implementation.");
  }

  virtual T get_alpha() const { return alpha_; }
  virtual T get_beta() const { return beta_; }

  /** Calculates the energy density. */
  virtual T CalcEnergyDensity() const = 0;

  /** Calculates the First Piola stress. */
  virtual Matrix3<T> CalcFirstPiola() const = 0;

  /** Calculates the First Piola stress Differential dP(dF). */
  virtual Matrix3<T> CalcFirstPiolaDifferential(
      const Eigen::Ref<const Matrix3<T>>& dF) const = 0;

  /** Calculates the First Piola stress derivative dP(dF). */
  virtual Eigen::Matrix<T, 9, 9> CalcFirstPiolaDerivative() const = 0;

 private:
  T alpha_{0};
  T beta_{0};
};

}  // namespace fem
}  // namespace drake
