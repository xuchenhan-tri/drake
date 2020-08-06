#include "drake/fem/corotated_linear_model.h"

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {
template <typename T>
void CorotatedLinearElasticity<T>::UpdateState(
    const Matrix3<T>& F, const Eigen::Matrix<T, 3, 4>& q) {
  F_ = F;
  Matrix4<T> Q;
  Q.template topLeftCorner<3, 4>() = q;
  Q.template bottomRows<1>() = Vector4<T>::Ones();
  R_ = (Q * inv_P_).template topLeftCorner<3, 3>();
  Matrix3<T> RtF = R_.transpose() * F_;
  strain_ = 0.5 * (RtF + RtF.transpose()) - Matrix3<T>::Identity();
  trace_strain_ = strain_.trace();
}

template <typename T>
T CorotatedLinearElasticity<T>::CalcEnergyDensity() const {
  return mu_ * strain_.squaredNorm() +
         0.5 * lambda_ * trace_strain_ * trace_strain_;
}

template <typename T>
Matrix3<T> CorotatedLinearElasticity<T>::CalcFirstPiola() const {
  Matrix3<T> P = 2.0 * mu_ * R_ * strain_ + lambda_ * trace_strain_ * R_;
  return P;
}

template <typename T>
Matrix3<T> CorotatedLinearElasticity<T>::CalcFirstPiolaDifferential(
    const Matrix3<T> dF) const {
  Matrix3<T> dP = mu_ * dF + mu_ * R_ * dF.transpose() * R_ +
                  lambda_ * (R_.array() * dF.array()).sum() * R_;
  return dP;
}

template <typename T>

Eigen::Matrix<T, 9, 9> CorotatedLinearElasticity<T>::CalcFirstPiolaDerivative()
    const {
  // TODO(xuchenhan-tri): implement me.
  Eigen::Matrix<T, 9, 9> dPdF;
  return dPdF;
}
}  // namespace fem
}  // namespace drake
