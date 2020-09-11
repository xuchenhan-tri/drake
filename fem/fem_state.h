#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/hyperelastic_cache.h"

namespace drake {
namespace fem {
template <typename T>
class FemState {
 public:
  const Matrix3X<T>& get_v0() const { return v0_; }
  Matrix3X<T>& get_mutable_v0() {
    set_q_star_out_of_date(true);
    return v0_;
  }

  const Matrix3X<T>& get_v_star() const { return v_star_; }
  Matrix3X<T>& get_mutable_v_star() { return v_star_; }

  const Matrix3X<T>& get_v() const { return v_; }
  Matrix3X<T>& get_mutable_v() { return v_; }

  const Matrix3X<T>& get_q0() const { return q0_; }
  Matrix3X<T>& get_mutable_q0() {
    // Mark dependent cache out of date.
    set_F0_out_of_date(true);
    set_q_star_out_of_date(true);
    return q0_;
  }

  const Matrix3X<T>& get_q() const { return q_; }
  Matrix3X<T>& get_mutable_q() {
    // Mark dependent cache out of date.
    set_F_out_of_date(true);
    return q_;
  }

  const T get_time() const { return time_; }
  void set_time(T time) { time_ = time; }

  // ----------- Cache -----------
  // ---------- Vertex quantities ----------
  const Matrix3X<T>& get_q_star() const { return q_star_; }
  Matrix3X<T>& get_mutable_q_star() {
    set_q_star_out_of_date(true);
    return q_star_;
  }

  // ---------- Quadrature quantities ----------
  const std::vector<Matrix3<T>>& get_F() const { return F_; }
  std::vector<Matrix3<T>>& get_mutable_F() {
    set_F_out_of_date(true);
    return F_;
  }

  const std::vector<Matrix3<T>>& get_F0() const { return F0_; }
  std::vector<Matrix3<T>>& get_mutable_F0() {
    set_F0_out_of_date(true);
    return F0_;
  }

  const std::vector<std::unique_ptr<HyperelasticCache<T>>>&
  get_hyperelastic_cache() const {
    return hyperelastic_cache_;
  }
  std::vector<std::unique_ptr<HyperelasticCache<T>>>&
  get_mutable_hyperelastic_cache() {
    set_hyperelastic_cache_out_of_date(true);
    return hyperelastic_cache_;
  }

  const std::vector<T>& get_psi() const { return psi_; }
  std::vector<T>& get_mutable_psi() {
    set_psi_out_of_date(true);
    return psi_;
  }

  const std::vector<Matrix3<T>>& get_P() const { return P_; }
  std::vector<Matrix3<T>>& get_mutable_P() {
    set_P_out_of_date(true);
    return P_;
  }

  const std::vector<Eigen::Matrix<T, 9, 9>>& get_dPdF() const { return dPdF_; }
  std::vector<Eigen::Matrix<T, 9, 9>>& get_mutable_dPdF() {
    set_dPdF_out_of_date(true);
    return dPdF_;
  }

  const Eigen::SparseMatrix<T>& get_A() const { return A_; }
  Eigen::SparseMatrix<T>& get_mutable_A() {
    set_A_out_of_date(true);
    return A_;
  }

  const Eigen::SparseMatrix<T>& get_Jc() const { return Jc_; }
  Eigen::SparseMatrix<T>& get_mutable_Jc() { return Jc_; }

  void set_q_star_out_of_date(bool flag) { q_star_out_of_date_ = flag; }
  void set_F_out_of_date(bool flag) {
    F_out_of_date_ = flag;
    if (flag) {
      // Mark dependent cache out of date.
      set_hyperelastic_cache_out_of_date(true);
    }
  }
  void set_F0_out_of_date(bool flag) {
    F0_out_of_date_ = flag;
    if (flag) {
      // Mark dependent cache out of date.
      set_hyperelastic_cache_out_of_date(true);
    }
  }
  void set_hyperelastic_cache_out_of_date(bool flag) {
    hyperelastic_cache_out_of_date_ = flag;
    if (flag) {
      // Mark dependent cache out of date.
      set_psi_out_of_date(true);
      set_P_out_of_date(true);
      set_dPdF_out_of_date(true);
    }
  }
  void set_psi_out_of_date(bool flag) { psi_out_of_date_ = flag; }
  void set_P_out_of_date(bool flag) { P_out_of_date_ = flag; }
  void set_dPdF_out_of_date(bool flag) {
    dPdF_out_of_date_ = flag;
    if (flag) {
      // Mark dependent cache out of date.
      set_A_out_of_date(true);
    }
  }
  void set_A_out_of_date(bool flag) { A_out_of_date_ = flag; }

  bool q_star_out_of_date() { return q_star_out_of_date_; }
  bool F_out_of_date() { return F_out_of_date_; }
  bool F0_out_of_date() { return F0_out_of_date_; }
  bool hyperelastic_cache_out_of_date() {
    return hyperelastic_cache_out_of_date_;
  }
  bool psi_out_of_date() { return psi_out_of_date_; }
  bool P_out_of_date() { return P_out_of_date_; }
  bool dPdF_out_of_date() { return dPdF_out_of_date_; }
  bool A_out_of_date() { return A_out_of_date_; }

 private:
  //  ------------ States --------------
  // Vertex velocity from previous time step.
  Matrix3X<T> v0_;
  // Vertex velocity after dynamics solve.
  Matrix3X<T> v_star_;
  // Vertex velocities.
  Matrix3X<T> v_;
  // Vertex position from previous time step.
  Matrix3X<T> q0_;
  // Vertex positions.
  Matrix3X<T> q_;
  T time_{0.0};
  //  ------------ Cache --------------
  // q* = q₀+ dt * v₀.
  Matrix3X<T> q_star_;
  bool q_star_out_of_date_{true};
  // Deformation gradient.
  std::vector<Matrix3<T>> F_;
  bool F_out_of_date_{true};
  // Deformation gradient of the previous time step.
  std::vector<Matrix3<T>> F0_;
  bool F0_out_of_date_{true};
  // Scratch for computing energy/stress/stress-derivatives.
  std::vector<std::unique_ptr<HyperelasticCache<T>>> hyperelastic_cache_;
  bool hyperelastic_cache_out_of_date_{true};
  // Elastic energy density.
  std::vector<T> psi_;
  bool psi_out_of_date_{true};
  // First Piola Stress.
  std::vector<Matrix3<T>> P_;
  bool P_out_of_date_{true};
  // First Piola Stress derivative.
  std::vector<Eigen::Matrix<T, 9, 9>> dPdF_;
  bool dPdF_out_of_date_{true};

  Eigen::SparseMatrix<T> A_;
  bool A_out_of_date_{true};
  Eigen::SparseMatrix<T> Jc_;
};

}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::FemState)
