#pragma once

#include <array>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/mpm/bspline_weights.h"
#include "drake/multibody/mpm/grid_data.h"
#include "drake/multibody/mpm/particle_data.h"
#include "drake/multibody/mpm/soa_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* A Pad is a 3x3x3 subgrid. */
template <typename T>
using Pad = std::array<std::array<std::array<T, 3>, 3>, 3>;

template <typename T>
class WorkingSet {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(WorkingSet);

  constexpr static int kWorkingSetSize = 1024;

  WorkingSet(T dx, T D_inverse_dt)
      : dx_(dx), D_inverse_dt_(D_inverse_dt), bsplines_(dx) {
    DRAKE_DEMAND(dx > 0);
    DRAKE_DEMAND(D_inverse_dt > 0);
    ReserveAll(kWorkingSetSize);
  }

  int size() const { return data_size_; }
  const T& D_inverse_dt() const { return D_inverse_dt_; }
  const T* m_ptr() const { return m_.data(); }
  const T* w_ptr() const { return w_.data(); }
  SoaVec3<T>::ConstPtrs v_ptr() const { return v_.ptrs(); }
  SoaVec3<T>::ConstPtrs x_ptr() const { return x_.ptrs(); }
  SoaMat3<T>::ConstPtrs tau_ptr() const { return tau_.ptrs(); }
  SoaMat3<T>::ConstPtrs C_ptr() const { return C_.ptrs(); }
  SoaMat3<T>::ConstPtrs A_ptr() const { return A_.ptrs(); }
  SoaMat3<T>::Ptrs mutable_A_ptr() { return A_.ptrs(); }

  void Load(const ParticleData<T>& particle_data,
            const std::vector<int>& data_indices) {
    ClearAll();
    LoadScalar(particle_data.m(), data_indices, &m_);
    LoadVector(particle_data.v(), data_indices, &v_);
    LoadVector(particle_data.x(), data_indices, &x_);
    LoadMatrix(particle_data.tau_volume(), data_indices, &tau_);
    LoadMatrix(particle_data.C(), data_indices, &C_);

    data_size_ = data_indices.size();
    A_.Resize(data_size_);
    w_.resize(data_size_);
    bsplines_.Compute(x_);
  }

  void Load(int i, int j, int k) { bsplines_.ComputeWeights(i, j, k, &w_); }

 private:
  /* Clears all data in the working set. Parameters like dx_ and data_size_ are
   untouched. */
  void ClearAll() {
    m_.clear();
    w_.clear();

    v_.Clear();
    x_.Clear();
    tau_.Clear();
    C_.Clear();
    A_.Clear();
  }

  /* Reserves space for n elements for all data. */
  void ReserveAll(int n) {
    m_.reserve(n);
    w_.reserve(n);

    v_.Reserve(n);
    x_.Reserve(n);
    tau_.Reserve(n);
    C_.Reserve(n);
    A_.Reserve(n);
  }

  std::vector<T> m_;
  std::vector<T> w_;
  SoaVec3<T> v_;
  SoaVec3<T> x_;
  SoaMat3<T> tau_;
  SoaMat3<T> C_;
  SoaMat3<T> A_;

  T dx_{};
  T D_inverse_dt_{};
  int data_size_{};
  BsplineWeightsSimd<T> bsplines_;
};

template <typename T>
void P2G(const ParticleData<T>& particle_data,
         const std::vector<int>& data_indices, const Pad<Vector3<T>>& grid_x,
         Pad<GridData<T>>* grid_data, WorkingSet<T>* working_set);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
