#include "drake/multibody/mpm/transfer_kernels.h"

// This is the magic juju that compiles our impl functions for multiple CPUs.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "multibody/mpm/transfer_kernels.cc"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#pragma GCC diagnostic pop

#include "drake/common/hwy_dynamic_impl.h"

HWY_BEFORE_NAMESPACE();
namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {
namespace HWY_NAMESPACE {
// The hn namespace holds the CPU-specific function overloads. By defining it
// using a substitute-able macro, we achieve per-CPU instruction selection.
namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
void ComputeAngularMomentumGradientImpl(
    int data_size, const T* m, const typename SoaMat3<T>::ConstPtrs& C,
    const typename SoaMat3<T>::ConstPtrs& tau, T D_inverse_dt,
    typename SoaMat3<T>::Ptrs A) {
  using D = hn::ScalableTag<T>;
  const D d;
  const auto inv_dt_v = hn::Set(d, D_inverse_dt);
  const int vec_size = Lanes(d);

  // We'll process in two steps:
  // 1. Full vectors
  // 2. Tail, if any
  const int full_size = (data_size / vec_size) * vec_size;
  const int remainder = data_size - full_size;

  auto process_chunk = [&](int i, int n) {
    // Load using LoadN if n < vec_size, else LoadU
    auto LoadFunc = [&](const T* ptr) {
      return (n == vec_size) ? hn::LoadU(d, ptr + i) : hn::LoadN(d, ptr + i, n);
    };
    // Store using StoreN if n < vec_size, else StoreU
    auto StoreFunc = [&](const auto& vec, T* ptr) {
      return (n == vec_size) ? hn::StoreU(vec, d, ptr + i)
                             : hn::StoreN(vec, d, ptr + i, n);
    };

    // Load mass vector
    const auto m_v = LoadFunc(m);

    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        // Load C and tau components
        const auto C_v = LoadFunc(C[r][c]);
        const auto tau_v = LoadFunc(tau[r][c]);

        // A[r][c] = m * C[r][c] - D_inverse_dt * tau[r][c]
        const auto res_v = hn::Sub(hn::Mul(m_v, C_v), hn::Mul(inv_dt_v, tau_v));

        StoreFunc(res_v, A[r][c]);
      }
    }
  };

  // Process full-size chunks
  for (int i = 0; i < full_size; i += vec_size) {
    process_chunk(i, vec_size);
  }

  // Process tail if any
  if (remainder > 0) {
    process_chunk(full_size, remainder);
  }
}

void ComputeAngularMomentumGradientImpld(int data_size, const double* m,
                                         const SoaMat3<double>::ConstPtrs& C,
                                         const SoaMat3<double>::ConstPtrs& tau,
                                         double D_inverse_dt,
                                         SoaMat3<double>::Ptrs A) {
  ComputeAngularMomentumGradientImpl(data_size, m, C, tau, D_inverse_dt, A);
}

void ComputeAngularMomentumGradientImplf(int data_size, const float* m,
                                         const SoaMat3<float>::ConstPtrs& C,
                                         const SoaMat3<float>::ConstPtrs& tau,
                                         float D_inverse_dt,
                                         SoaMat3<float>::Ptrs A) {
  ComputeAngularMomentumGradientImpl(data_size, m, C, tau, D_inverse_dt, A);
}

/* Computes mi = m * w and mvi = mi * vp + A * (xi - xp) * w. */
template <typename T>
void ComputeGridDataImpl(int data_size, const T* mp,
                         const typename SoaVec3<T>::ConstPtrs& vp,
                         const typename SoaVec3<T>::ConstPtrs& xp,
                         const Vector3<T>& xi, const T* w,
                         const typename SoaMat3<T>::ConstPtrs& A,
                         GridData<T>* grid_data) {
  using D = hn::ScalableTag<T>;
  const D d;
  const int vec_size = hn::Lanes(d);

  const auto xi0_v = hn::Set(d, xi[0]);
  const auto xi1_v = hn::Set(d, xi[1]);
  const auto xi2_v = hn::Set(d, xi[2]);

  // We'll process in two steps:
  // 1. Full vectors
  // 2. Tail, if any
  const int full_size = (data_size / vec_size) * vec_size;
  const int remainder = data_size - full_size;

  // Local accumulators
  T m_acc = T(0);
  T v0_acc = T(0);
  T v1_acc = T(0);
  T v2_acc = T(0);

  auto process_chunk = [&](int i, int n) {
    // Load using LoadN if n < vec_size, else LoadU
    auto LoadFunc = [&](const T* ptr) {
      return (n == vec_size) ? hn::LoadU(d, ptr + i) : hn::LoadN(d, ptr + i, n);
    };

    // Load primary data
    const auto mp_v = LoadFunc(mp);
    const auto w_v = LoadFunc(w);

    const auto xp0_v = LoadFunc(xp[0]);
    const auto xp1_v = LoadFunc(xp[1]);
    const auto xp2_v = LoadFunc(xp[2]);

    // (xi - xp)*w
    const auto xi_minus_xp_w0_v = hn::Mul(hn::Sub(xi0_v, xp0_v), w_v);
    const auto xi_minus_xp_w1_v = hn::Mul(hn::Sub(xi1_v, xp1_v), w_v);
    const auto xi_minus_xp_w2_v = hn::Mul(hn::Sub(xi2_v, xp2_v), w_v);

    const auto A00_v = LoadFunc(A[0][0]);
    const auto A01_v = LoadFunc(A[0][1]);
    const auto A02_v = LoadFunc(A[0][2]);
    const auto A10_v = LoadFunc(A[1][0]);
    const auto A11_v = LoadFunc(A[1][1]);
    const auto A12_v = LoadFunc(A[1][2]);
    const auto A20_v = LoadFunc(A[2][0]);
    const auto A21_v = LoadFunc(A[2][1]);
    const auto A22_v = LoadFunc(A[2][2]);

    // z = A * ((xi - xp)*w)
    const auto z0_v = hn::MulAdd(
        xi_minus_xp_w0_v, A00_v,
        hn::MulAdd(xi_minus_xp_w1_v, A01_v, hn::Mul(xi_minus_xp_w2_v, A02_v)));
    const auto z1_v = hn::MulAdd(
        xi_minus_xp_w0_v, A10_v,
        hn::MulAdd(xi_minus_xp_w1_v, A11_v, hn::Mul(xi_minus_xp_w2_v, A12_v)));
    const auto z2_v = hn::MulAdd(
        xi_minus_xp_w0_v, A20_v,
        hn::MulAdd(xi_minus_xp_w1_v, A21_v, hn::Mul(xi_minus_xp_w2_v, A22_v)));

    const auto vp0_v = LoadFunc(vp[0]);
    const auto vp1_v = LoadFunc(vp[1]);
    const auto vp2_v = LoadFunc(vp[2]);

    // mi = m * w
    const auto mi_v = hn::Mul(mp_v, w_v);

    // mvi = mi * vp + z
    const auto mvi0_v = hn::MulAdd(mi_v, vp0_v, z0_v);
    const auto mvi1_v = hn::MulAdd(mi_v, vp1_v, z1_v);
    const auto mvi2_v = hn::MulAdd(mi_v, vp2_v, z2_v);

    // Accumulate the reduced values
    m_acc += hn::ReduceSum(d, mi_v);
    v0_acc += hn::ReduceSum(d, mvi0_v);
    v1_acc += hn::ReduceSum(d, mvi1_v);
    v2_acc += hn::ReduceSum(d, mvi2_v);
  };

  // Process full-size chunks
  for (int i = 0; i < full_size; i += vec_size) {
    process_chunk(i, vec_size);
  }

  // Process tail if any
  if (remainder > 0) {
    process_chunk(full_size, remainder);
  }

  // Update grid_data outside the loop
  grid_data->m += m_acc;
  grid_data->v[0] += v0_acc;
  grid_data->v[1] += v1_acc;
  grid_data->v[2] += v2_acc;
}

void ComputeGridDataImpld(int data_size, const double* mp,
                          const SoaVec3<double>::ConstPtrs& vp,
                          const SoaVec3<double>::ConstPtrs& xp,
                          const Vector3<double>& xi, const double* w,
                          const SoaMat3<double>::ConstPtrs& A,
                          GridData<double>* grid_data) {
  ComputeGridDataImpl(data_size, mp, vp, xp, xi, w, A, grid_data);
}

void ComputeGridDataImplf(int data_size, const float* mp,
                          const SoaVec3<float>::ConstPtrs& vp,
                          const SoaVec3<float>::ConstPtrs& xp,
                          const Vector3<float>& xi, const float* w,
                          const SoaMat3<float>::ConstPtrs& A,
                          GridData<float>* grid_data) {
  ComputeGridDataImpl(data_size, mp, vp, xp, xi, w, A, grid_data);
}

}  // namespace HWY_NAMESPACE
}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
HWY_AFTER_NAMESPACE();

// This part of the file is only compiled once total, instead of once per CPU.
#if HWY_ONCE

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

HWY_EXPORT(ComputeAngularMomentumGradientImpld);
HWY_EXPORT(ComputeAngularMomentumGradientImplf);
HWY_EXPORT(ComputeGridDataImpld);
HWY_EXPORT(ComputeGridDataImplf);

template <typename T>
struct ChooseComputeAMatrixImpl {
  auto operator()() {
    if constexpr (std::is_same_v<T, float>) {
      return HWY_DYNAMIC_POINTER(ComputeAngularMomentumGradientImplf);
    } else {
      return HWY_DYNAMIC_POINTER(ComputeAngularMomentumGradientImpld);
    }
  }
};

template <typename T>
struct ChooseComputeGridDataImpl {
  auto operator()() {
    if constexpr (std::is_same_v<T, float>) {
      return HWY_DYNAMIC_POINTER(ComputeGridDataImplf);
    } else {
      return HWY_DYNAMIC_POINTER(ComputeGridDataImpld);
    }
  }
};

}  // namespace

template <typename T>
void P2G(const ParticleData<T>& particle_data,
         const std::vector<int>& data_indices, const Pad<Vector3<T>>& grid_x,
         Pad<GridData<T>>* grid_data, WorkingSet<T>* working_set) {
  working_set->Load(particle_data, data_indices);
  LateBoundFunction<ChooseComputeAMatrixImpl<T>>::Call(
      working_set->size(), working_set->m_ptr(), working_set->C_ptr(),
      working_set->tau_ptr(), working_set->D_inverse_dt(),
      working_set->mutable_A_ptr());
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        working_set->Load(i, j, k);
        LateBoundFunction<ChooseComputeGridDataImpl<T>>::Call(
            working_set->size(), working_set->m_ptr(), working_set->v_ptr(),
            working_set->x_ptr(), grid_x[i][j][k], working_set->w_ptr(),
            working_set->A_ptr(), &(*grid_data)[i][j][k]);
      }
    }
  }
}

template void P2G(const ParticleData<double>&, const std::vector<int>&,
                  const Pad<Vector3<double>>&, Pad<GridData<double>>*,
                  WorkingSet<double>*);
template void P2G(const ParticleData<float>&, const std::vector<int>&,
                  const Pad<Vector3<float>>&, Pad<GridData<float>>*,
                  WorkingSet<float>*);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

#endif  // HWY_ONCE
