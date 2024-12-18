#include "drake/multibody/mpm/bspline_weights.h"

#include "drake/math/autodiff_gradient.h"

// This is the magic juju that compiles our impl functions for multiple CPUs.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "multibody/mpm/bspline_weights.cc"
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
void ComputeXReferenceImpl(const T* x0, const T* x1, const T* x2, int data_size,
                           T dx, T* x_ref0, T* x_ref1, T* x_ref2) {
  using D = hn::ScalableTag<T>;
  const D d;
  const int vec_size = hn::Lanes(d);

  const auto dx_v = hn::Set(d, dx);

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
    auto process_vector = [&](const T* x, T* x_ref) {
      const auto x_v = LoadFunc(x);
      const auto x_ref_v = hn::Div(x_v, dx_v);
      StoreFunc(x_ref_v, x_ref);
    };
    process_vector(x0, x_ref0);
    process_vector(x1, x_ref1);
    process_vector(x2, x_ref2);
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

void ComputeXReferenceImpld(const double* x0, const double* x1,
                            const double* x2, int data_size, double dx,
                            double* x_ref0, double* x_ref1, double* x_ref2) {
  ComputeXReferenceImpl(x0, x1, x2, data_size, dx, x_ref0, x_ref1, x_ref2);
}

void ComputeXReferenceImplf(const float* x0, const float* x1, const float* x2,
                            int data_size, double dx, float* x_ref0,
                            float* x_ref1, float* x_ref2) {
  ComputeXReferenceImpl(x0, x1, x2, data_size, static_cast<float>(dx), x_ref0,
                        x_ref1, x_ref2);
}

template <typename T>
void EvalBsplineImpl(int data_size, const T* x_reference, int base_node, T* w0,
                     T* w1, T* w2) {
  using D = hn::ScalableTag<T>;
  const D d;
  const int vec_size = hn::Lanes(d);
  const auto base_node_v = hn::Set(d, static_cast<T>(base_node));

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

    const auto x_v = LoadFunc(x_reference);
    const auto d_v = hn::Sub(base_node_v, x_v);
    const auto half_v = hn::Set(d, static_cast<T>(0.5));
    const auto d1_v = hn::Add(half_v, d_v);
    auto w_v = hn::Mul(hn::Mul(d1_v, d1_v), half_v);
    StoreFunc(w_v, w0);
    const auto three_quarters_v = hn::Set(d, static_cast<T>(0.75));
    w_v = hn::Sub(three_quarters_v, hn::Mul(d_v, d_v));
    StoreFunc(w_v, w1);
    const auto d2_v = hn::Sub(half_v, d_v);
    w_v = hn::Mul(hn::Mul(d2_v, d2_v), half_v);
    StoreFunc(w_v, w2);
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

void EvalBsplineImpld(int data_size, const double* x_reference, int base_node,
                      double* w0, double* w1, double* w2) {
  EvalBsplineImpl(data_size, x_reference, base_node, w0, w1, w2);
}
void EvalBsplineImplf(int data_size, const float* x_reference, int base_node,
                      float* w0, float* w1, float* w2) {
  EvalBsplineImpl(data_size, x_reference, base_node, w0, w1, w2);
}

template <typename T>
void ComputeWeightsImpl(int data_size, const T* w0, const T* w1, const T* w2,
                        T* w) {
  using D = hn::ScalableTag<T>;
  const D d;
  const int vec_size = hn::Lanes(d);

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

    const auto w0_v = LoadFunc(w0);
    const auto w1_v = LoadFunc(w1);
    const auto w2_v = LoadFunc(w2);
    const auto w_v = hn::Mul(hn::Mul(w0_v, w1_v), w2_v);
    StoreFunc(w_v, w);
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

void ComputeWeightsImpld(int data_size, const double* w0, const double* w1,
                         const double* w2, double* w) {
  ComputeWeightsImpl(data_size, w0, w1, w2, w);
}
void ComputeWeightsImplf(int data_size, const float* w0, const float* w1,
                         const float* w2, float* w) {
  ComputeWeightsImpl(data_size, w0, w1, w2, w);
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

HWY_EXPORT(ComputeXReferenceImpld);
HWY_EXPORT(ComputeXReferenceImplf);
HWY_EXPORT(EvalBsplineImpld);
HWY_EXPORT(EvalBsplineImplf);
HWY_EXPORT(ComputeWeightsImpld);
HWY_EXPORT(ComputeWeightsImplf);

template <typename T>
struct ChooseComputeXReferenceImpl {
  auto operator()() {
    if constexpr (std::is_same_v<T, float>) {
      return HWY_DYNAMIC_POINTER(ComputeXReferenceImplf);
    } else {
      return HWY_DYNAMIC_POINTER(ComputeXReferenceImpld);
    }
  }
};

template <typename T>
struct ChooseEvalBsplineImpl {
  auto operator()() {
    if constexpr (std::is_same_v<T, float>) {
      return HWY_DYNAMIC_POINTER(EvalBsplineImplf);
    } else {
      return HWY_DYNAMIC_POINTER(EvalBsplineImpld);
    }
  }
};

template <typename T>
struct ChooseComputeWeightsImpl {
  auto operator()() {
    if constexpr (std::is_same_v<T, float>) {
      return HWY_DYNAMIC_POINTER(ComputeWeightsImplf);
    } else {
      return HWY_DYNAMIC_POINTER(ComputeWeightsImpld);
    }
  }
};

}  // namespace

BsplineWeights<double> MakeBsplineWeights(const Vector3<AutoDiffXd>& x,
                                          double dx) {
  const auto x_double = math::DiscardZeroGradient(x);
  return BsplineWeights<double>(x_double, dx);
}

template <typename T>
void ComputeXReference(const T* x0, const T* x1, const T* x2, int data_size,
                       double dx, T* x_ref0, T* x_ref1, T* x_ref2) {
  LateBoundFunction<ChooseComputeXReferenceImpl<T>>::Call(
      x0, x1, x2, data_size, dx, x_ref0, x_ref1, x_ref2);
}

template <typename T>
void EvalBspline(int data_size, const T* x_reference, int base_node, T* w0,
                 T* w1, T* w2) {
  LateBoundFunction<ChooseEvalBsplineImpl<T>>::Call(data_size, x_reference,
                                                    base_node, w0, w1, w2);
}

template <typename T>
void ComputeWeights(int data_size, const T* w0, const T* w1, const T* w2,
                    T* weights) {
  LateBoundFunction<ChooseComputeWeightsImpl<T>>::Call(data_size, w0, w1, w2,
                                                       weights);
}

template void ComputeXReference(const double* x0, const double* x1,
                                const double* x2, int data_size, double dx,
                                double* x_ref0, double* x_ref1, double* x_ref2);
template void ComputeXReference(const float* x0, const float* x1,
                                const float* x2, int data_size, double dx,
                                float* x_ref0, float* x_ref1, float* x_ref2);
template void EvalBspline(int data_size, const double* x_reference,
                          int base_node, double* w0, double* w1, double* w2);
template void EvalBspline(int data_size, const float* x_reference,
                          int base_node, float* w0, float* w1, float* w2);
template void ComputeWeights(int data_size, const double* w0, const double* w1,
                             const double* w2, double* weights);
template void ComputeWeights(int data_size, const float* w0, const float* w1,
                             const float* w2, float* weights);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

#endif  // HWY_ONCE
