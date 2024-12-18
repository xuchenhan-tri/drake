#pragma once

#include <array>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* A dedicated data structure for a 3D vector in "Structure of Arrays" (SoA)
 form. Memory layout for N elements looks like:
   x: [x1, x2, ..., xN]
   y: [y1, y2, ..., yN]
   z: [z1, z2, ..., zN] */
template <typename T>
class SoaVec3 {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SoaVec3);
  using Ptrs = std::array<T*, 3>;
  using ConstPtrs = std::array<const T*, 3>;

  SoaVec3() = default;

  /* Reserves space for n elements in each component vector. */
  void Reserve(int n) {
    data_[0].reserve(n);
    data_[1].reserve(n);
    data_[2].reserve(n);
  }

  /* Clears all component arrays. */
  void Clear() {
    data_[0].clear();
    data_[1].clear();
    data_[2].clear();
  }

  /* Resizes each component vector to have n elements. */
  void Resize(int n) {
    data_[0].resize(n);
    data_[1].resize(n);
    data_[2].resize(n);
  }

  /* Pushes back a single v = (x,y,z) triple into the SoA. */
  void PushBack(const Vector3<T>& v) {
    data_[0].push_back(v.x());
    data_[1].push_back(v.y());
    data_[2].push_back(v.z());
  }

  /* Returns raw pointer to the i-th component's data array, for SIMD usage.
   @pre 0 <= i < 3. */
  T* ptr(int i) { return data_[i].data(); }
  const T* ptr(int i) const { return data_[i].data(); }

  /* Returns raw pointers to each component's data array in the matrix form. */
  Ptrs ptrs() { return {data_[0].data(), data_[1].data(), data_[2].data()}; }
  ConstPtrs ptrs() const {
    return {data_[0].data(), data_[1].data(), data_[2].data()};
  }

  /* Current number of elements per component. */
  int size() const { return static_cast<int>(data_[0].size()); }

 private:
  /* data_[0] = x-components, data_[1] = y-components, data_[2] = z-components
   */
  std::array<std::vector<T>, 3> data_;
};

/* A dedicated data structure for a 3×3 matrix in SoA form. Memory layout for N
 elements looks like
   m[0][0]: [m_00_1, m_00_2, ..., m_00_N]
   m[0][1]: [m_01_1, m_01_2, ..., m_01_N]
   ...
   m[2][2]: [m_22_1, m_22_2, ..., m_22_N] */
template <typename T>
class SoaMat3 {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SoaMat3);
  using Ptrs = std::array<std::array<T*, 3>, 3>;
  using ConstPtrs = std::array<std::array<const T*, 3>, 3>;

  SoaMat3() = default;

  /* Reserves space for n elements in each component vector. */
  void Reserve(int n) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        data_[r][c].reserve(n);
      }
    }
  }

  /* Clears all component vectors. */
  void Clear() {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        data_[r][c].clear();
      }
    }
  }

  /* Resizes each component vector to have n elements. */
  void Resize(int n) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        data_[r][c].resize(n);
      }
    }
  }

  /* Pushes back one 3×3 matrix into the SoA. */
  void PushBack(const Matrix3<T>& mat) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        data_[r][c].push_back(mat(r, c));
      }
    }
  }

  /* Returns pointer to the (r,c)-th component's data array, for SIMD usage.
   @pre 0 <= r,c < 3 */
  T* ptr(int r, int c) { return data_[r][c].data(); }
  const T* ptr(int r, int c) const { return data_[r][c].data(); }

  /* Returns raw pointers to each component's data array in the matrix form. */
  Ptrs ptrs() {
    return Ptrs{
        {{{data_[0][0].data(), data_[0][1].data(), data_[0][2].data()}},
         {{data_[1][0].data(), data_[1][1].data(), data_[1][2].data()}},
         {{data_[2][0].data(), data_[2][1].data(), data_[2][2].data()}}}};
  }
  ConstPtrs ptrs() const {
    return ConstPtrs{
        {{{data_[0][0].data(), data_[0][1].data(), data_[0][2].data()}},
         {{data_[1][0].data(), data_[1][1].data(), data_[1][2].data()}},
         {{data_[2][0].data(), data_[2][1].data(), data_[2][2].data()}}}};
  }

 private:
  /* data_[r][c] = [m_rc_1, m_rc_2, ..., m_rc_N] */
  std::array<std::array<std::vector<T>, 3>, 3> data_;
};

template <typename T>
void LoadScalar(const std::vector<T>& data,
                const std::vector<int>& data_indices, std::vector<T>* scalar) {
  for (int i : data_indices) {
    scalar->push_back(data[i]);
  }
}

template <typename T>
void LoadVector(const std::vector<Vector3<T>>& data,
                const std::vector<int>& data_indices, SoaVec3<T>* v) {
  for (int i : data_indices) {
    v->PushBack(data[i]);
  }
}

template <typename T>
void LoadMatrix(const std::vector<Matrix3<T>>& data,
                const std::vector<int>& data_indices, SoaMat3<T>* m) {
  for (int i : data_indices) {
    m->PushBack(data[i]);
  }
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
