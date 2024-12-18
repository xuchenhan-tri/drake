#pragma once

#include "particles.h"
#include "sparse_grid.h"

#include "drake/common/parallelism.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

// TODO(xuchenhan-tri): Move this into a cc files.
namespace HWY_NAMESPACE {
// The hn namespace holds the CPU-specific function overloads. By defining it
// using a substitute-able macro, we achieve per-CPU instruction selection.
namespace hn = hwy::HWY_NAMESPACE;

// Computes m * C - D_inverse_dt * tau using highway simd instructions.
template <typename T>
void ComputeAngularMomentumGradient(const T* m, const T* C[3][3],
                                    const T* tau[3][3], T D_inverse_dt,
                                    int data_size, T* A[3][3]) {
  using D = hn::ScalableTag<T>;
  const D d;
  const auto inv_dt_v = hn::Set(d, D_inverse_dt);
  const int vec_size = Lanes(d);

  for (int i = 0; i < data_size; i += vec_size) {
    // Load mass vector
    const auto m_v = hn::Load(d, m + i);

    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        // Load C and tau components
        const auto C_v = hn::Load(d, C[r][c] + i);
        const auto tau_v = hn::Load(d, tau[r][c] + i);

        // A[r][c] = m * C[r][c] - D_inverse_dt * tau[r][c]
        const auto res_v = hn::Sub(hn::Mul(m_v, C_v), hn::Mul(inv_dt_v, tau_v));

        hn::Store(res_v, d, A[r][c] + i);
      }
    }
  }
}
}  // namespace HWY_NAMESPACE

template <typename T>
void LoadScalar(const std::vector<T>& data,
                const std::vector<int>& data_indices, std::vector<T>* scalar);
{
  for (int i : data_indices) {
    scalar->push_back(data[i]);
  }
}

template <typename T>
void LoadVector(const std::vector<Vector3<T>>& data,
                const std::vector<int>& data_indices, std::vector<T>* v[3]) {
  for (int i : data_indices) {
    v[0]->push_back(data[i][0]);
    v[1]->push_back(data[i][1]);
    v[2]->push_back(data[i][2]);
  }
}

template <typename T>
void LoadMatrix(const std::vector<Matrix3<T>>& data,
                const std::vector<int>& data_indices, std::vector<T>* m[3][3]) {
  for (int i : data_indices) {
    const Matrix3<T>& tau = data[i];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        m[i][j]->push_back(tau(i, j));
      }
    }
  }
}

template <typename T>
class WorkingSet {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(WorkingSet);

  constexpr static int kWorkingSetSize = 32;

  WorkingSet(T dx, T D_inverse_dt) : dx_(dx), D_inverse_dt_(D_inverse_dt) {
    DRAKE_DEMAND(dx > 0);
    DRAKE_DEMAND(D_inverse_dt > 0);
    m_.reserve(kWorkingSetSize);
    m_ptr_ = m_.data();
    for (int i = 0; i < 3; ++i) {
      v_[i].reserve(kWorkingSetSize);
      v_ptr_[i] = v_[i].data();
    }
    for (int i = 0; i < 3; ++i) {
      x_[i].reserve(kWorkingSetSize);
      x_ptr_[i] = x_[i].data();
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        w_[i][j].reserve(kWorkingSetSize);
        w_ptr_[i][j] = w_[i][j].data();
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        tau_[i][j].reserve(kWorkingSetSize);
        tau_ptr_[i][j] = tau_[i][j].data();
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        C_[i][j].reserve(kWorkingSetSize);
        C_ptr_[i][j] = C_[i][j].data();
      }
    }
  }

  void Load(const ParticleData<T>& particle_data,
            const std::vector<int>& data_indices,
            const Pad<Vector3<T>>& grid_x) {
    m_.clear();
    for (int i = 0; i < 3; ++i) {
      v_[i].clear();
    }
    for (int i = 0; i < 3; ++i) {
      x_[i].clear();
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        w_[i][j].clear();
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        C_[i][j].clear();
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        tau_[i][j].clear();
      }
    }
    LoadScalar(particle_data.m, data_indices, &m_ptr_);
    LoadVector(particle_data.v, data_indices, &v_ptr_);
    LoadVector(particle_data.x, data_indices, &x_ptr_);
    LoadMatrix(particle_data.tau_v0, data_indices, &tau_ptr_);
    LoadMatrix(particle_data.C, data_indices, &C_ptr_);
    data_size_ = data_indices.size();
    // Computes A = m * C - D_inverse_dt * tau;
    ComputeAngularMomentumGradient(m_ptr_, C_ptr_, tau_ptr_, D_inverse_dt_,
                                   data_size_, &A_);
  }

  Vector4<T> P2G(int i, int j, int k) const;

 private:
  std::vector<T> m_;
  std::vector<T> v_[3];
  std::vector<T> x_[3];
  std::vector<T> w_[3][3];
  std::vector<T> tau_[3][3];
  std::vector<T> C_[3][3];
  T* m_ptr{};
  T* v_ptr_[3]{};
  T* x_ptr_[3]{};
  T* w_ptr_[3][3]{};
  T* tau_ptr_[3][3]{};
  T* C_ptr_[3][3]{};

  T dx_{};
  T D_inverse_dt_{};
  int data_size_{};
};

/* Transfer class for particle-to-grid and grid-to-particle transfer in a
 single Moving Least Squares Material Point Method (MLS-MPM) [Hu et al. 2018]
 step using Affine Particle In Cell (APIC) [Jiang et al. 2016].

 [Hu et al. 2018] Hu, Y., Fang, Y., Ge, Z., Qu, Z., Zhu, Y., Pradhana, A., &
 Jiang, C. (2018). A moving least squares material point method with
 displacement discontinuity and two-way rigid body coupling. ACM Transactions
 on Graphics (TOG), 37(4), 1-14.

 [Jiang et al. 2016] Jiang, C., Schroeder, C., Teran, J., Stomakhin, A., &
 Selle, A. (2016). The material point method for simulating continuum
 materials. In ACM SIGGRAPH 2016 courses.
 @tparam <T, U> = <double, double> or <float, float> or <double, AutoDiffXd>
 @tparam Grid is either SparseGrid or MockSparseGrid (for testing). */
template <typename T, template <typename> class Grid = SparseGrid>
class Transfer {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Transfer);

  /* Constructs a Transfer object for particle-to-grid and grid-to-particle
   transfer between `grid` and `particles`.
   The constructor prepares the grid for transfer by sorting particle indices
   and allocating memory for grid data.
   @pre grid and particles are not null. */
  Transfer(T dt, Grid<T>* grid, Particles<T>* particles,
           bool reset_grid = true);
  Transfer(T dt, Grid<T>* grid, ContactParticleData<T>* particles);

  const Grid<T>& grid() const {
    DRAKE_DEMAND(grid_ != nullptr);
    return *grid_;
  }

  const Particles<T>& particles() const {
    DRAKE_DEMAND(particles_ != nullptr);
    return *particles_;
  }

  Grid<T>& mutable_grid() {
    DRAKE_DEMAND(grid_ != nullptr);
    return *grid_;
  }

  Particles<T>& mutable_particles() {
    DRAKE_DEMAND(particles_ != nullptr);
    return *particles_;
  }

  /* The transfer functions below each come in four flavors as the cross
   product of two options:
   1. Serial or Parallel: Serial functions are single-threaded, while Parallel
   functions use the number of threads specified by the parallelize argument.
   Thread level parallelism is achieved using OpenMP parallel for over grid
   blocks.
   2. Scalar or Simd: Scalar functions use scalar types (double or float),
   while Simd functions use SIMD types (SimdScalar<double> or
   SimdScalar<float>). The Simd functions are vectorized over particles that
   share the same base grid node. */

  /* Particle to grid transfer (P2G). After the call to P2G, the grid store
   the mass and momentum transfered from the particles using APIC. */
  void SerialParticleToGrid();
  void ParallelSimdParticleToGrid(Parallelism parallelize);

  /* Grid to particle transfer (G2P). After the call to G2P, the particles
   store the mass and momentum transfered from the grid using APIC.
   @pre the grid stores mass and velocity (not momentum). Hence, the velocity
   from the grid needs to be processed after P2G and before G2P. */
  void SerialGridToParticle();
  void ParallelSimdGridToParticle(Parallelism parallelize);

  void ContactP2G2P();

 private:
  T dt_{0.0};
  Grid<T>* grid_{};
  Particles<T>* particles_{};
  ContactParticleData<T>* contact_particles_{};
  /* The D inverse matrix in computing the affine matrix. See page 42 in the
   MPM course notes referenced in the class documentation. */
  T D_inverse_{0.0};
  T D_inverse_dt_{0.0};
  T kApicRatio{1.0};
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
