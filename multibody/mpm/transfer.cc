#include "transfer.h"

#include <array>
#include <vector>

#include "mock_sparse_grid.h"
#include "simd_scalar.h"
#include "sort_particles.h"
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "drake/common/ssize.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T, template <typename> class Grid>
Transfer<T, Grid>::Transfer(T dt, Grid<T>* grid, Particles<T>* particles,
                            bool reset_grid)
    : dt_(dt), grid_(grid), particles_(particles) {
  DRAKE_DEMAND(dt > 0);
  DRAKE_DEMAND(grid != nullptr);
  DRAKE_DEMAND(particles != nullptr);
  particles->Sort(*grid);
  if (reset_grid) {
    grid_->Allocate(particles->sorter);
  }
  D_inverse_ = 4.0 / (grid_->dx() * grid_->dx());
  D_inverse_dt_ = D_inverse_ * dt_;
}

template <typename T, template <typename> class Grid>
void Transfer<T, Grid>::SerialParticleToGrid() {
  using Scalar = decltype(grid_->dx());
  const ParticleSorter& sorter = particles_->sorter;
  auto p2g_kernel = [&](const Pad<Vector3<Scalar>>& grid_x,
                        Pad<GridData<T>>* grid_data,
                        const ParticleData<T>* particle_data, int data_index) {
    const T& m = particle_data->m[data_index];
    const Vector3<T>& x = particle_data->x[data_index];
    const Vector3<T>& v = particle_data->v[data_index];
    const Matrix3<T>& C = particle_data->C[data_index];
    const Matrix3<T>& tau_v0 = particle_data->tau_v0[data_index];
    bool participating = particle_data->in_constraint[data_index];
    const BsplineWeights<Scalar> bspline = MakeBsplineWeights(x, grid_->dx());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const Scalar& w = bspline.weight(i, j, k);
          const Vector3<T>& xi = grid_x[i][j][k];
          /* The mass transfer is as described equation (126) in [Jiang et al.
           2016]. The momentum transfer is as described in equation (171) in
           [Jiang et al. 2016] with the force term equivalent to equation (18)
           in [Hu et al. 2018], but simplified. We sketch the proof of the
           equivalence here:

           The new grid momentum is given by mvᵢⁿ + fᵢdt with mvᵢⁿ being the
           grid momentum from the current time step transferred from the
           particles. That is,

           mvᵢⁿ = Σₚ mₚvₚ + Cₚ(xᵢ - xₚ) wᵢₚ (equation 178 [Jiang et al. 2016])

           where wᵢₚ is the weight of the particle p to the grid node i. fᵢdt is
           the change in momentum with the force given by fᵢ = -∂E/∂xᵢ

           E = ∑ₚ VₚΨ(Fₚ) where Vₚ is the volume of the particle p in the
           reference configuration and Ψ is the strain energy density.

           Noting that
             Fₚ = (I + dtCₚ)Fₚⁿ (equation 17 [Hu et al. 2018])
             Cₚ = Bₚ * D⁻¹ (equation 173 [Jiang et al. 2016]), and
             Bₚ = ∑ᵢ wᵢₚ vᵢ(xᵢ − xₚ) (equation 176 [Jiang et al. 2016]),

           we compute -∂E/∂xᵢ and get

            fᵢ = -∑ₚ Vₚ * Pₚ * Fₚⁿᵀ * D⁻¹ * (xᵢ − xₚ) * wᵢₚ

           with Pₚ = ∂Ψ/∂Fₚ. Noting that Pₚ * Fₚⁿᵀ is the Kirchhoff stress, we
           group Vₚ * Pₚ * Fₚⁿᵀ into a single term `tau_v0`. Rearranging terms
           reveals that mvᵢⁿ + fᵢdt is given by the equation in the code below.
          */
          const T mi = m * w;
          (*grid_data)[i][j][k].v +=
              mi * v + (m * C - D_inverse_dt_ * tau_v0) * (xi - x) * w;
          (*grid_data)[i][j][k].m += mi;
          /* Set all participating grid node to have grid node index -2. */
          // TODO(xuchenhan-tri): This is a temporary solution to mark the
          // participating grid nodes. We should use a special flag instead of a
          // hard-coded number.
          if (participating) (*grid_data)[i][j][k].index = -2;
        }
      }
    }
  };

  sorter.Iterate(grid_, &particles_->data, true, std::move(p2g_kernel));
}

template <>
void Transfer<AutoDiffXd, MockSparseGrid>::ParallelSimdParticleToGrid(
    const Parallelism parallelize) {
  throw std::runtime_error("simd p2g Not implemented");
}

template <typename T, template <typename> class Grid>
void Transfer<T, Grid>::ParallelSimdParticleToGrid(
    const Parallelism parallelism) {
  const ParticleSorter& sorter = particles_->sorter;
  auto p2g_kernel = [&](const Pad<Vector3<T>>& grid_x,
                        Pad<GridData<T>>* grid_data,
                        const ParticleData<T>* particle_data,
                        const std::vector<int>& data_indices) {
    const SimdScalar<T> m = Load(particle_data->m, data_indices);
    const Vector3<SimdScalar<T>> x = Load(particle_data->x, data_indices);
    const Vector3<SimdScalar<T>> v = Load(particle_data->v, data_indices);
    const Matrix3<SimdScalar<T>> C = Load(particle_data->C, data_indices);
    const bool participating = Load(particle_data->in_constraint, data_indices);
    const Matrix3<SimdScalar<T>> tau_v0 =
        Load(particle_data->tau_v0, data_indices);
    const BsplineWeights<SimdScalar<T>> bspline =
        BsplineWeights<SimdScalar<T>>(x, grid_->dx());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const SimdScalar<T>& w = bspline.weight(i, j, k);
          const Vector3<T>& xi = grid_x[i][j][k];
          // TODO(xuchenhan-tri): Better document this. The formula isn't
          // exactly the same as the paper spells out.
          /* Tse the grid velocity data to store momentum. */
          const SimdScalar<T> mi = m * w;
          const Vector3<SimdScalar<T>> mvi =
              mi * v + (m * C - D_inverse_dt_ * tau_v0) * (xi - x) * w;
          (*grid_data)[i][j][k].m += ReduceSum(mi);
          (*grid_data)[i][j][k].v += ReduceSum(mvi);
          if (participating) (*grid_data)[i][j][k].index = -2;
        }
      }
    }
  };
  sorter.IterateParallelSimd(grid_, &particles_->data, true, parallelism,
                             std::move(p2g_kernel));
}

template <typename T, template <typename> class Grid>
void Transfer<T, Grid>::SerialGridToParticle() {
  using Scalar = decltype(grid_->dx());
  const ParticleSorter& sorter = particles_->sorter;
  auto g2p_kernel = [&](const Pad<Vector3<Scalar>>& grid_x,
                        Pad<GridData<T>>* grid_data,
                        ParticleData<T>* particle_data, int data_index) {
    Vector3<T>& x = particle_data->x[data_index];
    const BsplineWeights<Scalar> bspline = MakeBsplineWeights(x, grid_->dx());
    Vector3<T>& v = particle_data->v[data_index];
    Matrix3<T>& C = particle_data->C[data_index];
    v.setZero();
    C.setZero();
    Matrix3<T>& F = particle_data->F[data_index];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const Vector3<T>& vi = (*grid_data)[i][j][k].v;
          const Vector3<Scalar>& xi = grid_x[i][j][k];
          const Scalar& w = bspline.weight(i, j, k);
          v += w * vi;
          C += (w * vi) * (xi - x).transpose();
        }
      }
    }
    x += v * dt_;
    C *= D_inverse_;
    F += C * dt_ * F;
    /* We use 0.5 * r * (C + Cᵀ) + 0.5 * (C - Cᵀ) to update the affine
     matrix C. With r = 0, the transfer reduces to RPIC transfer described
     in [Jiang et al. 2015]. With r = 1, the transfer reduces to APIC
     transfer. RPIC, APIC, and any linear combination thereof is
     linear/angular momentum conserving. RPIC dissipates more energy than
     APIC. We use a linear combination of RPIC and APIC with r as a
     parameter for numerical damping. */
    const T c1 = (1 + kApicRatio) * 0.5;
    const T c2 = (kApicRatio - 1) * 0.5;
    C = (c1 * C + c2 * C.transpose()).eval();
  };
  sorter.Iterate(grid_, &particles_->data, false, std::move(g2p_kernel));
}

template <>
void Transfer<AutoDiffXd, MockSparseGrid>::ParallelSimdGridToParticle(
    const Parallelism parallelize) {
  throw std::runtime_error("simd g2p Not implemented.");
}

template <typename T, template <typename> class Grid>
void Transfer<T, Grid>::ParallelSimdGridToParticle(
    const Parallelism parallelism) {
  const ParticleSorter& sorter = particles_->sorter;
  auto g2p_kernel = [&](const Pad<Vector3<T>>& grid_x,
                        Pad<GridData<T>>* grid_data,
                        ParticleData<T>* particle_data,
                        const std::vector<int>& data_indices) {
    Vector3<SimdScalar<T>> v = Vector3<SimdScalar<T>>::Zero();
    Matrix3<SimdScalar<T>> B = Matrix3<SimdScalar<T>>::Zero();
    Vector3<SimdScalar<T>> x = Load(particle_data->x, data_indices);
    const BsplineWeights<SimdScalar<T>> bspline =
        BsplineWeights<SimdScalar<T>>(x, grid_->dx());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const Vector3<T>& vi = (*grid_data)[i][j][k].v;
          const Vector3<T>& xi = grid_x[i][j][k];
          const SimdScalar<T> w = bspline.weight(i, j, k);
          v += w * vi;
          B += (w * vi) * (xi - x).transpose();
        }
      }
    }
    Matrix3<SimdScalar<T>> C = B * D_inverse_;
    x += v * dt_;
    Matrix3<SimdScalar<T>> F = Load(particle_data->F, data_indices);
    F += C * dt_ * F;
    const T c1 = (1 + kApicRatio) * 0.5;
    const T c2 = (kApicRatio - 1) * 0.5;
    C = (c1 * C + c2 * C.transpose()).eval();
    Store(v, &particle_data->v, data_indices);
    Store(x, &particle_data->x, data_indices);
    Store(C, &particle_data->C, data_indices);
    Store(F, &particle_data->F, data_indices);
  };
  sorter.IterateParallelSimd(grid_, &particles_->data, false, parallelism,
                             std::move(g2p_kernel));
}

template <>
void Transfer<AutoDiffXd, MockSparseGrid>::ContactP2G2P() {
  throw std::runtime_error("p2g2p Not implemented.");
}

template <typename T, template <typename> class Grid>
void Transfer<T, Grid>::ContactP2G2P() {
  const ParticleSorter& sorter = particles_->sorter;
  auto p2g_kernel = [&](const Pad<Vector3<T>>& grid_x,
                        Pad<GridData<T>>* grid_data,
                        ParticleData<T>* particle_data, int data_index) {
    const Vector3<T>& x = particle_data->x[data_index];
    const Vector3<T>& f = particle_data->f[data_index];
    BsplineWeights<T> bspline(x, grid_->dx());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const T& w = bspline.weight(i, j, k);
          (*grid_data)[i][j][k].v += f * w;
        }
      }
    }
  };
  sorter.Iterate(grid_, &particles_->data, true, std::move(p2g_kernel));

  /* G2P */
  auto g2p_kernel = [&](const Pad<Vector3<T>>& grid_x,
                        Pad<GridData<T>>* grid_data,
                        ParticleData<T>* particle_data, int data_index) {
    Vector3<T>& x = particle_data->x[data_index];
    Vector3<T>& v = particle_data->v[data_index];
    const BsplineWeights<T> bspline(x, grid_->dx());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const Vector3<T>& vi = (*grid_data)[i][j][k].v;
          const T& mi = (*grid_data)[i][j][k].m;
          const T& w = bspline.weight(i, j, k);
          v += w * vi / mi;
        }
      }
    }
  };
  sorter.Iterate(grid_, &particles_->data, false, std::move(g2p_kernel));
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::internal::Transfer<double>;
template class drake::multibody::mpm::internal::Transfer<float>;
template class drake::multibody::mpm::internal::Transfer<
    drake::AutoDiffXd, drake::multibody::mpm::internal::MockSparseGrid>;
