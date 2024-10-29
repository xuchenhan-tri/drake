#pragma once

#include "transfer.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* Newton-Raphson solver for implicit MPM. */
template <typename T>
class MpmSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmSolver);

  /* Initializes an MpmSolver for an implicit MPM step.
   @param[in] dt          Time step.
   @param[in] sparse_grid The background Eulerian grid, must be non-null.
   @param[in] particles   The particles at the previous time step, must be
                          non-null. */
  MpmSolver(T dt, SparseGrid<T>* sparse_grid, ParticleData<T>* particles,
            Parallelism parallelism)
      : dt_(dt), state_(sparse_grid, particles), parallelism_(parallelism) {}

  /* Solves the equation

     M*dv-f(x(v+dv),v+dv)*dt = 0

   with unknown variable dv on the grid with a Newton-Raphson solver. */
  int SolveFreeMotion() {
    VectorX<T> b = VectorX<T>::Zero(state_.num_dofs());
    state_.CalcResidual(&b);
    T residual_norm = b.norm();
    if (residual_norm < abs_tolerance_) {
      return 0;
    }
    VectorX<T> ddv = VectorX<T>::Zero(state_.num_dofs());
    const T initial_residual_norm = residual_norm;
    Block3x3SparseSymmetricMatrix tangent_matrix = state_.MakeTangentMatrix();
    LinearSolver linear_solver;
    int iter = 0;

    while (iter < max_iterations &&
           /* On first iteration, this is equivalent to residual_norm <
              abs_tolerance_, which we have ruled out earlier. */
           !solver_converged(residual_norm, initial_residual_norm)) {
      state_.CalcTangentMatrix(&tangent_matrix);
      if (iter == 0) {
        linear_solver.SetMatrix(tangent_matrix);
      } else {
        linear_solver.UpdateMatrix(tangent_matrix);
      }
      const bool factored = linear_solver.Factor();
      if (!factored) {
        throw std::runtime_error(
            "Tangent matrix factorization failed in MpmSolver because the MPM "
            "tangent matrix is not symmetric positive definite (SPD). This may "
            "be triggered by a combination of a stiff nonlinear constitutive "
            "model and a large time step.");
      }
      /* Solve for the change in unknowns. */
      ddv = linear_solver.Solve(-b);
      state_.IncrementDv(ddv);
      state_.CalcResidual(&b);
      residual_norm = b.norm();
      ++iter;
    }
  }

 private:
  class State {
   public:
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(State);
    /* Constructs a State that's used in MpmSolver. */
    State(T dt, SparseGrid<T>* grid, ParticleData<T>* particles,
          Parallelism parallelism)
        : grid_(grid), particles_(particles), parallelism_(parallelism) {
      DRAKE_DEMAND(grid != nullptr);
      DRAKE_DEMAND(particles != nullptr);
      Transfer<T> transfer(dt, grid, particles);
      transfer.ParallelSimdParticleToGrid(parallelism);
      grid->SetNodeIndices();
      constexpr int kSpatialDim = 3;
      dv_ = VectorX<T>::Zero(grid->num_active_nodes() * kSpatialDim);
      F_ = particles_->F;
      tau_v0_ = particles_->tau_v0;
      volume_scaled_stress_derivatives_.resize(F_.size());
      UpdateParticleState();
    }

    int num_dofs() const { return dv_.size(); }

    /* Sets dv = dv + ddv. */
    void IncrementDv(const VectorX<T>& ddv) {
      DRAKE_DEMAND(ddv.size() == dv_.size());
      dv_ += ddv;
      UpdateParticleState();
    }

    /* Makes a Block3x3SparseSymmetricMatrix that has the sparsity pattern of
     the grid induced by the particles. Each entry of the returned matrix is set
     to zero. Note that there exists a non-zero entry between grid node i and j
     iff there exists a particle that transfers to both i and j. With quadratic
     B-spline, a node can have up to 125 neighbors, including itself.
     @pre SetNodeIndices() has been called on the grid referenced by this
     transfer. */
    contact_solvers::internal::Block3x3SparseSymmetricMatrix MakeTangentMatrix()
        const;

    /* Computes the residual vector b = M * dv - f(x(v+dv), v+dv) * dt. */
    void CalcResidual(VectorX<T>* b) {
      DRAKE_DEMAND(b != nullptr);
      b->resizeLike(dv);
      b->setZero();
      // TODO(xuchenhan-tri): Implement this function.
    }

    /* Computes the tangent matrix of the residual vector b = M * dv -
     f(x(v+dv), v+dv) * dt. */
    void CalcTangentMatrix(Block3x3SparseSymmetricMatrix* tangent_matrix) {
      DRAKE_DEMAND(tangent_matrix != nullptr);
      // TODO(xuchenhan-tri): Implement this function.
    }

   private:
    /* Computes the particle deformation gradient, stress, and stress
     derivatives based on grid data and dv. */
    void UpdateParticleState() {
      /* First update the deformation gradient. */
      const int lanes = SimdScalar<T>::lanes();
      const std::vector<int>& sentinel_particles =
          particles_->sentinel_particles;
      const std::vector<int>& data_indices = particles_->data_indices;
      const std::vector<uint64_t>& base_node_offsets =
          particles_->base_node_offsets;
      const int num_blocks = sparse_grid_->num_blocks();
      [[maybe_unused]] const int num_threads = parallelize.num_threads();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
  for (int b = 0; b < num_blocks; ++b) {
    bool need_new_pad = true;
    Pad<Vector3<T>> grid_x;
    Pad<GridData<T>> grid_data;
    const int particle_start = sentinel_particles[b];
    const int particle_end = sentinel_particles[b + 1];
    std::vector<int> indices;
    indices.reserve(lanes);
    int p = particle_start;
    while (p < particle_end) {
      int next_p = p + 1;
      while (base_node_offsets[next_p] == base_node_offsets[p] &&
             next_p - p < lanes && next_p < particle_end) {
        ++next_p;
      }
      if (need_new_pad) {
        grid_data = sparse_grid_->GetPadData(base_node_offsets[p]);
        grid_x = sparse_grid_->GetPadNodes(particles_->x[data_indices[p]]);
      }
      indices.clear();
      for (int i = p; i < next_p; ++i) {
        indices.push_back(data_indices[i]);
      }
      Matrix3<SimdScalar<T>> B = Matrix3<SimdScalar<T>>::Zero();
      Vector3<SimdScalar<T>> x = Load(particles_->x, indices);
      const BsplineWeights<SimdScalar<T>> bspline =
          BsplineWeights<SimdScalar<T>>(x, sparse_grid_->dx());
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            const int grid_index = grid_data[i][j][k].index;
            const Vector3<T>& vi = grid_data[i][j][k].v + dv_.segment<3>(3 * grid_index);
            const Vector3<T>& xi = grid_x[i][j][k];
            const SimdScalar<T> w = bspline.weight(i, j, k);
            B += (w * vi) * (xi - x).transpose();
          }
        }
      }
      Matrix3<SimdScalar<T>> C = B * D_inverse_;
      Matrix3<SimdScalar<T>> F = Load(F_, indices);
      F += C * dt_ * F;
      Store(F, &F_, indices);

      need_new_pad = (next_p == particle_end) ||
                     base_node_offsets[next_p] != base_node_offsets[p];
      p = next_p;
    }}
    /* Then update stress and stress derivatives. */
    particles_->UpdateStress(F_, &tau_v0_, parallelism_);
    particles_->UpdateStressDerivatives(F_, &volume_scaled_stress_derivatives_, parallelism_);

    }

    VectorX<T> dv_;
    SparseGrid<T>* grid_;
    ParticleData<T>* particles_;
    Parallelism parallelism_{};
    /* Scratch data. */
    std::vector<Matrix3<T>> F_;
    std::vector<Matrix3<T>> tau_v0_;
    std::vector<Eigen::Matrix<T, 9, 9>> volume_scaled_stress_derivatives_;
  };

  T dt_{};
  State state_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake