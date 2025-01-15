#include "drake/multibody/mpm/transfer.h"

// #include "drake/common/test_utilities/limit_malloc.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
Transfer<T>::Transfer(T dt, SparseGrid<T>* grid, Particles<T>* particles)
    : dt_(dt), grid_(grid), particles_(particles) {
  DRAKE_DEMAND(dt > 0);
  DRAKE_DEMAND(grid != nullptr);
  DRAKE_DEMAND(particles != nullptr);
  particles->Sort(*grid);
  D_inverse_ = 4.0 / (grid_->dx() * grid_->dx());
  D_inverse_dt_ = D_inverse_ * dt_;
}

template <typename T>
void Transfer<T>::ScalarParticleToGrid(const Particles<T>& particles,
                                       SparseGrid<T>* grid) {
  const auto& particle_data = particles.data;
  const auto& sorter = particles.sorter;
  const int num_particles = particle_data.num_particles();
  const std::vector<int>& data_indices = sorter.data_indices();
  const std::vector<uint64_t>& base_node_offsets = sorter.base_node_offsets();
  const std::vector<Vector3<T>>& particle_x = particle_data.x();
  DRAKE_DEMAND(ssize(data_indices) == num_particles);
  bool need_new_pad = true;
  Pad<Vector3<T>> grid_x;
  Pad<GridData<T>> grid_data;
  for (int p = 0; p < num_particles; ++p) {
    const int data_index = data_indices[p];
    if (need_new_pad) {
      grid_data = grid->GetPadData(base_node_offsets[p]);
      grid_x = grid->GetPadNodes(particle_x[data_index]);
    }
    ScalarP2G(grid_x, particle_data, data_index, &grid_data);
    need_new_pad = (p + 1 == num_particles) ||
                   (base_node_offsets[p] != base_node_offsets[p + 1]);
    if (need_new_pad) {
      grid->SetPadData(base_node_offsets[p], grid_data);
    }
  }
}

template <typename T>
void Transfer<T>::SimdParticleToGrid(const Particles<T>& particles,
                                     SparseGrid<T>* grid) {
  const auto& particle_data = particles.data;
  const auto& sorter = particles.sorter;
  const int num_particles = particle_data.num_particles();
  const std::vector<int>& data_indices = sorter.data_indices();
  const std::vector<uint64_t>& base_node_offsets = sorter.base_node_offsets();
  const std::vector<Vector3<T>>& particle_x = particle_data.x();
  DRAKE_DEMAND(ssize(data_indices) == num_particles);
  Pad<Vector3<T>> grid_x;
  Pad<GridData<T>> grid_data;
  std::vector<int> working_set_indices;
  working_set_indices.reserve(1024);
  WorkingSet<T> working_set(grid_->dx(), D_inverse_dt_);
  // test::LimitMallocParams params;
  // params.max_num_allocations = 3;
  // test::LimitMalloc guard;
  int p = 0;
  while (p < num_particles) {
    const uint64_t offset = base_node_offsets[p];
    grid_data = grid->GetPadData(base_node_offsets[p]);
    grid_x = grid->GetPadNodes(particle_x[data_indices[p]]);
    while (p < num_particles && base_node_offsets[p] == offset) {
      working_set_indices.push_back(data_indices[p]);
      ++p;
    }
    P2G(particle_data, working_set_indices, grid_x, &grid_data, &working_set);
    working_set_indices.clear();
    grid->SetPadData(offset, grid_data);
  }
}

template <typename T>
void Transfer<T>::ScalarP2G(const Pad<Vector3<T>>& grid_x,
                            const ParticleData<T>& particle_data,
                            int data_index, Pad<GridData<T>>* grid_data) {
  const T& m = particle_data.m()[data_index];
  const Vector3<T>& x = particle_data.x()[data_index];
  const Vector3<T>& v = particle_data.v()[data_index];
  const Matrix3<T>& C = particle_data.C()[data_index];
  const Matrix3<T>& tau_volume = particle_data.tau_volume()[data_index];
  const BsplineWeights<T> bspline = MakeBsplineWeights(x, grid_->dx());
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        const T& w = bspline.weight(i, j, k);
        const Vector3<T>& xi = grid_x[i][j][k];
        const T mi = m * w;
        (*grid_data)[i][j][k].v +=
            mi * v + (m * C - D_inverse_dt_ * tau_volume) * (xi - x) * w;
        (*grid_data)[i][j][k].m += mi;
      }
    }
  }
}

template class Transfer<double>;
template class Transfer<float>;

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
