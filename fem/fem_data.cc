#include "drake/fem/fem_data.h"
namespace drake {
namespace fem {
template <typename T>
void FemData<T>::SetMassFromDensity(const int object_id, const T density) {
  const auto& vertex_range = vertex_indices_[object_id];
  // Clear old mass values.
  for (int i = 0; i < static_cast<int>(vertex_range.size()); ++i) {
    mass_[vertex_range[i]] = 0;
  }
  // Add the mass contribution of each element.
  const auto& element_range = element_indices_[object_id];
  for (int i = 0; i < static_cast<int>(element_range.size()); ++i) {
    const auto& element = elements_[element_range[i]];
    const Vector4<int>& local_indices = element.get_indices();
    const T fraction = 1.0 / static_cast<T>(local_indices.size());
    for (int j = 0; j < static_cast<int>(local_indices.size()); ++j) {
      mass_[local_indices[j]] +=
          density * element.get_element_measure() * fraction;
    }
  }
}

template <typename T>
int FemData<T>::AddUndeformedObject(const std::vector<Vector4<int>>& indices,
                                    const Matrix3X<T>& positions,
                                    const MaterialConfig& config) {
  // Add new elements and record the element indices for this object.
  std::vector<int> local_element_indices(indices.size());
  Vector4<int> vertex_offset{num_vertices_, num_vertices_, num_vertices_,
                               num_vertices_};
  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    mesh_.push_back(indices[i] + vertex_offset);
    // TODO(xuchenhan-tri): Support customized constitutive models.
    elements_.emplace_back(
        indices[i], positions,
        std::make_unique<CorotatedLinearElasticity<T>>(
            config.youngs_modulus, config.poisson_ratio, config.mass_damping,
            config.stiffness_damping),
        config.density, num_vertices_);
    local_element_indices[i] = num_elements_++;
  }
  element_indices_.push_back(local_element_indices);

  // Record the vertex indices for this object.
  std::vector<int> local_vertex_indices(positions.cols());
  for (int i = 0; i < positions.cols(); ++i)
    local_vertex_indices[i] = num_vertices_++;
  vertex_indices_.push_back(local_vertex_indices);

  // Allocate for reference positions.
  Q_.conservativeResize(3, Q_.cols() + positions.cols());
  Q_.rightCols(positions.cols()) = positions;

  // Set mass.
  mass_.conservativeResize(mass_.size() + positions.cols());
  const int object_id = num_objects_;
  num_objects_++;
  SetMassFromDensity(object_id, config.density);
  return object_id;
}
}  // namespace fem
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::drake::fem::FemData)
