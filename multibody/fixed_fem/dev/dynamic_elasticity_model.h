#pragma once

#include <array>
#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/multibody/fixed_fem/dev/damping_model.h"
#include "drake/multibody/fixed_fem/dev/elasticity_model.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
/** The FEM model for dynamic 3D elasticity problems. Implements the interface
 in FemModel. It is assumed that elements are only added to, but never deleted
 from, the model.
 @tparam Element    The type of DynamicElasticityElement used in this
 %DynamicElasticityModel, must be an instantiation of DynamicElasticityElement.
 */
template <class Element>
class DynamicElasticityModel : public ElasticityModel<Element> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DynamicElasticityModel);

  using T = typename Element::Traits::T;
  using ConstitutiveModel = typename Element::Traits::ConstitutiveModel;

  DynamicElasticityModel() = default;
  ~DynamicElasticityModel() = default;

  /** Add tetrahedral DynamicElasticityElements to the %DynamicElasticityModel
   from a mesh. The positions of the vertices in the mesh are used as reference
   positions for the volume as well as the generalized positions for the model
   in MakeFemState().
   @param mesh    The input tetrahedral mesh that describes the connectivity and
   the positions of the vertices. Each geometry::VolumeElement in the input
   `mesh` will generate a DynamicElasticityElement in this
   %DynamicElasticityModel.
   @param constitutive_model    The ConstitutiveModel to be used for all the
   DynamicElasticityElements created from the input `mesh`.
   @param density    The mass density of the new elements, in unit kg/m³.
   @param damping_model    The DampingModel to be used for all the
   DynamicElasticityElements created from the input `mesh`.
   @throw std::exception if Element::Traits::kNumNodes != 4. */
  void AddDynamicElasticityElementsFromTetMesh(
      const geometry::VolumeMesh<T>& mesh,
      const ConstitutiveModel& constitutive_model, const T& density,
      const DampingModel<T>& damping_model) {
    /* Alias for more readability. */
    constexpr int kDim = Element::Traits::kSolutionDimension;
    constexpr int kNumNodes = Element::Traits::kNumNodes;
    DRAKE_THROW_UNLESS(kNumNodes == 4);

    using geometry::VolumeElementIndex;
    /* Record the reference positions of the input mesh. */
    const int node_offset = this->ParseTetMesh(mesh);
    const NodeIndex node_index_offset(node_offset);

    /* Builds and adds new elements. */
    const VectorX<T>& X = this->reference_positions();
    std::array<NodeIndex, kNumNodes> element_node_indices;
    for (VolumeElementIndex i(0); i < mesh.num_elements(); ++i) {
      for (int j = 0; j < kNumNodes; ++j) {
        const int node_id = mesh.element(i).vertex(j) + node_index_offset;
        element_node_indices[j] = NodeIndex(node_id);
      }
      const Vector<T, kDim* kNumNodes> element_reference_positions =
          Element::ExtractElementDofs(element_node_indices, X);
      const ElementIndex next_element_index =
          ElementIndex(this->num_elements());
      const auto& element_reference_positions_reshaped =
          Eigen::Map<const Eigen::Matrix<T, kDim, kNumNodes>>(
              element_reference_positions.data(), kDim, kNumNodes);
      this->AddElement(next_element_index, element_node_indices,
                       constitutive_model, element_reference_positions_reshaped,
                       density, this->gravity(), damping_model);
    }
  }

 private:
  /* Implements FemModel::DoMakeFemState(). Generalized positions are
   initialized to be reference positions of the input mesh vertices. Velocities
   and accelerations are initialized to 0. */
  FemState<Element> DoMakeFemState() const final {
    const VectorX<T>& X = this->reference_positions();
    return FemState<Element>(X, VectorX<T>::Zero(X.size()),
                             VectorX<T>::Zero(X.size()));
  }
};
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
