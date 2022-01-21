#pragma once

#include <array>
#include <memory>
#include <utility>

#include "drake/multibody/fixed_fem/dev/elasticity_element.h"
#include "drake/multibody/fixed_fem/dev/fem_model.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* The FEM model for dynamic 3D volumetric elasticity problems.
 @tparam Element  The type of FEM element used in this model. Must be of type
 VolumetricElement. */
template <class Element>
class VolumetricModel : public FemModel<Element> {
 public:
  using Traits = typename Element::Traits;
  using T = typename Traits::T;
  constexpr int kSpatialDimension = Traits::kSpatialDimension;
  constexpr int num_nodes = Traits::num_nodes;

  static_assert(
      std::is_same_v<
          VolumetricElement<Traits::IsoparametricElement, Traits::Quadrature,
                            Traits::ConstitutiveModel, Element, Traits>,
          Element>,
      "The template parameter `Element` must be of type VolumetricElement.");

  /* Calculates the total elastic potential energy (in joules) in this
   VolumetricModel. */
  T CalcElasticEnergy(const FemState<Element>& state) const {
    T energy(0);
    for (ElementIndex i(0); i < this->num_elements(); ++i) {
      const Element& e = this->element(i);
      energy += e.CalcElasticEnergy(state);
    }
    return energy;
  }

  /* Sets the gravity vector for all elements, existing and future, in the
   model. */
  void SetGravityVector(const Vector<T, kSpatialDimension>& gravity) {
    /* Store gravity so that all elements added after the call to this method
     get the "new" gravity constant. */
    gravity_ = gravity;
    /* Update the gravity vector in elements added before the call to
     this method. */
    for (ElementIndex e(0); e < this->num_elements(); ++e) {
      this->mutable_element(e).set_gravity_vector(gravity);
    }
  }

  /* Returns the gravity vector for all elements, existing and future, in the
   model. */
  const Vector<T, kSpatialDimension> gravity() const { return gravity_; }

 protected:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(VolumetricModel);

  explicit VolumetricModel(
      std::unique_ptr<internal::StateUpdater<T>> state_updater)
      : FemModel<Element>(std::move(state_updater)) {}

  virtual ~VolumetricModel() = default;

  /** Parse a tetrahedral volume mesh, store the positions of the vertices in
   the mesh as the reference positions for the vertices, and increment the total
   number of vertices in the model. Returns the total number of vertices
   *before* the input `mesh` is parsed.
   @param mesh    The input tetrahedral mesh that describes the connectivity and
   the positions of the vertices. Each geometry::VolumeElement in the input
   `mesh` will generate an VolumetricElement in this VolumetricModel.
   @throws std::exception if called on models with non-tetrahedral elements. */
  int ParseTetMesh(const geometry::VolumeMesh<T>& mesh) {
    if constexpr (num_nodes != 4) {
      throw std::logic_error(
          "ParseTetMesh() only supports tetrahedral elements.");
    } else {
      /* Alias for more readability. */
      constexpr int kDim = kSpatialDimension;
      /* Record the reference positions of the input mesh. */
      const int num_new_vertices = mesh.num_vertices();
      reference_positions_.conservativeResize(reference_positions_.size() +
                                              kDim * num_new_vertices);
      const NodeIndex node_index_offset = NodeIndex(this->num_nodes());
      for (int i = 0; i < num_new_vertices; ++i) {
        reference_positions_.template segment<kDim>(
            kDim * (i + node_index_offset)) = mesh.vertex(i);
      }
      /* Record the number of vertices *before* the input mesh is parsed. */
      const int num_vertices = this->num_nodes();
      this->increment_num_nodes(num_new_vertices);
      return num_vertices;
    }
  }

  /* Returns the reference positions of all nodes in the model. */
  const VectorX<T>& reference_positions() const { return reference_positions_; }

 private:
  VectorX<T> reference_positions_{};
  Vector<T, kSpatialDimension> gravity_{0, 0, -9.81};
};
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
