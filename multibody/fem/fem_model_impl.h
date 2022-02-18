#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Sparse>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/element_data.h"
#include "drake/multibody/fem/element_data_impl.h"
#include "drake/multibody/fem/fem_element.h"
#include "drake/multibody/fem/fem_indexes.h"
#include "drake/multibody/fem/fem_model.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* FemModelImpl provides a fixed size implementaion of FemModel by
 templatizing on the type of FemElement. See FemModel for more information
 on the user-facing APIs of this class. Many methods provided by FemModel
 (e.g. FemModel::CalcTangentMatrix) involve evaluating computationally
 intensive loops over FemElement, and the overhead caused by virtual methods may
 be significant. Therefore, this class is templated on the FemElement to avoid
 the overhead of virtual methods. The type information at compile time also
 helps eliminate heap allocations.
 @tparam Element    The type of FEM elements that makes up this FemModelImpl.
 This template parameter must be an instantiation of FemElement, which provides
 the scalar type and the compile time constants such as the natural dimension
 and the number of nodes/quadrature points in each
 element. See FemElements for more details. */
template <class Element>
class FemModelImpl : public FemModel<typename Element::Traits::T> {
 public:
  static_assert(
      std::is_base_of_v<FemElement<Element, typename Element::Traits>, Element>,
      "The template parameter Element should be derived from FemElement. ");
  using T = typename Element::Traits::T;

  int num_elements() const final { return elements_.size(); }

 protected:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemModelImpl);

  /* Creates an empty FemModelImpl with no elements. */
  FemModelImpl() = default;

  virtual ~FemModelImpl() = default;

  const Element& element(ElementIndex i) const {
    DRAKE_ASSERT(i.is_valid());
    DRAKE_ASSERT(i < num_elements());
    return elements_[i];
  }

  Element& mutable_element(ElementIndex i) {
    DRAKE_ASSERT(i.is_valid());
    DRAKE_ASSERT(i < num_elements());
    return elements_[i];
  }

  /* Moves the input `element` into the vector of elements held by this
   FemModelImpl. */
  void AddElement(Element&& element) {
    elements_.emplace_back(std::move(element));
  }

  /* Alternative signature for adding a new element to this FemModelImpl.
   Forwards the arguments `args` to the constructor of the Element and
   potentially creates the new element in place. */
  template <typename... Args>
  void AddElement(Args&&... args) {
    elements_.emplace_back(std::forward<Args>(args)...);
  }

 private:
  void DoCalcResidual(const FemData<T>& fem_data,
                      EigenPtr<VectorX<T>> residual) const final {
    /* The values are accumulated in the residual, so it is important to clear
     the old data. */
    residual->setZero();
    /* Aliases to improve readability. */
    constexpr int kNumDofs = Element::Traits::num_dofs;
    constexpr int kNumNodes = Element::Traits::num_nodes;
    constexpr int kDim = Element::Traits::kSpatialDimension;
    /* Scratch space to store the contribution to the residual from each
     element. */
    Vector<T, kNumDofs> element_residual;
    for (ElementIndex e(0); e < num_elements(); ++e) {
      elements_[e].CalcResidual(fem_data, &element_residual);
      const std::array<NodeIndex, kNumNodes>& element_node_indices =
          elements_[e].node_indices();
      for (int i = 0; i < kNumNodes; ++i) {
        const int ei = element_node_indices[i];
        residual->template segment<kDim>(ei * kDim) +=
            element_residual.template segment<kDim>(i * kDim);
      }
    }
  }

  void DoCalcTangentMatrix(const FemData<T>& fem_data,
                           const Vector3<T>& weights,
                           Eigen::SparseMatrix<T>* tangent_matrix) const final {
    /* The values are accumulated in the tangent_matrix, so it is important to
     clear the old data. */
    using Iterator = typename Eigen::SparseMatrix<T>::InnerIterator;
    for (int k = 0; k < tangent_matrix->outerSize(); ++k) {
      for (Iterator it(*tangent_matrix, k); it; ++it) {
        it.valueRef() = 0;
      }
    }
    /* Aliases to improve readability. */
    constexpr int kNumDofs = Element::Traits::num_dofs;
    constexpr int kNumNodes = Element::Traits::num_nodes;
    constexpr int kDim = Element::Traits::kSpatialDimension;
    /* Scratch space to store the contribution to the tangent matrix from each
     element. */
    Eigen::Matrix<T, kNumDofs, kNumDofs> element_tangent_matrix;
    for (ElementIndex e(0); e < num_elements(); ++e) {
      element_tangent_matrix.setZero();
      elements_[e].CalcTangentMatrix(fem_data, weights,
                                     &element_tangent_matrix);
      const std::array<NodeIndex, kNumNodes>& element_node_indices =
          elements_[e].node_indices();
      for (int a = 0; a < kNumNodes; ++a) {
        for (int i = 0; i < kDim; ++i) {
          for (int b = 0; b < kNumNodes; ++b) {
            for (int j = 0; j < kDim; ++j) {
              tangent_matrix->coeffRef(element_node_indices[a] * kDim + i,
                                       element_node_indices[b] * kDim + j) +=
                  element_tangent_matrix(a * kDim + i, b * kDim + j);
            }
          }
        }
      }
    }
  }

  void DoCalcTangentMatrix(
      const FemData<T>& fem_data, const Vector3<T>& weights,
      PetscSymmetricBlockSparseMatrix* tangent_matrix) const {
    if constexpr (!std::is_same_v<typename Element::T, double>) {
      throw std::logic_error(
          "The PetscSymmetricBlockSparseMatrix overload of "
          "FemModelImpl::CalcTangentMatrixForConcreteData() only supports "
          "scalar "
          "type `double`.");
    } else {
      /* The values are accumulated in the tangent_matrix, so it is important to
       clear the old data. */
      tangent_matrix->SetZero();

      /* Aliases to improve readability. */
      constexpr int kNumDofs = Element::Traits::num_dofs;
      constexpr int kNumNodes = Element::Traits::num_nodes;

      /* Scratch space to store the contribution to the tangent matrix from each
       element. */
      Vector<int, kNumNodes> block_indices;
      Eigen::Matrix<T, kNumDofs, kNumDofs> element_tangent_matrix;
      for (ElementIndex e(0); e < num_elements(); ++e) {
        elements_[e].CalcTangentMatrix(fem_data, weights,
                                       &element_tangent_matrix);
        const std::array<NodeIndex, kNumNodes>& element_node_indices =
            elements_[e].node_indices();
        // TODO(xuchenhan-tri): Avoid this index copy.
        for (int a = 0; a < kNumNodes; ++a) {
          block_indices(a) = element_node_indices[a];
        }
        tangent_matrix->AddToBlock(block_indices, element_tangent_matrix);
      }
    }
  }

  /* Implements FemModel::DoMakeEigenSparseTangentMatrix(). */
  Eigen::SparseMatrix<T> DoMakeEigenSparseTangentMatrix() const final {
    Eigen::SparseMatrix<T> tangent_matrix(this->num_dofs(), this->num_dofs());
    std::vector<Eigen::Triplet<T>> non_zero_entries;
    /* Alias for readability. */
    constexpr int element_num_dofs = Element::Traits::num_dofs;
    constexpr int element_num_nodes = Element::Traits::num_nodes;
    constexpr int kDim = Element::Traits::kSpatialDimension;
    /* Get an upper bound for the number of nonzero entries and allocate
     memories for them in the vector of triplets. */
    non_zero_entries.reserve(num_elements() * element_num_dofs *
                             element_num_dofs);
    /* Create a nonzero block for each pair of nodes that are connected by an
     edge in the mesh. */
    for (int e = 0; e < num_elements(); ++e) {
      const std::array<NodeIndex, element_num_nodes>& element_node_indices =
          elements_[e].node_indices();
      for (int a = 0; a < element_num_nodes; ++a) {
        for (int i = 0; i < kDim; ++i) {
          const int row_index = kDim * element_node_indices[a] + i;
          for (int b = 0; b < element_num_nodes; ++b) {
            for (int j = 0; j < kDim; ++j) {
              const int col_index = kDim * element_node_indices[b] + j;
              non_zero_entries.emplace_back(row_index, col_index, 0);
            }
          }
        }
      }
    }
    tangent_matrix.setFromTriplets(non_zero_entries.begin(),
                                   non_zero_entries.end());
    tangent_matrix.makeCompressed();
    return tangent_matrix;
  }

  /* Implements FemModel::MakePetscSymmetricBlockSparseTangentMatrix(). */
  std::unique_ptr<PetscSymmetricBlockSparseMatrix>
  DoMakePetscSymmetricBlockSparseTangentMatrix() const final {
    std::vector<std::unordered_set<int>> neighbor_nodes(this->num_nodes());
    /* Alias for readability. */
    constexpr int element_num_nodes = Element::Traits::num_nodes;
    constexpr int element_num_dofs = Element::Traits::num_dofs;
    constexpr int kDim = Element::Traits::kSpatialDimension;
    /* Create a nonzero block for each pair of nodes that are connected by an
     edge in the mesh. */
    for (int e = 0; e < num_elements(); ++e) {
      const std::array<NodeIndex, element_num_nodes>& element_node_indices =
          elements_[e].node_indices();
      for (int a = 0; a < element_num_nodes; ++a) {
        for (int b = a; b < element_num_nodes; ++b) {
          const int block_row =
              std::min(element_node_indices[a], element_node_indices[b]);
          const int block_col =
              std::max(element_node_indices[a], element_node_indices[b]);
          neighbor_nodes[block_row].insert(block_col);
        }
      }
    }
    std::vector<int> nonzero_blocks(this->num_nodes());
    for (int i = 0; i < this->num_nodes(); ++i) {
      nonzero_blocks[i] = neighbor_nodes[i].size();
    }
    auto tangent_matrix = std::make_unique<PetscSymmetricBlockSparseMatrix>(
        this->num_dofs(), kDim, nonzero_blocks);

    /* Populate the tangent matrix with zeros at appropriate places to allocate
     memory. */
    Vector<int, element_num_nodes> block_indices;
    const Eigen::Matrix<double, element_num_dofs, element_num_dofs>
        zero_matrix =
            Eigen::Matrix<double, element_num_dofs, element_num_dofs>::Zero();
    for (ElementIndex e(0); e < num_elements(); ++e) {
      const std::array<NodeIndex, element_num_nodes>& element_node_indices =
          elements_[e].node_indices();
      // TODO(xuchenhan-tri): Avoid this index copy.
      for (int a = 0; a < element_num_nodes; ++a) {
        block_indices(a) = element_node_indices[a];
      }
      tangent_matrix->AddToBlock(block_indices, zero_matrix);
    }
    return tangent_matrix;
  }

  /* Implements FemModel::SetGravityVector(). */
  void DoSetGravityVector(const Vector3<T>& gravity) {
    /* Update the gravity vector of all existing elements. */
    for (ElementIndex e(0); e < num_elements(); ++e) {
      elements_[e].set_gravity_vector(gravity);
    }
  }

  /* FemElements owned by this model. */
  std::vector<Element> elements_{};
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
