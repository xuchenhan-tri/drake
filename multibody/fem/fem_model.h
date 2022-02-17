#pragma once

#include <array>
#include <memory>
#include <utility>

#include <Eigen/Sparse>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/dirichlet_boundary_condition.h"
#include "drake/multibody/fem/element_data.h"
#include "drake/multibody/fem/fem_data.h"
#include "drake/multibody/fem/fem_state.h"
#include "drake/multibody/fem/petsc_symmetric_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace fem {

/** %FemModel calculates the components of the discretized FEM equations for
 dynamic elasticity problems. Typically, in dynamic elasticity problems, we are
 interested in the mapping that describes the motion of a material

    ϕ(⋅,t) : Ω⁰ → Ωᵗ,

 where Ω⁰ and Ωᵗ are subsets of R³, along with it's first and second derivatives
 (velocity and acceleration respectively):

    V(X,t) = ∂ϕ(X,t)/∂t,
    A(X,t) = ∂²ϕ(X,t)/∂t².

 The governing equations of interest are conservation of mass and conservation
 of momentum:

    R(X,t)J(X,t) = R(X,0),
    R(X,0)A(X,t) = fᵢₙₜ(X,t) + fₑₓₜ(X,t),

 where R is mass density and fᵢₙₜ and fₑₓₜ are internal and external force
 densities respectively. Using finite element method to discretize space, one
 gets

    ϕ(X,t) = ∑ᵢ xᵢ(t)Nᵢ(X)
    V(X,t) = ∑ᵢ vᵢ(t)Nᵢ(X)
    A(X,t) = ∑ᵢ aᵢ(t)Nᵢ(X)

where xᵢ, vᵢ, aᵢ ∈ R³ are nodal values of the spatially discretized position,
velocity and acceleration, and Nᵢ(X):Ω⁰ → R are the the basis functions. With
this spatial discretization, the PDE is turned in an ODE of the form

    G(x, v, a) = 0,            (1)

where x, v, a are the stacked xᵢ, vᵢ, aᵢ. %FemModel provides methods to
query various information about equation (1) given an FEM state (x, v, a) such
as the residual, G(x, v, a) (see CalcResidual()); the stiffness matrix, ∂G/∂x
(see CalcStiffnessMatrix()); the damping matrix, ∂G/∂v (see
CalcDampingMatrix()); the mass matrix, ∂G/∂a (see CalcMassMatrix()).
@tparam_nonsymbolic_scalar */
template <typename T>
class FemModel {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemModel);
  virtual ~FemModel() = default;

  /** The number of nodes that are associated with this model. */
  int num_nodes() const { return num_nodes_; }

  /** The number of degrees of freedom in this model. */
  int num_dofs() const { return 3 * num_nodes_; }

  /** The number of FEM elements in this model. */
  virtual int num_elements() const = 0;

  /** Creates a default FemData compatible with this model. */
  FemData<T> MakeFemData() const {
    /* FEM state. */
    const Matrix3X<T> model_state = MakeFemState();
    const DiscreteStateIndex q_index =
        system_.DeclareDiscreteState(model_state.row(0));
    const DiscreteStateIndex v_index =
        system_.DeclareDiscreteState(model_state.row(1));
    const DiscreteStateIndex a_index =
        system_.DeclareDiscreteState(model_state.row(2));
    /* FEM element data. */
    const std::unique_ptr<const ElementData<T>> model_element_data =
        MakeElementData();
    const auto& element_data_cache_entry = system->DeclareCacheEntry(
        "FEM state dependent element data",
        systems::ValueProducer(
            *model_element_data,
            std::function<void(const systems::Context<T>&, ElementData<T>*)>{
                [q_index, v_index, a_index, this](
                    const systems::Context<T>& context,
                    ElementData<T>* element_data) {
                  const VectorX<T>& q =
                      context.get_discrete_state(q_index).value();
                  const VectorX<T>& v =
                      context.get_discrete_state(v_index).value();
                  const VectorX<T>& a =
                      context.get_discrete_state(a_index).value();
                  this->CalcElementData(q, v, a, element_data);
                }}),
        {system_.discrete_state_ticket(q_index),
         system_.discrete_state_ticket(v_index),
         system_.discrete_state_ticket(a_index)});
    auto element_data_index = element_data_cache_entry.cache_index();
    return FemData<T>(
        {&system_, model_id_, q_index, v_index, a_index, element_data_index});
  }

  /** Calculates the residual at the given FEM state.
  @pre residual != nullptr.
  @throw std::exception if the FEM state is incompatible with this model.
  @note Use MakeFemState() to create an FEM state compatible with this
  model. */
  void CalcResidual(const FemData<T>& fem_data,
                    EigenPtr<VectorX<T>> residual) const;

  /** Calculates the tangent matrix at the given FEM state. The tangent matrix
   is given by a weight sum of stiffness matrix, damping matrix, and mass
   matrix.
   @param[in] state            The FemData at which the tangent matrix is
                               evaluated.
   @param[in] weights          The weight used to combine stiffness, damping,
                               and tangent matrices (in that order) into the
                               tangent matrix.
   @param[out] tangent_matrix  The output tangent_matrix.
   @pre tangent_matrix != nullptr.
   @pre The size of `tangent_matrix` is `num_dofs()` * `num_dofs()`.
   @throw std::exception if the FEM state is incompatible with this model.
   @note Use MakeFemData() to create an FEM state compatible with this
   model. */
  void CalcTangentMatrix(const FemData<T>& fem_data, const Vector3<T>& weights,
                         Eigen::SparseMatrix<T>* tangent_matrix) const;

  /* Alternative signature for calculating tangent matrix that writes to a
   PETSc matrix.
   @param[in] state            The FemData at which the tangent matrix is
                               evaluated.
   @param[in] weights          The weight used to combine stiffness, damping,
                               and tangent matrices (in that order) into the
                               tangent matrix.
   @param[out] tangent_matrix  The output tangent_matrix.
   @pre tangent_matrix != nullptr.
   @pre The size of `tangent_matrix` is `num_dofs()` by `num_dofs()`.
   @throw std::exception if the FEM state is incompatible with this model.
   @note Use AllocateFemData() to create an FEM state compatible with this
   model. */
  void CalcTangentMatrix(
      const FemData<T>& fem_data, const Vector3<T>& weights,
      internal::PetscSymmetricBlockSparseMatrix* tangent_matrix) const;

  /** Creates an Eigen::SparseMatrix that has the sparsity pattern of the
   tangent matrix of this FEM model. In particular, the size of the tangent
   matrix is `num_dofs()` by `num_dofs()`. */
  Eigen::SparseMatrix<T> MakeEigenSparseTangentMatrix() const;

  // TODO(xuchenhan-tri): We are returning a pointer to internal objects in a
  //  public method in a non-internal class.
  /** Creates a PetscSymmetricBlockSparseMatrix that has the sparsity pattern of
   the tangent matrix of this FEM model. In particular, the size of the tangent
   matrix is `num_dofs()` by `num_dofs()`. */
  std::unique_ptr<internal::PetscSymmetricBlockSparseMatrix>
  MakePetscSymmetricBlockSparseTangentMatrix() const;

  /** Applies boundary condition set for this %FemModel to the input
   `state`. No-op if no boundary condition is set.
   @pre state != nullptr. */
  void ApplyBoundaryCondition(FemState<T>* state) const;

  // TODO(xuchenhan-tri): Internal object in public method in non-internal
  //  class.
  /** Sets the Dirichlet boundary condition that this model is subject to. */
  void SetDirichletBoundaryCondition(
      internal::DirichletBoundaryCondition<T> dirichlet_bc) {
    dirichlet_bc_ = std::move(dirichlet_bc);
  }

  /** Returns the dirichlet boundary condition that this model is subject to. */
  const internal::DirichletBoundaryCondition<T>& dirichlet_boundary_condition()
      const {
    return dirichlet_bc_;
  }

  /** Returns the gravity vector for all elements in this model. */
  const Vector3<T>& gravity() const { return gravity_; }

  /** Sets the gravity vector of all existing and future elements in this model.
   */
  void SetGravityVector(const Vector3<T>& gravity);

  /** (Internal use only) Throws std::exception to report a mismatch between
  the concrete types of `this` FemModel and the FemState that was
  passed to API method `func`. */
  void ThrowIfModelStateIncompatible(const char* func,
                                     const FemData<T>& fem_data) const {
    if (fem_data.model() != this) {
      throw std::logic_error(std::string(func) +
                             "(): The FEM data and model are not compatible");
    }
  }

 protected:
  FemModel() : model_id_(ModelId::get_new_id()) {}

  /** Creates a default FEM state for this model, where the positions are set to
   the reference positions and the velocity and the accelerations are set to
   zero. */
  virtual FemState<T> MakeFemState() const = 0;

  /** Creates a default FEM ElementData that matches the size of this model. */
  virtual std::unique_ptr<ElementData<T>> MakeElementData() const = 0;

  /* Calculates the per-element data given the FEM state. */
  void CalcElementData(const FemState<T>& state,
                       ElementData<T>* element_data) const {
    DRAKE_DEMAND(element_data != nullptr);
    DRAKE_DEMAND(element_data->size() == this->num_elements());
    DRAKE_DEMAND(state.num_dofs() == this->num_dofs());
    DoCalcElementData(state, element_data);
  }

  /** Derived classes must override this method to provide an implementation
   for the NVI CalcResidual(). The input `state` is guaranteed to be
   compatible with `this` FEM model. */
  virtual void DoCalcResidual(const FemData<T>& fem_data,
                              EigenPtr<VectorX<T>> residual) const = 0;

  /** Derived classes must override this method to provide an implementation for
   the NVI CalcTangentMatrix(). The input `state` is guaranteed to be compatible
   with `this` FEM model. */
  virtual void DoCalcTangentMatrix(
      const FemData<T>& fem_data, const Vector3<T>& weights,
      Eigen::SparseMatrix<T>* tangent_matrix) const = 0;

  /** Derived classes must override this method to provide an implementation for
   the NVI CalcTangentMatrix(). The input `state` is guaranteed to be compatible
   with `this` FEM model. */
  virtual void DoCalcTangentMatrix(
      const FemData<T>& fem_data, const Vector3<T>& weights,
      internal::PetscSymmetricBlockSparseMatrix* tangent_matrix) const = 0;

  /** Derived classes must override this method to provide an implementation for
   the NVI MakeEigenSparseTangentMatrix(). */
  virtual Eigen::SparseMatrix<T> DoMakeEigenSparseTangentMatrix() const = 0;

  /** Derived classes must override this method to provide an implementation for
   the NVI MakePetscSymmetricBlockSparseTangentMatrix(). */
  virtual std::unique_ptr<internal::PetscSymmetricBlockSparseMatrix>
  DoMakePetscSymmetricBlockSparseTangentMatrix() const = 0;

  /** Derived classes must override this method to set the gravity vector for
   all existing elements in the model. */
  virtual void DoSetGravityVector(const Vector3<T>& gravity) = 0;

  /** Derived classes must override this method to calculate per element data in
   the model from the state. */
  virtual void DoCalcElementData(const FemState<T>& state,
                                 ElementData<T>* element_data) const = 0;

  /** Derived classes must invoke this method to update the number of nodes in
   the model when they add more nodes to the FEM model. */
  void increment_num_nodes(int num_new_nodes) { num_nodes_ += num_new_nodes; }

 private:
  /* A system that can publicly declare cache entries. Used to manage the
   FemData created and consumed by this FEM model. */
  class CachingSystem<T> : public systems::LeafSystem<T> {
   public:
    using SystemBase::DeclareCacheEntry;
  };
  CachingSystem<T> system_;

  ModelId model_id_;
  /* The total number of nodes in the system. */
  int num_nodes_{0};
  /* The Dirichlet boundary condition that the model is subject to. */
  internal::DirichletBoundaryCondition<T> dirichlet_bc_;
  /* Returns the gravity vector for all elements in the model. */
  Vector3<T> gravity_{0, 0, -9.81};
};
}  // namespace fem
}  // namespace multibody
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::FemModel);
