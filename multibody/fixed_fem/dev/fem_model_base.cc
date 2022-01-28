#include "drake/multibody/fixed_fem/dev/fem_model_base.h"

namespace drake {
namespace multibody {
namespace fem {

template <typename T>
std::unique_ptr<FemStateBase<T>> FemModelBase<T>::MakeFemStateBase() const {
  return DoMakeFemStateBase();
}

template <typename T>
void FemModelBase<T>::CalcResidual(const FemStateBase<T>& state,
                                   EigenPtr<VectorX<T>> residual) const {
  DRAKE_DEMAND(residual != nullptr);
  ThrowIfModelStateIncompatible(__func__, state);
  DoCalcResidual(state, residual);
  dirichlet_bc_.ApplyBoundaryConditionToResidual(residual);
}

template <typename T>
void FemModelBase<T>::CalcTangentMatrix(
    const FemStateBase<T>& state, const Vector3<T>& weights,
    Eigen::SparseMatrix<T>* tangent_matrix) const {
  DRAKE_DEMAND(tangent_matrix != nullptr);
  DRAKE_DEMAND(tangent_matrix->rows() == num_dofs());
  DRAKE_DEMAND(tangent_matrix->cols() == num_dofs());
  ThrowIfModelStateIncompatible(__func__, state);
  DoCalcTangentMatrix(state, weights, tangent_matrix);
  dirichlet_bc_.ApplyBoundaryConditionToTangentMatrix(tangent_matrix);
}

template <typename T>
void FemModelBase<T>::CalcTangentMatrix(
    const FemStateBase<T>& state, const Vector3<T>& weights,
    internal::PetscSymmetricBlockSparseMatrix* tangent_matrix) const {
  DRAKE_DEMAND(tangent_matrix != nullptr);
  DRAKE_DEMAND(tangent_matrix->rows() == num_dofs());
  DRAKE_DEMAND(tangent_matrix->cols() == num_dofs());
  ThrowIfModelStateIncompatible(__func__, state);
  DoCalcTangentMatrix(state, weights, tangent_matrix);
  dirichlet_bc_.ApplyBoundaryConditionToTangentMatrix(tangent_matrix);
}

template <typename T>
Eigen::SparseMatrix<T> FemModelBase<T>::MakeEigenSparseTangentMatrix() const {
  return DoMakeEigenSparseTangentMatrix();
}

template <typename T>
std::unique_ptr<internal::PetscSymmetricBlockSparseMatrix>
FemModelBase<T>::MakePetscSymmetricBlockSparseTangentMatrix() const {
  return DoMakePetscSymmetricBlockSparseTangentMatrix();
}

template <typename T>
void FemModelBase<T>::ApplyBoundaryCondition(FemStateBase<T>* state) const {
  DRAKE_DEMAND(state != nullptr);
  ThrowIfModelStateIncompatible(__func__, *state);
  dirichlet_bc_.ApplyBoundaryConditionToState(state);
}

template <typename T>
void FemModelBase<T>::SetGravityVector(const Vector3<T>& gravity) {
  /* Store gravity so that all elements added after the call to this method
   get the "new" gravity constant. */
  gravity_ = gravity;
  /* Update the gravity vector in elements added before the call to
   this method. */
  DoSetGravityVector(gravity);
}

}  // namespace fem
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::FemModelBase);
