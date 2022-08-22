#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/query_results/deformable_rigid_contact.h"
#include "drake/multibody/fem/discrete_time_integrator.h"
#include "drake/multibody/fem/petsc_symmetric_block_sparse_matrix.h"
#include "drake/multibody/fem/schur_complement.h"
#include "drake/multibody/fem/velocity_newmark_scheme.h"
#include "drake/multibody/plant/contact_pair_kinematics.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/discrete_update_manager.h"
#include "drake/systems/framework/context.h"

namespace drake {
namespace multibody {
namespace internal {

/* A data structure that contains a stacked list of Eigen::VectorX<T>'s with
 offset information to retrieve individual vectors. */
template <typename T>
class StackedVectors {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(StackedVectors);

  /* Constructs an empty StackedVectors. */
  StackedVectors() = default;

  StackedVectors(VectorX<T> stacked_vectors, std::vector<int> sizes,
                 std::vector<int> offsets) {
    Set(std::move(stacked_vectors), std::move(sizes), std::move(offsets));
  }

  /* Sets `this` StackedVectors to the given data. */
  void Set(VectorX<T> stacked_vectors, std::vector<int> sizes,
           std::vector<int> offsets) {
    data_ = std::move(stacked_vectors);
    sizes_ = std::move(sizes);
    offsets_ = std::move(offsets);
  }

  void Set(std::vector<VectorX<T>> vectors) {
    Clear();
    offsets_.emplace_back(0);
    int total_size = 0;
    for (const VectorX<T>& v : vectors) {
      sizes_.emplace_back(v.size());
      total_size += v.size();
      offsets_.emplace_back(total_size);
    }
    data_.resize(total_size);
    for (int i = 0; i < static_cast<int>(vectors.size()); ++i) {
      data_.segment(offsets_[i], sizes_[i]) = std::move(vectors[i]);
    }
  }

  StackedVectors(const std::vector<VectorX<T>>& data);

  const VectorX<T>& stacked_vector() const { return data_; }

  VectorX<T> vector(int i) const {
    return data_.segment(offsets_[i], sizes_[i]);
  }

  int size(int i) const { return sizes_[i]; }

  int offset(int i) const { return offsets_[i]; }

 private:
  void Clear() {
    data_.resize(0);
    sizes_.clear();
    offsets_.clear();
  }
  VectorX<T> data_;
  std::vector<int> sizes_;
  std::vector<int> offsets_;
};

/*
 @tparam_nonsymbolic_scalar */
template <typename T>
class DeformableDriver : public ScalarConvertibleComponent<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DeformableDriver)

  /* Constructs a deformable driver that solves for the dynamics of the given
   `deformable_model`. The newly constructed driver is used in the given
   `manager` to perform discrete updates. The given `deformable_model` and
   `manager` must outlive this driver.
   @pre deformable_model != nullptr.
   @pre manager != nullptr. */
  DeformableDriver(const DeformableModel<T>* deformable_model,
                   const DiscreteUpdateManager<T>* manager);

  ~DeformableDriver() = default;

  bool is_cloneable_to_double() const final { return true; }
  bool is_cloneable_to_autodiff() const final { return false; }
  bool is_cloneable_to_symbolic() const final { return false; }

  /* Declare cache entries used by this DeformableDriver through the given
   manager.
   @pre `manager` is not nullptr and points to the same DiscreteUpdateManager
   provided at construction. */
  void DeclareCacheEntries(DiscreteUpdateManager<T>* manager);

  /* Calculates the velocities of all participating dofs. The dofs are ordered
   in the increasing order of deformable body ids. That is, the participating
   dofs of the deformable body with the smallest body id come first, followed by
   those of the deformable body with the second smallest id and so on.
   @pre participating_v != nullptr. */
  void CalcParticipatingVelocities(const systems::Context<T>& context,
                                   StackedVectors<T>* participating_v) const;

  /* Eval version of CalcParticipatingVelocities(). */
  const StackedVectors<T>& EvalParticipatingVelocities(
      const systems::Context<T>& context) const;

  /* Calculates the free motion velocities of all participating dofs. The dofs
   are ordered in the increasing order of deformable body ids. That is, the
   participating dofs of the deformable body with the smallest body id come
   first, followed by those of the deformable body with the second smallest id
   and so on.
   @pre participating_v_star != nullptr. */
  void CalcParticipatingFreeMotionVelocities(
      const systems::Context<T>& context,
      StackedVectors<T>* participating_v_star) const;

  /* Eval version of CalcParticipatingFreeMotionVelocities(). */
  const StackedVectors<T>& EvalParticipatingFreeMotionVelocities(
      const systems::Context<T>& context) const;

  /* Given the configuration stored in `context`, this function appends
   discrete pairs corresponding to deformable rigid contact into `pairs`.
   @pre pairs != nullptr. */
  void AppendDiscreteContactPairs(
      const systems::Context<T>& context,
      std::vector<internal::DiscreteContactPair<T>>* pairs) const;

  /* Appends the contact kinematics information between deformable and rigid
   bodies to the given vector.
   @pre contact_kinematics != nullptr. */
  void AppendContactKinematics(
      const systems::Context<T>& context,
      std::vector<ContactPairKinematics<T>>* contact_kinematics) const;

  /* Appends the linear dynamics matrices of each deformable body registered
   in this model to `A` in increasing order of deformable body ids.
   @pre A != nullptr. */
  void AppendLinearDynamicsMatrix(const systems::Context<T>& context,
                                  std::vector<MatrixX<T>>* A) const;

  /* Updates the discrete states of all deformable bodies.
   @pre new_states != nullptr. */
  void UpdateDiscreteStates(const systems::Context<T>& context,
                            systems::DiscreteValues<T>* new_states) const;

 private:
  /* Struct used to conglomerate the indexes of cache entries declared by
   the manager. */
  struct CacheIndexes {
    systems::CacheIndex deformable_rigid_contacts;
    /* Participating v of all bodies, concatenated. */
    systems::CacheIndex stacked_participating_v;
    /* Participating v* of all bodies, concatenated. */
    systems::CacheIndex stacked_participating_v_stars;

    /* Per body cache entries. */
    std::vector<systems::CacheIndex> fem_states;
    std::vector<systems::CacheIndex> free_motion_fem_states;
    std::vector<systems::CacheIndex> next_fem_states;
    std::vector<systems::CacheIndex> fem_solver_scratches;
    std::vector<systems::CacheIndex> free_motion_tangent_matrices;
    std::vector<systems::CacheIndex>
        free_motion_tangent_matrix_schur_complements;
    std::vector<systems::CacheIndex> velocity_permutations;
    std::vector<systems::CacheIndex> participating_v;
    std::vector<systems::CacheIndex> participating_v_stars;
  };

  /* Provide private access for unit testing only. */
  friend class DeformableDriverTester;

  // TODO(xuchenhan-tri): Implement this.
  std::unique_ptr<DiscreteUpdateManager<double>> CloneToDouble() const;

  /* Computes the contact information between each registered deformable body
   and all rigid bodies that have collision representations that can
   meanfully interact with deformable bodies.
   @pre The geometry query input port of the MultibodyPlant that owns the
        manager associated with this DeformableDriver is connected.
   @pre deformable_rigid_contact != nullptr. */
  void CalcDeformableRigidContact(
      const systems::Context<T>& context,
      std::vector<geometry::internal::DeformableRigidContact<T>>*
          deformable_rigid_contact) const;

  /* Eval version of CalcDeformableRigidContact(). */
  const geometry::internal::DeformableRigidContact<T>&
  EvalDeformableRigidContact(const systems::Context<T>& context,
                             DeformableBodyIndex index) const;

  /* Computes the partial permutation that maps back and forth between
   participating velocities of the deformable body with the given `id` and
   the full vector of *all* velocity degrees of freedom of that deformable
   body.
   @param[out] permutation  Given a vector v of the velocities of all the
                            vertices of a deformable body, ordered by vertex
                            indexes, permutation.Apply(v, &vp) writes the
                            participating velocities of this deformable
                            body.
   @pre permutation != nullptr. */
  void CalcVelocityPermutation(
      const systems::Context<T>& context, DeformableBodyIndex index,
      contact_solvers::internal::PartialPermutation* permutation) const;

  /* Eval version of CalcVelocityPermutation(). */
  const contact_solvers::internal::PartialPermutation& EvalVelocityPermutation(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  /* Computes the velocities of the deformable body with the given `id` that
   participates in contact.
   @pre participating_v != nullptr.
   @note `participating_v` is cleared and resized if necessary before
   new values are populated. */
  void CalcParticipatingVelocities(const systems::Context<T>& context,
                                   DeformableBodyIndex index,
                                   VectorX<T>* participating_v) const;

  /* Eval version of CalcParticipatingVelocities(). */
  const VectorX<T>& EvalParticipatingVelocities(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  /* Computes the "free motion" velocities of the deformable body with the
   given `id` that participates in contact.
   @pre participating_v_star != nullptr.
   @note `participating_v_star` is cleared and resized if necessary before
   new values are populated. */
  void CalcParticipatingFreeMotionVelocities(
      const systems::Context<T>& context, DeformableBodyIndex index,
      VectorX<T>* participating_v_star) const;

  /* Eval version of CalcParticipatingFreeMotionVelocities(). */
  const VectorX<T>& EvalParticipatingFreeMotionVelocities(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  /* Copies the state of the deformable body with `id` in the given `context`
   to the `fem_state`.
   @pre fem_state != nullptr and has size compatible with the state of the
        deformable body with the given `id`. */
  void CalcFemState(const systems::Context<T>& context,
                    DeformableBodyIndex index,
                    fem::FemState<T>* fem_state) const;

  /* Eval version of CalcFemState(). */
  const fem::FemState<T>& EvalFemState(const systems::Context<T>& context,
                                       DeformableBodyIndex index) const;

  /* Given the state of the deformable body with `id` in the given `context`,
   computes the "free motion" state of the deformable body at the next time
   step.
   @pre fem_state_star != nullptr and has size compatible with the state of
        the deformable body with the given `id`. */
  void CalcFreeMotionFemState(const systems::Context<T>& context,
                              DeformableBodyIndex index,
                              fem::FemState<T>* fem_state_star) const;

  /* Eval version of CalcFreeMotionFemState(). */
  const fem::FemState<T>& EvalFreeMotionFemState(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  /* Given the state of the deformable body with `id` in the given `context`,
   computes the state of the deformable body at the next time step.
   @note The state of the deformable body will the same as the "free motion"
         state if no constraints is added on the body.
   @pre next_fem_state != nullptr and has size compatible with the state of
        the deformable body with the given `id`. */
  void CalcNextFemState(const systems::Context<T>& context,
                        DeformableBodyIndex index,
                        fem::FemState<T>* next_fem_state) const;

  /* Eval version of CalcNextFemState(). */
  const fem::FemState<T>& EvalNextFemState(const systems::Context<T>& context,
                                           DeformableBodyIndex index) const;

  /* Computes the tangent matrix (see FemModel::CalcTangentMatrix) of
   deformable body with the given `id` at the free motion state.
   @pre tangent_matrix != nullptr. */
  void CalcFreeMotionTangentMatrix(
      const systems::Context<T>& context, DeformableBodyIndex index,
      fem::internal::PetscSymmetricBlockSparseMatrix* tangent_matrix) const;

  /* Eval version of CalcFreeMotionTangentMatrix(). */
  const fem::internal::PetscSymmetricBlockSparseMatrix&
  EvalFreeMotionTangentMatrix(const systems::Context<T>& context,
                              DeformableBodyIndex index) const;

  /* Computes the Schur complement of the tangent matrix of
   deformable body with the given `id` at the free motion state based on
   contact participation (see
   PetscSymmetricBlockSparseMatrix::CalcSchurComplement). The dofs not
   participating in contact are eliminated in favor of those that do
   participate in contact.
   @pre tangent_matrix != nullptr. */
  void CalcFreeMotionTangentMatrixSchurComplement(
      const systems::Context<T>& context, DeformableBodyIndex index,
      fem::internal::SchurComplement<T>* schur_complement) const;

  /* Eval version of CalcFreeMotionTangentMatrixSchurComplement(). */
  const fem::internal::SchurComplement<T>&
  EvalFreeMotionTangentMatrixSchurComplement(const systems::Context<T>& context,
                                             DeformableBodyIndex index) const;

  CacheIndexes cache_indexes_;
  /* Modelling information about all deformable bodies. */
  const DeformableModel<T>* deformable_model_{};
  const DiscreteUpdateManager<T>* manager_{};
  /* The integrator used to advance deformable body states in time. */
  std::unique_ptr<fem::internal::DiscreteTimeIntegrator<T>>
      acceleration_integrator_;
  std::unique_ptr<fem::internal::DiscreteTimeIntegrator<T>>
      velocity_integrator_;
      mutable std::vector<Matrix3X<T>> temp_;
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::internal::DeformableDriver);
