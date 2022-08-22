#include "drake/multibody/plant/deformable_driver.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/contact_solvers/contact_solver_results.h"
#include "drake/multibody/fem/fem_model.h"
#include "drake/multibody/fem/fem_solver.h"
#include "drake/multibody/fem/velocity_newmark_scheme.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/proximity_property_queries.h"
#include "drake/systems/framework/context.h"

using drake::geometry::GeometryId;
using drake::geometry::internal::DeformableRigidContact;
using drake::math::RotationMatrix;
using drake::multibody::contact_solvers::internal::ContactSolverResults;
using drake::multibody::contact_solvers::internal::PartialPermutation;
using drake::multibody::fem::FemModel;
using drake::multibody::fem::FemState;
using drake::multibody::fem::internal::FemSolver;
using drake::multibody::fem::internal::FemSolverScratchData;
using drake::multibody::fem::internal::PetscSymmetricBlockSparseMatrix;
using drake::multibody::fem::internal::SchurComplement;
using drake::multibody::internal::DiscreteContactPair;
using drake::systems::Context;

namespace drake {
namespace multibody {
namespace internal {

template <typename T>
DeformableDriver<T>::DeformableDriver(
    const DeformableModel<T>* deformable_model,
    const DiscreteUpdateManager<T>* manager)
    : deformable_model_(deformable_model), manager_(manager) {
  DRAKE_DEMAND(deformable_model != nullptr);
  DRAKE_DEMAND(manager != nullptr);
  // Set the time integrator for advancing deformable states in time to be the
  // mid-point rule, i.e., x = x₀ + δt/2 *(v₀ + v).
  integrator_ = std::make_unique<fem::internal::VelocityNewmarkScheme<T>>(
      manager_->plant().time_step(), 1.0, 0.5);
}

template <typename T>
void DeformableDriver<T>::DeclareCacheEntries(
    DiscreteUpdateManager<T>* manager) {
  if constexpr (!std::is_same_v<T, double>) {
    unused(manager);
    throw std::logic_error(
        "DeformableDriver only supports updates for deformable bodies "
        "when T == double.");
  } else {
    DRAKE_DEMAND(manager_ == manager);
    const auto& deformable_rigid_contact_cache_entry =
        manager->DeclareCacheEntry(
            "Deformable Rigid Contact Data.",
            systems::ValueProducer(
                this, std::vector<DeformableRigidContact<T>>(),
                &DeformableDriver<T>::CalcDeformableRigidContact),
            {systems::System<T>::configuration_ticket()});
    cache_indexes_.deformable_rigid_contacts =
        deformable_rigid_contact_cache_entry.cache_index();

    std::set<systems::DependencyTicket> stacked_participating_v_dependencies;
    std::set<systems::DependencyTicket>
        stacked_participating_v_stars_dependencies;
    // Declare per deformable body cache entries.
    for (DeformableBodyIndex i(0); i < deformable_model_->num_bodies(); ++i) {
      const DeformableBodyId id = deformable_model_->GetBodyId(i);
      const fem::FemModel<T>& fem_model = deformable_model_->GetFemModel(id);
      std::unique_ptr<fem::FemState<T>> model_state = fem_model.MakeFemState();
      const auto& fem_state_cache_entry = manager->DeclareCacheEntry(
          fmt::format("FEM state for body with index {}", i),
          systems::ValueProducer(
              *model_state,
              std::function<void(const systems::Context<T>&,
                                 fem::FemState<T>*)>{
                  [this, i](const systems::Context<T>& context,
                            fem::FemState<T>* state) {
                    this->CalcFemState(context, i, state);
                  }}),
          {systems::System<T>::xd_ticket()});
      cache_indexes_.fem_states.emplace_back(
           fem_state_cache_entry.cache_index());

      const auto& free_motion_fem_state_cache_entry =
          manager->DeclareCacheEntry(
              fmt::format("Free motion FEM state for body with index {}", i),
              systems::ValueProducer(
                  *model_state,
                  std::function<void(const systems::Context<T>&,
                                     fem::FemState<T>*)>{
                      [this, i](const systems::Context<T>& context,
                                fem::FemState<T>* free_motion_state) {
                        this->CalcFreeMotionFemState(context, i,
                                                     free_motion_state);
                      }}),
              {fem_state_cache_entry.ticket()});
      cache_indexes_.free_motion_fem_states.emplace_back(
          free_motion_fem_state_cache_entry.cache_index());

      const auto& next_fem_state_cache_entry = manager->DeclareCacheEntry(
          fmt::format("Next FEM state for body with index {}", i),
          systems::ValueProducer(
              *model_state,
              std::function<void(const systems::Context<T>&,
                                 fem::FemState<T>*)>{
                  [this, i](const systems::Context<T>& context,
                            fem::FemState<T>* next_fem_state) {
                    this->CalcNextFemState(context, i, next_fem_state);
                  }}),
          {free_motion_fem_state_cache_entry.ticket()});
      cache_indexes_.next_fem_states.emplace_back(
          next_fem_state_cache_entry.cache_index());

      FemSolverScratchData scratch(fem_model);
      const auto& scratch_entry = manager->DeclareCacheEntry(
          fmt::format("FEM solver scratches for body with index {}", i),
          systems::ValueProducer(scratch, &systems::ValueProducer::NoopCalc),
          {systems::SystemBase::nothing_ticket()});
      cache_indexes_.fem_solver_scratches.emplace_back(
          scratch_entry.cache_index());

      std::unique_ptr<PetscSymmetricBlockSparseMatrix> model_tangent_matrix =
          fem_model.MakePetscSymmetricBlockSparseTangentMatrix();
      model_tangent_matrix->AssembleIfNecessary();
      const auto& free_motion_tangent_matrix_cache_entry =
          manager->DeclareCacheEntry(
              fmt::format("Free motion tangent matrix for body with index {}",
                          id),
              systems::ValueProducer(
                  *model_tangent_matrix,
                  std::function<void(const systems::Context<T>&,
                                     PetscSymmetricBlockSparseMatrix*)>{
                      [this, i](
                          const systems::Context<T>& context,
                          PetscSymmetricBlockSparseMatrix* tangent_matrix) {
                        this->CalcFreeMotionTangentMatrix(context, i,
                                                          tangent_matrix);
                      }}),
              {free_motion_fem_state_cache_entry.ticket()});
      cache_indexes_.free_motion_tangent_matrices.emplace_back(
          free_motion_tangent_matrix_cache_entry.cache_index());

      const auto& schur_complement_cache_entry = manager->DeclareCacheEntry(
          fmt::format("Free motion tangent matrix Schur complement for body "
                      "with index {}",
                      i),
          systems::ValueProducer(
              SchurComplement<T>(),
              std::function<void(const systems::Context<T>&,
                                 SchurComplement<T>*)>{
                  [this, i](const systems::Context<T>& context,
                            SchurComplement<T>* schur_complement) {
                    this->CalcFreeMotionTangentMatrixSchurComplement(
                        context, i, schur_complement);
                  }}),
          {free_motion_tangent_matrix_cache_entry.ticket(),
           deformable_rigid_contact_cache_entry.ticket()});
      cache_indexes_.free_motion_tangent_matrix_schur_complements.emplace_back(
          schur_complement_cache_entry.cache_index());

      const auto& velocity_permutation_cache_entry = manager->DeclareCacheEntry(
          fmt::format("Partial permutation for velocities of body {} based on "
                      "participation in contact",
                      i),
          systems::ValueProducer(std::function<void(const systems::Context<T>&,
                                                    PartialPermutation*)>{
              [this, i](const systems::Context<T>& context,
                        PartialPermutation* velocity_permutation) {
                this->CalcVelocityPermutation(context, i, velocity_permutation);
              }}),
          {deformable_rigid_contact_cache_entry.ticket()});
      cache_indexes_.velocity_permutations.emplace_back(
          velocity_permutation_cache_entry.cache_index());

      const auto& participating_v_cache_entry = manager->DeclareCacheEntry(
          fmt::format("Participating velocities for body {}", i),
          systems::ValueProducer(
              std::function<void(const systems::Context<T>&, VectorX<T>*)>{
                  [this, i](const systems::Context<T>& context,
                            VectorX<T>* participating_v) {
                    this->CalcParticipatingVelocities(context, i,
                                                      participating_v);
                  }}),
          {velocity_permutation_cache_entry.ticket(),
           fem_state_cache_entry.ticket()});
      cache_indexes_.participating_v.emplace_back(
          participating_v_cache_entry.cache_index());
      stacked_participating_v_dependencies.insert(
          participating_v_cache_entry.ticket());

      const auto& participating_v_star_cache_entry = manager->DeclareCacheEntry(
          fmt::format("Participating free motion velocities for body {}", i),
          systems::ValueProducer(
              std::function<void(const systems::Context<T>&, VectorX<T>*)>{
                  [this, i](const systems::Context<T>& context,
                            VectorX<T>* participating_v_star) {
                    this->CalcParticipatingFreeMotionVelocities(
                        context, i, participating_v_star);
                  }}),
          {velocity_permutation_cache_entry.ticket(),
           free_motion_fem_state_cache_entry.ticket()});
      cache_indexes_.participating_v_stars.emplace_back(
          participating_v_star_cache_entry.cache_index());
      stacked_participating_v_stars_dependencies.insert(
          participating_v_star_cache_entry.ticket());
    }
    const auto& stacked_participating_v_cache_entry =
        manager->DeclareCacheEntry(
            fmt::format("Participating velocities for all bodies"),
            systems::ValueProducer(
                this, StackedVectors<T>(),
                &DeformableDriver<T>::CalcParticipatingVelocities),
            stacked_participating_v_dependencies);
    cache_indexes_.stacked_participating_v =
        stacked_participating_v_cache_entry.cache_index();

    const auto& stacked_participating_v_stars_cache_entry =
        manager->DeclareCacheEntry(
            fmt::format("Participating free motion velocities for all bodies"),
            systems::ValueProducer(
                this, StackedVectors<T>(),
                &DeformableDriver<T>::CalcParticipatingFreeMotionVelocities),
            stacked_participating_v_stars_dependencies);
    cache_indexes_.stacked_participating_v_stars =
        stacked_participating_v_stars_cache_entry.cache_index();
  }
}

template <typename T>
void DeformableDriver<T>::CalcParticipatingVelocities(
    const systems::Context<T>& context,
    StackedVectors<T>* participating_v) const {
  DRAKE_DEMAND(participating_v != nullptr);
  std::vector<VectorX<T>> vs;
  for (DeformableBodyIndex index(0); index < deformable_model_->num_bodies();
       ++index) {
    vs.emplace_back(EvalParticipatingVelocities(context, index));
  }
  participating_v->Set(std::move(vs));
}

template <typename T>
const StackedVectors<T>& DeformableDriver<T>::EvalParticipatingVelocities(
    const systems::Context<T>& context) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.stacked_participating_v)
      .template Eval<StackedVectors<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcParticipatingFreeMotionVelocities(
    const systems::Context<T>& context,
    StackedVectors<T>* participating_v_star) const {
  DRAKE_DEMAND(participating_v_star != nullptr);
  std::vector<VectorX<T>> v_stars;
  for (DeformableBodyIndex index(0); index < deformable_model_->num_bodies();
       ++index) {
    v_stars.emplace_back(EvalParticipatingFreeMotionVelocities(context, index));
  }
  participating_v_star->Set(std::move(v_stars));
}

template <typename T>
const StackedVectors<T>&
DeformableDriver<T>::EvalParticipatingFreeMotionVelocities(
    const systems::Context<T>& context) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.stacked_participating_v_stars)
      .template Eval<StackedVectors<T>>(context);
}

template <typename T>
void DeformableDriver<T>::AppendDiscreteContactPairs(
    const systems::Context<T>& context,
    std::vector<DiscreteContactPair<T>>* result) const {
  DRAKE_DEMAND(result != nullptr);
  std::vector<DiscreteContactPair<T>>& contact_pairs = *result;

  const geometry::QueryObject<T>& query_object =
      manager_->plant()
          .get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
  const geometry::SceneGraphInspector<T>& inspector = query_object.inspector();

  // Simple utility to detect 0 / 0. As it is used in this method, denom
  // can only be zero if num is also zero, so we'll simply return zero.
  auto safe_divide = [](const T& num, const T& denom) {
    return denom == 0.0 ? T(0.0) : num / denom;
  };

  for (DeformableBodyIndex index(0); index < deformable_model_->num_bodies();
       ++index) {
    const DeformableRigidContact<T>& deformable_rigid_contact =
        EvalDeformableRigidContact(context, index);
    // Skip deformable bodies that are not in contact.
    if (deformable_rigid_contact.num_contact_points() == 0) continue;

    const geometry::GeometryId deformable_geometry_id =
        deformable_rigid_contact.deformable_id();
    for (int i = 0; i < deformable_rigid_contact.num_contact_points(); ++i) {
      const geometry::GeometryId rigid_geometry_id = deformable_rigid_contact.rigid_ids()[i];
      const T kA = GetPointContactStiffness(
          rigid_geometry_id, inspector, manager_->default_contact_stiffness());
      const T kB =
          GetPointContactStiffness(deformable_geometry_id, inspector,
                                   manager_->default_contact_stiffness());
      const T k = CombineStiffnesses(kA, kB);
      const T tauA = GetDissipationTimeConstant(
          rigid_geometry_id, inspector,
          manager_->default_dissipation_time_constant(), "rigid_body");
      const T tauB = GetDissipationTimeConstant(
          deformable_geometry_id, inspector,
          manager_->default_dissipation_time_constant(), "deformable_body");
      const T tau = CombineDissipationTimeConstant(tauA, tauB);

      // Combine friction coefficients.
      const double muA = GetCoulombFriction(rigid_geometry_id, inspector);
      const double muB = GetCoulombFriction(deformable_geometry_id, inspector);
      const T mu = T(safe_divide(2.0 * muA * muB, muA + muB));
      const Vector3<T>& p_WC =
          deformable_rigid_contact.contact_points_W()[i];
      // Rigid geometry is A. Deformable geometry is B. We want the normal to
      // point from B into A, but DeformableRigidContact provides the normal
      // from rigid into deformable, so we need to flip the sign.
      const Vector3<T>& nhat_BA_W =
          -deformable_rigid_contact.nhats_W()[i];
      const T phi0 = deformable_rigid_contact.signed_distances()[i];
      const T fn0 = NAN;  // not used.
      const T d = NAN;    // not used.
      contact_pairs.push_back({rigid_geometry_id, deformable_geometry_id, p_WC,
                               nhat_BA_W, phi0, fn0, k, d, tau, mu});
    }
  }
}

template <typename T>
void DeformableDriver<T>::AppendContactKinematics(
    const systems::Context<T>& context,
    std::vector<ContactPairKinematics<T>>* contact_kinematics) const {
  if constexpr (!std::is_same_v<T, double>) {
    unused(context, contact_kinematics);
    throw std::logic_error(
        "Deformable simulation only supports double as the scalar type.");
  } else {
    DRAKE_DEMAND(contact_kinematics != nullptr);
    // Scratch workspace variables.
    const int nv = manager_->plant().num_velocities();
    Matrix3X<T> Jv_WAc_W(3, nv);
    const Frame<T>& frame_W = manager_->plant().world_frame();

    // Let the rigid body be A and deformable body be B.
    // Since v_AcBc_W = v_WBc - v_WAc the relative velocity Jacobian will be:
    //   J_AcBc_W = Jv_WBc_W - Jv_WAc_W.
    // That is the relative velocity at C is v_AcBc_W = J_AcBc_W * v.
    // Finally J_AcBc_C = R_CW * J_AcBc_W.
    // Below we calculate the jacobian blocks that make up J_AcBc_C. Notice that
    // the set of dofs for deformable bodies and rigid bodies are mutually
    // exclusive, and Jv_WBc_W = 0 for rigid dofs and Jv_WAc_W = 0 for
    // deformable dofs.
    for (DeformableBodyIndex index(0); index < deformable_model_->num_bodies(); ++index) {
      const DeformableRigidContact<T>& deformable_rigid_contact =
          EvalDeformableRigidContact(context, index);
      // Skip deformable bodies that are not in contact.
      if (deformable_rigid_contact.num_contact_points() == 0) continue;

      const int deformable_clique_index = nv + index;
      // Scratch workspace and data needed for computing the jacobian w.r.t
      // deforamble dofs.
      Matrix3X<T> Jv_WBc_W = Matrix3X<T>::Zero(
          3, deformable_rigid_contact.num_vertices_in_contact() * 3);
      const PartialPermutation vertex_permutation =
          deformable_rigid_contact.CalcVertexPartialPermutation();

      for (int i = 0; i < deformable_rigid_contact.num_contact_points();
           ++i) {
        // We have at most two blocks per contact.
        std::vector<typename ContactPairKinematics<T>::JacobianCliqueBlock>
            jacobian_blocks;
        jacobian_blocks.reserve(2);
        // Data used for computing both the rigid and the deformable block.
        const Vector3<T>& p_WC =
            deformable_rigid_contact.contact_points_W()[i];
        const math::RotationMatrix<T>& R_CW =
            deformable_rigid_contact.R_CWs()[i];
        // Calculate the jacobian block for the deformable body.
        Jv_WBc_W.setZero();

        Vector4<int> participating_vertices =
            deformable_rigid_contact.contact_vertex_indexes()[i];
        const Vector4<T>& b =
            deformable_rigid_contact.barycentric_coordinates()[i];
        for (int v = 0; v < 4; ++v) {
          // Map indexes to the permuted domain.
          participating_vertices(v) =
              vertex_permutation.permuted_index(participating_vertices(v));
          // v_WBc = (b₀ * v₀ + b₁ * v₁ + b₂ * v₂ + b₃ * v₃) where v₀, v₁, v₂,
          // v₃ are the velocities of the vertices forming the tetrahedron
          // containing the contact point are the b's are their corresponding
          // barycentric weights.
          Jv_WBc_W.template middleCols<3>(3 * participating_vertices(v)) =
              b(v) * Matrix3<T>::Identity();
        }
        jacobian_blocks.emplace_back(deformable_clique_index,
                                     R_CW.matrix() * Jv_WBc_W);

        // Calculate the jacobian block for the rigid body if necessary.
        const GeometryId rigid_geometry_id = deformable_rigid_contact.rigid_ids()[i];
        const BodyIndex rigid_body_index =
            manager_->geometry_id_to_body_index().at(rigid_geometry_id);
        const TreeIndex tree_index =
            manager_->tree_topology().body_to_tree_index(rigid_body_index);
        if (tree_index.is_valid()) {
          const Body<T>& rigid_body =
              manager_->plant().get_body(rigid_body_index);
          manager_->internal_tree().CalcJacobianTranslationalVelocity(
              context, JacobianWrtVariable::kV, rigid_body.body_frame(),
              frame_W, p_WC, frame_W, frame_W, &Jv_WAc_W);
          Matrix3X<T> J =
              -R_CW.matrix() *
              Jv_WAc_W.middleCols(
                  manager_->tree_topology().tree_velocities_start(tree_index),
                  manager_->tree_topology().num_tree_velocities(tree_index));
          jacobian_blocks.emplace_back(tree_index, std::move(J));
        }
        contact_kinematics->emplace_back(
            deformable_rigid_contact.signed_distances()[i],
            std::move(jacobian_blocks), R_CW.transpose());
      }
    }
  }
}

template <typename T>
void DeformableDriver<T>::AppendLinearDynamicsMatrix(
    const systems::Context<T>& context, std::vector<MatrixX<T>>* A) const {
  DRAKE_DEMAND(A != nullptr);
  const int num_bodies = deformable_model_->num_bodies();
  A->reserve(A->size() + num_bodies);
  for (DeformableBodyIndex index(0); index < num_bodies; ++index) {
    const SchurComplement<T>& schur_complement =
        EvalFreeMotionTangentMatrixSchurComplement(context, index);
    A->emplace_back(schur_complement.get_D_complement());
    fmt::print("A = {} \n", schur_complement.get_D_complement());
  }
}

template <typename T>
void DeformableDriver<T>::UpdateDiscreteStates(
    const systems::Context<T>& context,
    systems::DiscreteValues<T>* new_states) const {
  if constexpr (!std::is_same_v<T, double>) {
    unused(context, new_states);
    throw std::logic_error(
        "Deformable simulation only supports double as the scalar type.");
  } else {
    const int num_bodies = deformable_model_->num_bodies();
    for (DeformableBodyIndex index(0); index < num_bodies; ++index) {
      const FemState<T>& next_fem_state = EvalNextFemState(context, index);
      const int num_dofs = next_fem_state.num_dofs();
      // Update the discrete values.
      VectorX<T> discrete_value(num_dofs * 3);
      discrete_value.head(num_dofs) = next_fem_state.GetPositions();
      discrete_value.segment(num_dofs, num_dofs) =
          next_fem_state.GetVelocities();
      discrete_value.tail(num_dofs) = next_fem_state.GetAccelerations();
      const DeformableBodyId id = deformable_model_->GetBodyId(index);
      new_states->set_value(deformable_model_->GetDiscreteStateIndex(id),
                            discrete_value);
    }
  }
}

template <typename T>
void DeformableDriver<T>::CalcDeformableRigidContact(
    const systems::Context<T>& context,
    std::vector<DeformableRigidContact<T>>* deformable_rigid_contact) const {
  if constexpr (!std::is_same_v<T, double>) {
    unused(context, deformable_rigid_contact);
    throw std::logic_error(
        "Deformable simulation only supports double as the scalar type.");
  } else {
    deformable_rigid_contact->clear();
    const geometry::QueryObject<T>& query_object =
        manager_->plant()
            .get_geometry_query_input_port()
            .template Eval<geometry::QueryObject<T>>(context);
    query_object.ComputeDeformableRigidContact(deformable_rigid_contact);
  }
}

template <typename T>
const DeformableRigidContact<T>&
DeformableDriver<T>::EvalDeformableRigidContact(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  const std::vector<DeformableRigidContact<T>>& all_contact_data =
      manager_->plant()
          .get_cache_entry(cache_indexes_.deformable_rigid_contacts)
          .template Eval<std::vector<DeformableRigidContact<T>>>(context);
  return all_contact_data.at(index);
}

template <typename T>
void DeformableDriver<T>::CalcVelocityPermutation(
    const systems::Context<T>& context, DeformableBodyIndex index,
    PartialPermutation* permutation) const {
  if constexpr (!std::is_same_v<T, double>) {
    unused(context, index, permutation);
    throw std::logic_error(
        "Deformable simulation only supports double as the scalar type.");
  } else {
    const DeformableRigidContact<T>& contact_data =
        EvalDeformableRigidContact(context, index);
    *permutation = contact_data.CalcDofPartialPermutation();
  }
}

template <typename T>
const PartialPermutation& DeformableDriver<T>::EvalVelocityPermutation(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.velocity_permutations.at(index))
      .template Eval<PartialPermutation>(context);
}

template <typename T>
void DeformableDriver<T>::CalcParticipatingVelocities(
    const systems::Context<T>& context, DeformableBodyIndex index,
    VectorX<T>* participating_v) const {
  DRAKE_DEMAND(participating_v != nullptr);
  const fem::FemState<T>& fem_state = EvalFemState(context, index);
  const VectorX<T>& v = fem_state.GetVelocities();
  const PartialPermutation& permutation =
      EvalVelocityPermutation(context, index);
  participating_v->resize(permutation.permuted_domain_size());
  permutation.Apply(v, participating_v);
}

template <typename T>
const VectorX<T>& DeformableDriver<T>::EvalParticipatingVelocities(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.participating_v.at(index))
      .template Eval<VectorX<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcParticipatingFreeMotionVelocities(
    const systems::Context<T>& context, DeformableBodyIndex index,
    VectorX<T>* participating_v_star) const {
  DRAKE_DEMAND(participating_v_star != nullptr);
  const fem::FemState<T>& free_motion_state =
      EvalFreeMotionFemState(context, index);
  const VectorX<T>& v_star = free_motion_state.GetVelocities();
  const PartialPermutation& permutation =
      EvalVelocityPermutation(context, index);
  participating_v_star->resize(permutation.permuted_domain_size());
  permutation.Apply(v_star, participating_v_star);
}

template <typename T>
const VectorX<T>& DeformableDriver<T>::EvalParticipatingFreeMotionVelocities(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.participating_v_stars.at(index))
      .template Eval<VectorX<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcFemState(const systems::Context<T>& context,
                                       DeformableBodyIndex index,
                                       FemState<T>* fem_state) const {
  const DeformableBodyId id = deformable_model_->GetBodyId(index);
  const systems::BasicVector<T>& discrete_state =
      context.get_discrete_state().get_vector(
          deformable_model_->GetDiscreteStateIndex(id));
  const auto& discrete_value = discrete_state.get_value();
  DRAKE_DEMAND(discrete_value.size() % 3 == 0);
  const int num_dofs = discrete_value.size() / 3;
  const auto& q = discrete_value.head(num_dofs);
  const auto& qdot = discrete_value.segment(num_dofs, num_dofs);
  const auto& qddot = discrete_value.tail(num_dofs);
  fem_state->SetPositions(q);
  fem_state->SetVelocities(qdot);
  fem_state->SetAccelerations(qddot);
}

template <typename T>
const FemState<T>& DeformableDriver<T>::EvalFemState(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.fem_states.at(index))
      .template Eval<FemState<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcFreeMotionFemState(
    const systems::Context<T>& context, DeformableBodyIndex index,
    FemState<T>* fem_state_star) const {
  if constexpr (std::is_same_v<T, double>) {
    const FemState<T>& fem_state = EvalFemState(context, index);
    const DeformableBodyId id = deformable_model_->GetBodyId(index);
    const FemModel<T>& model = deformable_model_->GetFemModel(id);
    const FemSolver<T> solver(&model, integrator_.get());
    FemSolverScratchData<T>& scratch =
        manager_->plant()
            .get_cache_entry(cache_indexes_.fem_solver_scratches.at(index))
            .get_mutable_cache_entry_value(context)
            .template GetMutableValueOrThrow<FemSolverScratchData<T>>();
    solver.AdvanceOneTimeStep(fem_state, fem_state_star, &scratch);
  } else {
    unused(context, index, fem_state_star);
    throw std::logic_error(
        "DeformableDriver only supports simulation with deformable "
        "bodies with T == double.");
  }
}

template <typename T>
const FemState<T>& DeformableDriver<T>::EvalFreeMotionFemState(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.free_motion_fem_states.at(index))
      .template Eval<FemState<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcNextFemState(const systems::Context<T>& context,
                                           DeformableBodyIndex index,
                                           FemState<T>* next_fem_state) const {
  if constexpr (std::is_same_v<T, double>) {
    const DeformableRigidContact<T>& contact_data =
        EvalDeformableRigidContact(context, index);
    if (contact_data.num_contact_points() == 0) {
      const FemState<T>& free_motion_state =
          EvalFreeMotionFemState(context, index);
      next_fem_state->SetPositions(free_motion_state.GetPositions());
      next_fem_state->SetVelocities(free_motion_state.GetVelocities());
      next_fem_state->SetAccelerations(free_motion_state.GetAccelerations());
    } else {
      const ContactSolverResults<T>& results =
          manager_->EvalContactSolverResults(context);
      const int num_participating_dofs =
          results.v_next.size() - manager_->plant().num_velocities();
      const auto stacked_v_next = results.v_next.tail(num_participating_dofs);
      /* Find how to extract next time step velocity corresponding to the given
       `index` from the stacked vectors. */
      const StackedVectors<T>& stacked_vectors =
          EvalParticipatingVelocities(context);
      const int offset = stacked_vectors.offset(index);
      const int size = stacked_vectors.size(index);

      const auto participating_v = stacked_v_next.segment(offset, size);
      // Compute the value of the post-constraint non-participating
      // velocities using Schur complement.
      const SchurComplement<T>& schur_complement =
          EvalFreeMotionTangentMatrixSchurComplement(context, index);
      const VectorX<T>& nonparticipating_v =
          schur_complement.SolveForY(participating_v);
      // Concatenate the participating and non-participating velocities and
      // then apply the inverse permutation to put the dofs in their
      // original order.
      const DeformableRigidContact<T>& deformable_rigid_contact =
          EvalDeformableRigidContact(context, index);
      const PartialPermutation& velocity_permutation =
          deformable_rigid_contact.CalcDofFullPermutation();
      const int num_dofs = velocity_permutation.domain_size();
      VectorX<T> permuted_velocities(num_dofs);
      permuted_velocities << participating_v, nonparticipating_v;
      VectorX<T> new_velocities(num_dofs);
      velocity_permutation.ApplyInverse(permuted_velocities, &new_velocities);
      const FemState<T>& fem_state = EvalFemState(context, index);
      integrator_->AdvanceOneTimeStep(fem_state, new_velocities,
                                      next_fem_state);
    }
  } else {
    unused(context, index, next_fem_state);
    throw std::logic_error(
        "DeformableDriver only supports simulation with deformable "
        "bodies with T == double.");
  }
}

template <typename T>
const FemState<T>& DeformableDriver<T>::EvalNextFemState(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.next_fem_states.at(index))
      .template Eval<FemState<T>>(context);
}

template <typename T>
void DeformableDriver<T>::CalcFreeMotionTangentMatrix(
    const systems::Context<T>& context, DeformableBodyIndex index,
    PetscSymmetricBlockSparseMatrix* tangent_matrix) const {
  if constexpr (!std::is_same_v<T, double>) {
    unused(context, index, tangent_matrix);
    throw std::logic_error(
        "DeformableDriver only supports updates for deformable bodies "
        "when T == double.");
  } else {
      const DeformableBodyId id = deformable_model_->GetBodyId(index);
    const FemModel<T>& fem_model = deformable_model_->GetFemModel(id);
    const FemState<T>& state = EvalFreeMotionFemState(context, index);
    fem_model.CalcTangentMatrix(state, integrator_->GetWeights(),
                                tangent_matrix);
  }
}

template <typename T>
const PetscSymmetricBlockSparseMatrix&
DeformableDriver<T>::EvalFreeMotionTangentMatrix(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(cache_indexes_.free_motion_tangent_matrices.at(index))
      .template Eval<PetscSymmetricBlockSparseMatrix>(context);
}

template <typename T>
void DeformableDriver<T>::CalcFreeMotionTangentMatrixSchurComplement(
    const systems::Context<T>& context, DeformableBodyIndex index,
    SchurComplement<T>* schur_complement) const {
  if constexpr (!std::is_same_v<T, double>) {
    unused(context, index, schur_complement);
    throw std::logic_error(
        "DeformableDriver only supports updates for deformable bodies "
        "when T == double.");
  } else {
    const DeformableRigidContact<T>& contact_data =
        EvalDeformableRigidContact(context, index);
    if (contact_data.num_contact_points() == 0) {
      // Avoid the expensive tangent matrix and Schur complement calculation if
      // there's no contact at all.
      *schur_complement = SchurComplement<T>();
      return;
    }
    const PetscSymmetricBlockSparseMatrix& tangent_matrix =
        EvalFreeMotionTangentMatrix(context, index);
    std::vector<int> participating_vertices;
    std::vector<int> non_participating_vertices;
    const PartialPermutation permutation =
        contact_data.CalcVertexPartialPermutation();
    DRAKE_DEMAND(3 * permutation.domain_size() == tangent_matrix.cols());
    for (int i = 0; i < permutation.domain_size(); ++i) {
      if (permutation.participates(i)) {
        participating_vertices.emplace_back(i);
      } else {
        non_participating_vertices.emplace_back(i);
      }
    }
    *schur_complement = tangent_matrix.CalcSchurComplement(
        non_participating_vertices, participating_vertices);
  }
}

template <typename T>
const SchurComplement<T>&
DeformableDriver<T>::EvalFreeMotionTangentMatrixSchurComplement(
    const systems::Context<T>& context, DeformableBodyIndex index) const {
  return manager_->plant()
      .get_cache_entry(
          cache_indexes_.free_motion_tangent_matrix_schur_complements.at(index))
      .template Eval<SchurComplement<T>>(context);
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::internal::DeformableDriver);
