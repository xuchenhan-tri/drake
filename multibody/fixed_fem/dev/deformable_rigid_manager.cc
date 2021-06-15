#include "drake/multibody/fixed_fem/dev/deformable_rigid_manager.h"

#include "drake/multibody/contact_solvers/block_sparse_linear_operator.h"
#include "drake/multibody/contact_solvers/block_sparse_matrix.h"
#include "drake/multibody/fixed_fem/dev/inverse_operator.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace multibody {
namespace fixed_fem {

namespace {
template <typename T>
VectorX<T> PermuteWithVertexOrdering(const VectorX<T>& v,
                                     const std::vector<int>& vertex_mapping) {
  DRAKE_DEMAND(static_cast<int>(vertex_mapping.size() * 3) == v.size());
  VectorX<T> permuted_v(v.size());
  for (int i = 0; i < static_cast<int>(vertex_mapping.size()); ++i) {
    permuted_v.template segment<3>(3 * vertex_mapping[i]) =
        v.template segment<3>(3 * i);
  }
  return permuted_v;
}
}  // namespace

using multibody::contact_solvers::internal::BlockSparseLinearOperator;
using multibody::contact_solvers::internal::BlockSparseMatrix;
using multibody::contact_solvers::internal::BlockSparseMatrixBuilder;
using multibody::contact_solvers::internal::ContactSolverResults;
using multibody::contact_solvers::internal::InverseOperator;
using multibody::contact_solvers::internal::PointContactData;
using multibody::contact_solvers::internal::SystemDynamicsData;

template <typename T>
void DeformableRigidManager<T>::RegisterCollisionObjects(
    const geometry::SceneGraph<T>& scene_graph) const {
  const geometry::SceneGraphInspector<T>& inspector =
      scene_graph.model_inspector();
  /* Make sure that the owning plant is registered at the given scene graph.
   */
  DRAKE_THROW_UNLESS(
      inspector.SourceIsRegistered(this->plant().get_source_id().value()));
  for (const auto& per_body_collision_geometries :
       this->collision_geometries()) {
    for (geometry::GeometryId id : per_body_collision_geometries) {
      /* Sanity check that the geometry comes from the owning MultibodyPlant
       indeed. */
      DRAKE_DEMAND(
          inspector.BelongsToSource(id, this->plant().get_source_id().value()));
      const geometry::Shape& shape = inspector.GetShape(id);
      const geometry::ProximityProperties* props =
          dynamic_cast<const geometry::ProximityProperties*>(
              inspector.GetProperties(id, geometry::Role::kProximity));
      /* Collision geometry must have proximity properties attached to it. */
      DRAKE_DEMAND(props != nullptr);
      collision_objects_.AddCollisionObject(id, shape, *props);
    }
  }
}

template <typename T>
void DeformableRigidManager<T>::ExtractModelInfo() {
  bool extracted_deformable_model = false;
  const std::vector<std::unique_ptr<multibody::internal::PhysicalModel<T>>>&
      physical_models = this->plant().physical_models();
  for (const auto& model : physical_models) {
    const auto* deformable_model =
        dynamic_cast<const DeformableModel<T>*>(model.get());
    if (deformable_model != nullptr) {
      if (extracted_deformable_model) {
        throw std::logic_error(
            "More than one DeformableModel are specified in the "
            "MultibodyPlant.");
      }
      deformable_model_ = deformable_model;
      MakeFemSolvers();
      RegisterDeformableGeometries();
      extracted_deformable_model = true;
    }
  }
  if (!extracted_deformable_model) {
    throw std::logic_error(
        "The owning MultibodyPlant does not have any deformable model.");
  }
  // TODO(xuchenhan-tri): Remove this when tangent_matrix_schur_complement
  //  becomes a cache entry.
  tangent_matrix_schur_complements_.resize(deformable_model_->num_bodies());
}

template <typename T>
void DeformableRigidManager<T>::MakeFemSolvers() {
  DRAKE_DEMAND(deformable_model_ != nullptr);
  for (int i = 0; i < deformable_model_->num_bodies(); ++i) {
    const FemModelBase<T>& fem_model =
        deformable_model_->fem_model(SoftBodyIndex(i));
    fem_solvers_.emplace_back(std::make_unique<FemSolver<T>>(&fem_model));
  }
}

template <typename T>
void DeformableRigidManager<T>::RegisterDeformableGeometries() {
  DRAKE_DEMAND(deformable_model_ != nullptr);
  deformable_meshes_ = deformable_model_->reference_configuration_meshes();
}

template <typename T>
void DeformableRigidManager<T>::DeclareCacheEntries(MultibodyPlant<T>* plant) {
  DRAKE_DEMAND(deformable_model_ != nullptr);
  for (SoftBodyIndex deformable_body_id(0);
       deformable_body_id < deformable_model_->num_bodies();
       ++deformable_body_id) {
    const FemModelBase<T>& fem_model =
        deformable_model_->fem_model(deformable_body_id);
    auto allocate_fem_state_base = [&]() {
      return AbstractValue::Make(*fem_model.MakeFemStateBase());
    };
    /* Lambda function to extract the q, qdot, and qddot from context and copy
     them to the cached fem state. */
    auto copy_to_fem_state = [this, deformable_body_id](
                                 const systems::ContextBase& context_base,
                                 AbstractValue* cache_value) {
      const auto& context =
          dynamic_cast<const systems::Context<T>&>(context_base);
      const systems::DiscreteValues<T>& all_discrete_states =
          context.get_discrete_state();
      /* Extract q, qdot and qddot from context. */
      const systems::BasicVector<T>& discrete_state =
          all_discrete_states.get_vector(
              deformable_model_->discrete_state_indexes()[deformable_body_id]);
      const auto& discrete_value = discrete_state.get_value();
      DRAKE_DEMAND(discrete_value.size() % 3 == 0);
      const int num_dofs = discrete_value.size() / 3;
      const auto& q = discrete_value.head(num_dofs);
      const auto& qdot = discrete_value.segment(num_dofs, num_dofs);
      const auto& qddot = discrete_value.tail(num_dofs);
      auto& fem_state =
          cache_value->template get_mutable_value<FemStateBase<T>>();
      fem_state.SetQ(q);
      fem_state.SetQdot(qdot);
      fem_state.SetQddot(qddot);
    };
    const auto& fem_state_cache_entry = plant->DeclareCacheEntry(
        "FEM state", allocate_fem_state_base, std::move(copy_to_fem_state),
        {systems::System<T>::xd_ticket()});
    fem_state_cache_indexes_.emplace_back(fem_state_cache_entry.cache_index());

    /* Lambda function to calculate the free-motion velocity for the deformable
     body. */
    auto calc_fem_state_star = [this, deformable_body_id](
                                   const systems::ContextBase& context_base,
                                   AbstractValue* cache_value) {
      const auto& context =
          dynamic_cast<const systems::Context<T>&>(context_base);
      const FemStateBase<T>& fem_state =
          EvalFemStateBase(context, deformable_body_id);
      auto& fem_state_star =
          cache_value->template get_mutable_value<FemStateBase<T>>();
      // TODO(xuchenhan-tri): FemState needs a SetFrom() method.
      fem_state_star.SetQ(fem_state.q());
      fem_state_star.SetQdot(fem_state.qdot());
      fem_state_star.SetQddot(fem_state.qddot());
      /* Obtain the contact-free state for the deformable body. */
      fem_solvers_[deformable_body_id]->AdvanceOneTimeStep(fem_state,
                                                           &fem_state_star);
    };
    /* Declares the free-motion cache entry which only depends on the fem state.
     */
    const auto& free_motion_fem_state_cache_entry = plant->DeclareCacheEntry(
        "Free motion FEM state", std::move(allocate_fem_state_base),
        std::move(calc_fem_state_star), {fem_state_cache_entry.ticket()});
    free_motion_cache_indexes_.emplace_back(
        free_motion_fem_state_cache_entry.cache_index());

    /* Lambda to allocate for the tangent matrix of a deformable body. */
    auto allocate_tangent_matrix = [&fem_model]() {
      Eigen::SparseMatrix<T> tangent_matrix(fem_model.num_dofs(),
                                            fem_model.num_dofs());
      fem_model.SetTangentMatrixSparsityPattern(&tangent_matrix);
      return AbstractValue::Make(EigenSparseMatrixWrapper<T>{tangent_matrix});
    };
    /* Lambda to calculate the tangent matrix of a deformable body at the free
     motion state. */
    auto calc_tangent_matrix = [&fem_model, deformable_body_id, this](
                                   const systems::ContextBase& context_base,
                                   AbstractValue* cache_value) {
      const auto& context =
          dynamic_cast<const systems::Context<T>&>(context_base);
      auto& tangent_matrix =
          cache_value->template get_mutable_value<EigenSparseMatrixWrapper<T>>()
              .sparse_matrix;
      const FemStateBase<T>& fem_state =
          EvalFreeMotionFemStateBase(context, deformable_body_id);
      fem_model.CalcTangentMatrix(fem_state, &tangent_matrix);
    };
    /* Declares the free-motion tangent matrix cache entry. */
    const auto& tangent_matrix_cache_entry = plant->DeclareCacheEntry(
        "Deformable tangent matrix at free motion state",
        std::move(allocate_tangent_matrix), std::move(calc_tangent_matrix),
        {free_motion_fem_state_cache_entry.ticket()});
    tangent_matrix_cache_indexes_.emplace_back(
        tangent_matrix_cache_entry.cache_index());
  }

  /* Lambda to allocate for a std::vector<internal::DeformableContactData<T>>.
   */
  auto allocate_contact_data = []() {
    return AbstractValue::Make(
        std::vector<internal::DeformableContactData<T>>());
  };
  /* Lambda to calculate the contact data for all deformable bodies. */
  auto calc_contact_data = [this](const systems::ContextBase& context_base,
                                  AbstractValue* cache_value) {
    const auto& context =
        dynamic_cast<const systems::Context<T>&>(context_base);
    auto& deformable_contact_data = cache_value->template get_mutable_value<
        std::vector<internal::DeformableContactData<T>>>();
    this->UpdateCollisionObjectPoses(context);
    this->UpdateDeformableVertexPositions(context);
    const int num_bodies = this->deformable_model_->num_bodies();
    deformable_contact_data.clear();
    deformable_contact_data.reserve(num_bodies);
    for (SoftBodyIndex deformable_body_id(0); deformable_body_id < num_bodies;
         ++deformable_body_id) {
      deformable_contact_data.emplace_back(
          this->CalcDeformableContactData(deformable_body_id));
    }
  };
  /* Declares the deformable contact data cache entry. */
  const auto& deformable_contact_data_cache_entry = plant->DeclareCacheEntry(
      "Deformable contact data", std::move(allocate_contact_data),
      std::move(calc_contact_data), {systems::System<T>::xd_ticket()});
  deformable_contact_data_cache_index_ =
      deformable_contact_data_cache_entry.cache_index();
}

#if 0
template <typename T>
void DeformableRigidManager<T>::DoCalcContactSolverResults(
    const systems::Context<T>& context,
    contact_solvers::internal::ContactSolverResults<T>* results) const {
  /* Assert this method was called on a context storing discrete state. */
  this->plant().ValidateContext(context);
  DRAKE_ASSERT(context.num_continuous_states() == 0);
  const int nv = this->plant().num_velocities();

  // TODO(xuchenhan-tri): This is not true when there are deformable dofs.
  //  Modify it when deformable-rigid contact is supported.
  /* Quick exit if there are no moving objects. */
  if (nv == 0) return;

  // TODO(xuchenhan-tri): Incorporate deformable-rigid contact pairs.
  /* Compute all rigid-rigid contact pairs, including both penetration pairs
   and quadrature pairs for discrete hydroelastic. */
  const std::vector<multibody::internal::DiscreteContactPair<T>>
      rigid_contact_pairs = this->CalcDiscreteContactPairs(context);
  const int num_rigid_contacts = rigid_contact_pairs.size();

  /* Extract all information needed by the contact/TAMSI solver. */
  multibody::internal::ContactJacobians<T> rigid_contact_jacobians;
  VectorX<T> v(nv);
  MatrixX<T> M(nv, nv);
  VectorX<T> minus_tau(nv);
  VectorX<T> mu(num_rigid_contacts);
  VectorX<T> phi(num_rigid_contacts);
  VectorX<T> fn(num_rigid_contacts);
  VectorX<T> stiffness(num_rigid_contacts);
  VectorX<T> damping(num_rigid_contacts);
  CalcContactQuantities(context, rigid_contact_pairs, &rigid_contact_jacobians,
                        &v, &M, &minus_tau, &mu, &phi, &fn, &stiffness,
                        &damping);

  /* Call the contact solver if one exists. Otherwise, invoke the TAMSI
   solver. */
  if (contact_solver_ != nullptr) {
    this->CallContactSolver(contact_solver_.get(), context.get_time(), v, M,
                            minus_tau, phi, rigid_contact_jacobians.Jc,
                            stiffness, damping, mu, results);
  } else {
    this->CallTamsiSolver(
        context.get_time(), v, M, minus_tau, fn, rigid_contact_jacobians.Jn,
        rigid_contact_jacobians.Jt, stiffness, damping, mu, results);
  }
}
#endif

template <typename T>
void DeformableRigidManager<T>::DoCalcContactSolverResults(
    const systems::Context<T>& context,
    contact_solvers::internal::ContactSolverResults<T>* results) const {
  /* Assert this method was called on a context storing discrete state. */
  this->plant().ValidateContext(context);
  DRAKE_ASSERT(context.num_continuous_states() == 0);
  DRAKE_DEMAND(deformable_model_ != nullptr);

  const int rigid_dofs = this->plant().num_velocities();
  /* Quick exit if there are no moving rigid or deformable objects. */
  if (rigid_dofs == 0 && deformable_model_->num_bodies() == 0) return;

  /* Compute all rigid-rigid and rigid-deformable contact pairs. */
  const std::vector<multibody::internal::DiscreteContactPair<T>>
      rigid_contact_pairs = this->CalcDiscreteContactPairs(context);
  const std::vector<internal::DeformableContactData<T>>&
      deformable_contact_data = EvalDeformableRigidContact(context);

  /* Extract all information needed by the contact solver. */
  /* Point contact data. */
  BlockSparseMatrix<T> Jc = CalcContactJacobian(context, rigid_contact_pairs,
                                                deformable_contact_data);
  DRAKE_DEMAND(Jc.rows() % 3 == 0);
  const int nc = Jc.rows() / 3;

  VectorX<T> mu(nc);
  VectorX<T> phi0(nc);
  VectorX<T> stiffness(nc);
  VectorX<T> damping(nc);
  CalcPerContactPointData(context, rigid_contact_pairs, deformable_contact_data,
                          &mu, &phi0, &stiffness, &damping);

  /* System dynamics data. */
  BlockSparseMatrix<T> A = CalcTangentMatrix(context, deformable_contact_data);
  const int nv = A.cols();
  DRAKE_DEMAND(A.rows() == nv);
  if (Jc.rows() != 0) {
    DRAKE_DEMAND(Jc.cols() == nv);
  }
  VectorX<T> v_star(nv);
  CalcFreeMotionVelocities(context, deformable_contact_data, &v_star);

  // TODO(xuchenhan-tri): The inverse is currently calculated inefficiently.
  //  However, since the inverse tangent matrix won't be needed for the new
  //  contact solver. We do not try to optimize it right now.
  InverseOperator<T> A_inv_op("A_inv", A.MakeDenseMatrix());
  BlockSparseLinearOperator<T> Jop("Jc", &Jc);
  SystemDynamicsData<T> dynamics_data(&A_inv_op, &v_star);
  PointContactData<T> contact_data(&phi0, &Jop, &stiffness, &damping, &mu);

  ContactSolverResults<T> two_way_coupled_results;
  two_way_coupled_results.Resize(nv, nc);
  contact_solver_->SolveWithGuess(this->plant().time_step(), dynamics_data,
                                  contact_data, v_star,
                                  &two_way_coupled_results);

  /* Extract the results related to the rigid dofs from the full two-way coupled
   deformable-rigid results. */
  const int num_rigid_contacts = rigid_contact_pairs.size();
  results->Resize(rigid_dofs, num_rigid_contacts);
  results->v_next = two_way_coupled_results.v_next.head(rigid_dofs);
  results->fn = two_way_coupled_results.fn.head(num_rigid_contacts);
  results->ft = two_way_coupled_results.ft.head(2 * num_rigid_contacts);
  results->vn = two_way_coupled_results.vn.head(num_rigid_contacts);
  results->vt = two_way_coupled_results.vt.head(2 * num_rigid_contacts);
  results->tau_contact = two_way_coupled_results.tau_contact.head(rigid_dofs);

  // TODO(xuchenhan-tri): Consider putting the deformable velocities in the
  //  return value instead of storing them as state in the manager.
  /* Extract the deformable velocities. */
  int dof_offset = rigid_dofs;
  deformable_participating_velocities_.resize(deformable_model_->num_bodies());
  for (int i = 0; i < deformable_model_->num_bodies(); ++i) {
    deformable_participating_velocities_[i] =
        two_way_coupled_results.v_next.segment(
            dof_offset,
            deformable_contact_data[i].num_vertices_in_contact() * 3);
    dof_offset += deformable_contact_data[i].num_vertices_in_contact() * 3;
  }
  /* Sanity check that all velocities are extracted. */
  DRAKE_DEMAND(dof_offset == nv);
}

template <typename T>
void DeformableRigidManager<T>::CalcContactQuantities(
    const systems::Context<T>& context,
    const std::vector<multibody::internal::DiscreteContactPair<T>>&
        rigid_contact_pairs,
    multibody::internal::ContactJacobians<T>* rigid_contact_jacobians,
    EigenPtr<VectorX<T>> v, EigenPtr<MatrixX<T>> M,
    EigenPtr<VectorX<T>> minus_tau, EigenPtr<VectorX<T>> mu,
    EigenPtr<VectorX<T>> phi, EigenPtr<VectorX<T>> fn,
    EigenPtr<VectorX<T>> stiffness, EigenPtr<VectorX<T>> damping) const {
  DRAKE_DEMAND(v != nullptr);
  DRAKE_DEMAND(M != nullptr);
  DRAKE_DEMAND(minus_tau != nullptr);
  DRAKE_DEMAND(mu != nullptr);
  DRAKE_DEMAND(phi != nullptr);
  DRAKE_DEMAND(fn != nullptr);
  DRAKE_DEMAND(stiffness != nullptr);
  DRAKE_DEMAND(damping != nullptr);
  /* Compute the contact Jacobians for the rigid dofs. */
  *rigid_contact_jacobians = this->EvalContactJacobians(context);

  /* Compute the generalized velocities. */
  auto x =
      context.get_discrete_state(this->multibody_state_index()).get_value();
  const int nv = this->plant().num_velocities();
  DRAKE_DEMAND(v->size() == nv);
  DRAKE_DEMAND(M->rows() == nv);
  DRAKE_DEMAND(M->cols() == nv);
  DRAKE_DEMAND(minus_tau->size() == nv);
  *v = x.bottomRows(nv);

  /* Compute the mass matrix. */
  this->plant().CalcMassMatrix(context, M);

  /* Computes the negative generalized non-contact forces on the rigid dofs,
   `minus_tau`. */
  MultibodyForces<T> forces(this->internal_tree());
  this->CalcNonContactForces(context, true /* discrete */, &forces);
  /* Workspace for inverse dynamics: Bodies' accelerations, ordered by
   BodyNodeIndex. */
  std::vector<SpatialAcceleration<T>> A_WB_array(this->plant().num_bodies());
  /* Generalized accelerations. */
  const VectorX<T> vdot = VectorX<T>::Zero(nv);
  /* Body forces (alias to forces). */
  std::vector<SpatialForce<T>>& F_BBo_W_array = forces.mutable_body_forces();
  /* With vdot = 0, this computes:
     -tau = C(q, v)v - tau_app - ∑ J_WBᵀ(q) Fapp_Bo_W. */
  *minus_tau = forces.mutable_generalized_forces();
  this->internal_tree().CalcInverseDynamics(
      context, vdot, F_BBo_W_array, *minus_tau, &A_WB_array,
      &F_BBo_W_array, /* Note: these arrays get overwritten on output. */
      minus_tau);

  /* Computes friction coefficient. Static friction is ignored by the time
   stepping scheme. */
  const int num_contacts = rigid_contact_pairs.size();
  DRAKE_DEMAND(mu->size() == num_contacts);
  std::vector<CoulombFriction<double>> combined_friction_pairs =
      this->CalcCombinedFrictionCoefficients(context, rigid_contact_pairs);
  std::transform(combined_friction_pairs.begin(), combined_friction_pairs.end(),
                 mu->data(),
                 [](const CoulombFriction<double>& coulomb_friction) {
                   return coulomb_friction.dynamic_friction();
                 });

  /* Compute penetration, normal contact force, stiffness, and damping. */
  DRAKE_DEMAND(phi->size() == num_contacts);
  DRAKE_DEMAND(fn->size() == num_contacts);
  DRAKE_DEMAND(stiffness->size() == num_contacts);
  DRAKE_DEMAND(damping->size() == num_contacts);
  for (int i = 0; i < num_contacts; ++i) {
    (*phi)[i] = rigid_contact_pairs[i].phi0;
    (*fn)[i] = rigid_contact_pairs[i].fn0;
    (*stiffness)[i] = rigid_contact_pairs[i].stiffness;
    (*damping)[i] = rigid_contact_pairs[i].damping;
  }
}

template <typename T>
BlockSparseMatrix<T> DeformableRigidManager<T>::CalcContactJacobian(
    const systems::Context<T>& context,
    const std::vector<multibody::internal::DiscreteContactPair<T>>&
        rigid_contact_pairs,
    const std::vector<internal::DeformableContactData<T>>
        deformable_contact_data) const {
  /* Return an empty matrix if there's no rigid dofs. */
  if (this->plant().num_velocities() == 0) {
    return BlockSparseMatrix<T>();
  }

  /* Each deformable body in contact forms a deformable dof group. */
  const int num_deformable_dof_groups = std::count_if(
      deformable_contact_data.begin(), deformable_contact_data.end(),
      [](const internal::DeformableContactData<T>& contact_data) {
        return contact_data.num_contact_points() != 0;
      });

  /* Return an empty matrix if no contact exists. */
  if (num_deformable_dof_groups == 0 && rigid_contact_pairs.empty()) {
    return BlockSparseMatrix<T>();
  }

  // TODO(xuchenhan-tri): Further exploit the sparsity in contact jacobian
  // using contact graph for rigid dofs.
  /* For now, all rigid-rigid contacts are viewed as a single contact group
   for the purpose of calculating contact jacobian. */
  const int num_rigid_dof_groups = 1;
  const int num_rigid_contact_groups = rigid_contact_pairs.empty() ? 0 : 1;
  /* Each deformable dof group contact forms one contact group. */
  const int num_deformable_contact_groups = num_deformable_dof_groups;

  const int num_contact_groups =
      num_deformable_contact_groups + num_rigid_contact_groups;
  const int num_bodies = num_rigid_dof_groups + num_deformable_dof_groups;

  /* Each deformable contact group gives two blocks in the jacobian (one
   deformable block and one rigid block). The rigid vs. rigid contact group
   gives only one block. The number of blocks therefore is bounded above by
   twice the number of contact groups. */
  BlockSparseMatrixBuilder<T> builder(num_contact_groups, num_bodies,
                                      2 * num_contact_groups);

  if (num_rigid_contact_groups != 0) {
    /* The rigid-rigid block. */
    builder.PushBlock(0, 0, this->EvalContactJacobians(context).Jc);
  }

  /* The block rows corresponding to deformable-rigid contacts start after
   rigid-rigid contacts. */
  int row_block = num_rigid_contact_groups;
  /* The block columns corresponding to deformable dofs start after the
   rigid dofs. */
  int column_block_deformable = num_rigid_dof_groups;
  const int column_block_rigid = 0;
  for (int i = 0; i < static_cast<int>(deformable_contact_data.size()); ++i) {
    const internal::DeformableContactData<T>& contact_data =
        deformable_contact_data[i];
    /* Skip deformable bodies that are not in contact. */
    if (contact_data.num_contact_points() == 0) {
      continue;
    }
    builder.PushBlock(row_block, column_block_rigid,
                      CalcContactJacobianRigidBlock(context, contact_data));
    builder.PushBlock(row_block, column_block_deformable,
                      CalcContactJacobianDeformableBlock(contact_data));
    ++row_block;
    ++column_block_deformable;
  }
  return builder.Build();
}

template <typename T>
MatrixX<T> DeformableRigidManager<T>::CalcContactJacobianDeformableBlock(
    const internal::DeformableContactData<T>& contact_data) const {
  MatrixX<T> Jc = MatrixX<T>::Zero(3 * contact_data.num_contact_points(),
                                   3 * contact_data.num_vertices_in_contact());
  int contact_point_offset = 0;
  for (const auto& contact_pair : contact_data.contact_pairs()) {
    const DeformableContactSurface<T>& contact_surface =
        contact_pair.contact_surface;
    for (int ic = 0; ic < contact_surface.num_polygons(); ++ic) {
      const ContactPolygonData<T>& polygon_data =
          contact_surface.polygon_data(ic);
      /* The contribution to the contact velocity from the deformable object
       A is R_CW * v_WAq. Note ₖₗ
         v_WAq = b₀ * v_WVᵢ₀ + b₁ * v_WVᵢ₁ + b₂ * v_WVᵢ₂ + b₃ * v_WVᵢ₃,
       where bⱼ is the barycentric weight corresponding to the vertex iⱼ and
       v_WVᵢⱼ is the velocity of the vertex iⱼ. */
      const Vector4<T>& barycentric_weights = polygon_data.b_centroid;
      const geometry::VolumeElement tet_element =
          deformable_meshes_[contact_pair.deformable_id].element(
              polygon_data.tet_index);
      for (int j = 0; j < 4; ++j) {
        const int permuted_vertex_index =
            contact_data.permuted_vertex_indexes()[tet_element.vertex(j)];
        Jc.template block<3, 3>(3 * contact_point_offset,
                                3 * permuted_vertex_index) =
            contact_pair.R_CWs[ic].matrix() * barycentric_weights(j);
      }
      ++contact_point_offset;
    }
  }
  return Jc;
}

template <typename T>
MatrixX<T> DeformableRigidManager<T>::CalcContactJacobianRigidBlock(
    const systems::Context<T>& context,
    const internal::DeformableContactData<T>& contact_data) const {
  const int rigid_dofs = this->plant().num_velocities();
  MatrixX<T> Jc =
      MatrixX<T>::Zero(3 * contact_data.num_contact_points(), rigid_dofs);
  int contact_point_offset = 0;
  for (const auto& contact_pair : contact_data.contact_pairs()) {
    const DeformableContactSurface<T>& contact_surface =
        contact_pair.contact_surface;
    if (contact_surface.num_polygons() == 0) {
      continue;
    }
    const geometry::GeometryId rigid_id = contact_pair.rigid_id;
    const Frame<T>& body_frame =
        this->GetBodyFrameFromCollisionGeometry(rigid_id);
    const int num_contact_points = contact_surface.num_polygons();
    /* The contact points in world frame. */
    Matrix3X<T> p_WCs(3, num_contact_points);
    for (int i = 0; i < contact_surface.num_polygons(); ++i) {
      p_WCs.col(i) = contact_surface.polygon_data(i).centroid;
    }
    /* The contact jacobian associated with the rigid geometry with
     `rigid_id` to be written to. */
    auto Jc_block = Jc.block(3 * contact_point_offset, 0,
                             3 * num_contact_points, rigid_dofs);
    this->internal_tree().CalcJacobianTranslationalVelocity(
        context, JacobianWrtVariable::kV, body_frame,
        this->plant().world_frame(), p_WCs, this->plant().world_frame(),
        this->plant().world_frame(), &Jc_block);
    /* Rotates to the contact frame at each contact point. */
    for (int i = 0; i < contact_pair.num_contact_points(); ++i) {
      Jc_block.block(3 * i, 0, 3, rigid_dofs) =
          -contact_pair.R_CWs[i].matrix() *
          Jc_block.block(3 * i, 0, 3, rigid_dofs);
    }
    contact_point_offset += num_contact_points;
  }
  /* Sanity check that all rows of the Jacobian has been written to. */
  DRAKE_DEMAND(3 * contact_point_offset == Jc.rows());
  return Jc;
}

template <typename T>
void DeformableRigidManager<T>::CalcPerContactPointData(
    const systems::Context<T>& context,
    const std::vector<multibody::internal::DiscreteContactPair<T>>&
        rigid_contact_pairs,
    const std::vector<internal::DeformableContactData<T>>
        deformable_contact_data,
    EigenPtr<VectorX<T>> mu, EigenPtr<VectorX<T>> phi,
    EigenPtr<VectorX<T>> stiffness, EigenPtr<VectorX<T>> damping) const {
  DRAKE_DEMAND(mu != nullptr);
  DRAKE_DEMAND(phi != nullptr);
  DRAKE_DEMAND(stiffness != nullptr);
  DRAKE_DEMAND(damping != nullptr);
  const int num_rigid_contacts = rigid_contact_pairs.size();
  // TODO(xuchenhan-tri): num_deformable_contacts can be cached. Or we can skip
  //  the check altogether since this is internal.
  int num_deformable_contacts = 0;
  for (const auto& data : deformable_contact_data) {
    num_deformable_contacts += data.num_contact_points();
  }
  const int nc = num_rigid_contacts + num_deformable_contacts;
  DRAKE_DEMAND(mu->size() == nc);
  DRAKE_DEMAND(phi->size() == nc);
  DRAKE_DEMAND(stiffness->size() == nc);
  DRAKE_DEMAND(damping->size() == nc);

  /* First write the rigid contact data. */
  std::vector<CoulombFriction<double>> combined_friction_pairs =
      this->CalcCombinedFrictionCoefficients(context, rigid_contact_pairs);
  std::transform(combined_friction_pairs.begin(), combined_friction_pairs.end(),
                 mu->data(),
                 [](const CoulombFriction<double>& coulomb_friction) {
                   return coulomb_friction.dynamic_friction();
                 });
  for (int i = 0; i < num_rigid_contacts; ++i) {
    (*phi)[i] = rigid_contact_pairs[i].phi0;
    (*stiffness)[i] = rigid_contact_pairs[i].stiffness;
    (*damping)[i] = rigid_contact_pairs[i].damping;
  }

  int contact_offset = num_rigid_contacts;
  /* Then write the deformable contact data. */
  for (int i = 0; i < static_cast<int>(deformable_contact_data.size()); ++i) {
    const std::vector<internal::DeformableRigidContactPair<T>>& contact_pairs =
        deformable_contact_data[i].contact_pairs();
    for (int j = 0; j < deformable_contact_data[i].num_contact_pairs(); ++j) {
      const internal::DeformableRigidContactPair<T>& contact_pair =
          contact_pairs[j];
      const int num_contact_points_in_pair = contact_pair.num_contact_points();
      // TODO(xuchenhan-tri): Set phi to actual values when they are
      //  calculated.
      phi->segment(contact_offset, num_contact_points_in_pair) =
          VectorX<T>::Zero(num_contact_points_in_pair);
      mu->segment(contact_offset, num_contact_points_in_pair) =
          VectorX<T>::Ones(num_contact_points_in_pair) * contact_pair.friction;
      stiffness->segment(contact_offset, num_contact_points_in_pair) =
          VectorX<T>::Ones(num_contact_points_in_pair) * contact_pair.stiffness;
      damping->segment(contact_offset, num_contact_points_in_pair) =
          VectorX<T>::Ones(num_contact_points_in_pair) *
          contact_pair.dissipation;
      contact_offset += num_contact_points_in_pair;
    }
  }
}

template <typename T>
void DeformableRigidManager<T>::DoCalcDiscreteValues(
    const systems::Context<T>& context,
    systems::DiscreteValues<T>* updates) const {
  /* Get the rigid dofs from context. */
  auto x =
      context.get_discrete_state(this->multibody_state_index()).get_value();
  const auto& q = x.topRows(this->plant().num_positions());

  const contact_solvers::internal::ContactSolverResults<T>& solver_results =
      this->EvalContactSolverResults(context);
  /* Update rigid states. */
  {
    const auto& v_next = solver_results.v_next;

    VectorX<T> qdot_next(this->plant().num_positions());
    this->plant().MapVelocityToQDot(context, v_next, &qdot_next);
    const double dt = this->plant().time_step();
    const auto& q_next = q + dt * qdot_next;

    VectorX<T> x_next(this->plant().num_multibody_states());
    x_next << q_next, v_next;
    updates->get_mutable_vector(this->multibody_state_index())
        .SetFromVector(x_next);
  }

  /* Update deformable states. */
  {
    const std::vector<internal::DeformableContactData<T>>&
        deformable_contact_data = EvalDeformableRigidContact(context);
    const std::vector<systems::DiscreteStateIndex>& discrete_state_indexes =
        deformable_model_->discrete_state_indexes();
    for (SoftBodyIndex deformable_body_id(0);
         deformable_body_id < free_motion_cache_indexes_.size();
         ++deformable_body_id) {
      const FemStateBase<T>& fem_state_star =
          EvalFreeMotionFemStateBase(context, deformable_body_id);
      const auto& contact_data = deformable_contact_data[deformable_body_id];
      /* If the deformable body is not in contact, then the free motion velocity
       is the final velocity. */
      const VectorX<T>& v_star = fem_state_star.qdot();
      const int num_total_dofs = v_star.size();
      VectorX<T> v_next;
      if (contact_data.num_contact_points() == 0) {
        v_next = v_star;
      } else {
        /* Otherwise, use Schur complement to obtain the velocity change in
         vertices not participating in contact. */
        VectorX<T> permuted_v_star = PermuteWithVertexOrdering(
            v_star, contact_data.permuted_vertex_indexes());
        const int num_participating_dofs =
            3 * contact_data.num_vertices_in_contact();
        const auto participating_v_star =
            permuted_v_star.head(num_participating_dofs);
        const auto nonparticipating_v_star =
            permuted_v_star.tail(num_total_dofs - num_participating_dofs);
        const VectorX<T>& participating_v =
            deformable_participating_velocities_[deformable_body_id];
        DRAKE_DEMAND(participating_v.size() == participating_v_star.size());
        const auto participating_delta_v =
            participating_v - participating_v_star;
        const VectorX<T> nonparticipating_delta_v =
            tangent_matrix_schur_complements_[deformable_body_id].SolveForY(
                participating_delta_v);
        const auto nonparticipating_v =
            nonparticipating_delta_v + nonparticipating_v_star;
        VectorX<T> permuted_v(num_total_dofs);
        permuted_v << participating_v, nonparticipating_v;
        v_next = PermuteWithVertexOrdering(
            permuted_v, contact_data.permuted_to_original_indexes());
      }
      DRAKE_DEMAND(v_next.size() == fem_state_star.num_generalized_positions());

      // TODO(xuchenhan-tri): This should be handled by FemModel.
      /* Advance to the next time step with the new velocity. */
      const auto& fem_state0 = EvalFemStateBase(context, deformable_body_id);
      const VectorX<T>& q0 = fem_state0.q();
      const VectorX<T>& v0 = fem_state0.qdot();
      const auto& q_next =
          q0 + this->plant().time_step() * (0.5 * v_next + 0.5 * v0);
      const auto& a_next = (v_next - v0) / this->plant().time_step();

      systems::BasicVector<T>& next_discrete_state =
          updates->get_mutable_vector(
              discrete_state_indexes[deformable_body_id]);
      Eigen::VectorBlock<VectorX<T>> next_discrete_value =
          next_discrete_state.get_mutable_value();
      next_discrete_value.head(num_total_dofs) = q_next;
      next_discrete_value.segment(num_total_dofs, num_total_dofs) = v_next;
      next_discrete_value.tail(num_total_dofs) = a_next;
    }
  }
}

template <typename T>
void DeformableRigidManager<T>::UpdateCollisionObjectPoses(
    const systems::Context<T>& context) const {
  const geometry::QueryObject<T>& query_object =
      this->plant()
          .get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
  const std::vector<geometry::GeometryId>& geometry_ids =
      collision_objects_.geometry_ids();
  for (const geometry::GeometryId id : geometry_ids) {
    const math::RigidTransform<T>& X_WG = query_object.GetPoseInWorld(id);
    collision_objects_.set_pose_in_world(id, X_WG);
  }
}

template <typename T>
void DeformableRigidManager<T>::UpdateDeformableVertexPositions(
    const systems::Context<T>& context) const {
  const std::vector<VectorX<T>>& vertex_positions =
      deformable_model_->get_vertex_positions_output_port()
          .template Eval<std::vector<VectorX<T>>>(context);
  DRAKE_DEMAND(vertex_positions.size() == deformable_meshes_.size());
  for (int i = 0; i < static_cast<int>(vertex_positions.size()); ++i) {
    const VectorX<T>& q = vertex_positions[i];
    const auto p_WVs = Eigen::Map<const Matrix3X<T>>(q.data(), 3, q.size() / 3);
    std::vector<geometry::VolumeElement> tets =
        deformable_meshes_[i].tetrahedra();
    // TODO(xuchenhan-tri): We assume the deformable body frame is always
    // the
    //  same as the world frame for now. It probably makes sense to have the
    //  frame move with the body in the future.
    std::vector<geometry::VolumeVertex<T>> vertices_D;
    for (int j = 0; j < p_WVs.cols(); ++j) {
      vertices_D.push_back(geometry::VolumeVertex<T>(p_WVs.col(j)));
    }
    deformable_meshes_[i] = {std::move(tets), std::move(vertices_D)};
  }
}

template <typename T>
internal::DeformableRigidContactPair<T>
DeformableRigidManager<T>::CalcDeformableRigidContactPair(
    geometry::GeometryId rigid_id, SoftBodyIndex deformable_id) const {
  DeformableContactSurface<T> contact_surface = ComputeTetMeshTriMeshContact(
      deformable_meshes_[deformable_id], collision_objects_.mesh(rigid_id),
      collision_objects_.pose_in_world(rigid_id));

  const auto get_point_contact_parameters =
      [this](const geometry::ProximityProperties& props) -> std::pair<T, T> {
    return std::make_pair(props.template GetPropertyOrDefault<T>(
                              geometry::internal::kMaterialGroup,
                              geometry::internal::kPointStiffness,
                              this->default_contact_stiffness()),
                          props.template GetPropertyOrDefault<T>(
                              geometry::internal::kMaterialGroup,
                              geometry::internal::kHcDissipation,
                              this->default_contact_dissipation()));
  };
  /* Extract the stiffness, dissipation and friction parameters of the
   deformable body. */
  DRAKE_DEMAND(deformable_model_ != nullptr);
  const geometry::ProximityProperties& deformable_props =
      deformable_model_->proximity_properties()[deformable_id];
  const auto [deformable_stiffness, deformable_dissipation] =
      get_point_contact_parameters(deformable_props);
  const CoulombFriction<T> deformable_mu =
      deformable_props.GetProperty<CoulombFriction<T>>(
          geometry::internal::kMaterialGroup, geometry::internal::kFriction);

  /* Extract the stiffness, dissipation and friction parameters of the rigid
   body. */
  const geometry::ProximityProperties& rigid_proximity_properties =
      collision_objects_.proximity_properties(rigid_id);
  const auto [rigid_stiffness, rigid_dissipation] =
      get_point_contact_parameters(rigid_proximity_properties);
  const CoulombFriction<T> rigid_mu =
      rigid_proximity_properties.GetProperty<CoulombFriction<T>>(
          geometry::internal::kMaterialGroup, geometry::internal::kFriction);

  /* Combine the stiffness, dissipation and friction parameters for the
   contact points. */
  auto [k, d] = multibody::internal::CombinePointContactParameters(
      deformable_stiffness, rigid_stiffness, deformable_dissipation,
      rigid_dissipation);
  const CoulombFriction<T> mu =
      CalcContactFrictionFromSurfaceProperties(deformable_mu, rigid_mu);
  return internal::DeformableRigidContactPair<T>(std::move(contact_surface),
                                                 rigid_id, deformable_id, k, d,
                                                 mu.dynamic_friction());
}

template <typename T>
internal::DeformableContactData<T>
DeformableRigidManager<T>::CalcDeformableContactData(
    SoftBodyIndex deformable_id) const {
  /* Calculate the all rigid-deformable contact pair for this deformable
   * body.
   */
  const std::vector<geometry::GeometryId>& rigid_ids =
      collision_objects_.geometry_ids();
  const int num_rigid_bodies = rigid_ids.size();
  std::vector<internal::DeformableRigidContactPair<T>>
      deformable_rigid_contact_pairs;
  for (int i = 0; i < num_rigid_bodies; ++i) {
    deformable_rigid_contact_pairs.emplace_back(
        CalcDeformableRigidContactPair(rigid_ids[i], deformable_id));
  }
  return {std::move(deformable_rigid_contact_pairs),
          deformable_meshes_[deformable_id]};
}

template <typename T>
std::vector<internal::DeformableContactData<T>>
DeformableRigidManager<T>::CalcDeformableRigidContact(
    const systems::Context<T>& context) const {
  UpdateCollisionObjectPoses(context);
  UpdateDeformableVertexPositions(context);
  std::vector<internal::DeformableContactData<T>> deformable_contact_data;
  for (SoftBodyIndex deformable_body_id(0);
       deformable_body_id < deformable_model_->num_bodies();
       ++deformable_body_id) {
    deformable_contact_data.emplace_back(
        CalcDeformableContactData(deformable_body_id));
  }
  return deformable_contact_data;
}

template <typename T>
BlockSparseMatrix<T> DeformableRigidManager<T>::CalcTangentMatrix(
    const systems::Context<T>& context,
    const std::vector<internal::DeformableContactData<T>>&
        deformable_contact_data) const {
  DRAKE_DEMAND(deformable_model_ != nullptr);
  int num_deformable_body_in_contact = 0;
  for (const auto& contact_data : deformable_contact_data) {
    if (contact_data.num_contact_points() > 0) {
      ++num_deformable_body_in_contact;
    }
  }
  const int num_diagonal_blocks = 1 + num_deformable_body_in_contact;
  BlockSparseMatrixBuilder<T> builder(num_diagonal_blocks, num_diagonal_blocks,
                                      num_diagonal_blocks);

  // TODO(xuchenhan-tri): use a Eval method here.
  const int nv = this->plant().num_velocities();
  MatrixX<T> M(nv, nv);
  this->plant().CalcMassMatrix(context, &M);
  builder.PushBlock(0, 0, M);

  int block_index = 1;
  for (SoftBodyIndex i(0); i < deformable_model_->num_bodies(); ++i) {
    if (deformable_contact_data[i].num_contact_points() == 0) {
      continue;
    }
    const Eigen::SparseMatrix<T>& tangent_matrix =
        EvalTangentMatrixAtFreeMotionState(context, i);
    internal::BlockTangentMatrix<T> permuted_tangent_matrix =
        internal::PermuteTangentMatrix(
            tangent_matrix,
            deformable_contact_data[i].permuted_vertex_indexes(),
            deformable_contact_data[i].num_vertices_in_contact());
    // TODO(xuchenhan-tri): Make tangent_matrix_schur_complements_
    //  a cache entry and use the Eval method here.
    tangent_matrix_schur_complements_[i] = internal::SchurComplement<T>(
        permuted_tangent_matrix.participating_block,
        permuted_tangent_matrix.off_diagonal_block,
        permuted_tangent_matrix.nonparticipating_block);
    builder.PushBlock(block_index, block_index,
                      tangent_matrix_schur_complements_[i].get_D_complement());
    ++block_index;
  }
  return builder.Build();
}

template <typename T>
void DeformableRigidManager<T>::CalcFreeMotionVelocities(
    const systems::Context<T>& context,
    const std::vector<internal::DeformableContactData<T>>&
        deformable_contact_data,
    EigenPtr<VectorX<T>> v_star) const {
  DRAKE_DEMAND(v_star != nullptr);
  const int nv = this->plant().num_velocities();
  // TODO(xuchenhan-tri): num_dofs can be cached. Or consider removing this
  // check altogether since this is internal calculation.
  int num_dofs = nv;
  for (int i = 0; i < static_cast<int>(deformable_contact_data.size()); ++i) {
    num_dofs += 3 * deformable_contact_data[i].num_vertices_in_contact();
  }
  DRAKE_DEMAND(v_star->size() == num_dofs);

  auto x =
      context.get_discrete_state(this->multibody_state_index()).get_value();
  const VectorX<T> rigid_v0 = x.bottomRows(nv);
  // TODO(xuchenhan-tri): Switch to a Eval method.
  /* Compute the mass matrix. */
  MatrixX<T> M(nv, nv);
  this->plant().CalcMassMatrix(context, &M);

  /* Computes the negative generalized non-contact forces on the rigid dofs,
   `minus_tau`. */
  MultibodyForces<T> forces(this->internal_tree());
  this->CalcNonContactForces(context, true /* discrete */, &forces);
  /* Workspace for inverse dynamics: Bodies' accelerations, ordered by
   BodyNodeIndex. */
  std::vector<SpatialAcceleration<T>> A_WB_array(this->plant().num_bodies());
  /* Generalized accelerations. */
  const VectorX<T> vdot = VectorX<T>::Zero(nv);
  /* Body forces (alias to forces). */
  std::vector<SpatialForce<T>>& F_BBo_W_array = forces.mutable_body_forces();
  /* With vdot = 0, this computes:
   -tau = C(q, v)v - tau_app - ∑ J_WBᵀ(q) Fapp_Bo_W. */
  VectorX<T> minus_tau = forces.mutable_generalized_forces();
  this->internal_tree().CalcInverseDynamics(
      context, vdot, F_BBo_W_array, minus_tau, &A_WB_array,
      &F_BBo_W_array, /* Note: these arrays get overwritten on output. */
      &minus_tau);
  v_star->segment(0, nv) =
      rigid_v0 - this->plant().time_step() * M.llt().solve(minus_tau);

  int dof_offset = nv;
  for (SoftBodyIndex i(0); i < deformable_contact_data.size(); ++i) {
    const FemStateBase<T>& fem_state_star =
        EvalFreeMotionFemStateBase(context, i);
    const VectorX<T>& v_star_deformable = fem_state_star.qdot();
    const std::vector<int>& permuted_to_original_indexes =
        deformable_contact_data[i].permuted_to_original_indexes();
    for (int j = 0; j < deformable_contact_data[i].num_vertices_in_contact();
         ++j) {
      v_star->template segment<3>(dof_offset) =
          v_star_deformable.template segment<3>(
              3 * permuted_to_original_indexes[j]);
      dof_offset += 3;
    }
  }
}

}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fixed_fem::DeformableRigidManager);
