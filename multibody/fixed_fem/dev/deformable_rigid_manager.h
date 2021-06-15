#pragma once

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/contact_solvers/block_sparse_matrix.h"
#include "drake/multibody/contact_solvers/contact_solver.h"
#include "drake/multibody/fixed_fem/dev/collision_objects.h"
#include "drake/multibody/fixed_fem/dev/deformable_contact_data.h"
#include "drake/multibody/fixed_fem/dev/deformable_model.h"
#include "drake/multibody/fixed_fem/dev/deformable_rigid_contact_pair.h"
#include "drake/multibody/fixed_fem/dev/fem_solver.h"
#include "drake/multibody/fixed_fem/dev/permute_tangent_matrix.h"
#include "drake/multibody/fixed_fem/dev/schur_complement.h"
#include "drake/multibody/plant/contact_jacobians.h"
#include "drake/multibody/plant/discrete_update_manager.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
/** %DeformableRigidManager implements the interface in DiscreteUpdateManager
 and performs discrete update for deformable and rigid bodies with a two-way
 coupling scheme.
 @tparam_nonsymbolic_scalar. */
template <typename T>
class DeformableRigidManager final
    : public multibody::internal::DiscreteUpdateManager<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DeformableRigidManager)

  DeformableRigidManager() = default;

  /** Sets the given `contact_solver` as the solver that `this`
    %DeformableRigidManager uses to solve contact. */
  void SetContactSolver(
      std::unique_ptr<multibody::contact_solvers::internal::ContactSolver<T>>
          contact_solver) {
    contact_solver_ = std::move(contact_solver);
  }

  // TODO(xuchenhan-tri): Remove this method when SceneGraph owns all rigid
  // and
  //  deformable geometries.
  // TODO(xuchenhan-tri): Reconcile the names of "deformable geometries" and
  //  "collision objects" when moving out of dev.
  /** Registers all the rigid collision geometries from the owning
   MultibodyPlant that are registered in the given SceneGraph into `this`
   %DeformableRigidManager. The registered rigid collision geometries will be
   used to generate rigid-deformable contact pairs to be used for the
   rigid-deformable two-way coupled contact solve. A common workflow to set up
   a simulation where deformable and rigid bodies interact with each other
   through contact looks like the following:
   ```
   // Set up a deformable model assciated with a MultibodyPlant.
   auto deformable_model = std::make_unique<DeformableModel<double>>(&plant);
   // Add deformable bodies to the model.
   deformable_model->RegisterDeformableBody(...);
   deformable_model->RegisterDeformableBody(...);
   // Done building the model. Move the DeformableModel into the
   MultibodyPlant. plant.AddPhysicalModel(std::move(deformable_model));
   // Register the plant as a source for scene graph for rigid geometries.
   plant.RegisterAsSourceForSceneGraph(&scene_graph);
   // Add rigid bodies.
   Parser parser(&plant, &scene_graph);
   parser.AddModelFromFile(...);
   parser.AddModelFromFile(...);
   // Done building the plant.
   plant.Finalize();
   // Use a DeformableRigidManager to perform the discrete updates.
   auto& deformable_rigid_manager = plant.SetDiscreteUpdateManager(
        std::make_unqiue<DeformableRigidManager<double>());
   // Register all rigid collision geometries at the manager.
   deformable_rigid_manager.RegisterCollisionObjects(scene_graph);
   ```
   @pre `This` %DeformableRigidManager is owned by a MultibodyPlant via the call
   MultibodyPlant::SetDiscreteUpdateManager().
   @pre The owning MultibodyPlant is registered as a source of the given
   `scene_graph`. */
  void RegisterCollisionObjects(
      const geometry::SceneGraph<T>& scene_graph) const;

 private:
  friend class DeformableRigidManagerTest;
  friend class DeformableRigidContactJacobianTest;

  template <typename Scalar, int Options = 0, typename StorageIndex = int>
  struct EigenSparseMatrixWrapper {
    using NonTypeTemplateParameter = std::integral_constant<int, Options>;
    Eigen::SparseMatrix<Scalar, Options, StorageIndex> sparse_matrix;
  };

  // TODO(xuchenhan-tri): Implement CloneToDouble() and CloneToAutoDiffXd() and
  //  the corresponding is_cloneable methods.

  /* Implements DiscreteUpdateManager::ExtractModelInfo(). Verifies that
   exactly one DeformableModel is registered in the owning plant and
   sets up FEM solvers for deformable bodies. */
  void
  ExtractModelInfo() final;

  /* Make the FEM solvers that solve the deformable FEM models. */
  void MakeFemSolvers();

  // TODO(xuchenhan-tri): Remove this temporary geometry solutions when
  //  SceneGraph manages all deformable geometries.
  /* Registers the geometries in the DeformableModel in the owning
   MultibodyPlant at `this` DeformableRigidManager. */
  void RegisterDeformableGeometries();

  void DeclareCacheEntries() final;

  void DoCalcContactSolverResults(
      const systems::Context<T>& context,
      contact_solvers::internal::ContactSolverResults<T>* results) const final;

  /* Calculates all contact quantities needed by the contact solver and the
   TAMSI solver from the given `context` and `rigid_contact_pairs`.
   @pre All pointer parameters are non-null.
   @pre The size of `v` and `minus_tau` are equal to the number of rigid
        generalized velocities.
   @pre `M` is square and has rows and columns equal to the number of rigid
        generalized velocities.
   @pre `mu`, `phi`, `fn`, `stiffness`, `damping`, and `rigid_contact_pairs`
        have the same size. */
  void CalcContactQuantities(
      const systems::Context<T>& context,
      const std::vector<multibody::internal::DiscreteContactPair<T>>&
          rigid_contact_pairs,
      multibody::internal::ContactJacobians<T>* rigid_contact_jacobians,
      EigenPtr<VectorX<T>> v, EigenPtr<MatrixX<T>> M,
      EigenPtr<VectorX<T>> minus_tau, EigenPtr<VectorX<T>> mu,
      EigenPtr<VectorX<T>> phi, EigenPtr<VectorX<T>> fn,
      EigenPtr<VectorX<T>> stiffness, EigenPtr<VectorX<T>> damping) const;

  void CalcPerContactPointData(
      const systems::Context<T>& context,
      const std::vector<multibody::internal::DiscreteContactPair<T>>&
          rigid_contact_pairs,
      const std::vector<internal::DeformableContactData<T>>
          deformable_contact_data,
      EigenPtr<VectorX<T>> mu, EigenPtr<VectorX<T>> phi,
      EigenPtr<VectorX<T>> stiffness, EigenPtr<VectorX<T>> damping) const;

  // TODO(xuchenhan-tri): Implement this once AccelerationKinematicsCache
  //  also caches acceleration for deformable dofs.
  void DoCalcAccelerationKinematicsCache(
      const systems::Context<T>&,
      multibody::internal::AccelerationKinematicsCache<T>*) const final {
    throw std::logic_error(
        "DoCalcAcclerationKinematicsCache() hasn't be implemented for "
        "DeformableRigidManager yet.");
  }

  void DoCalcDiscreteValues(const systems::Context<T>& context,
                            systems::DiscreteValues<T>* updates) const final;

  /* Evaluates the FEM state of the deformable body with index `id`. */
  const FemStateBase<T>& EvalFemStateBase(const systems::Context<T>& context,
                                          SoftBodyIndex id) const {
    return this->plant()
        .get_cache_entry(fem_state_cache_indexes_[id])
        .template Eval<FemStateBase<T>>(context);
  }

  /* Evaluates the free motion FEM state of the deformable body with index
   `id`. */
  const FemStateBase<T>& EvalFreeMotionFemStateBase(
      const systems::Context<T>& context, SoftBodyIndex id) const {
    return this->plant()
        .get_cache_entry(free_motion_cache_indexes_[id])
        .template Eval<FemStateBase<T>>(context);
  }

  /* Evaluates the tangent matrix of the deformable body with index `id` at free
   motion state. */
  const Eigen::SparseMatrix<T>& EvalTangentMatrixAtFreeMotionState(
      const systems::Context<T>& context, SoftBodyIndex id) const {
    return this->plant()
        .get_cache_entry(tangent_matrix_cache_indexes_[id])
        .template Eval<EigenSparseMatrixWrapper<T>>(context).sparse_matrix;
  }

  // TODO(xuchenhan-tri): Remove this method when SceneGraph takes control
  //  of all geometries. SceneGraph should be responsible for obtaining the
  //  most up-to-date rigid body poses.
  /* Updates the world poses of all rigid collision geometries registered
   in `this` DeformableRigidManager. */
  void UpdateCollisionObjectPoses(const systems::Context<T>& context) const;

  // TODO(xuchenhan-tri): This method (or similar) should belong to
  // SceneGraph
  //  when SceneGraph takes control of all geometries.
  /* Updates the vertex positions for all deformable meshes. */
  void UpdateDeformableVertexPositions(
      const systems::Context<T>& context) const;

  // TODO(xuchenhan-tri): Make proper distinction between id and index in
  //  variable names.
  /* Calculates the contact information for the contact pair consisting of the
   rigid body identified by `rigid_id` and the deformable body identified by
   `deformable_id`. */
  internal::DeformableRigidContactPair<T> CalcDeformableRigidContactPair(
      geometry::GeometryId rigid_id, SoftBodyIndex deformable_id) const;

  /* Calculates and returns the DeformableContactData that contains information
   about all deformable-rigid contacts associated with the deformable body
   identified by `deformable_id`. */
  internal::DeformableContactData<T> CalcDeformableContactData(
      SoftBodyIndex deformable_id) const;

  // TODO(xuchenhan-tri): Skip empty contact data altogether.
  /* Calculates all deforamble-rigid contacts and returns a vector of
   DeformableContactData in which the i-th entry contains contact information
   about the i-th deformable body against all rigid bodies. If the i-th
   deformable body is not in contact with any rigid body, then the i-th entry
   (data[i]) in the return value will have `data[i].num_contact_points() == 0`.
  */
  std::vector<internal::DeformableContactData<T>> CalcDeformableRigidContact(
      const systems::Context<T>& context) const;

  /* Eval version of the method CalcDeformableRigidContact(). */
  const std::vector<internal::DeformableContactData<T>>&
  EvalDeformableRigidContact(const systems::Context<T>& context) const {
    return this->plant()
        .get_cache_entry(deformable_contact_data_cache_index_)
        .template Eval<std::vector<internal::DeformableContactData<T>>>(
            context);
  }

  // TODO(xuchenhan-tri): Modify the description of the contact jacobian when
  //  sparsity for rigid dofs are exploited.
  /* Calculates and returns the contact jacobian as a block sparse matrix.
   As an illustrating example, consider a scene with n rigid bodies and m
   deformable bodies. Four of those deformable bodies are in contact with rigid
   geometries. Then the sparsity pattern of the contact jacobian looks like:
                                | RR               |
                                | RD0  D0          |
                                | RD1     D1       |
                                | RD2        D2    |
                                | RD3           D3 |
   where "RR" represents the rigid-rigid block, "RDi" represents the the
   rigid-deformable block with respect to the rigid dofs for the i-th deformable
   body, and "Di" represents the rigid-deformable block with respect to the
   deformable dofs for the i-th deformable body.

   More specifically, the contact jacobian is ordered in the following way:
    1. The contact jacobian has 3 * (ncr + ncd) rows, where ncr is the number of
       contact points among rigid objects, and ncd is the number of contact
       points between deformable bodies and rigid bodies, i.e, the sum of
       DeformableContactData::num_contact_points() for all deformable bodies in
       contact.
    2. Rows 3*i, 3*i+1, and 3*i+2 correspond to the i-th rigid-rigid
       contact point for i = 0, ..., ncr-1. These contact points are ordered as
       given by the result of `EvalDiscreteContactPairs()`. Rows 3*(i+ncr),
       3*(i+ncr)+1, and 3*(i+ncr)+2 correspond to the i-th rigid-deformable
       contact points for i = 0, ..., ncd-1. These contact points are ordered as
       given by the result of `CalcDeformableRigidContact()`.
    3. The contact jacobian has nvr + 3 * nvd columns, where nvr is the number
       of rigid velocity degrees of freedom and nvd is the number of deformable
       vertices participating in contact. A vertex of a deformable body is said
       to be participating in contact if it is incident to a tetrahedron that
       contains a contact point.
    4. The first nvr columns of the contact jacobian correspond to the rigid
       velocity degrees of freedom. The last 3 * nvd columns correspond to the
       participating deformable velocity degrees of freedom. The participating
       deformable velocity dofs come in blocks. The i-th block corresponds to
       i-th deformable body and has number of dofs equal to 3 * number of
       participating vertices for deformable body i (see
       DeformableContactData::num_vertices_in_contact()). Within the i-th block,
       the 3*j, 3*j+1, and 3*j+2 velocity dofs belong to the j-th permuted
       vertex. (see DeformableContactData::permuted_vertex_indexes()). */
  multibody::contact_solvers::internal::BlockSparseMatrix<T>
  CalcContactJacobian(const systems::Context<T>& context) const;

  /* Given the contact data for a deformable body, calculates the contact
   jacobian for the contact points associated with that deformable body with
   respect to the deformable degrees of freedom participating in the contact. */
  MatrixX<T> CalcContactJacobianDeformableBlock(
      const internal::DeformableContactData<T>& contact_data) const;

  /* Given the contact data for a deformable body, calculates the contact
   jacobian for the contact points associated with that deformable body with
   respect to all rigid degrees of freedom. */
  MatrixX<T> CalcContactJacobianRigidBlock(
      const systems::Context<T>& context,
      const internal::DeformableContactData<T>& contact_data) const;

  /* Calculates the tangent matrix fed into the contact solver. */
  multibody::contact_solvers::internal::BlockSparseMatrix<T> CalcTangentMatrix(
      const systems::Context<T>& context,
      const std::vector<internal::DeformableContactData<T>>&
          deformable_contact_data) const;

  void CalcFreeMotionVelocities(
      const systems::Context<T>& context,
      const std::vector<internal::DeformableContactData<T>>&
          deformable_contact_data,
      EigenPtr<VectorX<T>> v_star) const;

  /* Given the GeometryId of a rigid collision geometry, returns the body frame
   of the collision geometry.
   @pre The collision geometry with the given `id` is already registered with
   `this` DeformableRigidManager. */
  const Frame<T>& GetBodyFrameFromCollisionGeometry(
      geometry::GeometryId id) const {
    BodyIndex body_index = this->geometry_id_to_body_index().at(id);
    const Body<T>& body = this->plant().get_body(body_index);
    return body.body_frame();
  }

  /* The deformable models being solved by `this` manager. */
  const DeformableModel<T>* deformable_model_{nullptr};
  /* Cached FEM state quantities. */
  std::vector<systems::CacheIndex> fem_state_cache_indexes_;
  std::vector<systems::CacheIndex> free_motion_cache_indexes_;
  std::vector<systems::CacheIndex> tangent_matrix_cache_indexes_;
  /* Cached contact query results. */
  systems::CacheIndex deformable_contact_data_cache_index_;
  /* Solvers for all deformable bodies. */
  std::vector<std::unique_ptr<FemSolver<T>>> fem_solvers_{};
  std::unique_ptr<multibody::contact_solvers::internal::ContactSolver<T>>
      contact_solver_{nullptr};

  // TODO(xuchenhan-tri): Consider bumping these up to cache entries.
  mutable std::vector<internal::SchurComplement<T>>
      tangent_matrix_schur_complements_{};
  // TODO(xuchenhan-tri): Consider storing them in ContactSolverResults instead
  //  of keep them here as mutable members.
  mutable std::vector<VectorX<T>> deformable_participating_velocities_{};

  /* Geometries temporarily managed by DeformableRigidManager. In the
   future, SceneGraph will manage all the geometries. */
  mutable std::vector<geometry::VolumeMesh<T>> deformable_meshes_{};
  mutable internal::CollisionObjects<T> collision_objects_;
};
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fixed_fem::DeformableRigidManager)
