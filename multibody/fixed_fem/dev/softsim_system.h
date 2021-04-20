#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/unused.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/multibody/fixed_fem/dev/collision_objects.h"
#include "drake/multibody/fixed_fem/dev/contact_solver_data.h"
#include "drake/multibody/fixed_fem/dev/deformable_body_config.h"
#include "drake/multibody/fixed_fem/dev/deformable_rigid_contact_data.h"
#include "drake/multibody/fixed_fem/dev/dirichlet_boundary_condition.h"
#include "drake/multibody/fixed_fem/dev/dynamic_elasticity_element.h"
#include "drake/multibody/fixed_fem/dev/dynamic_elasticity_model.h"
#include "drake/multibody/fixed_fem/dev/fem_solver.h"
#include "drake/multibody/fixed_fem/dev/linear_simplex_element.h"
#include "drake/multibody/fixed_fem/dev/simplex_gaussian_quadrature.h"
#include "drake/multibody/fixed_fem/softsim_base.h"
#include "drake/systems/framework/leaf_system.h"
namespace drake {
namespace multibody {
namespace fixed_fem {
/** A minimum Drake system (see systems::System) for simulating the dynamics of
 deformable bodies. Each deformable body is modeled as a volumetric mesh and
 spatially discretized with Finite Element Method. Currently, %SoftsimSystem is
 modeled as a discrete system with periodic updates. The discrete update
 interval `dt` is passed in at construction and must be positive.

 Deformable bodies can only be added, but not deleted. Each deformable body is
 uniquely identified by its index, which is equal to the number of deformable
 bodies existing in the %SoftsimSystem at the time of its registration.

 The current positions of the vertices of the mesh representing the deformable
 bodies can be queried via the `vertex_positions` output port. The output port
 is an abstract-valued port containing std::vector<VectorX<T>>. There is one
 VectorX for each deformable body registered with the system. The i-th body
 corresponds to the i-th VectorX. The i-th VectorX has 3N values where N is the
 number of vertices in the i-th mesh. For mesh i, the x-, y-, and z-positions
 (measured and expressed in the world frame) of the j-th vertex are 3j, 3j + 1,
 and 3j + 2 in the i-th VectorX from the output port.

 The connectivity of the meshes representing the deformable bodies and their
 initial positions can be queried via `initial_meshes()` which returns an
 std::vector of volume meshes. The i-th mesh stores the connectivity for body i,
 which does not change throughout the simulation, as well as the initial
 positions of the vertices of the i-th body.

 Simple zero DirichletBoundaryCondition can be configured via
 `SetRegisteredBodyInWall()`. Collision and contact are currently not supported
 in %SoftsimSystem. A default gravity value of (0, 0, -9.81) is assumed.

 @system
 name: SoftsimSystem
 output_ports:
 - vertex_positions
 @endsystem

 @tparam_non_symbolic T.*/
template <typename T>
class SoftsimSystem final : public SoftsimBase<T>,
                            public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SoftsimSystem)

  /* Construct a %SoftsimSystem with the fixed prescribed discrete time step.
   @pre dt > 0. */
  explicit SoftsimSystem(multibody::MultibodyPlant<T>* mbp);

  // TODO(xuchenhan-tri): Identify deformable bodies with actual identifiers,
  //  which would make deleting deformable bodies easier to track in the future.
  /** Adds a deformable body modeled with linear simplex element, linear
   quadrature rule, mid-point integration rule and the given `config`. Returns
   the index of the newly added deformable body.
   @param[in] mesh        The volume mesh representing the deformable body
                          geometry.
   @param[in] name        The name of the deformable body.
   @param[in] config      The physical properties of the deformable body.
   @param[in] properties  The proximity properties associated with the collision
                          geometry. They *must* include the (`material`,
                          `coulomb_friction`) property of type
                          CoulombFriction<double>.
   @pre `name` is distinct from names of all previously registered bodies.
   @pre config.IsValid() == true. */
  SoftBodyIndex RegisterDeformableBody(
      const geometry::VolumeMesh<T>& mesh, std::string name,
      const DeformableBodyConfig<T>& config,
      geometry::ProximityProperties properties);

  /** Set zero Dirichlet boundary conditions for a given body. All vertices in
   the mesh of corresponding to the deformable body with `body_id` within
   `distance_tolerance` (measured in meters) from the halfspace defined by point
   `p_WQ` and outward normal `n_W` will be set with a wall boundary condition.
   @pre n_W.norm() > 1e-10.
   @throw std::exception if body_id >= num_bodies(). */
  void SetWallBoundaryCondition(SoftBodyIndex body_id, const Vector3<T>& p_WQ,
                                const Vector3<T>& n_W,
                                double distance_tolerance = 1e-6);

  double dt() const { return dt_; }

  const systems::OutputPort<T>& get_vertex_positions_output_port() const {
    return systems::System<T>::get_output_port(vertex_positions_port_);
  }

  /** Returns the number of deformable bodies in the %SoftsimSystem. */
  int num_bodies() const { return meshes_.size(); }

  /** The volume meshes of the deformable bodies.
   The meshes have the same order as the registration of their corresponding
   deformable bodies. */
  const std::vector<geometry::VolumeMesh<T>>& meshes() const { return meshes_; }

  /** The names of all the registered bodies in the same order as the bodies
   were registered. */
  const std::vector<std::string>& names() const { return names_; }

 private:
  friend class SoftsimSystemTest;

  /* Register a deformable body with the given type of constitutive model.
   @tparam Model    The type of constitutive model for the new deformable body,
   must be derived from ConstitutiveModel. */
  template <template <class, int> class Model>
  void RegisterDeformableBodyHelper(const geometry::VolumeMesh<T>& mesh,
                                    std::string name,
                                    const DeformableBodyConfig<T>& config);

  /* Advance the dynamics of all registered bodies by one time step and store
   the states at the new time step in the given `next_states`. */
  void AdvanceOneTimeStep(const systems::Context<T>& context,
                          systems::DiscreteValues<T>* next_states) const;

  /* Copies the generalized positions of each body to the given `output`.
   The order of the body positions follows that in which the bodies were
   added. */
  void CopyVertexPositionsOut(const systems::Context<T>& context,
                              std::vector<VectorX<T>>* output) const;

  /* Implements SoftsimBase::BuildRigidGeometryRepresentations(). */
  void RegisterCollisionObject(
      geometry::GeometryId geometry_id, const geometry::Shape& shape,
      const geometry::ProximityProperties& properties) final;

  /* Updates the positions of the vertices of the deformable mesh associated
   with the deformable body with index `id` given the generalized positions of
   the deforamble body. */
  void UpdateMesh(SoftBodyIndex id, const VectorX<T>& q) const {
    const auto p_WVs = Eigen::Map<const Matrix3X<T>>(q.data(), 3, q.size() / 3);
    std::vector<geometry::VolumeElement> tets = meshes_[id].tetrahedra();
    // We assume the deformable body frame is always the same as the world frame
    // for now. It probably makes sense to have the frame move with the body in
    // the future.
    std::vector<geometry::VolumeVertex<T>> vertices_D;
    for (int i = 0; i < p_WVs.cols(); ++i) {
      vertices_D.push_back(geometry::VolumeVertex<T>(p_WVs.col(i)));
    }
    meshes_[id] = {std::move(tets), std::move(vertices_D)};
  }

  /* Updates the world pose of all rigid collision geometries registered in
   `this` SoftsimSystem. */
  void UpdatePoseForAllCollisionObjects(const systems::Context<T>& context);

  void AssembleContactSolverData(const systems::Context<T>& context0,
                                 const VectorX<T>& v0, const MatrixX<T>& M0,
                                 VectorX<T>&& minus_tau, VectorX<T>&& phi0,
                                 const MatrixX<T>& contact_jacobians,
                                 VectorX<T>&& stiffness, VectorX<T>&& damping,
                                 VectorX<T>&& mu) final;

  /* Calculates the ContactSolverData associated with the contacts between all
  rigid bodies and the deformable body with `deformable_id`.
  @param context                          The system state at the time the
                                          contact solver data is evaluated.
  @param deformable_id                    The SoftBodyIndex of the deformable
                                          body of interest.
  @param deformable_dof_offset            The starting index of the dofs of the
                                          deformable body of interest.
  @param num_total_dofs                   The total number of dofs, equal to the
                                          sum of the number of rigid dofs plus
                                          the number of deformable dofs. */
  internal::ContactSolverData<T> CalcContactSolverData(
      const systems::Context<T>& context, SoftBodyIndex deformable_id,
      int deformable_dof_offset, int num_total_dofs) const;

  /* Calculates the DeformableRigidContactData with the contacts between the
   rigid geometry with `rigid_id` and the deformable body with
   `deformable_id`. */
  internal::DeformableRigidContactData<T> CalcDeformableRigidContactData(
      geometry::GeometryId rigid_id, SoftBodyIndex deformable_id) const;

  /* Uses the `contact_data` to construct the contact jacobian entries with
   respect to the rigid dofs and then offset the row indices of the entries by
   `row_offset` and append the entries to `contact_jacobian_triplets`. */
  void AppendContactJacobianRigid(
      const systems::Context<T>& context,
      const internal::DeformableRigidContactData<T>& contact_data,
      int row_offset,
      std::vector<Eigen::Triplet<T>>* contact_jacobian_triplets) const;

  /* Uses the `contact_data` to construct the contact jacobian entries with
   respect to the deformable dofs and then offset the row/column indices of the
   entries by `row_offset`/`col_offset` and append the entries to
   `contact_jacobian_triplets`. */
  void AppendContactJacobianDeformable(
      const internal::DeformableRigidContactData<T>& contact_data,
      int row_offset, int col_offset,
      std::vector<Eigen::Triplet<T>>* contact_jacobian_triplets) const;

  void SolveContactProblem(
      const contact_solvers::internal::ContactSolver<T>& contact_solver,
      contact_solvers::internal::ContactSolverResults<T>* results) const final;

  double dt_{0};
  const Vector3<T> gravity_{0, 0, -9.81};
  /* Scratch space for the time n and time n+1 FEM states to avoid repeated
   allocation. */
  mutable std::vector<std::unique_ptr<FemStateBase<T>>> prev_fem_states_{};
  mutable std::vector<std::unique_ptr<FemStateBase<T>>> next_fem_states_{};
  /* Volume meshes for all deformablebodies. */
  mutable std::vector<geometry::VolumeMesh<T>> meshes_{};
  std::vector<geometry::ProximityProperties> deformable_proximity_properties_{};
  /* Solvers for all bodies. */
  std::vector<std::unique_ptr<FemSolver<T>>> fem_solvers_{};
  /* Names of all registered bodies. */
  std::vector<std::string> names_{};
  /* Port Indexes. */
  systems::OutputPortIndex vertex_positions_port_;
  /* All rigid collision objects used in rigid-deformable contact. */
  internal::CollisionObjects<T> collision_objects_;
  /* Contact solver data for both rigid-rigid contacts and rigid-deformable
   contacts. */
  internal::ContactSolverData<T> contact_solver_data_;
};
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fixed_fem::SoftsimSystem);
