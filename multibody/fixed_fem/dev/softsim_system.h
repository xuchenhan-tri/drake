#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_frame.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/contact_solvers/contact_solver.h"
#include "drake/multibody/fixed_fem/dev/contact_data_calculator.h"
#include "drake/multibody/fixed_fem/dev/deformable_body_config.h"
#include "drake/multibody/fixed_fem/dev/dirichlet_boundary_condition.h"
#include "drake/multibody/fixed_fem/dev/dynamic_elasticity_element.h"
#include "drake/multibody/fixed_fem/dev/dynamic_elasticity_model.h"
#include "drake/multibody/fixed_fem/dev/dynamics_data_calculator.h"
#include "drake/multibody/fixed_fem/dev/fem_solver.h"
#include "drake/multibody/fixed_fem/dev/linear_simplex_element.h"
#include "drake/multibody/fixed_fem/dev/simplex_gaussian_quadrature.h"
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
 initial positions can be queried via `meshes()` which returns an
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
class SoftsimSystem final : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SoftsimSystem)

  /** (Advanced) Construct a %SoftsimSystem with the fixed prescribed discrete
   time step. No SceneGraph pointer is provided so collision objects cannot be
   visualized.
   @pre dt > 0. */
  explicit SoftsimSystem(double dt);

  /** Constructs a %SoftsimSystem with the fixed prescribed discrete time step
   and the given SceneGraph pointer.
   @pre dt > 0. */
  SoftsimSystem(double dt, geometry::SceneGraph<T>* scene_graph);

  // TODO(xuchenhan-tri): Identify deformable bodies with actual identifiers,
  //  which would make deleting deformable bodies easier to track in the future.
  /** Adds a deformable body modeled with linear simplex element, linear
   quadrature rule, mid-point integration rule and the given `config`. Returns
   the index of the newly added deformable body.
   @pre `name` is distinct from names of all previously registered bodies.
   @pre config.IsValid() == true. */
  SoftBodyIndex RegisterDeformableBody(const geometry::VolumeMesh<T>& mesh,
                                       std::string name,
                                       const DeformableBodyConfig<T>& config);

  /** Set zero Dirichlet boundary conditions for a given body. All vertices in
   the mesh of corresponding to the deformable body with `body_id` within
   `distance_tolerance` (measured in meters) from the halfspace defined by point
   `p_WQ` and outward normal `n_W` will be set with a wall boundary condition.
   @pre n_W.norm() > 1e-10.
   @throw std::exception if body_id >= num_bodies(). */
  void SetWallBoundaryCondition(SoftBodyIndex body_id, const Vector3<T>& p_WQ,
                                const Vector3<T>& n_W,
                                double distance_tolerance = 1e-6);

  /** Add a collision geometry with the given `shape`, `proximity_properties`
   and pose `X_WG`. */
  template <typename ShapeType>
  void RegisterCollisionGeometry(
      const ShapeType& shape,
      const geometry::ProximityProperties& proximity_properties,
      const std::function<void(const T&, math::RigidTransform<T>*,
                               SpatialVelocity<T>*)>& motion_update_callback) {
    static_assert(std::is_base_of_v<geometry::Shape, ShapeType>,
                  "The template parameter 'ShapeType' must be derived from "
                  "'geometry::Shape'.");
    std::optional<geometry::internal::hydroelastic::RigidGeometry>
        rigid_geometry =
            geometry::internal::hydroelastic::MakeRigidRepresentation(
                shape, proximity_properties);
    DRAKE_THROW_UNLESS(rigid_geometry.has_value());
    geometry::SurfaceMesh<double> surface_mesh = rigid_geometry.value().mesh();
    collision_objects_.emplace_back(surface_mesh, proximity_properties,
                                    motion_update_callback);
    if (scene_graph_ != nullptr) {
      frame_ids_.emplace_back(scene_graph_->RegisterFrame(
          source_id_,
          geometry::GeometryFrame("collision object" +
                                  std::to_string(collision_objects_.size()))));
      const geometry::GeometryId id = scene_graph_->RegisterGeometry(
          source_id_, frame_ids_.back(),
          std::make_unique<geometry::GeometryInstance>(
              math::RigidTransformd::Identity(), shape.Clone(),
              "collision object"));
      scene_graph_->AssignRole(source_id_, id,
                               geometry::MakePhongIllustrationProperties(
                                   Vector4<double>(0, 1, 1, 1)));
    }
  }

  geometry::SourceId source_id() const { return source_id_; }

  double dt() const { return dt_; }

  const systems::OutputPort<T>& get_vertex_positions_output_port() const {
    return systems::System<T>::get_output_port(vertex_positions_port_);
  }

  const systems::OutputPort<T>& get_collision_objects_pose_output_port() const {
    return systems::System<T>::get_output_port(collision_objects_pose_port_);
  }

  /** Returns the number of deformable bodies in the %SoftsimSystem. */
  int num_bodies() const { return meshes_.size(); }

  /** The volume meshes of the deformable bodies. The meshes have the same order
   as the registration of their corresponding deformable bodies. */
  const std::vector<geometry::VolumeMesh<T>>& meshes() const { return meshes_; }

  /** The names of all the registered bodies in the same order as the bodies
   were registered. */
  const std::vector<std::string>& names() const { return names_; }

 private:
  friend class SoftsimSystemTest;

  /* Updates the poses and velocities of all collision objects given the current
   time. */
  void UpdateAllCollisionObjects(const T& time) const {
    for (auto& object : collision_objects_) {
      object.UpdatePositionAndVelocity(time);
    }
  }

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

  void UpdateMesh(SoftBodyIndex id, const VectorX<T>& q) const {
    const auto q_reshaped =
        Eigen::Map<const Matrix3X<T>>(q.data(), 3, q.size() / 3);
    std::vector<geometry::VolumeElement> tets = meshes_[id].tetrahedra();
    // We assume the deformable body frame is always the same as the world frame
    // for now. It probably makes sense to have the frame move with the body in
    // the future.
    std::vector<geometry::VolumeVertex<T>> vertices_D;
    for (int i = 0; i < q_reshaped.cols(); ++i) {
      vertices_D.push_back(geometry::VolumeVertex<T>(q_reshaped.col(i)));
    }
    meshes_[id] = {std::move(tets), std::move(vertices_D)};
  }

  /* Copies the generalized positions of each body to the given `output`.
   The order of the body positions follows that in which the bodies were
   added. */
  void CopyVertexPositionsOut(const systems::Context<T>& context,
                              std::vector<VectorX<T>>* output) const;

  /* Copies the poses of each collision object to the given output `poses`.
   The order of the poses follows that in which the bodies were added. */
  void OutputCollisionObjectsPose(const systems::Context<T>& context,
                                  geometry::FramePoseVector<T>* poses) const {
    unused(context);
    poses->clear();
    // Set the frames to the poses of the collision objects.
    for (int i = 0; i < static_cast<int>(collision_objects_.size()); ++i) {
      poses->set_value(frame_ids_[i], collision_objects_[i].pose());
    }
  }

  double dt_{0};
  const Vector3<T> gravity_{0, 0, -9.81};
  /* Scratch space for the time n and time n+1 FEM states to avoid repeated
   allocation. */
  mutable std::vector<std::unique_ptr<FemStateBase<T>>> prev_fem_states_{};
  mutable std::vector<std::unique_ptr<FemStateBase<T>>> next_fem_states_{};
  // TODO(xuchenhan-tri): The meshes should be managed by scene graph when
  //  Softsim is integrated with SceneGraph.
  /* The volume tetmesh for all bodies. */
  mutable std::vector<geometry::VolumeMesh<T>> meshes_{};
  mutable std::vector<CollisionObject<T>> collision_objects_{};
  mutable internal::ContactDataCalculator<T> contact_data_calculator_{};
  mutable std::vector<std::unique_ptr<internal::DynamicsDataCalculator<T>>>
      dynamics_data_calculators_{};
  /* Solvers for all bodies. */
  std::vector<std::unique_ptr<FemSolver<T>>> fem_solvers_{};
  /* Names of all registered bodies. */
  std::vector<std::string> names_{};
  std::unique_ptr<contact_solvers::internal::ContactSolver<T>> contact_solver_;
  /* SceneGraph pointer. Used to visualize collision objects only. */
  geometry::SceneGraph<T>* scene_graph_;
  /* Geometry source identifier for SoftsimSystem to interact with SceneGraph.
   */
  geometry::SourceId source_id_{};
  std::vector<geometry::FrameId> frame_ids_{};
  /* Port Indexes. */
  systems::OutputPortIndex vertex_positions_port_;
  systems::OutputPortIndex collision_objects_pose_port_;
};
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fixed_fem::SoftsimSystem);
