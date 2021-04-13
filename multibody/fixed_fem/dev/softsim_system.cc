#include "drake/multibody/fixed_fem/dev/softsim_system.h"

#include "drake/geometry/query_object.h"
#include "drake/multibody/fixed_fem/dev/corotated_model.h"
#include "drake/multibody/fixed_fem/dev/deformable_contact.h"
#include "drake/multibody/fixed_fem/dev/linear_constitutive_model.h"
#include "drake/multibody/plant/coulomb_friction.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
template <typename T>
SoftsimSystem<T>::SoftsimSystem(MultibodyPlant<T>* mbp)
    : multibody::internal::DeformableSolverBase<T>(mbp) {
  DRAKE_DEMAND(this->dt() > 0);
}

template <typename T>
SoftBodyIndex SoftsimSystem<T>::RegisterDeformableBody(
    const geometry::VolumeMesh<T>& mesh, std::string name,
    const DeformableBodyConfig<T>& config,
    geometry::ProximityProperties properties) {
  DRAKE_THROW_UNLESS(properties.HasProperty(geometry::internal::kMaterialGroup,
                                            geometry::internal::kFriction));
  /* Throw if name is not unique. */
  for (int i = 0; i < num_bodies(); ++i) {
    if (name == names_[i]) {
      throw std::runtime_error(fmt::format(
          "{}(): A body with name '{}' already exists in the system.", __func__,
          name));
    }
  }
  SoftBodyIndex body_index(num_bodies());
  switch (config.material_model()) {
    case MaterialModel::kLinear:
      RegisterDeformableBodyHelper<LinearConstitutiveModel>(
          mesh, std::move(name), config);
      break;
    case MaterialModel::kCorotated:
      RegisterDeformableBodyHelper<CorotatedModel>(mesh, std::move(name),
                                                   config);
      break;
  }
  const auto& deformable_state = next_fem_states_.back();
  this->DeclareDeformableState(deformable_state->q(), deformable_state->qdot(),
                               deformable_state->qddot());
  deformable_proximity_properties_.emplace_back(std::move(properties));
  return body_index;
}

template <typename T>
void SoftsimSystem<T>::SetWallBoundaryCondition(SoftBodyIndex body_id,
                                                const Vector3<T>& p_WQ,
                                                const Vector3<T>& n_W,
                                                double distance_tolerance) {
  DRAKE_DEMAND(n_W.norm() > 1e-10);
  const Vector3<T>& n_hatW = n_W.normalized();
  DRAKE_THROW_UNLESS(body_id < num_bodies());
  const int kDim = 3;
  FemSolver<T>& fem_solver = *fem_solvers_[body_id];
  FemModelBase<T>& fem_model = fem_solver.mutable_model();
  const int num_nodes = fem_model.num_nodes();
  // TODO(xuchenhan-tri): FemModel should support an easier way to retrieve its
  //  reference positions.
  const std::unique_ptr<FemStateBase<T>> fem_state =
      fem_model.MakeFemStateBase();
  const VectorX<T>& initial_positions = fem_state->q();
  auto bc = std::make_unique<DirichletBoundaryCondition<T>>(/* ODE order */ 2);
  for (int n = 0; n < num_nodes; ++n) {
    const Vector3<T>& p_WV = initial_positions.template segment<kDim>(n * kDim);
    const T distance_to_wall = (p_WV - p_WQ).dot(n_hatW);
    if (distance_to_wall * distance_to_wall <
        distance_tolerance * distance_tolerance) {
      const int dof_index(kDim * n);
      for (int d = 0; d < kDim; ++d) {
        bc->AddBoundaryCondition(DofIndex(dof_index + d),
                                 Vector3<T>(p_WV(d), 0, 0));
      }
    }
  }
  fem_model.SetDirichletBoundaryCondition(std::move(bc));
}

template <typename T>
template <template <class, int> class Model>
void SoftsimSystem<T>::RegisterDeformableBodyHelper(
    const geometry::VolumeMesh<T>& mesh, std::string name,
    const DeformableBodyConfig<T>& config) {
  constexpr int kNaturalDimension = 3;
  constexpr int kSpatialDimension = 3;
  constexpr int kQuadratureOrder = 1;
  using QuadratureType =
      SimplexGaussianQuadrature<kNaturalDimension, kQuadratureOrder>;
  constexpr int kNumQuads = QuadratureType::num_quadrature_points();
  using IsoparametricElementType =
      LinearSimplexElement<T, kNaturalDimension, kSpatialDimension, kNumQuads>;
  using ConstitutiveModelType = Model<T, kNumQuads>;
  static_assert(std::is_base_of_v<
                    ConstitutiveModel<ConstitutiveModelType,
                                      typename ConstitutiveModelType::Traits>,
                    ConstitutiveModelType>,
                "The template parameter 'Model' must be derived from "
                "ConstitutiveModel.");
  using ElementType =
      DynamicElasticityElement<IsoparametricElementType, QuadratureType,
                               ConstitutiveModelType>;
  using FemModelType = DynamicElasticityModel<ElementType>;
  using StateType = FemState<ElementType>;

  const DampingModel<T> damping_model(config.mass_damping_coefficient(),
                                      config.stiffness_damping_coefficient());
  auto fem_model = std::make_unique<FemModelType>(this->dt());
  ConstitutiveModelType constitutive_model(config.youngs_modulus(),
                                           config.poisson_ratio());
  fem_model->AddDynamicElasticityElementsFromTetMesh(
      mesh, constitutive_model, config.mass_density(), damping_model);
  fem_model->SetGravity(this->gravity());
  const StateType state = fem_model->MakeFemState();

  prev_fem_states_.emplace_back(std::make_unique<StateType>(state));
  next_fem_states_.emplace_back(std::make_unique<StateType>(state));
  fem_solvers_.emplace_back(
      std::make_unique<FemSolver<T>>(std::move(fem_model)));
  meshes_.emplace_back(mesh);
  names_.emplace_back(std::move(name));
}

// TODO(xuchenhan-tri): This function should be changed to only advance free
//  positions/velocities of deformable objects.
template <typename T>
VectorX<T> SoftsimSystem<T>::CalcFreeMotion(const VectorX<T>& state0,
                                            int deformable_body_index) const {
  DRAKE_DEMAND(0 <= deformable_body_index &&
               deformable_body_index < num_bodies());
  const int num_dofs = state0.size() / 3;
  const auto& q = state0.head(num_dofs);
  const auto& qdot = state0.segment(num_dofs, num_dofs);
  const auto& qddot = state0.tail(num_dofs);
  /* Set up FemState and advance to the next time step. */
  FemStateBase<T>& prev_fem_state = *prev_fem_states_[deformable_body_index];
  FemStateBase<T>& next_fem_state = *next_fem_states_[deformable_body_index];
  prev_fem_state.SetQ(q);
  prev_fem_state.SetQdot(qdot);
  prev_fem_state.SetQddot(qddot);
  // TODO(xuchenhan-tri): FemState needs a SetFrom() method. Setting
  //  DiscreteValues from FemStateBase (and vice-versa) should also be made
  //  more compact.
  next_fem_state.SetQ(q);
  next_fem_state.SetQdot(qdot);
  next_fem_state.SetQddot(qddot);
  fem_solvers_[deformable_body_index]->AdvanceOneTimeStep(prev_fem_state,
                                                          &next_fem_state);
  /* Copy new state to output variable. */
  VectorX<T> next_state(3 * num_dofs);
  next_state.head(num_dofs) = next_fem_state.q();
  next_state.segment(num_dofs, num_dofs) = next_fem_state.qdot();
  next_state.tail(num_dofs) = next_fem_state.qddot();
  return next_state;
}

template <typename T>
void SoftsimSystem<T>::RegisterCollisionObject(
    geometry::GeometryId geometry_id, const geometry::Shape& shape,
    const geometry::ProximityProperties& properties) {
  std::cout << "Registering collision objects" << std::endl;
  collision_objects_.AddCollisionObject(geometry_id, shape, properties);
}

template <typename T>
void SoftsimSystem<T>::UpdatePoseForAllCollisionObjects(
    const systems::Context<T>& context) {
  const MultibodyPlant<T>& mbp = this->multibody_plant();
  const geometry::QueryObject<T>& query_object =
      mbp.get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
  const std::vector<geometry::GeometryId>& geometry_ids =
      collision_objects_.geometry_ids();
  for (geometry::GeometryId id : geometry_ids) {
    const math::RigidTransform<T>& pose = query_object.GetPoseInWorld(id);
    collision_objects_.UpdatePoseInWorld(id, pose);
  }
}

template <typename T>
void SoftsimSystem<T>::AssembleContactSolverData(
    const systems::Context<T>& context0, const VectorX<T>& v0,
    const MatrixX<T>& M0, VectorX<T>&& minus_tau, VectorX<T>&& phi0,
    const MatrixX<T>& contact_jacobian, VectorX<T>&& stiffness,
    VectorX<T>&& damping, VectorX<T>&& mu) {
  const int num_rigid_dofs = v0.size();
  const int num_rigid_contacts = phi0.size();
  DRAKE_DEMAND(num_rigid_dofs == minus_tau.size());
  DRAKE_DEMAND(num_rigid_contacts == stiffness.size());
  DRAKE_DEMAND(num_rigid_contacts == damping.size());
  DRAKE_DEMAND(num_rigid_contacts == mu.size());
  DRAKE_DEMAND(3 * num_rigid_contacts == contact_jacobian.rows());
  DRAKE_DEMAND(num_rigid_dofs == contact_jacobian.cols());
  DRAKE_DEMAND(num_rigid_dofs == M0.cols());
  DRAKE_DEMAND(num_rigid_dofs == M0.rows());

  // Make sure the pose for the collision objects are up-to-date.
  UpdatePoseForAllCollisionObjects(context0);

  // Get the positions offsets for the deformable dofs.
  // TODO(xuchenhan-tri): This should be precomputes.
  int num_deformable_dofs = 0;
  std::vector<int> deformable_dof_offsets(next_fem_states_.size());
  for (int i = 0; i < num_bodies(); ++i) {
    deformable_dof_offsets[i] = num_deformable_dofs + num_rigid_dofs;
    num_deformable_dofs += next_fem_states_[i]->num_generalized_positions();
  }
  const int num_total_dofs = num_rigid_dofs + num_deformable_dofs;

  internal::PointContactDataStorage<T> point_contact_data_storage(
      num_total_dofs);
  // Append rigid-rigid point contact data.
  point_contact_data_storage.AppendData(std::move(phi0), std::move(stiffness),
                                        std::move(damping), std::move(mu),
                                        contact_jacobian);
  // Append rigid-deformable point contact data.
  for (SoftBodyIndex i(0); i < num_bodies(); ++i) {
    AppendPointContactData(context0, i, deformable_dof_offsets[i],
                           &point_contact_data_storage);
  }
}

template <typename T>
void SoftsimSystem<T>::AppendPointContactData(
    const systems::Context<T>& context, SoftBodyIndex deformable_id,
    int deformable_dof_offset,
    internal::PointContactDataStorage<T>* point_contact_data_storage) const {
  const std::vector<geometry::GeometryId>& rigid_ids =
      collision_objects_.geometry_ids();
  // Number of contacts between all rigid bodies and the deformable body`.
  int nc = 0;
  std::vector<internal::DeformableRigidContactData<T>> contact_data;
  contact_data.reserve(rigid_ids.size());
  std::vector<int> contact_id_offsets(rigid_ids.size());
  for (int i = 0; i < static_cast<int>(rigid_ids.size()); ++i) {
    contact_data.emplace_back(
        CalcDeformableRigidContactData(rigid_ids[i], deformable_id));
    contact_id_offsets[i] = nc;
    nc += contact_data.back().contact_surface.num_polygons();
  }

  // TODO(xuchenhan-tri): Set penetration data properly. This requires changes
  // to DeformableContactSurface and ComputeTetMeshTriMeshContact() to support
  // calculation of penetration depth. Currently, phi0 is set to zero because
  // PGS does not use it.
  std::vector<T> phi0(nc, 0);
  std::vector<T> stiffness(nc);
  std::vector<T> dissipation(nc);
  std::vector<T> friction(nc);
  std::vector<Eigen::Triplet<T>> contact_jacobian_triplets;

  for (int i = 0; i < static_cast<int>(contact_data.size()); ++i) {
    const internal::DeformableRigidContactData<T>& data_i = contact_data[i];
    // The number of contacts in data_i.
    const int nci = data_i.contact_surface.num_polygons();

    std::fill(stiffness.begin() + contact_id_offsets[i],
              stiffness.begin() + contact_id_offsets[i] + nci,
              data_i.stiffness);
    std::fill(dissipation.begin() + contact_id_offsets[i],
              dissipation.begin() + contact_id_offsets[i] + nci,
              data_i.dissipation);
    std::fill(friction.begin() + contact_id_offsets[i],
              friction.begin() + contact_id_offsets[i] + nci, data_i.friction);

    AppendContactJacobianRigid(context, data_i, 3 * contact_id_offsets[i],
                               &contact_jacobian_triplets);
    AppendContactJacobianDeformable(data_i, 3 * contact_id_offsets[i],
                                    deformable_dof_offset,
                                    &contact_jacobian_triplets);
  }
  point_contact_data_storage->AppendData(
      std::move(phi0), std::move(stiffness), std::move(dissipation),
      std::move(friction), std::move(contact_jacobian_triplets));
}

template <typename T>
internal::DeformableRigidContactData<T>
SoftsimSystem<T>::CalcDeformableRigidContactData(
    geometry::GeometryId rigid_id, SoftBodyIndex deformable_id) const {
  DeformableContactSurface<T> contact_surface = ComputeTetMeshTriMeshContact(
      meshes_[deformable_id], collision_objects_.mesh(rigid_id),
      collision_objects_.pose(rigid_id));

  const auto get_point_contact_parameters =
      [&](const geometry::ProximityProperties& props) -> std::pair<T, T> {
    return std::make_pair(props.template GetPropertyOrDefault<T>(
                              geometry::internal::kMaterialGroup,
                              geometry::internal::kPointStiffness,
                              this->default_contact_stiffness()),
                          props.template GetPropertyOrDefault<T>(
                              geometry::internal::kMaterialGroup,
                              geometry::internal::kHcDissipation,
                              this->default_contact_dissipation()));
  };
  // Extract the stiffness, dissipation and friction parameters of the
  // deformable body.
  const geometry::ProximityProperties& deformable_props =
      deformable_proximity_properties_[deformable_id];
  const auto [deformable_stiffness, deformable_dissipation] =
      get_point_contact_parameters(deformable_props);
  DRAKE_THROW_UNLESS(deformable_props.HasProperty(
      geometry::internal::kMaterialGroup, geometry::internal::kFriction));
  const CoulombFriction<double> deformable_mu =
      deformable_props.GetProperty<CoulombFriction<double>>(
          geometry::internal::kMaterialGroup, geometry::internal::kFriction);

  // Extract the stiffness, dissipation and friction parameters of the rigid
  // body.
  const auto& rigid_proximity_properties =
      collision_objects_.proximity_properties(rigid_id);
  const auto [rigid_stiffness, rigid_dissipation] =
      get_point_contact_parameters(rigid_proximity_properties);
  const CoulombFriction<double> rigid_mu =
      rigid_proximity_properties.template GetProperty<CoulombFriction<double>>(
          geometry::internal::kMaterialGroup, geometry::internal::kFriction);

  // Combine the stiffness, dissipation and friction parameters for the
  // contact points.
  auto [k, d] =
      CombinePointContactParameters(deformable_stiffness, rigid_stiffness,
                                    deformable_dissipation, rigid_dissipation);
  const CoulombFriction<double> mu =
      CalcContactFrictionFromSurfaceProperties(deformable_mu, rigid_mu);
  return internal::DeformableRigidContactData<T>(
      std::move(contact_surface), rigid_id, deformable_id, std::move(k),
      std::move(d), mu.dynamic_friction());
}

template <typename T>
void SoftsimSystem<T>::AppendContactJacobianRigid(
    const systems::Context<T>& context,
    const internal::DeformableRigidContactData<T>& contact_data, int row_offset,
    std::vector<Eigen::Triplet<T>>* contact_jacobian_triplets) const {
  const DeformableContactSurface<T>& contact_surface =
      contact_data.contact_surface;
  for (int i = 0; i < contact_surface.num_polygons(); ++i) {
    // For point Rc (origin of rigid body's frame R shifted to the contact
    // point C), calculate Jv_v_WRc (Rc's translational velocity Jacobian
    // in the world frame W with respect to generalized velocities v).
    Matrix3X<T> Jv_v_WRc_dense(3, this->multibody_plant().num_velocities());
    /* The position of the contact point in the world frame. */
    const Vector3<T>& p_WC = contact_surface.polygon_data(i).centroid;
    this->CalcJacobianTranslationVelocity(context, p_WC, contact_data.rigid_id,
                                          &Jv_v_WRc_dense);
    /* The same Jacobian expressed in the contact frame. */
    const Eigen::SparseMatrix<T> Jv_v_WRc_C =
        (-contact_data.R_CWs[i] * Jv_v_WRc_dense).sparseView();
    std::vector<Eigen::Triplet<T>> triplets =
        internal::ConvertEigenSparseMatrixToTripletsWithOffsets(
            Jv_v_WRc_C, row_offset + 3 * i, 0);
    contact_jacobian_triplets->insert(contact_jacobian_triplets->end(),
                                      triplets.begin(), triplets.end());
  }
}

template <typename T>
void SoftsimSystem<T>::AppendContactJacobianDeformable(
    const internal::DeformableRigidContactData<T>& contact_data, int row_offset,
    int col_offset,
    std::vector<Eigen::Triplet<T>>* contact_jacobian_triplets) const {
  const DeformableContactSurface<T>& contact_surface =
      contact_data.contact_surface;
  for (int i = 0; i < contact_surface.num_polygons(); ++i) {
    const ContactPolygonData<T>& polygon_data = contact_surface.polygon_data(i);
    /* The contribution to the contact velocity from the deformable object
     A is R_CW * v_WAq. Note
       v_WAq = b₀ * v_WVᵢ₀ + b₁ * v_WVᵢ₁ + b₂ * v_WVᵢ₂ + b₃ * v_WVᵢ₃,
     where bₗ is the barycentric weight corresponding to vertex kₗ and
     v_WVₖₗ is the velocity of that vertex. */
    const Vector4<T>& barycentric_weights = polygon_data.b_centroid;
    const geometry::VolumeElement tet_element =
        meshes_[contact_data.deformable_id].element(polygon_data.tet_index);
    for (int l = 0; l < 4; ++l) {
      const int col = col_offset + 3 * tet_element.vertex(l);
      internal::AddMatrix3ToEigenTriplets<T>(
          contact_data.R_CWs[i] * barycentric_weights(l), row_offset + 3 * i,
          col, contact_jacobian_triplets);
    }
  }
}

template <typename T>
void SoftsimSystem<T>::SolveContactProblem(
    const contact_solvers::internal::ContactSolver<T>& contact_solver,
    contact_solvers::internal::ContactSolverResults<T>* results) const {
  unused(contact_solver);
  unused(results);
}
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fixed_fem::SoftsimSystem);
