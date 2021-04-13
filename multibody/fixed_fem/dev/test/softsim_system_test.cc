#include "drake/multibody/fixed_fem/dev/softsim_system.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#include "drake/multibody/fixed_fem/dev/deformable_rigid_contact_data.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
// TODO(xuchenhan-tri): Add a test to verify that the deformable body parameters
//  are properly passed to the FemModel.
/* Deformable body parameters. These parameters (with the exception of
 kMassDamping) are dummy in the sense that they do not affect the result of
 the test as long as they are valid. */
const double kYoungsModulus = 1.23;
const double kPoissonRatio = 0.456;
const double kDensity = 0.789;
/* The mass damping coefficient is set to zero so that the free fall test
 (CalcFreeMotion) produces an easy analytical solution. */
const double kMassDamping = 0.0;
const double kStiffnessDamping = 0.02;
/* Time step. */
const double kDt = 0.0123;
/* Number of vertices in the box mesh (see below). */
constexpr int kNumVertices = 8;
const double kContactStiffness = 4.21;
const double kContactDissipation = 1.24;
const double kFrictionStatic = 0.32;
const double kFrictionDynamic = 0.31;
const CoulombFriction<double> kFriction(kFrictionStatic, kFrictionDynamic);

using contact_solvers::internal::ExtractNormal;
using contact_solvers::internal::ExtractTangent;

class SoftsimSystemTest : public ::testing::Test {
 protected:
  /* Make a box and subdivide it into 6 tetrahedra. */
  static geometry::VolumeMesh<double> MakeBoxTetMesh() {
    const double length = 1;
    geometry::Box box(length, length, length);
    geometry::VolumeMesh<double> mesh =
        geometry::internal::MakeBoxVolumeMesh<double>(box, length);
    DRAKE_DEMAND(mesh.num_elements() == 6);
    DRAKE_DEMAND(mesh.num_vertices() == kNumVertices);
    return mesh;
  }

  /* Creates a dummy velocity vector that has size equal to the number of dofs
   in the box tet mesh. */
  static VectorX<double> dummy_velocity() {
    VectorX<double> v(3 * kNumVertices);
    for (int i = 0; i < v.size(); ++i) {
      v(i) = 0.1 * i;
    }
    return v;
  }

  /* Forwards the call of AppendContactJacobianDeformable() to the SoftsimSystem
   under test. */
  void AppendContactJacobianDeformable(
      const internal::DeformableRigidContactData<double>& contact_data,
      int row_offset, int col_offset,
      std::vector<Eigen::Triplet<double>>* contact_jacobian_triplets) const {
    softsim_system_.AppendContactJacobianDeformable(
        contact_data, row_offset, col_offset, contact_jacobian_triplets);
  }

  /* Forwards the call of AppendContactJacobianRigid() to the SoftsimSystem
   under test. */
  void AppendContactJacobianRigid(
      const systems::Context<double>& context,
      const internal::DeformableRigidContactData<double>& contact_data,
      int row_offset,
      std::vector<Eigen::Triplet<double>>* contact_jacobian_triplets) const {
    softsim_system_.AppendContactJacobianRigid(
        context, contact_data, row_offset, contact_jacobian_triplets);
  }

  /* Forwards the call of AppendPointContactData() to the SoftsimSystem
   under test. */
  void AppendPointContactData(const systems::Context<double>& context,
                              SoftBodyIndex deformable_id,
                              int deformable_dof_offset,
                              internal::PointContactDataStorage<double>*
                                  point_contact_data_storage) const {
    softsim_system_.AppendPointContactData(context, deformable_id,
                                           deformable_dof_offset,
                                           point_contact_data_storage);
  }

  /* Forwards the call of UpdatePoseForAllCollisionObjects() to the
   SoftsimSystem under test. */
  void UpdatePoseForAllCollisionObjects(
      const systems::Context<double>& context) {
    softsim_system_.UpdatePoseForAllCollisionObjects(context);
  }

  const internal::CollisionObjects<double>& get_collision_objects() const {
    return softsim_system_.collision_objects_;
  }

  /* Create a dummy DeformableConfig. */
  static DeformableBodyConfig<double> MakeDeformableConfig() {
    DeformableBodyConfig<double> config;
    config.set_youngs_modulus(kYoungsModulus);
    config.set_poisson_ratio(kPoissonRatio);
    config.set_mass_damping_coefficient(kMassDamping);
    config.set_stiffness_damping_coefficient(kStiffnessDamping);
    config.set_mass_density(kDensity);
    config.set_material_model(MaterialModel::kLinear);
    return config;
  }

  /* Forwards the call to CalcFreeMotion() to the SoftsimSystem under test. */
  VectorX<double> CalcFreeMotion(const VectorX<double>& state0,
                                 int deformable_body_index) const {
    return softsim_system_.CalcFreeMotion(state0, deformable_body_index);
  }

  double dt() const { return softsim_system_.dt(); }
  const Vector3<double>& gravity() const { return softsim_system_.gravity(); }

  /* Add a dummy box shaped deformable body with the given "name". */
  SoftBodyIndex AddDeformableBox(std::string name) {
    geometry::ProximityProperties dummy_proximity_props;
    geometry::AddContactMaterial(kContactStiffness, kContactDissipation,
                                 kContactStiffness, kFriction,
                                 &dummy_proximity_props);
    return softsim_system_.RegisterDeformableBody(
        MakeBoxTetMesh(), std::move(name), MakeDeformableConfig(),
        dummy_proximity_props);
  }

  /* The SoftsimSystem under test. */
  std::unique_ptr<MultibodyPlant<double>> mbp_{
      std::make_unique<MultibodyPlant<double>>(kDt)};
  SoftsimSystem<double> softsim_system_{mbp_.get()};
};

namespace {
TEST_F(SoftsimSystemTest, RegisterDeformableBody) {
  AddDeformableBox("box");
  EXPECT_EQ(softsim_system_.num_bodies(), 1);
  const std::vector<std::string>& registered_names = softsim_system_.names();
  EXPECT_EQ(registered_names.size(), 1);
  EXPECT_EQ(registered_names[0], "box");
  const std::vector<geometry::VolumeMesh<double>>& meshes =
      softsim_system_.meshes();
  EXPECT_EQ(meshes.size(), 1);
  EXPECT_TRUE(MakeBoxTetMesh().Equal(meshes[0]));
  EXPECT_EQ(dt(), kDt);
  EXPECT_TRUE(
      CompareMatrices(gravity(), mbp_.gravity_field().gravity_vector()));
}

/* Verifies that registering a deformable body returns the expected body id and
 that registering a body with an existing name throws an exception. */
TEST_F(SoftsimSystemTest, RegisterDeformableBodyUniqueNameRequirement) {
  EXPECT_EQ(AddDeformableBox("box1"), SoftBodyIndex(0));
  /* The returned body index should be the same as the number of deformable
   bodies in the system before the new one is added. */
  EXPECT_EQ(AddDeformableBox("box2"), SoftBodyIndex(1));
  EXPECT_EQ(softsim_system_.num_bodies(), 2);
  DRAKE_EXPECT_THROWS_MESSAGE(AddDeformableBox("box1"), std::exception,
                              "RegisterDeformableBody\\(\\): A body with name "
                              "'box1' already exists in the system.");
}

/* Verifies that the SoftsimSystem calculates the expected displacement for a
 deformable object under free fall over one time step. */
TEST_F(SoftsimSystemTest, CalcFreeMotion) {
  const SoftBodyIndex box_id = AddDeformableBox("box");
  const std::vector<geometry::VolumeMesh<double>>& meshes =
      softsim_system_.initial_meshes();
  const geometry::VolumeMesh<double>& box_mesh = meshes[box_id];
  const int num_dofs = kNumVertices * 3;
  VectorX<double> q0(num_dofs);
  for (geometry::VolumeVertexIndex i(0); i < box_mesh.num_vertices(); ++i) {
    q0.segment<3>(3 * i) = box_mesh.vertex(i).r_MV();
  }
  VectorX<double> x0 = VectorX<double>::Zero(3 * num_dofs);
  x0.head(num_dofs) = q0;
  const VectorX<double> x = CalcFreeMotion(x0, box_id);
  EXPECT_EQ(x.size(), 3 * num_dofs);
  const VectorX<double> q = x.head(num_dofs);
  /* The factor of 0.25 seems strange but is correct. For the default mid-point
   rule used by DynamicElasticityModel,
        x = xₙ + dt ⋅ vₙ + dt² ⋅ (0.25 ⋅ a + 0.25 ⋅ aₙ).
   In this test case vₙ and aₙ are both 0, so x - xₙ is given by
   0.25 ⋅ a ⋅ dt². */
  const Vector3<double> expected_displacement = 0.25 * kDt * kDt * gravity();
  const double kTol = 1e-14;
  for (int i = 0; i < kNumVertices; ++i) {
    const Vector3<double> displacement =
        q.segment<3>(3 * i) - q0.segment<3>(3 * i);
    EXPECT_TRUE(CompareMatrices(displacement, expected_displacement, kTol));
  }
}

/* Sets up arbitrary contact data and verify that in the absense of contribution
 from rigid body, the contact velocity is equal to the contact jacobian with
 respect to the deformable velocities times the deformable velocities. */
TEST_F(SoftsimSystemTest, AppendContactJacobianDeformable) {
  /* Creates two deformable objects to cover the case for non-zero offsets in
   the call to AppendContactJacobianDeformable(). */
  SoftBodyIndex box1 = AddDeformableBox("box1");
  SoftBodyIndex box2 = AddDeformableBox("box2");

  /* Create some fake contact data. */
  const Vector4<double> barycentric_centroid(0.1, 0.2, 0.3, 0.4);
  const Vector3<double> contact_normal1(1, 0, 0);
  const Vector3<double> contact_normal2(0, 1, 0);
  const geometry::VolumeElementIndex tet_index1(0);
  const geometry::VolumeElementIndex tet_index2(1);

  /* These data do not affect the result of the test. */
  const Vector3<double> dummy_centroid(0, 0, 0);
  const double dummy_area = 0;
  const double dummy_mu = 0;
  geometry::GeometryId dummy_rigid_id = geometry::GeometryId::get_new_id();

  /* The contact data for box1. */
  ContactPolygonData<double> polygon_data1{dummy_area, contact_normal1,
                                           dummy_centroid, barycentric_centroid,
                                           tet_index1};
  DeformableContactSurface<double> contact_surface1({polygon_data1});
  internal::DeformableRigidContactData<double> contact_data1(
      std::move(contact_surface1), dummy_rigid_id, box1, 0, 0, dummy_mu);

  /* The contact data for box2. */
  ContactPolygonData<double> polygon_data2{dummy_area, contact_normal2,
                                           dummy_centroid, barycentric_centroid,
                                           tet_index2};
  DeformableContactSurface<double> contact_surface2({polygon_data2});
  internal::DeformableRigidContactData<double> contact_data2(
      std::move(contact_surface2), dummy_rigid_id, box2, 0, 0, dummy_mu);

  /* Build the contact jacobian. */
  std::vector<Eigen::Triplet<double>> contact_jacobian_triplets;
  /* The first contact data has no offset. */
  AppendContactJacobianDeformable(contact_data1, 0, 0,
                                  &contact_jacobian_triplets);
  /* The second contact data has a row offset of 3 (there is one contact in the
   frist contact data) and a colume offset of 3 * kNumVertices (the first
   deformable body has 3 * kNumVertices dofs). */
  AppendContactJacobianDeformable(contact_data2, 3, 3 * kNumVertices,
                                  &contact_jacobian_triplets);
  Eigen::SparseMatrix<double> contact_jacobian(6, 3 * 2 * kNumVertices);
  contact_jacobian.setFromTriplets(contact_jacobian_triplets.begin(),
                                   contact_jacobian_triplets.end());

  /* Create arbitrary velocities for all the deformable dofs. */
  const VectorX<double> v1 = dummy_velocity();
  const VectorX<double> v2 = dummy_velocity();
  VectorX<double> v(v1.size() + v2.size());
  v << v1, v2;
  /* The contribution to the contact velocity from the deformable dofs. */
  const VectorX<double> contact_velocity = contact_jacobian * v;

  /* Now we calculate the expected contact velocities. */
  /* First evaluate the deformable velocity at the contact point in world from
   by interpolating from the vertices of the tet that the contact point is
   coincident. */
  const geometry::VolumeMesh<double>& mesh1 = softsim_system_.meshes()[box1];
  const geometry::VolumeMesh<double>& mesh2 = softsim_system_.meshes()[box2];
  const geometry::VolumeElement& element1 = mesh1.element(tet_index1);
  const geometry::VolumeElement& element2 = mesh2.element(tet_index2);
  const auto interpolate_velocity =
      [&](const VectorX<double>& velocities,
          const geometry::VolumeElement& element,
          const Vector4<double>& bary_weights) -> Vector3<double> {
    Vector3<double> interpolated_velocity = Vector3<double>::Zero();
    for (int i = 0; i < 4; ++i) {
      interpolated_velocity +=
          velocities.template segment<3>(3 * element.vertex(i)) *
          bary_weights(i);
    }
    return interpolated_velocity;
  };
  const Vector3<double> v_WC1 =
      interpolate_velocity(v1, element1, barycentric_centroid);
  const Vector3<double> v_WC2 =
      interpolate_velocity(v2, element2, barycentric_centroid);
  /* Then we convert the contact velocity into the contact frame. */
  const Vector3<double> v_WC1_C = contact_data1.R_CWs[0] * v_WC1;
  const Vector3<double> v_WC2_C = contact_data2.R_CWs[0] * v_WC2;

  constexpr double kTol = std::numeric_limits<double>::epsilon();
  EXPECT_TRUE(
      CompareMatrices(contact_velocity.template segment<3>(0), v_WC1_C, kTol));
  EXPECT_TRUE(
      CompareMatrices(contact_velocity.template segment<3>(3), v_WC2_C, kTol));
}

/* Sets up arbitrary contact data and verify that in the absense of contribution
 from deformable body, the contact velocity is equal to the contact jacobian
 with respect to the rigid velocities times the rigid velocities. */
TEST_F(SoftsimSystemTest, AppendContactJacobianRigid) {
  /* Creates two deformable objects to cover the case for non-zero offsets in
   the call to AppendContactJacobianRigid(). */
  SoftBodyIndex box1 = AddDeformableBox("box1");
  SoftBodyIndex box2 = AddDeformableBox("box2");

  /* Create some arbitrary contact data. */
  const Vector3<double> centroid1(0.1, 0.2, 0.3);
  const Vector3<double> centroid2(0.4, 0.5, 0.6);
  const Vector3<double> contact_normal1(1, 0, 0);
  const Vector3<double> contact_normal2(0, 1, 0);

  /* Add a rigid body in the MbP that we attach collision geometries to. */
  const RigidBody<double>& rigid_body =
      mbp_->AddRigidBody("rigid_body", SpatialInertia<double>());
  /* Even we register a geometry for collision, it is dummy in the sense that we
   do *not* use it to calculate the contact surface. Instead, we construct the
   contact surface with known quantities so that we can verify the contact
   jacobian with a known set of contact points. The sole purpose of the
   collision geometry registration is generating a rigid GeometryId that links
   to the underlying rigid body dof. */
  geometry::Box dummy_box(1, 1, 1);
  geometry::SceneGraph<double> dummy_scene_graph;
  mbp_->RegisterAsSourceForSceneGraph(&dummy_scene_graph);

  const geometry::GeometryId rigid_id = mbp_->RegisterCollisionGeometry(
      rigid_body, math::RigidTransform<double>(), dummy_box,
      "dummy_collision_geometry", CoulombFriction<double>(0, 0));
  mbp_->Finalize();
  std::unique_ptr<systems::Context<double>> context =
      mbp_->CreateDefaultContext();

  const Vector3<double> p_WB(0.84, 0.76, 0.44);
  const math::RollPitchYaw<double> rpy(Vector3<double>(0.91, 0.52, 0.23));
  const math::RigidTransform<double> X_WB(rpy, p_WB);
  mbp_->SetFreeBodyPose(context.get(), rigid_body, X_WB);
  /* Create some arbitrary spatial velocitiy for the rigid body. */
  const SpatialVelocity<double> V_WB(Vector3<double>(3.14, 1.59, 2.65),
                                     Vector3<double>(2.71, 8.28, 1.82));
  mbp_->SetFreeBodySpatialVelocity(context.get(), rigid_body, V_WB);

  /* These data do not affect the result of the test. */
  const geometry::VolumeElementIndex dummy_tet_index(0);
  const Vector4<double> dummy_barycentric_centroid(0.1, 0.2, 0.3, 0.4);
  const double dummy_area = 0;
  const double dummy_mu = 0;

  /* The contact data for box1. */
  ContactPolygonData<double> polygon_data1{
      dummy_area, contact_normal1, centroid1, dummy_barycentric_centroid,
      dummy_tet_index};
  DeformableContactSurface<double> contact_surface1({polygon_data1});
  internal::DeformableRigidContactData<double> contact_data1(
      std::move(contact_surface1), rigid_id, box1, 0, 0, dummy_mu);

  /* The contact data for box2. */
  ContactPolygonData<double> polygon_data2{
      dummy_area, contact_normal2, centroid2, dummy_barycentric_centroid,
      dummy_tet_index};
  DeformableContactSurface<double> contact_surface2({polygon_data2});
  internal::DeformableRigidContactData<double> contact_data2(
      std::move(contact_surface2), rigid_id, box2, 0, 0, dummy_mu);

  /* Build the contact jacobian. */
  std::vector<Eigen::Triplet<double>> contact_jacobian_triplets;
  /* The first contact data has no offset. */
  AppendContactJacobianRigid(*context, contact_data1, 0,
                             &contact_jacobian_triplets);
  /* The second contact data has a row offset of 3 (there is one contact in the
   frist contact data). */
  AppendContactJacobianRigid(*context, contact_data2, 3,
                             &contact_jacobian_triplets);
  Eigen::SparseMatrix<double> contact_jacobian(6, mbp_->num_velocities());
  contact_jacobian.setFromTriplets(contact_jacobian_triplets.begin(),
                                   contact_jacobian_triplets.end());
  const auto& rigid_velocities = mbp_->GetVelocities(*context);
  const VectorX<double> contact_velocity = contact_jacobian * rigid_velocities;

  /* Now we calculate the expected contact velocities. */
  /* First evaluate the velocity at the contact point in world by shifting the
   spatial velocity of the rigid body to the contact point and extracting the
   translational part. */
  const Vector3<double> v_WC1 = V_WB.Shift(centroid1 - p_WB).translational();
  const Vector3<double> v_WC2 = V_WB.Shift(centroid2 - p_WB).translational();
  /* Then we convert the contact velocity into the contact frame. */
  const Vector3<double> v_WC1_C = -contact_data1.R_CWs[0] * v_WC1;
  const Vector3<double> v_WC2_C = -contact_data2.R_CWs[0] * v_WC2;

  constexpr double kTol = std::numeric_limits<double>::epsilon();
  EXPECT_TRUE(
      CompareMatrices(contact_velocity.template segment<3>(0), v_WC1_C, kTol));
  EXPECT_TRUE(
      CompareMatrices(contact_velocity.template segment<3>(3), v_WC2_C, kTol));
}

/* Test the result of SoftsimSystem::AppendPointContactData() in a contact
 scenerio where the contact data can be calculated analytically.
 The objects in contact look like:
                  +Z
        B          |          C
          -------  |   -------
          |     |  |   |     |
          |  ---+--+---+---  |
          |  |  ●  |   ●  |  |
          |  |  |  |A  |  |  |
    -Y----+--+--+--+---+--+--+----+Y
          |  |  |  |   |  |  |
          |  |  ●  |   ●  |  |
          |  ---+--+---+---  |
          |     |  |   |     |
          -------  |   -------
                   |
                  -Z
 where object A is deformable and object B and C are rigid. A is an axis-aligned
 cube with side length 1 centered at origin. B and C are axis-aligned boxes with
 size (2, 0.5, 2), centered at (0, -0.5, 0) and (0, 0.5, 0) respectively.
 "●" represents a characteristic contact point. */
TEST_F(SoftsimSystemTest, AppendPointContactData) {
  /* Connects the softsim system to the MbP. */
  mbp_->set_softsim_base(&softsim_system_);
  systems::DiagramBuilder<double> builder;
  MultibodyPlant<double>& plant =
      AddMultibodyPlantSceneGraph(&builder, std::move(mbp_));
  /* Add the deformable body A. */
  SoftBodyIndex A = AddDeformableBox("A");
  /* Add the rigid bodies B and C. */
  const RigidBody<double>& B =
      plant.AddRigidBody("B", SpatialInertia<double>());
  const RigidBody<double>& C =
      plant.AddRigidBody("C", SpatialInertia<double>());
  const geometry::Box rigid_box_collision_geometry(2, 0.5, 2);

  geometry::ProximityProperties props_B;
  geometry::ProximityProperties props_C;
  const double kStiffnessB = 1.23;
  const double kStiffnessC = 4.56;
  const double kDissipationB = 7.89;
  const double kDissipationC = 1.011;
  const double kFrictionBStatic = 1.214;
  const double kFrictionBDynamic = 1.213;
  const CoulombFriction<double> kFrictionB(kFrictionBStatic, kFrictionBDynamic);
  const double kFrictionCStatic = 1.415;
  const double kFrictionCDynamic = 1.414;
  const CoulombFriction<double> kFrictionC(kFrictionCStatic, kFrictionCDynamic);
  geometry::AddContactMaterial(kStiffnessB, kDissipationB, kStiffnessB,
                               kFrictionB, &props_B);
  geometry::AddContactMaterial(kStiffnessC, kDissipationC, kStiffnessC,
                               kFrictionC, &props_C);

  const geometry::GeometryId B_id = plant.RegisterCollisionGeometry(
      B, math::RigidTransform<double>(), rigid_box_collision_geometry,
      "collision_box_B", props_B);
  const geometry::GeometryId C_id = plant.RegisterCollisionGeometry(
      C, math::RigidTransform<double>(), rigid_box_collision_geometry,
      "collision_box_C", props_C);
  plant.Finalize();
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  auto& mbp_context = plant.GetMyMutableContextFromRoot(context.get());

  const Vector3<double> p_WB(0, -0.5, 04);
  const Vector3<double> p_WC(0, 0.5, 04);
  const math::RigidTransform<double> X_WB(p_WB);
  const math::RigidTransform<double> X_WC(p_WC);
  plant.SetFreeBodyPose(&mbp_context, B, X_WB);
  plant.SetFreeBodyPose(&mbp_context, C, X_WC);
  /* Propagate the pose update to the collision objects stored in softsim
   system. */
  UpdatePoseForAllCollisionObjects(mbp_context);
  /* Create arbitrary spatial velocitiy for the rigid bodies. */
  const Vector3<double> w_WB(0, 0, 0);
  const Vector3<double> v_WB(0, 1, 0);
  const Vector3<double> w_WC(1, 0, 0);
  const Vector3<double> v_WC(0, 0, 0);
  const SpatialVelocity<double> V_WB(w_WB, v_WB);
  const SpatialVelocity<double> V_WC(w_WC, v_WC);
  plant.SetFreeBodySpatialVelocity(&mbp_context, B, V_WB);
  plant.SetFreeBodySpatialVelocity(&mbp_context, C, V_WC);

  const int num_deformable_dofs = kNumVertices * 3;
  const int num_rigid_dofs = plant.num_velocities();
  const int num_total_dofs = num_rigid_dofs + num_deformable_dofs;
  /* Calculates the PointContactDataStorage. */
  internal::PointContactDataStorage<double> point_contact_data_storage(
      num_total_dofs);
  AppendPointContactData(mbp_context, A, num_rigid_dofs,
                         &point_contact_data_storage);
  const int nc = point_contact_data_storage.num_contacts();
  const VectorX<double> phi0 = Eigen::Map<const Eigen::VectorXd>(
      point_contact_data_storage.phi0().data(), nc);
  const VectorX<double> stiffness = Eigen::Map<const Eigen::VectorXd>(
      point_contact_data_storage.stiffness().data(), nc);
  const VectorX<double> dissipation = Eigen::Map<const Eigen::VectorXd>(
      point_contact_data_storage.damping().data(), nc);
  const VectorX<double> mu = Eigen::Map<const Eigen::VectorXd>(
      point_contact_data_storage.mu().data(), nc);
  Eigen::SparseMatrix<double> contact_jacobian_sparse(
      point_contact_data_storage.num_contacts(),
      point_contact_data_storage.num_dofs());
  contact_jacobian_sparse.setFromTriplets(
      point_contact_data_storage.Jc_triplets().begin(),
      point_contact_data_storage.Jc_triplets().end());
  const MatrixX<double> contact_jacobian(contact_jacobian_sparse);

  /* Now we verify the calculated PointContactDataStorage matches the expected
   data from pen-and-paper calulation. */
  const internal::CollisionObjects<double>& collision_objects =
      get_collision_objects();
  const DeformableContactSurface<double> contact_surface_AB =
      ComputeTetMeshTriMeshContact(softsim_system_.meshes()[0],
                                   collision_objects.mesh(B_id),
                                   collision_objects.pose(B_id));
  const DeformableContactSurface<double> contact_surface_AC =
      ComputeTetMeshTriMeshContact(softsim_system_.meshes()[0],
                                   collision_objects.mesh(C_id),
                                   collision_objects.pose(C_id));

  /* Verify the number of contacts is as expected. */
  const int nc_AB = contact_surface_AB.num_polygons();
  const int nc_AC = contact_surface_AC.num_polygons();
  EXPECT_EQ(nc, nc_AB + nc_AC);

  // TODO(xuchenhan-tri): Modify this test when phi0 is properly calculated.
  /* The phi0 data is unused and set to zero for now. This will change in the
   future. */
  EXPECT_TRUE(CompareMatrices(phi0, VectorX<double>::Zero(nc)));

  /* Verify that the calculated stiffness and dissipation at the contact points
   matches expectation. */
  VectorX<double> expected_stiffness(nc);
  VectorX<double> expected_dissipation(nc);
  const auto [k_AB, d_AB] = CombinePointContactParameters(
      kContactStiffness, kStiffnessB, kContactDissipation, kDissipationB);
  const auto [k_AC, d_AC] = CombinePointContactParameters(
      kContactStiffness, kStiffnessC, kContactDissipation, kDissipationC);
  VectorX<double> expected_stiffness_AB = VectorX<double>::Ones(nc_AB) * k_AB;
  VectorX<double> expected_stiffness_AC = VectorX<double>::Ones(nc_AC) * k_AC;
  expected_stiffness << expected_stiffness_AB, expected_stiffness_AC;
  VectorX<double> expected_dissipation_AB = VectorX<double>::Ones(nc_AB) * d_AB;
  VectorX<double> expected_dissipation_AC = VectorX<double>::Ones(nc_AC) * d_AC;
  expected_dissipation << expected_dissipation_AB, expected_dissipation_AC;
  EXPECT_TRUE(CompareMatrices(stiffness, expected_stiffness));
  EXPECT_TRUE(CompareMatrices(dissipation, expected_dissipation));

  /* Verify that the calculated friction at the contact points matches
   expectation. */
  const CoulombFriction<double> expected_mu_AB =
      CalcContactFrictionFromSurfaceProperties(kFriction, kFrictionB);
  const CoulombFriction<double> expected_mu_AC =
      CalcContactFrictionFromSurfaceProperties(kFriction, kFrictionC);
  VectorX<double> expected_friction(nc);
  VectorX<double> expected_friction_AB =
      VectorX<double>::Ones(nc_AB) * expected_mu_AB.dynamic_friction();
  VectorX<double> expected_friction_AC =
      VectorX<double>::Ones(nc_AC) * expected_mu_AC.dynamic_friction();
  expected_friction << expected_friction_AB, expected_friction_AC;
  EXPECT_TRUE(CompareMatrices(mu, expected_friction));

  /* We verify the correctness of the contact jacobian by calculating the
   contact velocites by multiplying the contact jacobian and the generalized
   velocities and verify the contact velocities against pen-and-paper
   calculation. */
  VectorX<double> generalized_velocities(num_total_dofs);
  VectorX<double> deformable_velocities(num_deformable_dofs);
  const Vector3<double> v_WA(0, 0, 1);
  for (int i = 0; i < kNumVertices; ++i) {
    deformable_velocities.template segment<3>(3 * i) = v_WA;
  }
  generalized_velocities << deformable_velocities,
      plant.GetVelocities(mbp_context);
  const VectorX<double> contact_velocities =
      contact_jacobian * generalized_velocities;
  EXPECT_EQ(contact_velocities.size(), 3 * nc);

  VectorX<double> vn(nc);
  VectorX<double> vt(2 * nc);
  ExtractNormal(contact_velocities, &vn);
  ExtractTangent(contact_velocities, &vt);
  /* For contact between A and B, the normal component is expected to be -1 (due
   to the motion of B) and the tangnent component is expected to have norm 1
   (due to the motion of A). */
  for (int i = 0; i < nc_AB; ++i) {
    EXPECT_DOUBLE_EQ(vn(i), -1);
    EXPECT_DOUBLE_EQ(vt.template segment<2>(2 * i).norm(), v_WA.norm());
  }
  for (int i = 0; i < nc_AC; ++i) {
    const int contact_index = nc_AB + i;
    const ContactPolygonData<double>& data = contact_surface_AC.polygon_data(i);
    /* This is by definition the position of the contact point Cq measured in
     and expressed in the deformable frame, but we are using the assumption
     that the deformable frame is always the same as the world frame for now,
     and thus we denote the quantity with p_WCq. This assumption might change in
     the future. */
    const Vector3<double>& p_WCq = data.centroid;
    const Vector3<double> p_CoCq_W = p_WCq - p_WC;
    const Vector3<double> v_WCq = v_WC + w_WC.cross(p_CoCq_W);
    const double vn_WCq = v_WCq.dot(data.unit_normal);
    const Vector3<double> vt_WCq = v_WA - (v_WCq - data.unit_normal * vn_WCq);
    EXPECT_DOUBLE_EQ(vn(contact_index), -vn_WCq);
    EXPECT_DOUBLE_EQ(vt.template segment<2>(2 * contact_index).norm(),
                     vt_WCq.norm());
  }
}

}  // namespace
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
