#include "drake/multibody/fixed_fem/dev/softsim_system.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/geometry/proximity/make_box_mesh.h"
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
 (AdvanceOneTimeStep) produces an easy analytical solution. */
const double kMassDamping = 0.0;
const double kStiffnessDamping = 0.02;
/* Time step. */
const double kDt = 0.0123;
const double kGravity = -9.81;
/* Number of vertices in the box mesh (see below). */
constexpr int kNumVertices = 8;

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

  /* Calls the SoftsimSystem::AdvanceOneTimeStep and returns the positions of
   the vertices of each deformable body at the end of the time step. */
  const std::vector<VectorX<double>> AdvanceOneTimeStep(
      const systems::Context<double>& context) {
    std::unique_ptr<systems::DiscreteValues<double>> next_states =
        context.get_discrete_state().Clone();
    softsim_system_.AdvanceOneTimeStep(context, next_states.get());
    std::vector<VectorX<double>> positions(softsim_system_.num_bodies());
    for (int i = 0; i < softsim_system_.num_bodies(); ++i) {
      const auto& next_state_values = next_states->get_vector(i).get_value();
      const int num_dofs = next_state_values.size() / 3;
      positions[i] = next_state_values.head(num_dofs);
    }
    return positions;
  }

  /* Add a dummy box shaped deformable body with the given "name". */
  SoftBodyIndex AddDeformableBox(std::string name) {
    const CoulombFriction<double> friction(0, 0);
    geometry::ProximityProperties dummy_proximity_props;
    geometry::AddContactMaterial(std::nullopt, std::nullopt, std::nullopt,
                                 friction, &dummy_proximity_props);
    return softsim_system_.RegisterDeformableBody(
        MakeBoxTetMesh(), std::move(name), MakeDeformableConfig(),
        dummy_proximity_props);
  }

  /* The SoftsimSystem under test. */
  multibody::MultibodyPlant<double> mbp_{kDt};
  SoftsimSystem<double> softsim_system_{&mbp_};
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
  EXPECT_EQ(softsim_system_.dt(), kDt);
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
TEST_F(SoftsimSystemTest, AdvanceOneTimeStep) {
  std::optional<systems::PeriodicEventData> periodic_event_data =
      softsim_system_.GetUniquePeriodicDiscreteUpdateAttribute();
  ASSERT_TRUE(periodic_event_data.has_value());
  EXPECT_EQ(periodic_event_data.value().period_sec(), kDt);
  EXPECT_EQ(periodic_event_data.value().offset_sec(), 0);

  AddDeformableBox("box");
  auto context = softsim_system_.CreateDefaultContext();
  const auto& vertex_position_port =
      softsim_system_.get_vertex_positions_output_port();
  const std::vector<VectorX<double>> initial_positions =
      vertex_position_port.Eval<std::vector<VectorX<double>>>(*context);
  EXPECT_EQ(initial_positions.size(), 1);
  EXPECT_EQ(initial_positions[0].size(), kNumVertices * 3);
  const std::vector<VectorX<double>> current_positions =
      AdvanceOneTimeStep(*context);
  EXPECT_EQ(current_positions.size(), 1);
  EXPECT_EQ(current_positions[0].size(), kNumVertices * 3);
  /* The factor of 0.25 seems strange but is correct. For the default mid-point
   rule used by DynamicElasticityModel,
        x = xₙ + dt ⋅ vₙ + dt² ⋅ (0.25 ⋅ a + 0.25 ⋅ aₙ).
   In this test case vₙ and aₙ are both 0, so x - xₙ is given by
   0.25 ⋅ a ⋅ dt². */
  const Vector3<double> expected_displacement(0, 0,
                                              0.25 * kGravity * kDt * kDt);
  const double kTol = 1e-14;
  for (int i = 0; i < kNumVertices; ++i) {
    const Vector3<double> displacement =
        current_positions[0].segment<3>(3 * i) -
        initial_positions[0].segment<3>(3 * i);
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
  Vector4<double> barycentric_centroid(0.1, 0.2, 0.3, 0.4);
  Vector3<double> contact_normal1(1, 0, 0);
  Vector3<double> contact_normal2(0, 1, 0);
  const geometry::VolumeElementIndex tet_index1(0);
  const geometry::VolumeElementIndex tet_index2(1);

  /* These data do not affect the result of the test. */
  Vector3<double> dummy_centroid(0, 0, 0);
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
  VectorX<double> v1 = dummy_velocity();
  VectorX<double> v2 = dummy_velocity();
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
}  // namespace
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
