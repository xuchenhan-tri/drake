#include "drake/multibody/fixed_fem/dev/dynamics_data_calculator.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/multibody/fixed_fem/dev/collision_object.h"
#include "drake/multibody/fixed_fem/dev/dynamic_elasticity_element.h"
#include "drake/multibody/fixed_fem/dev/dynamic_elasticity_model.h"
#include "drake/multibody/fixed_fem/dev/fem_state_base.h"
#include "drake/multibody/fixed_fem/dev/linear_constitutive_model.h"
#include "drake/multibody/fixed_fem/dev/linear_simplex_element.h"
#include "drake/multibody/fixed_fem/dev/simplex_gaussian_quadrature.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
namespace {
constexpr int kNaturalDimension = 3;
constexpr int kSpatialDimension = 3;
constexpr int kSolutionDimension = 3;
constexpr int kQuadratureOrder = 1;
const Vector3<double> kRotationalVelocity = {1, 2, 3};
const Vector3<double> kTranslationalVelocity = {4, 5, 6};
using QuadratureType =
    SimplexGaussianQuadrature<kNaturalDimension, kQuadratureOrder>;
constexpr int kNumQuads = QuadratureType::num_quadrature_points();
using IsoparametricElementType =
    LinearSimplexElement<double, kNaturalDimension, kSpatialDimension,
                         kNumQuads>;
using ConstitutiveModelType = LinearConstitutiveModel<double, kNumQuads>;
using ElementType =
    DynamicElasticityElement<IsoparametricElementType, QuadratureType,
                             ConstitutiveModelType>;
/* Dummy values for physics parameters. They don't affect the test. */
const double kYoungsModulus = 1e7;
const double kPoissonRatio = 0.4;
const double kMassDamping = 0.01;
const double kStiffnessDamping = 0.02;
/* Mass density of the deformable object. */
const double kDensity = 1e3;
/* The geometry of the model under test is a cube and it has 8 vertices and 6
 elements. */
constexpr int kNumVertices = 8;
constexpr int kNumDeformableDofs = kNumVertices * kSolutionDimension;
const double kDt = 1e-3;
const double kL = 1;
const double kTol = 1e-14;

class DynamicsDataCalculatorTest : public ::testing::Test {
 protected:
  /* Make a box and subdivide it into tet mesh. */
  static geometry::VolumeMesh<double> MakeBoxTetMesh() {
    geometry::Box box(kL, kL, kL);
    geometry::VolumeMesh<double> mesh =
        geometry::internal::MakeBoxVolumeMesh<double>(box, kL);
    return mesh;
  }

  void AddBoxToModel() {
    geometry::VolumeMesh<double> mesh = MakeBoxTetMesh();
    const ConstitutiveModelType constitutive_model(kYoungsModulus,
                                                   kPoissonRatio);
    const DampingModel<double> damping_model(kMassDamping, kStiffnessDamping);
    model_.AddDynamicElasticityElementsFromTetMesh(mesh, constitutive_model,
                                                   kDensity, damping_model);
  }

  void SetUp() override {
    AddBoxToModel();

    /* Add a collision object. */
    geometry::Box box{kL, kL, kL};
    geometry::ProximityProperties proximity_properties;
    geometry::AddContactMaterial({}, {}, CoulombFriction<double>(),
                                 &proximity_properties);
    const SpatialVelocity<double> V_WB(kRotationalVelocity,
                                       kTranslationalVelocity);
    std::optional<geometry::internal::hydroelastic::RigidGeometry>
        rigid_geometry =
            geometry::internal::hydroelastic::MakeRigidRepresentation(
                box, proximity_properties);
    DRAKE_DEMAND(rigid_geometry.has_value());
    const geometry::SurfaceMesh<double> surface_mesh =
        rigid_geometry.value().mesh();
    const auto motion_callback =
        [=](const double& time, math::RigidTransform<double>* pose,
            SpatialVelocity<double>* spatial_velocity) {
          unused(time);
          pose->SetIdentity();
          *spatial_velocity = V_WB;
        };
    collision_objects_.emplace_back(surface_mesh, proximity_properties,
                                    motion_callback);
    /* Update the prescribed spatial velocity to its prescribed value. */
    collision_objects_[0].UpdatePositionAndVelocity(kDt);
  }

  /* Return an arbitrary value for the velocities on the deformable dofs. */
  static VectorX<double> dummy_qdot() {
    Vector<double, kNumDeformableDofs> qdot;
    qdot << 0.18, 0.63, 0.54, 0.13, 0.92, 0.17, 0.03, 0.86, 0.85, 0.25, 0.53,
        0.67, 0.81, 0.36, 0.45, 0.31, 0.29, 0.71, 0.30, 0.68, 0.58, 0.52, 0.35,
        0.76;
    return qdot;
  }

  DynamicElasticityModel<ElementType> model_{kDt};
  std::vector<CollisionObject<double>> collision_objects_;
  DynamicsDataCalculator<double> dynamics_data_calculator_{&model_,
                                                           &collision_objects_};
};

/* Tests the mesh has been successfully converted to elements. */
TEST_F(DynamicsDataCalculatorTest, ComputeDynamicsData) {
  const std::unique_ptr<FemStateBase<double>> fem_state_base =
      model_.MakeFemStateBase();
  EXPECT_EQ(fem_state_base->num_generalized_positions(), kNumDeformableDofs);
  fem_state_base->SetQdot(dummy_qdot());
  const contact_solvers::internal::SystemDynamicsData data =
      dynamics_data_calculator_.ComputeDynamicsData(*fem_state_base);
  EXPECT_EQ(data.num_velocities(), kNumDeformableDofs + 6);

  VectorX<double> expected_v_star(kNumDeformableDofs + 6);
  expected_v_star << dummy_qdot(), kRotationalVelocity, kTranslationalVelocity;
  /* The v_star from the dynamics data is copied from the state and the collsion
   object velocities and therefore should be exactly equal to the expected
   value. */
  EXPECT_TRUE(CompareMatrices(data.get_v_star(), expected_v_star, 0));

  Eigen::SparseMatrix<double> Ainv_matrix(data.num_velocities(),
                                          data.num_velocities());
  data.get_Ainv().AssembleMatrix(&Ainv_matrix);
  MatrixX<double> expected_Ainv_matrix =
      MatrixX<double>::Zero(data.num_velocities(), data.num_velocities());
  Eigen::SparseMatrix<double> deformable_tangent_matrix(kNumDeformableDofs,
                                                        kNumDeformableDofs);
  model_.SetTangentMatrixSparsityPattern(&deformable_tangent_matrix);
  model_.CalcTangentMatrix(*fem_state_base, &deformable_tangent_matrix);
  /* The block corresponding to the deformable object is the inverse of the
   * tangent matrix from the FEM model. */
  expected_Ainv_matrix.topLeftCorner(kNumDeformableDofs, kNumDeformableDofs) =
      MatrixX<double>(deformable_tangent_matrix).inverse();
  /* The block corresponding to collision object is zero. */
  expected_Ainv_matrix.bottomRightCorner(6, 6).setZero();
  EXPECT_TRUE(CompareMatrices(MatrixX<double>(Ainv_matrix),
                              expected_Ainv_matrix, kTol));
}
}  // namespace
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
