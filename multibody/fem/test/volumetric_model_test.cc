#include "drake/multibody/fem/volumetric_model.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/fem/acceleration_newmark_scheme.h"
#include "drake/multibody/fem/fem_state.h"
#include "drake/multibody/fem/linear_constitutive_model.h"
#include "drake/multibody/fem/linear_simplex_element.h"
#include "drake/multibody/fem/simplex_gaussian_quadrature.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace {

using Eigen::MatrixXd;

constexpr int kNaturalDimension = 3;
constexpr int kSpatialDimension = 3;
constexpr int kQuadratureOrder = 1;
using QuadratureType =
    internal::SimplexGaussianQuadrature<kNaturalDimension, kQuadratureOrder>;
constexpr int kNumQuads = QuadratureType::num_quadrature_points;

using AutoDiffIsoparametricElement =
    internal::LinearSimplexElement<AutoDiffXd, kNaturalDimension,
                                   kSpatialDimension, kNumQuads>;
using AutoDiffConstitutiveModel =
    internal::LinearConstitutiveModel<AutoDiffXd, kNumQuads>;
using AutoDiffElement =
    VolumetricElement<AutoDiffIsoparametricElement, QuadratureType,
                      AutoDiffConstitutiveModel>;

using DoubleIsoparametricElement =
    internal::LinearSimplexElement<double, kNaturalDimension, kSpatialDimension,
                                   kNumQuads>;
using DoubleConstitutiveModel =
    internal::LinearConstitutiveModel<double, kNumQuads>;
using DoubleElement =
    VolumetricElement<DoubleIsoparametricElement, QuadratureType,
                      DoubleConstitutiveModel>;

const double kYoungsModulus = 1.23;
const double kPoissonRatio = 0.456;
const double kDensity = 0.789;
/* The geometry of the model under test is a cube and it has 8 vertices and 6
 elements. */
constexpr int kNumCubeVertices = 8;
constexpr int kNumDofs = kNumCubeVertices * kSpatialDimension;
constexpr int kNumElements = 6;
/* Parameters for Newmark scheme. */
const double kDt = 1e-3;
const double kGamma = 0.5;
const double kBeta = 0.25;
/* Parameters for the damping model. */
const double kMassDamping = 0.01;
const double kStiffnessDamping = 0.02;

class VolumetricModelTest : public ::testing::Test {
 protected:
  /* Makes a box and subdivides it into 6 tetrahedra. */
  template <typename T>
  geometry::VolumeMesh<T> MakeBoxTetMesh() {
    const double length = 0.1;
    geometry::Box box(length, length, length);
    geometry::VolumeMesh<T> mesh =
        geometry::internal::MakeBoxVolumeMesh<T>(box, length);
    DRAKE_DEMAND(mesh.num_elements() == 6);
    return mesh;
  }

  /* Adds a FEM model of a box discretized into 6 tetrahedra into the given
   `fem_model_impl`. */
  template <typename FemModelType>
  void AddBoxToModel(FemModelType* fem_model) {
    using T = typename FemModelType::T;
    geometry::VolumeMesh<T> mesh = MakeBoxTetMesh<T>();
    const typename FemModelType::ConstitutiveModel constitutive_model(
        kYoungsModulus, kPoissonRatio);
    const DampingModel<T> damping_model(kMassDamping, kStiffnessDamping);
    fem_model->AddVolumetricElementsFromTetMesh(mesh, constitutive_model,
                                                kDensity, damping_model);
  }

  void SetUp() override { AddBoxToModel(&model_); }

  /* Returns an arbitrary vector of the given size. */
  static VectorX<double> perturbation(int size) {
    VectorX<double> delta(size);
    for (int i = 0; i < size; ++i) {
      delta(i) = 0.01 * i;
    }
    return delta;
  }

  /* Returns an arbitrary FEM state whose generalized positions are different
   from reference positions and whose velocities and acclerations are nonzero.
   In addition, set up autodiff derivatives for accelerations if the scalar type
   is AutoDiffXd. */
  template <typename FemModelType>
  std::unique_ptr<FemData<typename FemModelType::T>> MakeDeformedFemData(
      const FemModelType& fem_model) {
    using T = typename FemModelType::T;
    const int num_dofs = fem_model.num_dofs();
    if constexpr (std::is_same_v<T, AutoDiffXd>) {
      const auto fem_data_info = fem_model.AllocateFemData(&autodiff_system_);
      autodiff_context_ = autodiff_system_.CreateDefaultContext();
      auto fem_data = std::make_unique<FemData<AutoDiffXd>>(
          fem_data_info, autodiff_context_.get());
      /* Perturb a. */
      const VectorX<double> perturbed_a =
          math::ExtractValue(fem_data->GetAccelerations()) +
          perturbation(num_dofs);
      /* Set up AutodiffXd derivatives. */
      VectorX<AutoDiffXd> perturbed_a_autodiff(num_dofs);
      math::InitializeAutoDiff(perturbed_a, &perturbed_a_autodiff);
      /* It's important to set up the `deformed_state` with AdvanceOneTimeStep()
       so that the derivatives such as dq/da are set up. */
      integrator_.AdvanceOneTimeStep(fem_data->GetFemState(),
                                     perturbed_a_autodiff,
                                     &fem_data->GetMutableFemState());
      return fem_data;
    } else {
      const auto fem_data_info = fem_model.AllocateFemData(&double_system_);
      double_context_ = double_system_.CreateDefaultContext();
      auto fem_data = std::make_unique<FemData<double>>(
          fem_data_info, double_context_.get());
      /* Perturb a. */
      const VectorX<double> perturbed_a =
          fem_data->GetAccelerations() + perturbation(num_dofs);
      const AccelerationNewmarkScheme<double> double_integrator(kDt, kGamma,
                                                                kBeta);
      double_integrator.AdvanceOneTimeStep(fem_data->GetFemState(), perturbed_a,
                                           &fem_data->GetMutableFemState());
      return fem_data;
    }
    DRAKE_UNREACHABLE();
  }

  /* The system and context that allocates and stores FEM data. */
  systems::LeafSystem<double> double_system_{};
  systems::LeafSystem<AutoDiffXd> autodiff_system_{};
  std::unique_ptr<systems::Context<double>> double_context_{nullptr};
  std::unique_ptr<systems::Context<AutoDiffXd>> autodiff_context_{nullptr};
  /* The model under test. */
  VolumetricModel<AutoDiffElement> model_{};
  AccelerationNewmarkScheme<AutoDiffXd> integrator_{kDt, kGamma, kBeta};
};

/* Tests the mesh has been successfully converted to elements. */
TEST_F(VolumetricModelTest, Geometry) {
  EXPECT_EQ(model_.num_nodes(), kNumCubeVertices);
  EXPECT_EQ(model_.num_elements(), kNumElements);
}

/* Tests that the tangent matrix of the model is the derivative of the residual
 with respect to the change in a. */
TEST_F(VolumetricModelTest, TangentMatrixIsResidualDerivative) {
  using T = AutoDiffXd;

  std::unique_ptr<FemData<AutoDiffXd>> fem_data =
      MakeDeformedFemData(model_);
  VectorX<T> residual(fem_data->num_dofs());
  model_.CalcResidual(*fem_data, &residual);

  Eigen::SparseMatrix<T> tangent_matrix = model_.MakeEigenSparseTangentMatrix();
  model_.CalcTangentMatrix(*fem_data, integrator_.weights(), &tangent_matrix);

  /* In the discretization of the unit cube by 6 tetrahedra, there are 19 edges,
   and 8 nodes, creating 19*2 + 8 blocks of 3-by-3 nonzero entries. Hence the
   number of nonzero entries of the tangent matrix should be (19*2+8)*9. */
  const int nnz = (19 * 2 + 8) * 9;
  EXPECT_EQ(tangent_matrix.nonZeros(), nnz);

  const MatrixX<T> dense_tangent_matrix(tangent_matrix);
  for (int i = 0; i < fem_data->num_dofs(); ++i) {
    /* The tangent matrix should be the derivative of the residual. Notice that
     here we are comparing a VectorX<double> and the value of a
     VectorX<AutoDiffXd>. */
    EXPECT_TRUE(CompareMatrices(residual(i).derivatives(),
                                dense_tangent_matrix.col(i),
                                4 * std::numeric_limits<double>::epsilon()));
  }
}

/* Verifies that the tangent matrix calculated as PETSc matrix is the same as
 that calculated as Eigen::SparseMatrix. */
TEST_F(VolumetricModelTest, TangentMatrixParity) {
  std::unique_ptr<FemData<AutoDiffXd>> fem_data =
      MakeDeformedFemData(model_);
  Eigen::SparseMatrix<AutoDiffXd> eigen_tangent_matrix =
      model_.MakeEigenSparseTangentMatrix();
  model_.CalcTangentMatrix(*fem_data, integrator_.weights(),
                           &eigen_tangent_matrix);
  const MatrixX<AutoDiffXd> eigen_dense_autodiff_matrix = eigen_tangent_matrix;
  const MatrixXd eigen_dense_matrix =
      math::ExtractValue(eigen_dense_autodiff_matrix);

  VolumetricModel<DoubleElement> double_model;
  AddBoxToModel(&double_model);
  std::unique_ptr<FemData<double>> double_data =
      MakeDeformedFemData(double_model);
  const AccelerationNewmarkScheme<double> double_integrator_{kDt, kGamma,
                                                             kBeta};
  std::unique_ptr<internal::PetscSymmetricBlockSparseMatrix>
      petsc_tangent_matrix =
          double_model.MakePetscSymmetricBlockSparseTangentMatrix();
  double_model.CalcTangentMatrix(*double_data, double_integrator_.weights(),
                                 petsc_tangent_matrix.get());
  petsc_tangent_matrix->AssembleIfNecessary();
  const MatrixXd petsc_dense_matrix = petsc_tangent_matrix->MakeDenseMatrix();
  EXPECT_TRUE(CompareMatrices(eigen_dense_matrix, petsc_dense_matrix,
                              std::numeric_limits<double>::epsilon()));
}

/* Adds two copies of the same set of elements to test that the node offsets in
 AddVolumetricElementsFromTetMesh() are working as intended. */
TEST_F(VolumetricModelTest, MultipleMesh) {
  /* Add a second box mesh to the model. */
  AddBoxToModel(&model_);
  EXPECT_EQ(model_.num_nodes(), 2 * kNumCubeVertices);
  /* Each cube is split into 6 tetrahedra. */
  EXPECT_EQ(model_.num_elements(), 2 * kNumElements);

  const std::unique_ptr<FemData<AutoDiffXd>> fem_data =
      MakeDeformedFemData(model_);
  EXPECT_EQ(fem_data->num_dofs(), 2 * kNumDofs);
}

}  // namespace
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
