#include "drake/multibody/fem/volumetric_element.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/multibody/fem/corotated_model.h"
#include "drake/multibody/fem/fem_data.h"
#include "drake/multibody/fem/linear_simplex_element.h"
#include "drake/multibody/fem/simplex_gaussian_quadrature.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

constexpr int kNaturalDimension = 3;
constexpr int kSpatialDimension = 3;
constexpr int kQuadratureOrder = 1;
constexpr double kEpsilon = 1e-14;
const ElementIndex kZeroIndex(0);
using T = AutoDiffXd;
using QuadratureType =
    internal::SimplexGaussianQuadrature<kNaturalDimension, kQuadratureOrder>;
static constexpr int kNumQuads = QuadratureType::num_quadrature_points;
using IsoparametricElementType =
    internal::LinearSimplexElement<T, kNaturalDimension, kSpatialDimension,
                                   kNumQuads>;
using ConstitutiveModelType = CorotatedModel<T, kNumQuads>;
using DeformationGradientDataType = CorotatedModelData<T, kNumQuads>;

class VolumetricElementTest : public ::testing::Test {
 protected:
  using ElementType = VolumetricElement<IsoparametricElementType,
                                        QuadratureType, ConstitutiveModelType>;
  using Data = typename ElementType::Data;
  static constexpr int kNumDofs = ElementType::num_dofs;
  static constexpr int kNumNodes = ElementType::num_nodes;
  const std::array<NodeIndex, kNumNodes> kNodeIndices = {
      {NodeIndex(0), NodeIndex(1), NodeIndex(2), NodeIndex(3)}};
  const T kYoungsModulus{1};
  const T kPoissonRatio{0.25};
  const T kDensity{1.23};
  const T kMassDamping{1e-4};
  const T kStiffnessDamping{1e-3};

  void SetUp() override { SetupElement(); }

  void SetupElement() {
    Eigen::Matrix<T, kSpatialDimension, kNumNodes> X = reference_positions();
    ConstitutiveModelType constitutive_model(kYoungsModulus, kPoissonRatio);
    DampingModel<T> damping_model(0, 0);
    elements_.emplace_back(kZeroIndex, kNodeIndices, constitutive_model, X,
                           kDensity, damping_model);
  }

  /* Makes an FemData to be consumed by the unit tests with the given q, v, and
   a as model values. */
  FemData<T> MakeFemData(const VectorX<T>& q, const VectorX<T>& v,
                         const VectorX<T>& a) {
    using systems::DiscreteStateIndex;
    // Declare data for the element under test in the private system owned by
    // this tester.
    const DiscreteStateIndex q_index = system_.DeclareDiscreteState(q);
    const DiscreteStateIndex v_index = system_.DeclareDiscreteState(v);
    const DiscreteStateIndex a_index = system_.DeclareDiscreteState(a);
    /* FEM element data. */
    std::vector<Data> model_data(1);
    const auto& element_data_cache_entry = system_.DeclareCacheEntry(
        "FEM state dependent element data",
        systems::ValueProducer(
            model_data,
            std::function<void(const systems::Context<T>&, std::vector<Data>*)>{
                [this, q_index, v_index, a_index](
                    const systems::Context<T>& context,
                    std::vector<Data>* element_data) {
                  DRAKE_DEMAND(element_data != nullptr);
                  DRAKE_DEMAND(element_data->size() == 1);
                  const VectorX<T>& position =
                      context.get_discrete_state(q_index).value();
                  const VectorX<T>& velocity =
                      context.get_discrete_state(v_index).value();
                  const VectorX<T>& acceleration =
                      context.get_discrete_state(a_index).value();
                  (*element_data)[0] =
                      element().ComputeData(position, velocity, acceleration);
                }}),
        {system_.discrete_state_ticket(q_index),
         system_.discrete_state_ticket(v_index),
         system_.discrete_state_ticket(a_index)});
    auto element_data_index = element_data_cache_entry.cache_index();
    FemDataInfo<T> fem_data_info{&system_, ModelId::get_new_id(),
                                 q_index,  v_index,
                                 a_index,  element_data_index};
    return FemData<T>(fem_data_info);
  }

  /* Set up the state and data of a deformed element. */
  FemData<T> MakeDeformedData() {
    Vector<double, kNumDofs> perturbation;
    perturbation << 0.18, 0.63, 0.54, 0.13, 0.92, 0.17, 0.03, 0.86, 0.85, 0.25,
        0.53, 0.67;
    Eigen::Matrix<T, kSpatialDimension, kNumNodes> X = reference_positions();
    Vector<double, kNumDofs> x =
        Eigen::Map<Vector<double, kNumDofs>>(math::DiscardGradient(X).data(),
                                             reference_positions().size()) +
        perturbation;
    Vector<T, kNumDofs> x_autodiff;
    math::InitializeAutoDiff(x, &x_autodiff);
    /* Set up arbitrary velocity and acceleration. */
    const Vector<T, kNumDofs> v_autodiff = -1.23 * perturbation;
    const Vector<T, kNumDofs> a_autodiff = 4.56 * perturbation;
    return MakeFemData(x_autodiff, v_autodiff, a_autodiff);
  }

  /* Set up a state where the positions are the same as reference positions. */
  FemData<T> MakeReferenceData() {
    Eigen::Matrix<T, kSpatialDimension, kNumNodes> X = reference_positions();
    Vector<double, kNumDofs> x(Eigen::Map<Vector<double, kNumDofs>>(
        math::DiscardGradient(X).data(), reference_positions().size()));
    Vector<T, kNumDofs> x_autodiff;
    math::InitializeAutoDiff(x, &x_autodiff);
    const Vector<T, kNumDofs> v_autodiff = Vector<T, kNumDofs>::Zero();
    const Vector<T, kNumDofs> a_autodiff = Vector<T, kNumDofs>::Zero();
    return MakeFemData(x_autodiff, v_autodiff, a_autodiff);
  }

  /* Set arbitrary reference positions with the requirement that the tetrahedron
   is not inverted. */
  Eigen::Matrix<T, kSpatialDimension, kNumNodes> reference_positions() const {
    Eigen::Matrix<T, kSpatialDimension, kNumNodes> X(kSpatialDimension,
                                                     kNumNodes);
    // clang-format off
    X << -0.10, 0.90, 0.02, 0.10,
         1.33,  0.23, 0.04, 0.01,
         0.20,  0.03, 2.31, -0.12;
    // clang-format on
    return X;
  }

  /* Get the one and only element. */
  const ElementType& element() const {
    DRAKE_DEMAND(elements_.size() == 1);
    return elements_[0];
  }

  /* Calculates the negative elastic force acting on the nodes of the only
   element evaluated with the given `fem_data`. */
  Vector<T, kNumDofs> CalcNegativeElasticForce(
      const FemData<T>& fem_data) const {
    Vector<T, kNumDofs> neg_force = Vector<T, kNumDofs>::Zero();
    element().AddNegativeElasticForce(fem_data, &neg_force);
    return neg_force;
  }

  /* Calculates the negative elastic force derivative with respect to positions
   for the only element evaluated with the given `fem_data`. */
  Eigen::Matrix<T, kNumDofs, kNumDofs> CalcNegativeElasticForceDerivative(
      const FemData<T>& fem_data) const {
    Eigen::Matrix<T, kNumDofs, kNumDofs> neg_force_derivative =
        Eigen::Matrix<T, kNumDofs, kNumDofs>::Zero();
    element().AddScaledElasticForceDerivative(fem_data, -1,
                                              &neg_force_derivative);
    return neg_force_derivative;
  }

  /* Calculates the DeformationGradientData for the only element evaluated with
   the given node positions. */
  DeformationGradientDataType CalcDeformationGradientData(
      const VectorX<T>& q) const {
    const std::array<Matrix3<T>, kNumQuads> F =
        element().CalcDeformationGradient(q);
    DeformationGradientDataType deformation_gradient_data;
    deformation_gradient_data.UpdateData(F);
    return deformation_gradient_data;
  }

  /* Calculates and verifies the energy and elastic forces evaluated with the
   given `data` are zero. */
  void VerifyEnergyAndForceAreZero(const FemData<T>& fem_data) const {
    T energy = element().CalcElasticEnergy(fem_data);
    EXPECT_NEAR(energy.value(), 0, std::numeric_limits<double>::epsilon());
    Vector<T, kNumDofs> neg_elastic_force = CalcNegativeElasticForce(fem_data);
    EXPECT_TRUE(CompareMatrices(Vector<T, kNumDofs>::Zero(), neg_elastic_force,
                                std::numeric_limits<double>::epsilon()));
  }

  /* Returns the constitutive model of the only element. */
  const ConstitutiveModelType& constitutive_model() const {
    return element().constitutive_model();
  }

  /* Returns the density of the only element. */
  const T& density(const ElementType& e) const { return e.density_; }

  /* Returns the volume evaluated at each quadrature point in the reference
   configuration of the only element. */
  const std::array<T, kNumQuads>& reference_volume() const {
    return element().reference_volume_;
  }

  /* Returns the mass matrix of the only element. */
  const Eigen::Matrix<T, kNumDofs, kNumDofs>& get_mass_matrix() const {
    return element().mass_matrix_;
  }

  /* Returns the gravity force acting on the nodes of the only element with
   the given `fem_data`. */
  Vector<T, kNumDofs> CalcGravityForce(const FemData<T>& fem_data) const {
    Vector<T, kNumDofs> gravity_force = Vector<T, kNumDofs>::Zero();
    element().AddScaledGravityForce(fem_data, 1.0, &gravity_force);
    return gravity_force;
  }

  class CachingSystem : public systems::LeafSystem<T> {
   public:
    using LeafSystem::DeclareDiscreteState;
    using SystemBase::DeclareCacheEntry;
  };
  CachingSystem system_;
  std::vector<ElementType> elements_;
};

namespace {

TEST_F(VolumetricElementTest, Constructor) {
  EXPECT_EQ(element().node_indices(), kNodeIndices);
  EXPECT_EQ(element().element_index(), kZeroIndex);
  EXPECT_EQ(density(element()), kDensity);
  ElementType move_constructed_element(std::move(elements_[0]));
  EXPECT_EQ(move_constructed_element.node_indices(), kNodeIndices);
  EXPECT_EQ(move_constructed_element.element_index(), kZeroIndex);
  EXPECT_EQ(density(move_constructed_element), kDensity);
}

/* Any undeformed state gives zero energy and zero force. */
TEST_F(VolumetricElementTest, UndeformedState) {
  /* The initial state where the current position is equal to reference
   position is undeformed. */
  FemData<T> fem_data = MakeReferenceData();
  VerifyEnergyAndForceAreZero(fem_data);

  /* Any rigid transformation of a undeformed state is undeformed. */
  math::RigidTransform<T> transform(math::RollPitchYaw<T>(1, 2, 3),
                                    Vector3<T>(0.314, 0.159, 0.265));
  Eigen::Matrix<T, kSpatialDimension, kNumNodes> X = reference_positions();
  Eigen::Matrix<T, kSpatialDimension, kNumNodes> rigid_transformed_X;
  for (int i = 0; i < kNumNodes; ++i) {
    rigid_transformed_X.col(i) = transform * X.col(i);
  }
  fem_data.SetPositions(Eigen::Map<Vector<T, kNumDofs>>(
      rigid_transformed_X.data(), rigid_transformed_X.size()));
  VerifyEnergyAndForceAreZero(fem_data);
}

/* Tests that in a deformed state, the energy and forces agrees with
 hand-calculated results. */
TEST_F(VolumetricElementTest, DeformedState) {
  FemData<T> fem_data = MakeReferenceData();
  /* Deform the element by scaling the initial position by a factor of 2. */
  fem_data.SetPositions(fem_data.GetPositions() * 2.0);
  const auto deformation_gradient_data =
      CalcDeformationGradientData(fem_data.GetPositions());
  std::array<T, kNumQuads> energy_density_array;
  constitutive_model().CalcElasticEnergyDensity(deformation_gradient_data,
                                                &energy_density_array);
  const double energy_density = ExtractDoubleOrThrow(energy_density_array[0]);
  /* Set up a matrix to help with calculating volume of the element. */
  Matrix4<double> matrix_for_volume_calculation;
  matrix_for_volume_calculation.bottomRows<1>() = Vector4<double>::Ones();
  const auto X = reference_positions();
  const auto X_double = math::DiscardGradient(X);
  matrix_for_volume_calculation.topRows<3>() = X_double;
  const double reference_volume =
      1.0 / 6.0 * std::abs(matrix_for_volume_calculation.determinant());
  const double analytical_energy = energy_density * reference_volume;
  /* Verify calculated energy is close to energy calculated analytically. */
  EXPECT_NEAR(element().CalcElasticEnergy(fem_data).value(), analytical_energy,
              kEpsilon);

  const auto neg_elastic_force_autodiff = CalcNegativeElasticForce(fem_data);
  Vector<double, kNumDofs> neg_elastic_force =
      math::DiscardGradient(neg_elastic_force_autodiff);
  /* Force on node 0. */
  const Vector3<double> force0 = -neg_elastic_force.head<3>();
  /* The first piola stress. */
  std::array<Matrix3<T>, kNumQuads> P_array;
  constitutive_model().CalcFirstPiolaStress(deformation_gradient_data,
                                            &P_array);
  const Matrix3<double>& P = math::DiscardGradient(P_array[0]);
  /* The directional face area of the face formed by node 0, 1, and 2 in the
   reference configuration. The indices are carefully ordered so that the
   direction is pointing to the inward face normal. */
  const Vector3<double> face012 =
      0.5 * (X_double.col(1) - X_double.col(0))
                .cross(X_double.col(2) - X_double.col(0));
  const Vector3<double> face013 =
      0.5 * (X_double.col(3) - X_double.col(0))
                .cross(X_double.col(1) - X_double.col(0));
  const Vector3<double> face023 =
      0.5 * (X_double.col(2) - X_double.col(0))
                .cross(X_double.col(3) - X_double.col(0));
  /* The analytic force exerted on node 0 is the average of the total force
   exerted the faces incidenting node 0. */
  const Vector3<double> force0_expected =
      P * (face012 + face013 + face023) / 3.0;
  EXPECT_TRUE(CompareMatrices(force0, force0_expected, kEpsilon));
}

/* Tests that at any given state, the negative elastic force is the derivative
 elastic energy with respect to the generalized positions. */
TEST_F(VolumetricElementTest, NegativeElasticForceIsEnergyDerivative) {
  FemData<T> fem_data = MakeDeformedData();
  T energy = element().CalcElasticEnergy(fem_data);
  Vector<T, kNumDofs> neg_elastic_force = CalcNegativeElasticForce(fem_data);
  EXPECT_TRUE(
      CompareMatrices(energy.derivatives(), neg_elastic_force, kEpsilon));
}

/* Tests that at any given state, CalcNegativeElasticForceDerivative() does in
 fact calculates the derivative of the negative elastic force. */
TEST_F(VolumetricElementTest, ElasticForceCompatibleWithItsDerivative) {
  FemData<T> fem_data = MakeDeformedData();
  Vector<T, kNumDofs> neg_elastic_force = CalcNegativeElasticForce(fem_data);
  Eigen::Matrix<T, kNumDofs, kNumDofs> neg_elastic_force_derivative =
      CalcNegativeElasticForceDerivative(fem_data);
  for (int i = 0; i < kNumDofs; ++i) {
    EXPECT_TRUE(CompareMatrices(neg_elastic_force(i).derivatives().transpose(),
                                neg_elastic_force_derivative.row(i), kEpsilon));
  }
}

/* In each dimension, the entries of the mass matrix should sum up to the
 total mass assigned to the element. */
TEST_F(VolumetricElementTest, MassMatrixSumUpToTotalMass) {
  const Eigen::Matrix<T, kNumDofs, kNumDofs>& mass_matrix = get_mass_matrix();
  const double mass_matrix_sum = mass_matrix.sum().value();
  double total_mass = 0;
  for (int q = 0; q < kNumQuads; ++q) {
    total_mass += (reference_volume()[q] * kDensity).value();
  }
  /* The mass matrix repeats the mass in each spatial dimension and needs to
   be scaled accordingly. */
  EXPECT_EQ(mass_matrix_sum, total_mass * kSpatialDimension);
}

/* Tests that the gravity forces match the expected value. */
TEST_F(VolumetricElementTest, Gravity) {
  const Eigen::Matrix<T, kNumDofs, kNumDofs>& mass_matrix = get_mass_matrix();
  Vector<T, kNumDofs> element_gravity_acceleration;
  for (int i = 0; i < kNumNodes; ++i) {
    element_gravity_acceleration.template segment<kSpatialDimension>(
        i * kSpatialDimension) = element().gravity_vector();
  }
  const Vector<T, kNumDofs> expected_gravity_force =
      mass_matrix * element_gravity_acceleration;

  const FemData<T> reference_fem_data = MakeReferenceData();
  EXPECT_TRUE(CompareMatrices(expected_gravity_force,
                              CalcGravityForce(reference_fem_data)));
  const FemData<T> deformed_fem_data = MakeDeformedData();
  EXPECT_TRUE(CompareMatrices(expected_gravity_force,
                              CalcGravityForce(deformed_fem_data)));
}

}  // namespace
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
