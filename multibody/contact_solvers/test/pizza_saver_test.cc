#include <iostream>
#include <memory>

#include <fstream>

#include <Eigen/SparseCore>
#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#include "drake/multibody/contact_solvers/test/multibody_sim_driver.h"
#include "drake/multibody/contact_solvers/test/pgs_solver.h"
#include "drake/multibody/contact_solvers/mp_convex_solver.h"
#include "drake/multibody/contact_solvers/contact_solver.h"
#include "drake/multibody/plant/contact_results.h"

#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/nlopt_solver.h"
#include "drake/solvers/snopt_solver.h"

namespace drake {
namespace multibody {

using test::MultibodySimDriver;

namespace contact_solvers {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using math::RigidTransformd;

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

template <typename ContactSolverType>
class PizzaSaverTest : public ::testing::Test {
 public:
  void SetUp() override {
    const std::string model_file =
        "drake/multibody/contact_solvers/test/pizza_saver.sdf";
    driver_.BuildModel(dt_, model_file);
    const auto& plant = driver_.plant();
    const auto& particle = plant.GetBodyByName("body");
    // Add the ground, with the same friction as specified in the SDF file for
    // the particle.
    const std::vector<double> geometry_mu =
        driver_.GetDynamicFrictionCoefficients(particle);
    // For this test there should be three geometries with the same friction
    // coefficient.
    ASSERT_EQ(geometry_mu.size(), 3u);
    ASSERT_EQ(geometry_mu[0], geometry_mu[1]);
    ASSERT_EQ(geometry_mu[0], geometry_mu[2]);
    const double mu = geometry_mu[0];

    const std::vector<std::pair<double, double>> point_params =
        driver_.GetPointContactParameters(particle);
    ASSERT_EQ(point_params.size(), 3u);

    driver_.AddGround(point_params[0].first, point_params[0].second, mu);
    driver_.Finalize();
    const int nq = plant.num_positions();
    const int nv = plant.num_velocities();
    // Assert plant sizes.
    ASSERT_EQ(nq, 7);
    ASSERT_EQ(nv, 6);
    solver_ = &driver_.mutable_plant().set_contact_solver(
        std::make_unique<ContactSolverType>());
    SetParams(solver_);

    // MultibodyPlant state.
    SetInitialState();
  }

  // Set the particle to be in contact with the ground.
  void SetInitialState(double signed_distance = -1.0e-3) {
    const auto& plant = driver_.plant();
    const auto& body = plant.GetBodyByName("body");
    auto& context = driver_.mutable_plant_context();
    const Vector3d p_WB(0, 0, signed_distance);
    plant.SetFreeBodyPose(&context, body, math::RigidTransformd(p_WB));
    plant.SetFreeBodySpatialVelocity(&context, body,
                                     SpatialVelocity<double>::Zero());
  }

  VectorXd EvalGeneralizedContactForces() const {
    const auto& body = driver_.plant().GetBodyByName("body");
    const VectorXd tau_c = driver_.plant()
                           .get_generalized_contact_forces_output_port(
                               body.model_instance())
                           .Eval(driver_.plant_context());
    return tau_c;
  }

  void SetParams(MpConvexSolver<double>* solver) const {
    MpConvexSolverParameters params;
    //params.solver_id = solvers::NloptSolver::id();
    solver->set_parameters(params);
  }

  void PrintStats(const MpConvexSolver<double>& solver) const {
    auto& stats = solver.get_iteration_stats();
    std::cout << std::string(80, '-') << std::endl;  
    std::cout << std::string(80, '-') << std::endl;
    PRINT_VAR(stats.num_contacts);
    PRINT_VAR(stats.iteration_errors.vs_max);
    PRINT_VAR(stats.iteration_errors.id_rel_error);
  }

  void SetParams(PgsSolver<double>*) const {}

  void PrintStats(const PgsSolver<double>& solver) const {
    auto& stats = solver.get_iteration_stats();
    std::cout << std::string(80, '-') << std::endl;  
    std::cout << std::string(80, '-') << std::endl;

    PRINT_VAR(stats.iterations);
    fmt::print("{:>18} {:>18}\n", "vc", "gamma");
    for (const auto& errors : stats.iteration_errors) {
      fmt::print("{:18.6g} {:18.6g}\n", errors.vc_err, errors.gamma_err);
    }
  }


 protected:
  const double dt_{1.0e-3};
  const double kSignedDistance_{-1.0e-3};
  MultibodySimDriver driver_;
  ContactSolverType* solver_{nullptr};
  const double kContactVelocityAbsoluteTolerance_{1.0e-5};  // m/s.
  const double kContactVelocityRelativeTolerance_{1.0e-6};  // Unitless.
};

TYPED_TEST_SUITE_P(PizzaSaverTest);

TYPED_TEST_P(PizzaSaverTest, NoContact) {
  // No external forcing.
  const auto& body = this->driver_.plant().GetBodyByName("body");
  this->driver_.FixAppliedForce(body, SpatialForce<double>::Zero());

  // Set a no-contact state.
  this->SetInitialState(0.1);

  TypeParam& solver = *this->solver_;
  this->SetParams(&solver);

  // Verify forces in static equilibrium.
  // TODO(amcastro-tri): allow acces to scaling diagonals of W from
  // ContactSolver so that we can convert units.
  // Here we are assuming Wii = 1.0, which is OK for this problem.
  const VectorXd tau_c = this->EvalGeneralizedContactForces();

  this->PrintStats(solver);
}

TYPED_TEST_P(PizzaSaverTest, ZeroMoment) {
  // Inside a test, refer to TypeParam to get the type parameter.
  //TypeParam n = 0;

  // External forcing.
  const double Mz = 0.0;  // M_transition = 5.0
  const SpatialForce<double> F_Bo_W = SpatialForce<double>::Zero();
  const auto& body = this->driver_.plant().GetBodyByName("body");
  this->driver_.FixAppliedForce(body, F_Bo_W);

  //this->driver_.AdvanceNumSteps(1);

  TypeParam& solver = *this->solver_;
  this->SetParams(&solver);
  
  const double abs_tolerance = this->kContactVelocityAbsoluteTolerance_;

  // Verify forces in static equilibrium.
  // TODO(amcastro-tri): allow acces to scaling diagonals of W from
  // ContactSolver so that we can convert units.
  // Here we are assuming Wii = 1.0, which is OK for this problem.
  const VectorXd tau_c = this->EvalGeneralizedContactForces();

  this->PrintStats(solver);

  PRINT_VAR(tau_c.transpose());
  const double translational = tau_c.head<2>().norm();
  EXPECT_NEAR(translational, 0.0, abs_tolerance);
  const double torsional = tau_c(2);
  EXPECT_NEAR(torsional, 0.0, abs_tolerance);
  const double normal = tau_c(5);
  const double weight = 10.0;  // mass = 1.0 Kg and g = 10.0 m/s².
  EXPECT_NEAR(normal, weight, abs_tolerance);

#if 0
  const double translational = tau_c.head<2>().norm();
  EXPECT_NEAR(translational, 0.0, abs_tolerance);
  const double torsional = tau_c(2);
  EXPECT_NEAR(torsional + Mz, 0.0, abs_tolerance);
  const double normal = tau_c(5);
  const double weight = 10.0;  // mass = 1.0 Kg and g = 10.0 m/s².
  EXPECT_NEAR(normal, weight, abs_tolerance);


  // Verify contact velocities.
  const VectorXd& vc = solver.GetContactVelocities();

  VectorXd vn(nc);
  ExtractNormal(vc, &vn);
  EXPECT_LT(vn.norm() / nc, abs_tolerance);

  VectorXd vt(2 * nc);
  ExtractTangent(vc, &vt);

  // The problem has symmetry of revolution. Thus, for any rotation theta,
  // the three tangential velocities should have the same magnitude.
  const double v_slipA = vt.segment<2>(0).norm();
  const double v_slipB = vt.segment<2>(2).norm();
  const double v_slipC = vt.segment<2>(4).norm();  
  EXPECT_NEAR(v_slipA, v_slipB, abs_tolerance);
  EXPECT_NEAR(v_slipA, v_slipC, abs_tolerance);
  EXPECT_NEAR(v_slipC, v_slipB, abs_tolerance);

  // For this case where Mz < M_transition, we expect stiction (slip velocities
  // are smaller than the stiction tolerance).
  EXPECT_LT(v_slipA, abs_tolerance);
  EXPECT_LT(v_slipB, abs_tolerance);
  EXPECT_LT(v_slipC, abs_tolerance);  
#endif  
}

// This tests the solver when we apply a moment Mz about COM to the pizza saver.
// If Mz < mu * m * g * R, the saver should be in stiction (that is, the sliding
// velocity should be smaller than the regularization parameter). Otherwise the
// saver will start sliding. For this setup the transition occurs at
// M_transition = mu * m * g * R = 5.0
TYPED_TEST_P(PizzaSaverTest, SmallAppliedMoment) {
  // Inside a test, refer to TypeParam to get the type parameter.
  //TypeParam n = 0;

  // External forcing.
  const double Mz = 3.0;  // M_transition = 5.0
  const SpatialForce<double> F_Bo_W(Vector3d(0.0, 0.0, Mz), Vector3d::Zero());
  const auto& body = this->driver_.plant().GetBodyByName("body");
  this->driver_.FixAppliedForce(body, F_Bo_W);

  //this->driver_.AdvanceNumSteps(1);

  const TypeParam& solver = *this->solver_;
  const int nc = 3;
  const double abs_tolerance = this->kContactVelocityAbsoluteTolerance_;

  // Verify forces in static equilibrium.
  // TODO(amcastro-tri): allow acces to scaling diagonals of W from
  // ContactSolver so that we can convert units.
  // Here we are assuming Wii = 1.0, which is OK for this problem.
  const VectorXd tau_c = this->EvalGeneralizedContactForces();
  const double translational = tau_c.head<2>().norm();
  EXPECT_NEAR(translational, 0.0, abs_tolerance);
  const double torsional = tau_c(2);
  EXPECT_NEAR(torsional + Mz, 0.0, abs_tolerance);
  const double normal = tau_c(5);
  const double weight = 10.0;  // mass = 1.0 Kg and g = 10.0 m/s².
  EXPECT_NEAR(normal, weight, abs_tolerance);

  PRINT_VAR(tau_c.transpose());

  // Verify contact velocities.
  const VectorXd& vc = solver.GetContactVelocities();
  PRINT_VAR(vc.size());
  PRINT_VAR(vc.transpose());

  VectorXd vn(nc);
  ExtractNormal(vc, &vn);
  EXPECT_LT(vn.norm() / nc, abs_tolerance);

  VectorXd vt(2 * nc);
  ExtractTangent(vc, &vt);

  // The problem has symmetry of revolution. Thus, for any rotation theta,
  // the three tangential velocities should have the same magnitude.
  const double v_slipA = vt.segment<2>(0).norm();
  const double v_slipB = vt.segment<2>(2).norm();
  const double v_slipC = vt.segment<2>(4).norm();  
  EXPECT_NEAR(v_slipA, v_slipB, abs_tolerance);
  EXPECT_NEAR(v_slipA, v_slipC, abs_tolerance);
  EXPECT_NEAR(v_slipC, v_slipB, abs_tolerance);

  // For this case where Mz < M_transition, we expect stiction (slip velocities
  // are smaller than the stiction tolerance).
  EXPECT_LT(v_slipA, abs_tolerance);
  EXPECT_LT(v_slipB, abs_tolerance);
  EXPECT_LT(v_slipC, abs_tolerance);  

  this->PrintStats(solver);
}

// Exactly the same problem as in PizzaSaver::SmallAppliedMoment but with an
// applied moment Mz = 6.0 > M_transition = 5.0. In this case the pizza saver
// transitions to sliding with a net moment of Mz - M_transition during a
// period (time stepping interval) dt. Therefore we expect a change of angular
// velocity given by Δω = dt (Mz - Mtransition) / I.
TYPED_TEST_P(PizzaSaverTest, LargeAppliedMoment) {
  // Inside a test, refer to TypeParam to get the type parameter.
  //TypeParam n = 0;

  // External forcing.
  const double M_transition = 5.0;
  const double Mz = 8.0;
  const SpatialForce<double> F_Bo_W(Vector3d(0.0, 0.0, Mz), Vector3d::Zero());
  const auto& body = this->driver_.plant().GetBodyByName("body");
  this->driver_.FixAppliedForce(body, F_Bo_W);

  //this->driver_.AdvanceNumSteps(1);
  
  const double abs_tolerance = this->kContactVelocityAbsoluteTolerance_;

  // Verify forces in static equilibrium.
  // TODO(amcastro-tri): allow acces to scaling diagonals of W from
  // ContactSolver so that we can convert units.
  // Here we are assuming Wii = 1.0, which is OK for this problem.
  const VectorXd tau_c = this->EvalGeneralizedContactForces();
  PRINT_VAR(tau_c.transpose());
  const double translational = tau_c.head<2>().norm();
  EXPECT_NEAR(translational, 0.0, abs_tolerance);
  const double torsional = tau_c(2);
  EXPECT_NEAR(torsional + M_transition, 0.0, abs_tolerance);
  const double normal = tau_c(5);
  const double weight = 10.0;  // mass = 1.0 Kg and g = 10.0 m/s².
  EXPECT_NEAR(normal, weight, abs_tolerance);

  const TypeParam& solver = *this->solver_;
  const int nc = 3;

  // Verify contact velocities.
  const VectorXd& vc = solver.GetContactVelocities();
  PRINT_VAR(vc.transpose());

  PRINT_VAR(nc);
  PRINT_VAR(vc.size());
  VectorXd vn(nc);
  ExtractNormal(vc, &vn);
  EXPECT_LT(vn.norm() / nc, abs_tolerance);

  VectorXd vt(2 * nc);
  ExtractTangent(vc, &vt);

  PRINT_VAR(solver.GetImpulses().transpose());

  // The problem has symmetry of revolution. Thus, for any rotation theta,
  // the three tangential velocities should have the same magnitude.
  const double v_slipA = vt.segment<2>(0).norm();
  const double v_slipB = vt.segment<2>(2).norm();
  const double v_slipC = vt.segment<2>(4).norm();  
  EXPECT_NEAR(v_slipA, v_slipB, abs_tolerance);
  EXPECT_NEAR(v_slipA, v_slipC, abs_tolerance);
  EXPECT_NEAR(v_slipC, v_slipB, abs_tolerance);

  // For this case where Mz > M_transition, we expect sliding.
  EXPECT_GT(v_slipA, abs_tolerance);
  EXPECT_GT(v_slipB, abs_tolerance);
  EXPECT_GT(v_slipC, abs_tolerance);

  this->PrintStats(solver);
}

REGISTER_TYPED_TEST_SUITE_P(PizzaSaverTest, ZeroMoment, SmallAppliedMoment,
                            LargeAppliedMoment, NoContact);

typedef ::testing::Types<PgsSolver<double>, MpConvexSolver<double>>
    ContactSolverTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(ContactSolvers, PizzaSaverTest,
                               ContactSolverTypes);

}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
