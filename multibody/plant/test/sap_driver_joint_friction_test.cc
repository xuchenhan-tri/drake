#include <cmath>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff.h"
#include "drake/multibody/contact_solvers/contact_solver_results.h"
#include "drake/multibody/contact_solvers/sap/sap_contact_problem.h"
#include "drake/multibody/contact_solvers/sap/sap_joint_friction_constraint.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/sap_driver.h"
#include "drake/multibody/plant/test/compliant_contact_manager_tester.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"

/* @file This file tests SapDriver's support for joint dry friction.

  Constraints are only supported by the SAP solver. Therefore, to exercise the
  relevant code paths, we arbitrarily choose one contact approximation that uses
  the SAP solver. More precisely, in the unit tests below we call
  set_discrete_contact_approximation(DiscreteContactApproximation::kSap) on the
  MultibodyPlant used for testing. */

using drake::multibody::contact_solvers::internal::ContactSolverResults;
using drake::multibody::contact_solvers::internal::SapContactProblem;
using drake::multibody::contact_solvers::internal::SapJointFrictionConstraint;
using drake::systems::Context;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace drake {
namespace multibody {
namespace internal {

// Friend class used to provide access to a selection of private functions in
// SapDriver for testing purposes.
class SapDriverTest {
 public:
  static const ContactProblemCache<double>& EvalContactProblemCache(
      const SapDriver<double>& driver, const Context<double>& context) {
    return driver.EvalContactProblemCache(context);
  }
};

namespace {

// Dimensionless regularization used by the driver for joint friction
// constraints. This value must match the one in
// SapDriver::AddJointFrictionConstraints().
constexpr double kSigma = 1.0e-3;

// Relative tolerance for comparisons against analytical solutions. For the
// single DOF problems below the SAP cost is piecewise quadratic and Newton
// iterations converge to machine precision within a few iterations.
constexpr double kTolerance = 1.0e-10;

// Model parameters.
constexpr double kWheelInertia = 0.1;    // [kg⋅m²], about the wheel's axis.
constexpr double kWheelFriction = 0.5;   // [N⋅m]
constexpr double kSliderMass = 2.0;      // [kg]
constexpr double kSliderFriction = 1.0;  // [N]
constexpr double kElbowFriction = 0.3;   // [N⋅m]
constexpr double kLinkLength = 0.5;      // [m]

// Fixture with a model made of three kinematic trees, each with a joint that
// models dry friction:
//   1. A wheel free to spin about the world's z axis. A single revolute DOF.
//   2. A slider free to translate along the world's x axis. A single prismatic
//      DOF.
//   3. A two link arm in the x-y plane with a shoulder and an elbow revolute
//      joint about z. Only the elbow models dry friction. This tree is used to
//      verify the indexing of DOFs within a tree with more than one DOF.
// Gravity is set to zero so that actuation is the only force applied on the
// model besides friction. Therefore the wheel and slider dynamics have simple
// analytical solutions used to verify the implementation.
class SapDriverJointFrictionTest : public ::testing::Test {
 protected:
  void MakePlant(double time_step) {
    plant_ = std::make_unique<MultibodyPlant<double>>(time_step);
    // Only SAP supports the modeling of constraints.
    plant_->set_discrete_contact_approximation(
        DiscreteContactApproximation::kSap);
    plant_->mutable_gravity_field().set_gravity_vector(Vector3d::Zero());

    // Bodies with the same rotational inertia about every axis so that the
    // inertia about the joint's axis is known regardless of orientation.
    const SpatialInertia<double> M_wheel(
        1.0, Vector3d::Zero(),
        UnitInertia<double>::TriaxiallySymmetric(kWheelInertia));
    const SpatialInertia<double> M_slider(
        kSliderMass, Vector3d::Zero(),
        UnitInertia<double>::TriaxiallySymmetric(1.0));
    // The arm's inertial properties play no role in these tests.
    const SpatialInertia<double> M_link = SpatialInertia<double>::MakeUnitary();

    const RigidBody<double>& wheel = plant_->AddRigidBody("wheel", M_wheel);
    wheel_ = &plant_->AddJoint<RevoluteJoint>("wheel", plant_->world_body(),
                                              std::nullopt, wheel, std::nullopt,
                                              Vector3d::UnitZ());
    wheel_actuator_ = &plant_->AddJointActuator("wheel", *wheel_);
    plant_->GetMutableJointByName<RevoluteJoint>("wheel")
        .set_default_dry_friction(kWheelFriction);

    const RigidBody<double>& slider = plant_->AddRigidBody("slider", M_slider);
    slider_ = &plant_->AddJoint<PrismaticJoint>(
        "slider", plant_->world_body(), std::nullopt, slider, std::nullopt,
        Vector3d::UnitX());
    slider_actuator_ = &plant_->AddJointActuator("slider", *slider_);
    plant_->GetMutableJointByName<PrismaticJoint>("slider")
        .set_default_dry_friction(kSliderFriction);

    const RigidBody<double>& link1 = plant_->AddRigidBody("link1", M_link);
    const RigidBody<double>& link2 = plant_->AddRigidBody("link2", M_link);
    shoulder_ = &plant_->AddJoint<RevoluteJoint>(
        "shoulder", plant_->world_body(), std::nullopt, link1, std::nullopt,
        Vector3d::UnitZ());
    elbow_ = &plant_->AddJoint<RevoluteJoint>(
        "elbow", link1, math::RigidTransformd(Vector3d(kLinkLength, 0.0, 0.0)),
        link2, std::nullopt, Vector3d::UnitZ());
    plant_->GetMutableJointByName<RevoluteJoint>("elbow")
        .set_default_dry_friction(kElbowFriction);

    plant_->Finalize();

    auto owned_contact_manager =
        std::make_unique<CompliantContactManager<double>>();
    manager_ = owned_contact_manager.get();
    plant_->SetDiscreteUpdateManager(std::move(owned_contact_manager));

    context_ = plant_->CreateDefaultContext();
    SetActuation(0.0, 0.0);
  }

  const SapDriver<double>& sap_driver() const {
    return CompliantContactManagerTester::sap_driver(*manager_);
  }

  // Fixes the actuation input port so that `wheel_torque` is applied on the
  // wheel and `slider_force` is applied on the slider.
  void SetActuation(double wheel_torque, double slider_force) {
    VectorXd u = VectorXd::Zero(plant_->num_actuated_dofs());
    u(wheel_actuator_->input_start()) = wheel_torque;
    u(slider_actuator_->input_start()) = slider_force;
    plant_->get_actuation_input_port().FixValue(context_.get(), u);
  }

  // Advances the discrete dynamics by `num_steps` time steps, updating the
  // state stored in the context in place. The plant's forced update events
  // perform the same update as its periodic events. Time is not advanced,
  // which is irrelevant for these time invariant models.
  void Step(int num_steps) {
    for (int i = 0; i < num_steps; ++i) {
      plant_->ExecuteForcedEvents(context_.get(), /* publish = */ false);
    }
  }

  // Analytical solution for the velocity of a single DOF with effective inertia
  // `inertia` after one step from rest under a constant applied generalized
  // force `applied_force` and dry friction bound `friction`. In stiction the
  // regularized constraint leaves a residual creep velocity σ/(1+σ)⋅v*, where
  // v* is the free motion velocity, since the Delassus estimate w = 1/A is
  // exact for a single DOF and R = σ⋅w. Beyond the breakaway force
  // (1+σ)⋅friction the DOF slides with friction saturated at its bound.
  double ExpectedVelocityFromRest(double applied_force, double inertia,
                                  double friction) const {
    const double dt = plant_->time_step();
    const double v_star = dt * applied_force / inertia;
    if (std::abs(applied_force) <= (1.0 + kSigma) * friction) {
      return kSigma / (1.0 + kSigma) * v_star;
    }
    const double sign = applied_force > 0 ? 1.0 : -1.0;
    return v_star - sign * dt * friction / inertia;
  }

  std::unique_ptr<MultibodyPlant<double>> plant_;
  CompliantContactManager<double>* manager_{nullptr};
  std::unique_ptr<Context<double>> context_;
  const RevoluteJoint<double>* wheel_{nullptr};
  const PrismaticJoint<double>* slider_{nullptr};
  const RevoluteJoint<double>* shoulder_{nullptr};
  const RevoluteJoint<double>* elbow_{nullptr};
  const JointActuator<double>* wheel_actuator_{nullptr};
  const JointActuator<double>* slider_actuator_{nullptr};
};

// Verifies the driver adds one SapJointFrictionConstraint per single DOF joint
// with non-zero dry friction, using the value stored in the context and the
// right indexing within each tree.
TEST_F(SapDriverJointFrictionTest, ConstraintsAreAdded) {
  MakePlant(1.0e-3);

  // Change the wheel's friction through the context to verify the driver reads
  // context values rather than the defaults.
  const double wheel_friction = 2.0 * kWheelFriction;
  wheel_->SetDryFriction(context_.get(), wheel_friction);

  const ContactProblemCache<double>& cache =
      SapDriverTest::EvalContactProblemCache(sap_driver(), *context_);
  const SapContactProblem<double>& problem = *cache.sap_problem;

  // There is no contact and joints have no limits, thus the only constraints
  // are the friction constraints, added in joint index order.
  struct ExpectedConstraint {
    const Joint<double>* joint;
    double friction;
  };
  const std::vector<ExpectedConstraint> expected{{wheel_, wheel_friction},
                                                 {slider_, kSliderFriction},
                                                 {elbow_, kElbowFriction}};
  ASSERT_EQ(problem.num_constraints(), std::ssize(expected));

  const SpanningForest& forest =
      CompliantContactManagerTester::get_forest(*manager_);
  for (int i = 0; i < std::ssize(expected); ++i) {
    const auto* constraint =
        dynamic_cast<const SapJointFrictionConstraint<double>*>(
            &problem.get_constraint(i));
    ASSERT_NE(constraint, nullptr);
    const Joint<double>& joint = *expected[i].joint;
    const TreeIndex tree_index = forest.v_to_tree_index(joint.velocity_start());
    const SpanningForest::Tree& tree = forest.trees(tree_index);
    EXPECT_EQ(constraint->num_cliques(), 1);
    EXPECT_EQ(constraint->first_clique(), tree_index);
    EXPECT_EQ(constraint->num_velocities(0), tree.nv());
    EXPECT_EQ(constraint->clique_dof(),
              joint.velocity_start() - tree.v_start());
    EXPECT_EQ(constraint->parameters().friction, expected[i].friction);
    EXPECT_EQ(constraint->parameters().sigma, kSigma);
  }

  // The elbow is the second DOF of the arm's tree.
  const auto& elbow_constraint =
      dynamic_cast<const SapJointFrictionConstraint<double>&>(
          problem.get_constraint(2));
  EXPECT_EQ(elbow_constraint.num_velocities(0), 2);
  EXPECT_EQ(elbow_constraint.clique_dof(), 1);
}

// Joints with zero friction and locked joints do not lead to constraints.
TEST_F(SapDriverJointFrictionTest, ZeroFrictionAndLockedJointsAreSkipped) {
  MakePlant(1.0e-3);
  wheel_->SetDryFriction(context_.get(), 0.0);
  slider_->Lock(context_.get());

  const ContactProblemCache<double>& cache =
      SapDriverTest::EvalContactProblemCache(sap_driver(), *context_);
  const SapContactProblem<double>& problem = *cache.sap_problem;
  ASSERT_EQ(problem.num_constraints(), 1);
  const auto* constraint =
      dynamic_cast<const SapJointFrictionConstraint<double>*>(
          &problem.get_constraint(0));
  ASSERT_NE(constraint, nullptr);
  EXPECT_EQ(constraint->parameters().friction, kElbowFriction);

  // The locked slider must remain at rest even when actuated above its
  // friction bound.
  SetActuation(0.0, 10.0 * kSliderFriction);
  Step(3);
  EXPECT_EQ(slider_->get_translation_rate(*context_), 0.0);
}

// A locked joint in a tree that still has unlocked DOFs gets no constraint.
// This is the case that motivates skipping locked joints in the driver: the
// reduced problem removes the locked DOF's column from the constraint
// Jacobian, which would leave a zero Jacobian and a singular regularization.
TEST_F(SapDriverJointFrictionTest, LockedJointInTreeWithFreeDofs) {
  MakePlant(1.0e-3);
  elbow_->Lock(context_.get());
  shoulder_->set_angular_rate(context_.get(), 1.0);

  const ContactProblemCache<double>& cache =
      SapDriverTest::EvalContactProblemCache(sap_driver(), *context_);
  const SapContactProblem<double>& problem = *cache.sap_problem;
  ASSERT_EQ(problem.num_constraints(), 2);  // Wheel and slider only.
  for (int i = 0; i < problem.num_constraints(); ++i) {
    const auto& constraint =
        dynamic_cast<const SapJointFrictionConstraint<double>&>(
            problem.get_constraint(i));
    EXPECT_EQ(constraint.num_velocities(0), 1);
  }
  // The arm's tree participates in the reduced problem through the shoulder.
  ASSERT_NE(cache.sap_problem_locked, nullptr);

  // Stepping succeeds. Without gravity and friction on the shoulder, the arm
  // spins at a constant rate about the shoulder with the elbow locked.
  Step(3);
  EXPECT_EQ(elbow_->get_angular_rate(*context_), 0.0);
  EXPECT_NEAR(shoulder_->get_angular_rate(*context_), 1.0, kTolerance);
}

// A friction constraint on an unlocked joint whose tree has a locked DOF gets
// its Jacobian reduced (the locked column is removed) and must still behave as
// a single DOF friction joint.
TEST_F(SapDriverJointFrictionTest, FrictionJointInTreeWithLockedDof) {
  const double dt = 1.0e-3;
  MakePlant(dt);
  shoulder_->Lock(context_.get());
  const double initial_rate = 1.0;
  elbow_->set_angular_rate(context_.get(), initial_rate);

  const ContactProblemCache<double>& cache =
      SapDriverTest::EvalContactProblemCache(sap_driver(), *context_);
  ASSERT_EQ(cache.sap_problem->num_constraints(), 3);
  ASSERT_NE(cache.sap_problem_locked, nullptr);
  // The reduced problem still has the three friction constraints, with the
  // arm's constraint now acting on a single DOF clique.
  ASSERT_EQ(cache.sap_problem_locked->num_constraints(), 3);
  EXPECT_EQ(cache.sap_problem_locked->get_constraint(2).num_velocities(0), 1);

  // With the shoulder locked, link2 rotates about the elbow with its own
  // inertia about that axis (a unitary inertia at the elbow), decelerating
  // uniformly under the elbow's friction.
  const int num_steps = 3;
  Step(num_steps);
  EXPECT_EQ(shoulder_->get_angular_rate(*context_), 0.0);
  EXPECT_NEAR(elbow_->get_angular_rate(*context_),
              initial_rate - num_steps * dt * kElbowFriction, kTolerance);
}

// Under an applied force below the friction bound, joints stay in stiction:
// friction balances the applied force up to the regularization creep.
TEST_F(SapDriverJointFrictionTest, Stiction) {
  MakePlant(1.0e-3);
  const double wheel_torque = 0.5 * kWheelFriction;
  const double slider_force = -0.8 * kSliderFriction;
  SetActuation(wheel_torque, slider_force);

  // Velocities after the first step, evaluated without updating the context.
  const ContactSolverResults<double>& results =
      manager_->EvalContactSolverResults(*context_);
  const double wheel_v = results.v_next(wheel_->velocity_start());
  const double slider_v = results.v_next(slider_->velocity_start());
  const double wheel_v_expected =
      ExpectedVelocityFromRest(wheel_torque, kWheelInertia, kWheelFriction);
  const double slider_v_expected =
      ExpectedVelocityFromRest(slider_force, kSliderMass, kSliderFriction);
  EXPECT_NEAR(wheel_v, wheel_v_expected,
              kTolerance * std::abs(wheel_v_expected));
  EXPECT_NEAR(slider_v, slider_v_expected,
              kTolerance * std::abs(slider_v_expected));
  // The creep is a small fraction of the free motion velocity.
  const double dt = plant_->time_step();
  EXPECT_LT(std::abs(wheel_v),
            2.0 * kSigma * dt * std::abs(wheel_torque) / kWheelInertia);
  EXPECT_LT(std::abs(slider_v),
            2.0 * kSigma * dt * std::abs(slider_force) / kSliderMass);

  // The reported generalized forces include actuation and the friction force.
  // Friction balances the applied force so that the net force produces the
  // creep velocity, τ_net = A⋅v/δt.
  const MultibodyForces<double>& forces =
      manager_->EvalDiscreteUpdateMultibodyForces(*context_);
  const VectorXd& tau = forces.generalized_forces();
  EXPECT_NEAR(tau(wheel_->velocity_start()), kWheelInertia * wheel_v / dt,
              kTolerance * wheel_torque);
  EXPECT_NEAR(tau(slider_->velocity_start()), kSliderMass * slider_v / dt,
              kTolerance * std::abs(slider_force));
  // The arm is not actuated and remains at rest.
  EXPECT_EQ(results.v_next(shoulder_->velocity_start()), 0.0);
  EXPECT_EQ(results.v_next(elbow_->velocity_start()), 0.0);
  EXPECT_EQ(tau(elbow_->velocity_start()), 0.0);

  // Over many steps the joints creep but do not break away.
  Step(100);
  EXPECT_LT(std::abs(wheel_->get_angular_rate(*context_)),
            2.0 * kSigma * dt * std::abs(wheel_torque) / kWheelInertia);
  EXPECT_LT(std::abs(slider_->get_translation_rate(*context_)),
            2.0 * kSigma * dt * std::abs(slider_force) / kSliderMass);
}

// Under an applied force above the friction bound, joints slide with friction
// saturated at its bound and opposing motion.
TEST_F(SapDriverJointFrictionTest, Sliding) {
  MakePlant(1.0e-3);
  const double wheel_torque = 3.0 * kWheelFriction;
  const double slider_force = -2.5 * kSliderFriction;
  SetActuation(wheel_torque, slider_force);

  const ContactSolverResults<double>& results =
      manager_->EvalContactSolverResults(*context_);
  const double wheel_v = results.v_next(wheel_->velocity_start());
  const double slider_v = results.v_next(slider_->velocity_start());
  const double wheel_v_expected =
      ExpectedVelocityFromRest(wheel_torque, kWheelInertia, kWheelFriction);
  const double slider_v_expected =
      ExpectedVelocityFromRest(slider_force, kSliderMass, kSliderFriction);
  EXPECT_NEAR(wheel_v, wheel_v_expected,
              kTolerance * std::abs(wheel_v_expected));
  EXPECT_NEAR(slider_v, slider_v_expected,
              kTolerance * std::abs(slider_v_expected));

  // The reported generalized force is the applied force minus the saturated
  // friction force opposing motion.
  const MultibodyForces<double>& forces =
      manager_->EvalDiscreteUpdateMultibodyForces(*context_);
  const VectorXd& tau = forces.generalized_forces();
  EXPECT_NEAR(tau(wheel_->velocity_start()), wheel_torque - kWheelFriction,
              kTolerance * wheel_torque);
  EXPECT_NEAR(tau(slider_->velocity_start()), slider_force + kSliderFriction,
              kTolerance * std::abs(slider_force));

  // With constant applied forces the joints accelerate uniformly.
  const int num_steps = 50;
  Step(num_steps);
  EXPECT_NEAR(wheel_->get_angular_rate(*context_), num_steps * wheel_v_expected,
              kTolerance * num_steps * std::abs(wheel_v_expected));
  EXPECT_NEAR(slider_->get_translation_rate(*context_),
              num_steps * slider_v_expected,
              kTolerance * num_steps * std::abs(slider_v_expected));
}

// The response to an applied force is continuous through the breakaway force
// (1+σ)⋅τ_c and matches the analytical solution on both sides of it.
TEST_F(SapDriverJointFrictionTest, Breakaway) {
  MakePlant(1.0e-3);
  const double breakaway_torque = (1.0 + kSigma) * kWheelFriction;
  for (const double factor :
       {-1.5, -1.001, -0.999, -0.5, 0.5, 0.999, 1.001, 1.5}) {
    const double wheel_torque = factor * breakaway_torque;
    SetActuation(wheel_torque, 0.0);
    const ContactSolverResults<double>& results =
        manager_->EvalContactSolverResults(*context_);
    const double wheel_v = results.v_next(wheel_->velocity_start());
    const double wheel_v_expected =
        ExpectedVelocityFromRest(wheel_torque, kWheelInertia, kWheelFriction);
    EXPECT_NEAR(wheel_v, wheel_v_expected,
                kTolerance * std::abs(wheel_v_expected))
        << "factor = " << factor;
  }
}

// A spinning wheel without actuation decelerates uniformly under friction until
// it comes to rest, dissipating its kinetic energy over an angle independent of
// the time step. The velocity never reverses sign.
TEST_F(SapDriverJointFrictionTest, StoppingUnderFriction) {
  constexpr double kInitialRate = 2.0;  // [rad/s]
  // Uniform deceleration and the resulting analytical stopping time and angle.
  const double deceleration = kWheelFriction / kWheelInertia;
  const double stopping_time = kInitialRate / deceleration;
  const double stopping_angle =
      0.5 * kInitialRate * kInitialRate / deceleration;

  for (const double dt : {1.0e-3, 4.0e-3}) {
    MakePlant(dt);
    wheel_->set_angular_rate(context_.get(), kInitialRate);

    // Sliding phase: the rate decreases by δt⋅τ_c/I each step.
    const int num_sliding_steps = static_cast<int>(stopping_time / dt) - 1;
    for (int step = 1; step <= num_sliding_steps; ++step) {
      Step(1);
      const double expected_rate = kInitialRate - step * dt * deceleration;
      EXPECT_NEAR(wheel_->get_angular_rate(*context_), expected_rate,
                  kTolerance * kInitialRate)
          << "dt = " << dt << ", step = " << step;
    }

    // Continue past the stopping time. The wheel must come to rest (up to the
    // regularization creep, which decays geometrically) without reversing.
    const int extra_steps = static_cast<int>(0.1 * stopping_time / dt) + 5;
    for (int step = 0; step < extra_steps; ++step) {
      Step(1);
      EXPECT_GE(wheel_->get_angular_rate(*context_), 0.0);
    }
    EXPECT_LT(wheel_->get_angular_rate(*context_), 1.0e-12);

    // The kinetic energy ½⋅I⋅ω₀² is dissipated by friction over the stopping
    // angle, up to the first order discretization error of the symplectic
    // Euler scheme.
    EXPECT_NEAR(wheel_->get_angle(*context_), stopping_angle, kInitialRate * dt)
        << "dt = " << dt;
  }
}

// The driver is also instantiated on AutoDiffXd. The scalar converted plant
// must produce the same velocities as the double plant.
TEST_F(SapDriverJointFrictionTest, AutoDiffXd) {
  MakePlant(1.0e-3);
  const double wheel_torque = 3.0 * kWheelFriction;   // Sliding.
  const double slider_force = 0.5 * kSliderFriction;  // Stiction.
  SetActuation(wheel_torque, slider_force);

  std::unique_ptr<systems::System<AutoDiffXd>> system_ad =
      plant_->ToAutoDiffXd();
  const auto& plant_ad =
      dynamic_cast<const MultibodyPlant<AutoDiffXd>&>(*system_ad);
  std::unique_ptr<Context<AutoDiffXd>> context_ad =
      plant_ad.CreateDefaultContext();
  context_ad->SetTimeStateAndParametersFrom(*context_);
  VectorX<AutoDiffXd> u_ad =
      plant_->get_actuation_input_port().Eval(*context_).cast<AutoDiffXd>();
  plant_ad.get_actuation_input_port().FixValue(context_ad.get(), u_ad);

  Step(1);
  plant_ad.ExecuteForcedEvents(context_ad.get(), /* publish = */ false);

  const VectorXd v = plant_->GetVelocities(*context_);
  const VectorXd v_ad = math::ExtractValue(plant_ad.GetVelocities(*context_ad));
  EXPECT_TRUE(CompareMatrices(v_ad, v, kTolerance * v.norm()));
  // Sanity check that the model did move.
  EXPECT_GT(v(wheel_->velocity_start()), 0.0);
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
