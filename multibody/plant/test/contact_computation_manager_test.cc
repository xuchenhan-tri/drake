#include "drake/multibody/plant/contact_computation_manager.h"

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_access.h"
#include "drake/systems/analysis/simulator.h"

using Eigen::VectorXd;

namespace drake {
using systems::BasicVector;
using systems::Context;
using systems::DiscreteStateIndex;
using systems::DiscreteValues;
using systems::LeafSystem;
using systems::OutputPortIndex;

namespace multibody {

using contact_solvers::internal::ContactSolverResults;

namespace internal {
constexpr int kNumDofs = 6;
constexpr int kNumContacts = 4;
constexpr double kDummyVNext = 1.0;
constexpr double kDummyFn = 2.0;
constexpr double kDummyFt = 3.0;
constexpr double kDummyVn = 4.0;
constexpr double kDummyVt = 5.0;
constexpr double kDummyTau = 6.0;
constexpr double kDummyVdot = 7.0;
constexpr double kDt = 0.1;

/* A dummy manager class derived from ContactComputationManager for testing
 purpose. This class has a single discrete state of size `kNumDofs` that are
 intialized to zeros. The discrete update event increments each entry in the
 discrete state vector by 1. This class has a single abstract output port that
 reports the discrete states. */
class DummyContactComputationManager : public ContactComputationManager<double>,
                                       private MultibodyPlantAccess<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DummyContactComputationManager);

  ~DummyContactComputationManager() = default;

  explicit DummyContactComputationManager(MultibodyPlant<double>* plant)
      : MultibodyPlantAccess<double>(plant) {}

  const systems::OutputPort<double>& get_dummy_output_port() const {
    return this->get_output_port(output_port_index_);
  }

  const ContactSolverResults<double>& EvalContactSolverResults(
      const Context<double>& context) {
    return MultibodyPlantAccess<double>::EvalContactSolverResults(context);
  }

 private:
  void DeclareStateCacheAndPorts(MultibodyPlant<double>* plant) final {
    discrete_state_index_ =
        this->DeclareDiscreteState(VectorXd::Zero(kNumDofs));
    output_port_index_ =
        this->DeclareAbstractOutputPort(
                "dummy_output_port",
                []() {
                  VectorXd model_value = VectorXd::Zero(kNumDofs);
                  return AbstractValue::Make(model_value);
                },
                [this](const Context<double>& context, AbstractValue* output) {
                  VectorXd& data = output->get_mutable_value<VectorXd>();
                  data = context.get_discrete_state(discrete_state_index_)
                             .get_value();
                },
                // TODO(xuchenhan_tri): It may be worthwhile to expose the
                // discrete_state_ticket() method from MbP so that more
                // fine-grained ticket can be created.
                {systems::System<double>::xd_ticket()})
            .get_index();
  }

  // Assigns dummy values to the output ContactSolverResults.
  void CalcContactSolverResults(
      const Context<double>&,
      ContactSolverResults<double>* results) const final {
    results->Resize(kNumDofs, kNumContacts);
    results->v_next = VectorXd::Ones(kNumDofs) * kDummyVNext;
    results->fn = VectorXd::Ones(kNumContacts) * kDummyFn;
    results->ft = VectorXd::Ones(2 * kNumContacts) * kDummyFt;
    results->vn = VectorXd::Ones(kNumContacts) * kDummyVn;
    results->vt = VectorXd::Ones(2 * kNumContacts) * kDummyVt;
    results->tau_contact = VectorXd::Ones(kNumDofs) * kDummyTau;
  }

  // The dofs in the system has a constant velocity of one, therefore it should
  // have zero acceleration; however, we put a non-zero value in the
  // acceleration cache for testing purpose. */
  void CalcAccelerationKinematicsCache(
      const Context<double>& context,
      internal::AccelerationKinematicsCache<double>* ac) const final {
    VectorXd& vdot = ac->get_mutable_vdot();
    vdot = VectorXd::Ones(vdot.size()) * kDummyVdot;
  }

  // Increments all discrete state by 1.
  void CalcDiscreteValues(const Context<double>& context0,
                          DiscreteValues<double>* updates) const final {
    auto discrete_data =
        updates->get_mutable_vector(discrete_state_index_).get_mutable_value();
    discrete_data += VectorXd::Ones(discrete_data.size());
  }

  OutputPortIndex output_port_index_;
  DiscreteStateIndex discrete_state_index_;
};

class ContactComputationManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    plant_.AddRigidBody("rigid body", SpatialInertia<double>());
    plant_.Finalize();
    EXPECT_EQ(plant_.num_velocities(), kNumDofs);
    manager_ = &plant_.set_contact_manager(
        std::make_unique<DummyContactComputationManager>(&plant_));
  }

  MultibodyPlant<double> plant_{kDt};                 // A discrete MbP.
  DummyContactComputationManager* manager_{nullptr};  // The manager under test.
};

TEST_F(ContactComputationManagerTest, CalcDiscreteState) {
  auto context = plant_.CreateDefaultContext();
  auto simulator = systems::Simulator<double>(plant_, std::move(context));
  const int time_steps = 10;
  simulator.AdvanceTo(time_steps * kDt);
  const VectorXd final_discrete_state =
      manager_->get_dummy_output_port().Eval<VectorXd>(simulator.get_context());
  EXPECT_TRUE(CompareMatrices(final_discrete_state,
                              VectorXd::Ones(kNumDofs) * time_steps));
}

TEST_F(ContactComputationManagerTest, CalcContactSolverResults) {
  auto context = plant_.CreateDefaultContext();
  const ContactSolverResults<double>& results =
      manager_->EvalContactSolverResults(*context);
  EXPECT_TRUE(
      CompareMatrices(results.v_next, VectorXd::Ones(kNumDofs) * kDummyVNext));
  EXPECT_TRUE(
      CompareMatrices(results.fn, VectorXd::Ones(kNumContacts) * kDummyFn));
  EXPECT_TRUE(
      CompareMatrices(results.ft, VectorXd::Ones(2 * kNumContacts) * kDummyFt));
  EXPECT_TRUE(
      CompareMatrices(results.vn, VectorXd::Ones(kNumContacts) * kDummyVn));
  EXPECT_TRUE(
      CompareMatrices(results.vt, VectorXd::Ones(2 * kNumContacts) * kDummyVt));
  EXPECT_TRUE(CompareMatrices(results.tau_contact,
                              VectorXd::Ones(kNumDofs) * kDummyTau));
}

TEST_F(ContactComputationManagerTest, CalcAccelerationKinematicsCache) {
  auto context = plant_.CreateDefaultContext();
  const auto generalized_acceleration =
      plant_.get_generalized_acceleration_output_port().Eval(*context);
  EXPECT_TRUE(CompareMatrices(generalized_acceleration,
                              VectorXd::Ones(kNumDofs) * kDummyVdot));
}
}  // namespace internal
}  // namespace multibody
}  // namespace drake
