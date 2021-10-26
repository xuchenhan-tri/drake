#include "drake/multibody/plant/compliant_contact_computation_manager.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/contact_solvers/block_sparse_linear_operator.h"
#include "drake/multibody/contact_solvers/contact_solver.h"
#include "drake/multibody/contact_solvers/timer.h"
#include "drake/multibody/plant/contact_permutation_utils.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/context.h"
#include "drake/common/test_utilities/limit_malloc.h"

#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;

using drake::geometry::GeometryId;
using drake::math::RotationMatrix;
using drake::systems::Context;

namespace drake {
namespace multibody {

template <typename T>
CompliantContactComputationManager<T>::CompliantContactComputationManager(
    std::unique_ptr<contact_solvers::internal::ContactSolver<T>> contact_solver)
    : contact_solver_(std::move(contact_solver)) {}

template <typename T>
void CompliantContactComputationManager<T>::ExtractModelInfo() {
  const internal::MultibodyTreeTopology& topology =
      this->internal_tree().get_topology();

  internal::ComputeBfsToDfsPermutation(topology, &velocities_permutation_,
                                       &body_to_tree_map_);
  num_trees_ = velocities_permutation_.size();
  num_velocities_ = plant().num_velocities();

  // Allocate some needed workspace.
  const int nv = plant().num_velocities();
  workspace_.M.resize(nv, nv);
  if (theta_v_ == 0 && theta_qv_ == 0)
    workspace_.aux_plant_context_ = plant().CreateDefaultContext();
  tau_c_.resize(nv);
  tau_c_.setZero();
}

template <typename T>
void CompliantContactComputationManager<T>::DeclareCacheEntries() {
    auto& contact_jacobian_cache_entry = this->DeclareCacheEntry(
        std::string("Contact Jacobians Jc(q)."),
        systems::ValueProducer(
            internal::ContactJacobianCache<T>(),
            std::function<void(const systems::Context<T>&,
                               internal::ContactJacobianCache<T>*)>{
                [this](
                    const systems::Context<T>& context,
                    internal::ContactJacobianCache<T>* contact_jacobian_cache) {
                  const std::vector<internal::DiscreteContactPair<T>>&
                      contact_pairs = plant().EvalDiscreteContactPairs(context);
                  CalcContactJacobian(context, contact_pairs,
                                      &contact_jacobian_cache->Jc,
                                      &contact_jacobian_cache->R_WC_list);
                }}),        
        // N.B. We use xd_ticket() instead of q_ticket() since discrete
        // multibody plant does not have q's, but rather discrete state.
        // Therefore if we make it dependent on q_ticket() the Jacobian only
        // gets evaluated once at the start of the simulation.
        {systems::System<T>::xd_ticket(),
         systems::System<T>::all_parameters_ticket()});
    cache_indexes_.contact_jacobian =
        contact_jacobian_cache_entry.cache_index();
}

template <typename T>
void CompliantContactComputationManager<T>::CalcContactGraph(
    const geometry::QueryObject<T>& query_object,
    const std::vector<internal::DiscreteContactPair<T>>& contact_pairs) const {
  std::vector<SortedPair<int>> contacts;
  const geometry::SceneGraphInspector<T>& inspector = query_object.inspector();
  for (const auto& pp : contact_pairs) {
    const geometry::FrameId frameA = inspector.GetFrameId(pp.id_A);
    const geometry::FrameId frameB = inspector.GetFrameId(pp.id_B);
    const Body<T>* bodyA = plant().GetBodyFromFrameId(frameA);
    const Body<T>* bodyB = plant().GetBodyFromFrameId(frameB);
    DRAKE_DEMAND(bodyA != nullptr);
    DRAKE_DEMAND(bodyB != nullptr);
    const BodyIndex bodyA_index = bodyA->index();
    const BodyIndex bodyB_index = bodyB->index();

    const int treeA = body_to_tree_map_[bodyA_index];
    const int treeB = body_to_tree_map_[bodyB_index];
    // SceneGraph does not report collisions between anchored geometries.
    // We verify this.
    DRAKE_DEMAND(!(treeA < 0 && treeB < 0));
    contacts.push_back({treeA, treeB});
  }

  graph_ = internal::ComputeContactGraph(num_trees_, contacts,
                                         &participating_trees_);
  num_participating_trees_ = participating_trees_.size();

  num_participating_velocities_ = 0;
  participating_velocities_permutation_.resize(num_participating_trees_);
  for (int tp = 0; tp < num_participating_trees_; ++tp) {
    const int t = participating_trees_[tp];
    const int nt = velocities_permutation_[t].size();
    participating_velocities_permutation_[tp].resize(nt);
    // participating_velocities_permutation_[tp] =
    // num_participating_velocities_:num_participating_velocities_+nt-1.
    std::iota(participating_velocities_permutation_[tp].begin(),
              participating_velocities_permutation_[tp].end(),
              num_participating_velocities_);
    num_participating_velocities_ += nt;
  }
}

template <typename T>
void CompliantContactComputationManager<T>::
    CalcVelocityUpdateWithoutConstraints(
        const systems::Context<T>& context0, VectorX<T>* vstar,
        VectorX<T>* participating_vstar, VectorX<T>* v_guess,
        VectorX<T>* participating_v_guess) const {
  DRAKE_DEMAND(vstar != nullptr);
  DRAKE_DEMAND(participating_vstar != nullptr);
  DRAKE_DEMAND(vstar->size() == plant().num_velocities());
  DRAKE_DEMAND(v_guess->size() == plant().num_velocities());

  // MultibodyTreeSystem::CalcArticulatedBodyForceCache()
  MultibodyForces<T> forces0(plant());

  const internal::PositionKinematicsCache<T>& pc =
      plant().EvalPositionKinematics(context0);
  const internal::VelocityKinematicsCache<T>& vc =
      plant().EvalVelocityKinematics(context0);

  // Compute forces applied by force elements. Note that this resets forces
  // to empty so must come first.
  this->internal_tree().CalcForceElementsContribution(context0, pc, vc,
                                                      &forces0);

  // We need only handle MultibodyPlant-specific forces here.
  this->AddInForcesFromInputPorts(context0, &forces0);

  // Perform the tip-to-base pass to compute the force bias terms needed by ABA.
  const auto& tree_topology = this->internal_tree().get_topology();
  internal::ArticulatedBodyForceCache<T> aba_force_cache(tree_topology);
  this->internal_tree().CalcArticulatedBodyForceCache(context0, forces0,
                                                      &aba_force_cache);

  // MultibodyTreeSystem::CalcArticulatedBodyAccelerations()
  internal::AccelerationKinematicsCache<T> ac(tree_topology);
  this->internal_tree().CalcArticulatedBodyAccelerations(context0,
                                                         aba_force_cache, &ac);

  // Notice we use an explicit Euler scheme here since all forces are evaluated
  // at context0.
  const VectorX<T>& vdot0 = ac.get_vdot();
  const double dt = plant().time_step();

  const auto x0 = context0.get_discrete_state(0).get_value();
  // const VectorX<T> q0 = x0.topRows(this->num_positions());
  const VectorX<T> v0 = x0.bottomRows(plant().num_velocities());

  *vstar = v0 + dt * vdot0;

  // Extract "reduced" velocities for participating trees only.
  // TODO: rename to vstar_part?
  participating_vstar->resize(num_participating_velocities_);
  PermuteFullToParticipatingVelocities(*vstar, participating_vstar);

  *v_guess = v0;
  participating_v_guess->resize(num_participating_velocities_);
  PermuteFullToParticipatingVelocities(*v_guess, participating_v_guess);

// This strategy actually increased the number of iterations by about 20%.
// Thus var v_guess = v0 works best.
#if 0
  // Compute guess by using previous tau_c.
  // const auto& tau_c = plant().EvalContactSolverResults(context0).tau_contact;
  forces0.mutable_generalized_forces() += tau_c_;
  plant().internal_tree().CalcArticulatedBodyForceCache(context0, forces0,
                                                        &aba_force_cache);
  plant().internal_tree().CalcArticulatedBodyAccelerations(
      context0, aba_force_cache, &ac);
  *v_guess = v0 + dt * ac.get_vdot();
  participating_v_guess->resize(num_participating_velocities_);
  PermuteFullToParticipatingVelocities(*v_guess, participating_v_guess);
#endif
}

template <typename T>
void CompliantContactComputationManager<T>::
    CalcVelocityUpdateWithoutConstraintsUsingThetaMethod(
        const systems::Context<T>& context0, VectorX<T>* vstar,
        VectorX<T>* participating_vstar, VectorX<T>* v_guess,
        VectorX<T>* participating_v_guess,
        systems::Context<T>* context_star) const {
  // NR parameters. Expose if needed.
  const int max_iters = 30;
  const double rel_tol = 1.0e-12;
  const double abs_tol = 1.0e-16;

  // Context used to compute x*.
  systems::Context<T>& context = *context_star;

  // Set time, state and parameters at x0 and also fix inputs to u0.
  if constexpr (std::is_same<T, double>::value) {
    plant().FixInputPortsFrom(plant(), context0, &context);
  } else {
    throw std::runtime_error(
        "Context::FixInputPort() not supported for this scalar type");
  }

  const double dt = plant().time_step();

  // We want to approximate the ODEs Jacobian using the "most important" terms.
  // These are the mass matrix M, rotor inertia and joint dissipation D. For
  // stable simulations we'd like these to be implicit.
  // As a work-around to build M + theta*dt*D, I "trick" the system to believe
  // that the total rotor inertia is Ieff + theta*dt*D. I do this by temporarily
  // modifying the gear ration and rotor inertia parameters.
  for (JointActuatorIndex act_idx(0); act_idx < plant().num_actuators();
       ++act_idx) {
    const auto& act = plant().get_joint_actuator(act_idx);
    const T Irefl = act.calc_reflected_inertia(context);
    const auto& joint = act.joint();
    DRAKE_DEMAND(joint.damping().size() == 1);
    const double damping = joint.damping()[0];

    // This effectively modifies the mass matrix to M + Irefl + theta*dt*D.
    const T diagonal = Irefl + dt * theta_v_ * damping;
    act.SetGearRatio(&context, 1.0);
    act.SetRotorInertia(&context, diagonal);
  }

  // Fixed-point iterations (or quasi-Newton iterations)
  const int nq = plant().num_positions();
  const int nv = plant().num_velocities();
  const auto x0 = plant().GetPositionsAndVelocities(context0);
  const auto q0 = x0.head(nq);
  const auto v0 = x0.tail(nv);

  // We make x_theta to store [q(θq); v(θv)].
  VectorX<T> x_theta(plant().num_multibody_states());  // current iterate k.
  auto q_theta = x_theta.head(nq);
  auto v_theta = x_theta.tail(nv);

  // Workspace for previous iteration and xdot.
  VectorX<T> xkm(plant().num_multibody_states());  // previous iterate k-1.
  VectorX<T> xdot(plant().num_multibody_states());

  // auto x = plant().GetMutablePositionsAndVelocities(&context);
  // Initial guess to x = x0.
  context.SetTimeStateAndParametersFrom(context0);
  xkm = x0;  // previous iteration.
  auto q_theta_km = xkm.head(nq);
  auto v_theta_km = xkm.tail(nv);
  int k = 1;
  T err = 0;
  double relaxation = 1.0;
  for (; k <= max_iters; ++k) {
    xdot = CalcXdot(context);
    // Update state at θv.
    v_theta = (1.0 - relaxation) * v_theta_km +
              relaxation * (v0 + dt * theta_v_ * xdot.tail(nv));
    q_theta = (1.0 - relaxation) * q_theta_km +
              relaxation * (q0 + dt * theta_qv_ * xdot.head(nq));
    // this invalidates caching as desired.
    plant().SetPositionsAndVelocities(&context, x_theta);    
    err = (x_theta - xkm).norm();
    if (err < abs_tol + rel_tol * x_theta.norm()) {
      break;
    }
    xkm = x_theta;
  }
  PRINT_VAR(k);
  PRINT_VAR(err);
  if (k >= max_iters-1) {
    throw std::runtime_error("Computation of x_star did not converge.");
  }

  // At this points xk stores xθ = θx + (1−θ)x₀.
  // xdot stores ẋ(xθ), evaluated at xθ.
  // Therefore we can obtain x = x₀ + dtẋ
  plant().GetMutablePositionsAndVelocities(&context) = x0 + dt * xdot;

  // Extract velocities.
  *vstar = plant().GetVelocities(context);

  // Extract "reduced" velocities for participating trees only.
  // TODO: rename to vstar_part?
  participating_vstar->resize(num_participating_velocities_);
  PermuteFullToParticipatingVelocities(*vstar, participating_vstar);

  *v_guess = x0.tail(nv);
  participating_v_guess->resize(num_participating_velocities_);
  PermuteFullToParticipatingVelocities(*v_guess, participating_v_guess);
}

template <typename T>
void CompliantContactComputationManager<T>::CalcLinearDynamics(
    const systems::Context<T>& context,
    contact_solvers::internal::BlockSparseMatrix<T>* A) const {
  DRAKE_DEMAND(A != nullptr);
  plant().CalcMassMatrix(context, &workspace_.M);
  *A = internal::ExtractBlockDiagonalMassMatrix(
      workspace_.M, velocities_permutation_, participating_trees_);
}

template <typename T>
void CompliantContactComputationManager<T>::CalcContactQuantities(
    const systems::Context<T>& context,
    const std::vector<internal::DiscreteContactPair<T>>& contact_pairs,
    contact_solvers::internal::BlockSparseMatrix<T>* Jc, VectorX<T>* phi0,
    VectorX<T>* vc0, VectorX<T>* mu, VectorX<T>* stiffness,
    VectorX<T>* linear_damping) const {
  DRAKE_DEMAND(Jc != nullptr);
  const internal::ContactJacobianCache<T>& contact_jacobian =
      this->EvalContactJacobianCache(context);
  *Jc = internal::ExtractBlockJacobian(contact_jacobian.Jc, graph_,
                                       velocities_permutation_,
                                       participating_trees_);

  const std::vector<CoulombFriction<double>> combined_friction_pairs =
      this->CalcCombinedFrictionCoefficients(context,
                                                    contact_pairs);

  const int num_contacts = contact_pairs.size();
  phi0->resize(num_contacts);
  vc0->resize(3 * num_contacts);
  mu->resize(num_contacts);
  stiffness->resize(num_contacts);
  linear_damping->resize(num_contacts);

  const auto x0 = context.get_discrete_state(0).get_value();
  const VectorX<T> v0 = x0.bottomRows(plant().num_velocities());
  VectorX<T> v0_part(num_participating_velocities_);
  PermuteFullToParticipatingVelocities(v0, &v0_part);
  Jc->Multiply(v0_part, vc0);

  int k_permuted = 0;
  for (const auto& p : graph_.patches) {
    for (int k : p.contacts) {
      (*phi0)[k_permuted] = contact_pairs[k].phi0;
      (*stiffness)[k_permuted] = contact_pairs[k].stiffness;
      (*linear_damping)[k_permuted] = contact_pairs[k].damping;
      (*mu)[k_permuted] = combined_friction_pairs[k].dynamic_friction();
      ++k_permuted;
    }
  }
}

template <typename T>
void CompliantContactComputationManager<T>::DoCalcContactSolverResults(
    const systems::Context<T>& context,
    contact_solvers::internal::ContactSolverResults<T>* results) const {
  contact_solvers::internal::Timer total_timer;

  ContactManagerStats stats;
  stats.time = ExtractDoubleOrThrow(context.get_time());

  contact_solvers::internal::Timer timer;
  // const auto& query_object = plant().EvalGeometryQueryInput(context);
  const auto& query_object =
      plant()
          .get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
  const std::vector<internal::DiscreteContactPair<T>>& contact_pairs =
      plant().EvalDiscreteContactPairs(context);
  stats.geometry_time = timer.Elapsed();

  timer.Reset();
  CalcContactGraph(query_object, contact_pairs);
  stats.graph_time = timer.Elapsed();

  // After CalcContactGraph() we know the problem size and topology.
  const int nc = contact_pairs.size();

  stats.num_contacts = nc;

  timer.Reset();
  VectorX<T> vstar(plant().num_velocities());
  VectorX<T> participating_vstar(num_participating_velocities_);
  VectorX<T> v_guess(plant().num_velocities());
  VectorX<T> participating_v_guess(num_participating_velocities_);
  if (theta_v_ == 0 && theta_qv_ == 0) {
    CalcVelocityUpdateWithoutConstraints(context, &vstar, &participating_vstar,
                                         &v_guess, &participating_v_guess);
  } else {
    systems::Context<T>& context_star = *workspace_.aux_plant_context_;
    CalcVelocityUpdateWithoutConstraintsUsingThetaMethod(
        context, &vstar, &participating_vstar, &v_guess, &participating_v_guess,
        &context_star);
  }
  stats.vstar_time = timer.Elapsed();

  contact_solvers::internal::ContactSolverResults<T>
      participating_trees_results;
  if (nc > 0) {
    timer.Reset();
    // Computes the lienarized dynamics matrix A in A(v-v*) = Jᵀγ.
    contact_solvers::internal::BlockSparseMatrix<T> A;
    if (theta_v_ == 0 && theta_qv_ == 0) {
      CalcLinearDynamics(context, &A);
    } else {
      systems::Context<T>& context_star = *workspace_.aux_plant_context_;
      CalcLinearDynamics(context_star, &A);
    }
    stats.linear_dynamics_time = timer.Elapsed();

    timer.Reset();
    // Computes quantities to define contact constraints.
    contact_solvers::internal::BlockSparseMatrix<T> Jc;
    VectorX<T> phi0, vc0, mu, stiffness, linear_damping;
    CalcContactQuantities(context, contact_pairs, &Jc, &phi0, &vc0, &mu,
                          &stiffness, &linear_damping);
    stats.contact_jacobian_time = timer.Elapsed();

    // Create data structures to call the contact solver.
    contact_solvers::internal::BlockSparseLinearOperator<T> Aop("A", &A);
    contact_solvers::internal::BlockSparseLinearOperator<T> Jop("Jc", &Jc);
    contact_solvers::internal::SystemDynamicsData<T> dynamics_data(
        &Aop, nullptr, &participating_vstar);
    contact_solvers::internal::PointContactData<T> contact_data(
        &phi0, &vc0, &Jop, &stiffness, &linear_damping, &mu);

    // Initial guess.
    // const auto x0 = context.get_discrete_state(0).get_value();
    // const VectorX<T> v0 = x0.bottomRows(plant().num_velocities());
    // VectorX<T> participating_v0(num_participating_velocities_);
    // PermuteFullToParticipatingVelocities(v0, &participating_v0);

    timer.Reset();
    // Call contact solver.
    // TODO: consider using participating_v0 as the initial guess.
    participating_trees_results.Resize(num_participating_velocities_, nc);
    const contact_solvers::internal::ContactSolverStatus info =
        contact_solver_->SolveWithGuess(plant().time_step(), dynamics_data,
                                        contact_data, participating_v_guess,
                                        &participating_trees_results);
    if (info != contact_solvers::internal::ContactSolverStatus::kSuccess) {
      const std::string msg =
          fmt::format("MultibodyPlant's contact solver of type '" +
                          NiceTypeName::Get(*contact_solver_) +
                          "' failed to converge at "
                          "simulation time = {:7.3g} with discrete update "
                          "period = {:7.3g}.",
                      context.get_time(), plant().time_step());
      throw std::runtime_error(msg);
    }
    stats.contact_solver_time = timer.Elapsed();
  }  // nc > 0

  // ==========================================================================
  // ==========================================================================
  timer.Reset();
  // Permute "backwards" to the original ordering.
  // TODO: avoid heap allocations.
  results->Resize(num_velocities_, nc);
  // This effectively updates the non-participating velocities.
  results->v_next = vstar;
  // This effectively updates the non-participating generalized forces.
  results->tau_contact.setZero();
  tau_c_.setZero();

  // Update participating quantities.
  if (nc > 0) {
    PermuteParticipatingToFullVelocities(participating_trees_results.v_next,
                                         &results->v_next);
    PermuteParticipatingToFullVelocities(
        participating_trees_results.tau_contact, &results->tau_contact);

    // Save tau_c to compute initial guess in the next time step.
    tau_c_ = results->tau_contact;

    // TODO(amcastro-tri): Remove these when queries are computed in patch
    // order.
    PermuteFromPatches(1, participating_trees_results.fn, &results->fn);
    PermuteFromPatches(2, participating_trees_results.ft, &results->ft);
    PermuteFromPatches(1, participating_trees_results.vn, &results->vn);
    PermuteFromPatches(2, participating_trees_results.vt, &results->vt);
  }
  stats.pack_results_time = timer.Elapsed();

  stats.total_time = total_timer.Elapsed();

  stats_.push_back(stats);

  total_time_ += total_timer.Elapsed();
}

// forward = true --> vp(ip) = v(i).
// forward = false --> v(i) = vp(ip).
template <typename T>
void CompliantContactComputationManager<T>::PermuteVelocities(
    bool forward, VectorX<T>* v, VectorX<T>* vp) const {
  DRAKE_DEMAND(v != nullptr);
  DRAKE_DEMAND(vp != nullptr);
  DRAKE_DEMAND(v->size() == plant().num_velocities());
  DRAKE_DEMAND(vp->size() == num_participating_velocities_);
  for (int tp = 0; tp < num_participating_trees_; ++tp) {
    const int t = participating_trees_[tp];
    const int nt = participating_velocities_permutation_[tp].size();
    for (int vt = 0; vt < nt; ++vt) {
      const int i = velocities_permutation_[t][vt];
      const int ip = participating_velocities_permutation_[tp][vt];
      if (forward) {
        (*vp)(ip) = (*v)(i);
      } else {
        (*v)(i) = (*vp)(ip);
      }
    }
  }
}

template <typename T>
void CompliantContactComputationManager<
    T>::PermuteFullToParticipatingVelocities(const VectorX<T>& v,
                                             VectorX<T>* vp) const {
  PermuteVelocities(true, const_cast<VectorX<T>*>(&v), vp);
}

template <typename T>
void CompliantContactComputationManager<
    T>::PermuteParticipatingToFullVelocities(const VectorX<T>& vp,
                                             VectorX<T>* v) const {
  PermuteVelocities(false, v, const_cast<VectorX<T>*>(&vp));
}

// forward = true --> xp(kp) = x(k).
// forward = false --> x(k) = xp(kp).
template <typename T>
void CompliantContactComputationManager<T>::PermuteContacts(
    bool forward, int stride, VectorX<T>* x, VectorX<T>* xp) const {
  DRAKE_DEMAND(x != nullptr);
  DRAKE_DEMAND(xp != nullptr);
  DRAKE_DEMAND(x->size() == xp->size());
  int k_perm = 0;
  for (const auto& p : graph_.patches) {
    for (int k : p.contacts) {
      if (forward) {
        xp->segment(stride * k_perm, stride) = x->segment(stride * k, stride);
      } else {
        x->segment(stride * k, stride) = xp->segment(stride * k_perm, stride);
      }
      ++k_perm;
    }
  }
};

template <typename T>
void CompliantContactComputationManager<T>::PermuteIntoPatches(
    int stride, const VectorX<T>& x, VectorX<T>* xp) const {
  PermuteContacts(true, stride, const_cast<VectorX<T>*>(&x), xp);
}

template <typename T>
void CompliantContactComputationManager<T>::PermuteFromPatches(
    int stride, const VectorX<T>& xp, VectorX<T>* x) const {
  PermuteContacts(false, stride, x, const_cast<VectorX<T>*>(&xp));
}

template <typename T>
void CompliantContactComputationManager<T>::MinvOperator(
    const systems::Context<T>& context, const VectorX<T>& tau,
    VectorX<T>* a) const {
  const int nv = plant().num_velocities();
  const VectorX<T> zero_v = VectorX<T>::Zero(nv);

  // We set a constext to store state x = [q; v=0].
  systems::Context<T>& context_with_zero_v = *workspace_.aux_plant_context_;
  plant().SetVelocities(&context_with_zero_v, zero_v);
  plant().SetPositions(&context_with_zero_v, plant().GetPositions(context));
  // Or:
  // context_with_zero_v.SetTimeStateAndParametersFrom(context);
  // plant().GetVelocities(context_with_zero_v).setZero();

  // const internal::PositionKinematicsCache<T>& pc =
  //    plant().EvalPositionKinematics(context_with_zero_v);
  // const internal::VelocityKinematicsCache<T>& vc =
  //    plant().EvalVelocityKinematics(context_with_zero_v);

  // External forces only include generalized forces tau. Body forces F are
  // zero.
  MultibodyForces<T> forces(plant());
  forces.mutable_generalized_forces() = tau;

  // Perform the tip-to-base pass to compute the force bias terms needed by ABA.  
  const auto& tree_topology = this->internal_tree().get_topology();
  internal::ArticulatedBodyForceCache<T> aba_force_cache(tree_topology);
  this->internal_tree().CalcArticulatedBodyForceCache(
      context_with_zero_v, forces, &aba_force_cache);

  // MultibodyTreeSystem::CalcArticulatedBodyAccelerations()
  internal::AccelerationKinematicsCache<T> ac(tree_topology);
  this->internal_tree().CalcArticulatedBodyAccelerations(
      context_with_zero_v, aba_force_cache, &ac);

  // Notice we use an explicit Euler scheme here since all forces are evaluated
  // at context0.
  *a = ac.get_vdot();
}

template <typename T>
VectorX<T> CompliantContactComputationManager<T>::CalcXdot(
    const systems::Context<T>& context) const {  
  MultibodyForces<T> forces(plant());

  const internal::PositionKinematicsCache<T>& pc =
      plant().EvalPositionKinematics(context);
  const internal::VelocityKinematicsCache<T>& vc =
      plant().EvalVelocityKinematics(context);

  // Compute forces applied by force elements. Note that this resets forces
  // to empty so must come first.
  this->internal_tree().CalcForceElementsContribution(context, pc, vc,
                                                             &forces);

  // We need only handle MultibodyPlant-specific forces here.
  this->AddInForcesFromInputPorts(context, &forces);

  // Perform the tip-to-base pass to compute the force bias terms needed by ABA.
  const auto& tree_topology =
      this->internal_tree().get_topology();
  internal::ArticulatedBodyForceCache<T> aba_force_cache(tree_topology);
  this->internal_tree().CalcArticulatedBodyForceCache(
      context, forces, &aba_force_cache);

  // MultibodyTreeSystem::CalcArticulatedBodyAccelerations()
  internal::AccelerationKinematicsCache<T> ac(tree_topology);
  this->internal_tree().CalcArticulatedBodyAccelerations(
      context, aba_force_cache, &ac);

  const auto v = plant().GetVelocities(context);
  VectorX<T> qdot(plant().num_positions());
  plant().MapVelocityToQDot(context, v, &qdot);
  VectorX<T> xdot(plant().num_multibody_states());
  xdot << qdot, ac.get_vdot();
  return xdot;
}

template <typename T>
void CompliantContactComputationManager<T>::DoCalcAccelerationKinematicsCache(
    const drake::systems::Context<T>& context,
    internal::AccelerationKinematicsCache<T>* ac) const {
  // Evaluate contact results.
  const contact_solvers::internal::ContactSolverResults<T>& solver_results =
      this->EvalContactSolverResults(context);

  // Retrieve the solution velocity for the next time step.
  const VectorX<T>& v_next = solver_results.v_next;

  auto x0 = context.get_discrete_state(0).get_value();
  const VectorX<T> v0 = x0.bottomRows(plant().num_velocities());

  ac->get_mutable_vdot() = (v_next - v0) / plant().time_step();

  // N.B. Pool of spatial accelerations indexed by BodyNodeIndex.
  this->internal_tree().CalcSpatialAccelerationsFromVdot(
      context, plant().EvalPositionKinematics(context),
      plant().EvalVelocityKinematics(context), ac->get_vdot(),
      &ac->get_mutable_A_WB_pool());
}

template <typename T>
void CompliantContactComputationManager<T>::DoCalcDiscreteValues(
    const drake::systems::Context<T>& context0,
    drake::systems::DiscreteValues<T>* updates) const {
  // Get the system state as raw Eigen vectors
  // (solution at the previous time step).
  DRAKE_DEMAND(this->multibody_state_index() == 0);
  DRAKE_DEMAND(context0.num_discrete_state_groups() == 1);

  auto x0 = context0.get_discrete_state(0).get_value();
  const int nq = plant().num_positions();
  const int nv = plant().num_velocities();
  VectorX<T> q0 = x0.topRows(nq);
  VectorX<T> v0 = x0.bottomRows(nv);

  // For a discrete model this evaluates vdot = (v_next - v0)/time_step() and
  // includes contact forces.
  const VectorX<T>& vdot = plant().EvalForwardDynamics(context0).get_vdot();

  // TODO(amcastro-tri): Consider replacing this by:
  //   const VectorX<T>& v_next = solver_results.v_next;
  // to avoid additional vector operations.
  const VectorX<T>& v_next = v0 + plant().time_step() * vdot;
  const VectorX<T> v_theta = (1.0 - theta_q_) * v0 + theta_q_ * v_next;

  VectorX<T> qdot_next(nq);
  plant().MapVelocityToQDot(context0, v_theta, &qdot_next);
  VectorX<T> q_next = q0 + plant().time_step() * qdot_next;

  VectorX<T> x_next(nq + nv);
  x_next << q_next, v_next;
  updates->get_mutable_vector(0).SetFromVector(x_next);
}

template <typename T>
void CompliantContactComputationManager<T>::LogStats(
    const std::string& log_file_name) const {
  std::cout << fmt::format(
      "CompliantContactComputationManager total wall-clock: {:12.4g}\n",
      total_time());
  const std::vector<ContactManagerStats>& hist = get_stats_history();
  std::ofstream file(log_file_name);

  file << fmt::format("{} {} {} {} {} {} {} {} {} {}\n", "sim_time",
                      "num_contacts", "total_time", "geometry_time",
                      "vstar_time", "graph_time", "linear_dynamics_time",
                      "contact_jacobian_time", "contact_solver_time",
                      "pack_results_time");

  for (const auto& s : hist) {
    file << fmt::format(
        "{} {} {} {} {} {} {} {} {} {}\n", s.time, s.num_contacts, s.total_time,
        s.geometry_time, s.vstar_time, s.graph_time, s.linear_dynamics_time,
        s.contact_jacobian_time, s.contact_solver_time, s.pack_results_time);
  }
  file.close();
}

template <typename T>
void CompliantContactComputationManager<T>::CalcContactJacobian(
    const systems::Context<T>& context,
    const std::vector<internal::DiscreteContactPair<T>>& contact_pairs,
    MatrixX<T>* Jc_ptr, std::vector<RotationMatrix<T>>* R_WC_set) const {
  DRAKE_DEMAND(Jc_ptr != nullptr);

  const int num_contacts = contact_pairs.size();

  // Jn is defined such that vn = Jn * v, with vn of size nc.
  auto& Jc = *Jc_ptr;
  Jc.resize(3 * num_contacts, plant().num_velocities());

  if (R_WC_set != nullptr) {
    R_WC_set->clear();
    R_WC_set->reserve(num_contacts);
  }

  // Quick no-op exit. Notice we did resize Jn, Jt and R_WC_set to be zero
  // sized.
  if (num_contacts == 0) return;

  const int nv = plant().num_velocities();

  // Scratch workspace variables.
  Matrix3X<T> Jtmp(3, nv);
  Matrix3X<T> Jv_AcBc(3, nv);
  // Alloc workspace once for successive calls.
  internal::CalcJacobianWorkspace<T> jacobian_workspace(
      plant().num_bodies(), plant().num_velocities(), num_contacts);

#if 0
  std::vector<int> num_body_points(plant().num_bodies(), 0);
  for (int icontact = 0; icontact < num_contacts; ++icontact) {
    const auto& point_pair = contact_pairs[icontact];
    const GeometryId geometryA_id = point_pair.id_A;
    const GeometryId geometryB_id = point_pair.id_B;
    BodyIndex bodyA_index = plant().geometry_id_to_body_index_.at(geometryA_id);
    BodyIndex bodyB_index = plant().geometry_id_to_body_index_.at(geometryB_id);
    if (bodyA_index != world_index()) ++num_body_points[bodyA_index];
    if (bodyB_index != world_index()) ++num_body_points[bodyB_index];
  }

  std::vector<int> body_offset(plant().num_bodies(), 0);
  for (BodyIndex b(1); b < plant().num_bodies(); ++b) {
    body_offset[b] = body_offset[b - 1] + num_body_points[b - 1];
  }

  std::vector<BodyIndex> participating_bodies;
  participating_bodies.reserve(plant().num_bodies());  // simply reserve max.
  for (BodyIndex b(0); b < plant().num_bodies(); ++b) {
    if (num_body_points[b] > 0) participating_bodies.push_back(b);
  }  

  std::fill(num_body_points.begin(), num_body_points.end(), 0);  // reset to 0.
  // Collect contact point offsets p_WC.
  // At most 2nc poits (since we ignore the world)
  Matrix3X<T> p_WC(3, 2 * num_contacts);
  //std::vector<int> contact_map(2 * num_contacts);
  int num_entries = 0;
  std::vector<std::pair<int, int>> contact_map(num_contacts);
  for (int icontact = 0; icontact < num_contacts; ++icontact) {
    const auto& point_pair = contact_pairs[icontact];
    const GeometryId geometryA_id = point_pair.id_A;
    const GeometryId geometryB_id = point_pair.id_B;
    BodyIndex bodyA_index = plant().geometry_id_to_body_index_.at(geometryA_id);
    BodyIndex bodyB_index = plant().geometry_id_to_body_index_.at(geometryB_id);
    const Vector3<T>& p_WCi = point_pair.p_WC;
    contact_map[icontact] = {0, 0};
    if (bodyA_index != world_index()) {
      const int k = body_offset[bodyA_index] + num_body_points[bodyA_index];
      p_WC.col(k) = p_WCi;
      contact_map[icontact].first = k;
      ++num_body_points[bodyA_index];
      ++num_entries;
    }
    if (bodyB_index != world_index()) {
      const int k = body_offset[bodyB_index] + num_body_points[bodyB_index];
      p_WC.col(k) = p_WCi;
      contact_map[icontact].second = k;
      ++num_body_points[bodyB_index];
      ++num_entries;
    }
  }

  DRAKE_DEMAND(std::accumulate(num_body_points.begin(), num_body_points.end(),
                               0) == num_entries);

  const int num_contacting_bodies = participating_bodies.size();
  MatrixX<T> Jv_v_WBc(3 * num_entries, nv);
  //MatrixX<T> Jv_w_WB(3 * num_contacting_bodies, nv);
  Jv_v_WBc.setZero();
  //Jv_w_WB.setZero();
  const Frame<T>& frame_W = plant().world_frame();
  //int participating_body = 0;
  for (BodyIndex b(1); b < plant().num_bodies(); ++b) {
    if (num_body_points[b] > 0) {
      const Body<T>& body = plant().get_body(b);
      const int row_offset = 3 * body_offset[b];
      const int num_rows = 3 * num_body_points[b];
      auto Jv_v_WBi = Jv_v_WBc.block(row_offset, 0, num_rows, nv);
      const auto p_WBiC = p_WC.block(0, body_offset[b], 3, num_body_points[b]);
      plant().internal_tree().CalcJacobianTranslationalVelocity(
        context, JacobianWrtVariable::kV, body.body_frame(), frame_W, p_WBiC,
        frame_W, frame_W, &Jv_v_WBi);

#if 0
      auto Jv_w_WBi = Jv_w_WB.block(3 * participating_body, 0, 3, nv);
      plant().CalcJacobianAngularVelocity(context, JacobianWrtVariable::kV,
                                          body.body_frame(), frame_W, frame_W,
                                          &Jv_w_WBi);
      ++participating_body;
#endif      
    }
  }

  // From Jc = Jv_WBc - Jv_WAc
  for (int icontact = 0; icontact < num_contacts; ++icontact) {
    const auto& point_pair = contact_pairs[icontact];
    const GeometryId geometryA_id = point_pair.id_A;
    const GeometryId geometryB_id = point_pair.id_B;
    BodyIndex bodyA_index = plant().geometry_id_to_body_index_.at(geometryA_id);
    BodyIndex bodyB_index = plant().geometry_id_to_body_index_.at(geometryB_id);
    const int kA = contact_map[icontact].first;
    const int kB = contact_map[icontact].first;

    // TODO. set to zero first.
    // TODO: set to bodyB jacobian

    if (bodyA_index != world_index()) {
      const auto J_WAc = Jv_v_WBc.block(3 * kA, 0, 3, nv);
      Jc.block(3 * icontact, 0, 3, nv) -= J_WAc;
    }

    Jc.block(3 * icontact, 0, 3, nv) =
        R_WC.transpose() * Jc.block(3 * icontact, 0, 3, nv);

    // TODO: fix this to have all consistent signs.
    // Negate the normal direction so that this component corresponds to the
    // "separation" velocity.
    Jc.row(3 * icontact + 2) = -Jc.row(3 * icontact + 2);        
  }
#endif  

  //test::LimitMalloc guard({.max_num_allocations = 0});     

 const Frame<T>& frame_W = plant().world_frame();
 for (int icontact = 0; icontact < num_contacts; ++icontact) {
   const auto& point_pair = contact_pairs[icontact];

   const GeometryId geometryA_id = point_pair.id_A;
   const GeometryId geometryB_id = point_pair.id_B;

   BodyIndex bodyA_index = plant().geometry_id_to_body_index_.at(geometryA_id);
   const Body<T>& bodyA = plant().get_body(bodyA_index);
   BodyIndex bodyB_index = plant().geometry_id_to_body_index_.at(geometryB_id);
   const Body<T>& bodyB = plant().get_body(bodyB_index);

   // Penetration depth > 0 if bodies interpenetrate.
   const Vector3<T>& nhat_BA_W = point_pair.nhat_BA_W;
   const Vector3<T>& p_WC = point_pair.p_WC;

   // For point Ac (origin of frame A shifted to C), calculate Jv_v_WAc (Ac's
   // translational velocity Jacobian in the world frame W with respect to
   // generalized velocities v).  Note: Ac's translational velocity in W can
   // be written in terms of this Jacobian as v_WAc = Jv_v_WAc * v.
   plant().internal_tree().CalcJacobianTranslationalVelocity(
       context, JacobianWrtVariable::kV, bodyA.body_frame(), frame_W, p_WC,
       frame_W, frame_W, &Jtmp, &jacobian_workspace);
   Jv_AcBc = -Jtmp;  // Jv_AcBc = -J_WAc.

   // Similarly, for point Bc (origin of frame B shifted to C), calculate
   // Jv_v_WBc (Bc's translational velocity Jacobian in W with respect to v).
   plant().internal_tree().CalcJacobianTranslationalVelocity(
       context, JacobianWrtVariable::kV, bodyB.body_frame(), frame_W, p_WC,
       frame_W, frame_W, &Jtmp, &jacobian_workspace);
   Jv_AcBc += Jtmp;  // Jv_AcBc = J_WBc - J_WAc.

   // Computation of the tangential velocities Jacobian Jt:
   //
   // Compute the orientation of a contact frame C at the contact point such
   // that the z-axis Cz equals to nhat_BA_W. The tangent vectors are
   // arbitrary, with the only requirement being that they form a valid right
   // handed basis with nhat_BA.
   const math::RotationMatrix<T> R_WC =
        math::RotationMatrix<T>::MakeFromOneVector(nhat_BA_W, 2);
   if (R_WC_set != nullptr) {
     R_WC_set->push_back(R_WC);
   }

   Jc.block(3 * icontact, 0, 3, nv).noalias() =
       R_WC.matrix().transpose() * Jv_AcBc;

   // TODO: fix this to have all consistent signs.
   // Negate the normal direction so that this component corresponds to the
   // "separation" velocity.
   Jc.row(3 * icontact + 2) = -Jc.row(3 * icontact + 2);
  }
}

template <typename T>
void CompliantContactComputationManager<T>::DoCalcContactResults(
    const systems::Context<T>& context,
    ContactResults<T>* contact_results) const {
  contact_results->Clear();
  if (plant().num_collision_geometries() == 0) return;

  //const auto& point_pairs = plant().EvalPointPairPenetrations(context);
  const std::vector<internal::DiscreteContactPair<T>>& contact_pairs =
      plant().EvalDiscreteContactPairs(context);
  const int num_contacts = contact_pairs.size();

  const std::vector<RotationMatrix<T>>& R_WC_set =
      EvalContactJacobianCache(context).R_WC_list;
  const contact_solvers::internal::ContactSolverResults<T>& solver_results =
      plant().EvalContactSolverResults(context);

  const VectorX<T>& fn = solver_results.fn;
  const VectorX<T>& ft = solver_results.ft;
  const VectorX<T>& vt = solver_results.vt;
  const VectorX<T>& vn = solver_results.vn;

  // The strict equality is true only when point contact is used alone.
  // Otherwise there are quadrature points in addition to the point pairs.
  DRAKE_DEMAND(fn.size() >= num_contacts);
  DRAKE_DEMAND(ft.size() >= 2 * num_contacts);
  DRAKE_DEMAND(vn.size() >= num_contacts);
  DRAKE_DEMAND(vt.size() >= 2 * num_contacts);

  contact_results->Clear();

  for (int icontact = 0; icontact < num_contacts; ++icontact) {
    const auto& pair = contact_pairs[icontact];
    const GeometryId geometryA_id = pair.id_A;
    const GeometryId geometryB_id = pair.id_B;
    const BodyIndex bodyA_index =
        plant().geometry_id_to_body_index_.at(geometryA_id);
    const BodyIndex bodyB_index =
        plant().geometry_id_to_body_index_.at(geometryB_id);

    const Vector3<T>& p_WC = pair.p_WC;
    const Vector3<T>& nhat_BA_W = pair.nhat_BA_W;
    const T phi0 = pair.phi0;
    const Vector3<T> p_WCa = p_WC + phi0 * nhat_BA_W;
    const Vector3<T> p_WCb = p_WC - phi0 * nhat_BA_W;

    const RotationMatrix<T>& R_WC = R_WC_set[icontact];

    // Contact forces applied on B at contact point C.
    const Vector3<T> f_Bc_C(ft(2 * icontact), ft(2 * icontact + 1),
                            -fn(icontact));
    const Vector3<T> f_Bc_W = R_WC * f_Bc_C;

    // Slip velocity.
    const T slip = vt.template segment<2>(2 * icontact).norm();

    // Separation velocity in the normal direction.
    const T separation_velocity = vn(icontact);

    drake::geometry::PenetrationAsPointPair<T> penetration_pair{
        geometryA_id, geometryB_id, p_WCa, p_WCb, nhat_BA_W, -phi0};

    // Add pair info to the contact results.
    contact_results->AddContactInfo({bodyA_index, bodyB_index, f_Bc_W, p_WC,
                                     separation_velocity, slip,
                                     penetration_pair});
  }
}

}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::CompliantContactComputationManager);
