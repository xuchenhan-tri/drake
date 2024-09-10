#pragma once

#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace examples {
namespace deformable {

/* We create a leaf system that outputs the desired state of a parallel jaw
 gripper to follow a close-lift-open motion sequence. The desired position is
 2-dimensional with the first element corresponding to the wrist degree of
 freedom and the second element corresponding to the left finger degree of
 freedom. This control is a time-based state machine, where desired state
 changes based on the context time. There are four states, executed in the
 following order:

  0. The fingers are open in the initial state.
  1. The fingers are closed to secure a grasp.
  2. The gripper is lifted to a prescribed final height.
  3. The fingers are open to loosen the grasp.

 The desired state is interpolated between these states. */
class ParallelGripperController : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ParallelGripperController);

  /* Constructs a ParallelGripperController system with the given parameters.
   @param[in] open_width   The width between fingers in the open state. (meters)
   @param[in] closed_width The width between fingers in the closed state.
                           (meters)
   @param[in] height       The height of the gripper in the lifted state.
                           (meters) */
  ParallelGripperController(double open_width, double closed_width,
                            double height);

 private:
  /* Computes the output desired state of the parallel gripper. */
  void CalcDesiredState(const systems::Context<double>& context,
                        systems::BasicVector<double>* output) const;

  /* The time at which the fingers reach the desired closed state. */
  const double fingers_closed_time_{1.5};
  /* The time at which the gripper reaches the desired "lifted" state. */
  const double gripper_lifted_time_{3.0};
  const double hold_time_{5.5};
  /* The time at which the fingers reach the desired open state. */
  const double fingers_open_time_{7.0};
  Eigen::Vector2d initial_configuration_;
  Eigen::Vector2d closed_configuration_;
  Eigen::Vector2d lifted_configuration_;
  Eigen::Vector2d open_configuration_;
};

class ExternalForceSource : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExternalForceSource);

  /* Constructs a ParallelGripperController system with the given parameters.
   @param[in] open_width   The width between fingers in the open state. (meters)
   @param[in] closed_width The width between fingers in the closed state.
                           (meters)
   @param[in] height       The height of the gripper in the lifted state.
                           (meters) */
  ExternalForceSource(multibody::BodyIndex body_index,
                      const Vector3<double>& p_BoBq_B, double torque,
                      double start_time, double end_time)
      : body_index_(body_index),
        p_BoBq_B_(p_BoBq_B),
        torque_(torque),
        start_time_(start_time),
        end_time_(end_time) {
    this->DeclareAbstractOutputPort(
        "desired state",
        std::vector<multibody::ExternallyAppliedSpatialForce<double>>{},
        &ExternalForceSource::CalcOutput);
  }

 private:
  /* Computes the output desired state of the parallel gripper. */
  void CalcOutput(const systems::Context<double>& context,
                  std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
                      output) const {
    output->clear();
    const double t = context.get_time();
    if (t >= start_time_ && t <= end_time_) {
      multibody::ExternallyAppliedSpatialForce<double> force;
      force.body_index = body_index_;
      force.p_BoBq_B = p_BoBq_B_;
      force.F_Bq_W = multibody::SpatialForce<double>(
          Vector3<double>(0, -torque_, 0), Vector3<double>::Zero());
      output->push_back(force);
    }
  }

  multibody::BodyIndex body_index_;
  Vector3<double> p_BoBq_B_{};
  double torque_{};
  double start_time_{};
  double end_time_{};
};

}  // namespace deformable
}  // namespace examples
}  // namespace drake
