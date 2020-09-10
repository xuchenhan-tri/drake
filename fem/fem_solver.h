#pragma once

#include <memory>
#include <vector>
#include <utility>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/contact_jacobian.h"
#include "drake/fem/fem_config.h"
#include "drake/fem/fem_data.h"
#include "drake/fem/fem_element.h"
#include "drake/fem/fem_force.h"
#include "drake/fem/newton_solver.h"
#include "drake/multibody/solvers/inverse_operator.h"
#include "drake/multibody/solvers/pgs_solver.h"
#include "drake/multibody/solvers/point_contact_data.h"
#include "drake/multibody/solvers/sparse_linear_operator.h"
#include "drake/multibody/solvers/system_dynamics_data.h"

namespace drake {
namespace fem {

template <typename T>
class FemSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemSolver);

  explicit FemSolver(double dt)
      : data_(dt),
        force_(data_.get_elements()),
        objective_(&data_, &force_),
        newton_solver_(&objective_),
        contact_jacobian_(data_.get_q(), data_.get_collision_objects()) {}
  /**
   The internal main loop for the FEM simulation that calls NewtonSolver to
   calculate the discrete velocity change. Update the position and velocity
   states from time n to time n+1.
   */
  void AdvanceOneTimeStep() {
    SolveFreeMotion();
    SolveContact();
    data_.set_time(data_.get_time() + data_.get_dt());
  }

  /**
   Add an object represented by a list of vertices connected by a simplex mesh
  to the simulation. Multiple calls to this method is allowed and the resulting
  vertices and elements will be properly indexed.
  @param[in] indices    The list of indices describing the connectivity of the
  mesh. @p indices[i] contains the indices of the 4 vertices in the i-th
  element.
  @param[in] positions  The list of positions of the vertices in the world frame
  in the reference configuration.
  @return The object_id of the newly added_object.
  */
  int AddUndeformedObject(const std::vector<Vector4<int>>& indices,
                          const Matrix3X<T>& positions,
                          const FemConfig& config) {
    return data_.AddUndeformedObject(indices, positions, config);
  }

  /**
      Set the initial positions and velocities of a given object.
      @param[in] object_id     The id the object whose initial conditions are
     being set.
      @param[in] set_position  The function that takes an index i that modifies
     the initial position of the i-th vertex in the chosen object.
      @param[in] set_velocity  The function that takes an index i that modifies
     the initial velocity of the i-th vertex in the chosen object.

      @pre @p object_id < number of existing objects.
   */
  void SetInitialStates(
      const int object_id,
      std::function<void(int, EigenPtr<Matrix3X<T>>)> set_position = nullptr,
      std::function<void(int, EigenPtr<Matrix3X<T>>)> set_velocity = nullptr);

  /**
      Set the boundary condition of a given object. If the current boundary
     condition being set is incompatible with a previous boundary condition, the
     new one will be applied.
      @param[in] object_id   The id the object whose initial conditions are
     being set.
      @param[in] bc          A function that takes an index i and the reference positions and determines whether the node with index i is under Dirichlet boundary condition.

      @pre @p object_id < number of existing objects.
   */
  void SetBoundaryCondition(
      const int object_id,
      std::function<bool(int, const Matrix3X<T>&)> bc);

  void AddCollisionObject(std::unique_ptr<CollisionObject<T>> object) {
    data_.add_collision_object(std::move(object));
  }

  int get_num_position_dofs() const { return data_.get_num_position_dofs(); }

  const std::vector<Vector4<int>>& get_mesh() const { return data_.get_mesh(); }

  const Matrix3X<T>& get_q() const { return data_.get_q(); }

 private:
  /* Solve for the momentum equation without considering collisions or contact
   * and change the underlying states. */
  void SolveFreeMotion();

  /* Solve for the contact constraint and change the underlying states. */
  void SolveContact();

  FemData<T> data_;
  FemForce<T> force_;
  BackwardEulerObjective<T> objective_;
  NewtonSolver<T> newton_solver_;
  ContactJacobian<T> contact_jacobian_;
};

}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::FemSolver)
