#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/contact_jacobian.h"
#include "drake/fem/fem_config.h"
#include "drake/fem/fem_data.h"
#include "drake/fem/fem_element.h"
#include "drake/fem/fem_force.h"
#include "drake/fem/fem_state.h"
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
        state_(),
        force_(data_, &state_),
        objective_(data_, &state_, force_),
        newton_solver_(objective_){}
  /**
   The internal main loop for the FEM simulation that calls NewtonSolver to
   calculate the discrete velocity change. Update the position and velocity
   states from time n to time n+1.
   */
  void AdvanceOneTimeStep() const{
    SolveFreeMotion();
    SolveContact();
    state_.set_time(state_.get_time() + data_.get_dt());
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
                          const MaterialConfig& config) {
    int object_id = data_.AddUndeformedObject(indices, positions, config);

    // Create hyperelastic cache for the new object.
    auto& hyperelastic_cache = state_.get_mutable_hyperelastic_cache();
    const auto& elements = data_.get_elements();
    const auto& element_indices = data_.get_element_indices()[object_id];
    const int num_new_elements = element_indices.size();
    for (int i = 0; i < num_new_elements; ++i) {
      const auto& eid = element_indices[i];
      const auto& element = elements[eid];
      const auto* model = element.get_constitutive_model();
      auto model_cache = model->CreateCache();
      hyperelastic_cache.emplace_back(std::move(model_cache));
    }
    const int num_new_vertices = data_.get_vertex_indices()[object_id].size();
    // Update size of state/cache on vertices.
    auto& q = state_.get_mutable_q();
    q.conservativeResize(3, q.cols() + num_new_vertices);
    auto& q0 = state_.get_mutable_q0();
    q0.conservativeResize(3, q0.cols() + num_new_vertices);
    auto& q_star = state_.get_mutable_q_star();
    q_star.conservativeResize(3, q_star.cols() + num_new_vertices);
    auto& v0 = state_.get_mutable_v0();
    v0.conservativeResize(3, v0.cols() + num_new_vertices);
    auto& v = state_.get_mutable_v();
    v.conservativeResize(3, v.cols() + num_new_vertices);
    // Update size of cache on quadratures.
    auto& F = state_.get_mutable_F();
    F.resize(F.size() + num_new_elements);
    auto& F0 = state_.get_mutable_F0();
    F0.resize(F0.size() + num_new_elements);
    auto& psi = state_.get_mutable_psi();
    psi.resize(psi.size() + num_new_elements);
    auto& P = state_.get_mutable_P();
    P.resize(P.size() + num_new_elements);
    auto& dPdF = state_.get_mutable_dPdF();
    dPdF.resize(dPdF.size() + num_new_elements);
    return object_id;
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
      @param[in] bc          A function that takes an index i and the reference
     positions and determines whether the node with index i is under Dirichlet
     boundary condition.

      @pre @p object_id < number of existing objects.
   */
  void SetBoundaryCondition(const int object_id,
                            std::function<bool(int, const Matrix3X<T>&)> bc);

  void AddCollisionObject(std::unique_ptr<CollisionObject<T>> object) {
    data_.add_collision_object(std::move(object));
  }

  int get_num_position_dofs() const { return data_.get_num_position_dofs(); }

  const std::vector<Vector4<int>>& get_mesh() const { return data_.get_mesh(); }

  const Matrix3X<T>& get_q() const { return state_.get_q(); }

 private:
  /* Solve for the momentum equation without considering collisions or contact
   * and change the underlying states. */
  void SolveFreeMotion() const;

  /* Solve for the contact constraint and change the underlying states. */
  void SolveContact() const;

  const std::vector<Matrix3<T>>& EvalF0() const;
  const Matrix3X<T>& Evalqstar() const;

  FemData<T> data_;
  mutable FemState<T> state_;
  FemForce<T> force_;
  BackwardEulerObjective<T> objective_;
  NewtonSolver<T> newton_solver_;
};

}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::FemSolver)
