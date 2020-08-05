#pragma once

#include "drake/fem/constitutive_model.h"
#include "drake/fem/fem_force.h"
#include "drake/fem/fem_element.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/newton_solver.h"

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {

template <typename T>
class FemSolver {
public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemSolver);

    FemSolver(T dt): dt_(dt),
    elements_({}),
    force_(elements_),
    objective_(force_),
    newton_solver_(objective_)
    {}
    /**
     Calls NewtonSolver to calculate the discrete velocity change.
     Update the position and velocity states from time n to time n+1.
     */
    void UpdateDiscreteState(const VectorX<T>& q_n, const VectorX<T>& v_n,
                             VectorX<T>* next_q, VectorX<T>* next_v) const;

    /**
     Add an object represented by a list of vertices connected by a simplex mesh
    to the simulation.
    @param[in] indices    The list of indices describing the connectivity of the
    mesh. @p indices[i] contains the indices of the 4 vertices in the i-th
    element.
    @param[in] positions  The list of positions of the vertices in the undeformed
    configuration.
    @param[in] model      The constitutive model that the object is endowed with.
    */
    int AddUndeformedMesh(const std::vector<Vector4<int>>& indices,
                          const std::vector<Vector3<T>>& positions,
                          const ConstitutiveModel<T>& model);

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
    void SetInitialStates(const int object_id,
                          std::function<void(int, VectorX<T>*)> set_position,
                          std::function<void(int, VectorX<T>*)> set_velocity);

    /**
        Set the boundary condition of a given object.
        @param[in] object_id   The id the object whose initial conditions are
       being set.
        @param[in] bc          The function that takes an index i and the time t
       that modifies the position and the velocity of the i-th vertex in the
       chosen object at time t.

        @pre @p object_id < number of existing objects.
     */
    void SetBoundaryCondition(
            const int object_id,
            std::function<void(int, T, VectorX<T>*, VectorX<T>*)> bc);

    const VectorX<T>& get_mass() const { return mass_; }

    const std::vector<FemElement<T>>& get_elements() const { return elements_; }

    std::vector<FemElement<T>>& get_mutable_elements() { return elements_; }

    const FemForce<T>& get_force() const { return force_; }

    FemForce<T>& get_mutable_force() { return force_; }

    T dt() const { return dt_; }

    void set_dt (T dt) { dt_ = dt; }

    Vector3<T> gravity() { return gravity_; }

    void set_gravity(Vector3<T>& gravity) { gravity_ = gravity; }

private:
    T dt_;
    std::vector<FemElement<T>> elements_;
    FemForce<T> force_;
    BackwardEulerObjective<T> objective_;
    NewtonSolver<T> newton_solver_;
    // element_indices_[i] gives the vertex indices corresponding to object i.
    std::vector<std::vector<int>> element_indices_;
    Matrix3X<T> q;
    Matrix3X<T> v;
    VectorX<T> mass_;
    Vector3<T> gravity_;
};

}  // namespace fem
}  // namespace drake
