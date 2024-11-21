#include "../mpm_solver.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::Vector3i;
using Eigen::VectorXd;

/* Adds a particle at position x0 to `particles`. All other data are arbitrary.
 */
template <typename T>
void AddParticle(Particles<T>* particles,
                 const Eigen::Ref<const Vector3<T>>& x0, bool in_constraint) {
  particles->data.x.push_back(x0);
  particles->data.F.push_back(Matrix3<T>::Identity());
  particles->data.m.push_back(1.0);
  particles->data.v.push_back(Vector3<T>(1.0, 2.0, 3.0));
  particles->data.C.push_back(Matrix3<T>::Zero());
  particles->data.in_constraint.push_back(in_constraint);
  particles->data.volume.push_back(0.01);
  particles->data.tau_v0.push_back(Matrix3<T>::Zero());
}

/* Adds a linear corotated model to all particles.
 @note this function should only be called once in the lifespan of Particles. */
template <typename T>
void AddDefaultMaterial(ParticleData<T>* particle_data) {
  const int num_particles = particle_data->m.size();
  const fem::internal::LinearConstitutiveModel<T> model(1e4, 0.45);
  particle_data->constitutive_models.emplace_back(model);
  for (int i = 0; i < num_particles; ++i) {
    particle_data->strain_data.emplace_back(model.MakeDefaultData());
  }
  particle_data->materials.emplace_back(0, num_particles);
}

GTEST_TEST(MpmSolverTest, Constructor) {
  MpmSolver<double> solver;
  EXPECT_EQ(solver.schur_complement().rows(), 0);
  EXPECT_EQ(solver.schur_complement().cols(), 0);
  EXPECT_EQ(solver.participating_v_star().size(), 0);
}

GTEST_TEST(MpmSolverTest, FreeMotionSolve) {
  const double dx = 0.01;
  Particles<double> particles;
  const Vector3d x0 = Vector3d(dx, dx, dx);
  const Vector3d x1 = Vector3d(4.0 * dx, dx, dx);
  AddParticle<double>(&particles, x0, false);
  AddParticle<double>(&particles, x1, true);
  AddDefaultMaterial(&particles.data);
  const double dt = 0.02;

  /* At steady state, the solver should converge in zero iteration. */
  {
    /* First confirm the residual is zero. */
    const MpmState<double> state(dt, dx, particles);
    VectorX<double> residual(state.num_dofs());
    SolverState<double> solver_state(state.num_dofs(), state.num_particles());
    state.UpdateSolverParticleState(&solver_state);
    state.CalcResidual(solver_state, &residual);
    EXPECT_TRUE(
        CompareMatrices(residual, VectorXd::Zero(state.num_dofs()), 1e-14));

    /* Set up a solver that allows max iterations 0 and confirm the solver still
     can compute the free motion. */
    MpmSolverParameters parameters{.max_iterations = 0};
    MpmSolver<double> solver(parameters);
    EXPECT_NO_THROW(solver.ComputeFreeMotionState(state));

    auto tangent_matrix = state.MakeTangentMatrix();
    state.CalcTangentMatrix(solver_state, &tangent_matrix);
    const MatrixX<double> dense_tangent_matrix =
        tangent_matrix.MakeDenseMatrix();
    /* We make use of our clear box understanding of how indexing works for
     SPGrid. The grid nodes influenced by particle 0 (non-participating) come
     before the grid nodes influenced by particle 1 (participating). So, the
     dofs of 27 grid nodes at the top-left corner corresponds to particle 0 and
     the dofs of the 27 grid nodes at the bottom-right corner corresponds to
     particle 1.  */
    const MatrixX<double> A = dense_tangent_matrix.topLeftCorner<81, 81>();
    const MatrixX<double> B = dense_tangent_matrix.topRightCorner<81, 81>();
    const MatrixX<double> D = dense_tangent_matrix.bottomRightCorner<81, 81>();
    const MatrixX<double> S = D - B.transpose() * A.inverse() * B;
    EXPECT_TRUE(CompareMatrices(solver.schur_complement(), S));
  }

  {
    /* Modify the deformation gradient to be non-identity, so that the system is
     no-longer in steady state. */
    particles.data.F[0] =
        (Matrix3d() << 1.1, 0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 1.0).finished();

    const MpmState<double> state(dt, dx, particles);
    MpmSolverParameters parameters{.max_iterations = 0};
    MpmSolver<double> solver(parameters);
    DRAKE_EXPECT_THROWS_MESSAGE(solver.ComputeFreeMotionState(state),
                                "MpmSolver failed to converge.*");
    /* However, since the model is linear, the solver still converges in 1
     iteration. */
    parameters.max_iterations = 1;
    solver.set_parameters(parameters);
    EXPECT_NO_THROW(solver.ComputeFreeMotionState(state));
  }
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
