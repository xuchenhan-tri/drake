#include "drake/multibody/fixed_fem/dev/contact_solver_data.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
const int kNumRigidDofs = 1;
const int kNumDeformableDofs = 3;
const int kNcRigidRigid = 2;
const int kNcRigidDeformable = 4;
const double kPhi0RR = 0.1;
const double kStiffnessRR = 0.2;
const double kDampingRR = 0.3;
const double kMuRR = 0.4;
const double kContactJacobianRR = 0.5;
const double kPhi0RD = 1.1;
const double kStiffnessRD = 1.2;
const double kDampingRD = 1.3;
const double kMuRD = 1.4;
const double kContactJacobianRD = 1.5;
// Dummy values for contact data. The postfix `_rr` indicates data for
// rigid-rigid contacts while the postfix `_rd` indicates data for
// rigid-deformable contact.
const VectorXd dummy_phi0_rr = VectorXd::Ones(kNcRigidRigid) * kPhi0RR;
const VectorXd dummy_stiffness_rr =
    VectorXd::Ones(kNcRigidRigid) * kStiffnessRR;
const VectorXd dummy_damping_rr = VectorXd::Ones(kNcRigidRigid) * kDampingRR;
const VectorXd dummy_mu_rr = VectorXd::Ones(kNcRigidRigid) * kMuRR;
const MatrixXd dummy_contact_jacobian_rr =
    MatrixXd::Ones(3 * kNcRigidRigid, kNumRigidDofs) * kContactJacobianRR;

const VectorXd dummy_phi0_rd = VectorXd::Ones(kNcRigidDeformable) * kPhi0RD;
const VectorXd dummy_stiffness_rd =
    VectorXd::Ones(kNcRigidDeformable) * kStiffnessRD;
const VectorXd dummy_damping_rd =
    VectorXd::Ones(kNcRigidDeformable) * kDampingRD;
const VectorXd dummy_mu_rd = VectorXd::Ones(kNcRigidDeformable) * kMuRD;
const MatrixXd dummy_contact_jacobian_rd =
    MatrixXd::Ones(3 * kNcRigidDeformable, kNumRigidDofs + kNumDeformableDofs) *
    kContactJacobianRD;

GTEST_TEST(ContactSolverDataTest, ConcatenateData) {
  // Set up the contact data for the rigid rigid contacts.
  VectorXd phi0_rr = dummy_phi0_rr;
  VectorXd stiffness_rr = dummy_stiffness_rr;
  VectorXd damping_rr = dummy_damping_rr;
  VectorXd mu_rr = dummy_mu_rr;
  const MatrixXd contact_jacobian_rr = dummy_contact_jacobian_rr;
  const ContactSolverData<double> contact_data_rr(
      std::move(phi0_rr), std::move(stiffness_rr), std::move(damping_rr),
      std::move(mu_rr), std::move(contact_jacobian_rr), kNumRigidDofs,
      kNumDeformableDofs);

  // Set up the contact data for the rigid deformable contacts.
  VectorXd phi0_rd = dummy_phi0_rd;
  VectorXd stiffness_rd = dummy_stiffness_rd;
  VectorXd damping_rd = dummy_damping_rd;
  VectorXd mu_rd = dummy_mu_rd;
  const MatrixXd contact_jacobian_rd_dense = dummy_contact_jacobian_rd;
  Eigen::SparseMatrix<double> contact_jacobian_rd =
      contact_jacobian_rd_dense.sparseView();
  const ContactSolverData<double> contact_data_rd(
      std::move(phi0_rd), std::move(stiffness_rd), std::move(damping_rd),
      std::move(mu_rd), std::move(contact_jacobian_rd),
      kNumRigidDofs + kNumDeformableDofs);

  std::vector<ContactSolverData<double>> data;
  data.push_back(contact_data_rr);
  data.push_back(contact_data_rd);
  // Combine the contact data with the method under test.
  const ContactSolverData<double> combined_data =
      ConcatenateContactSolverData(data);

  // The data concatenated with the method under test should match the manually
  // combined data.
  VectorXd combined_phi0(kNcRigidDeformable + kNcRigidRigid);
  combined_phi0 << dummy_phi0_rr, dummy_phi0_rd;
  VectorXd combined_stiffness(kNcRigidDeformable + kNcRigidRigid);
  combined_stiffness << dummy_stiffness_rr, dummy_stiffness_rd;
  VectorXd combined_damping(kNcRigidDeformable + kNcRigidRigid);
  combined_damping << dummy_damping_rr, dummy_damping_rd;
  VectorXd combined_mu(kNcRigidDeformable + kNcRigidRigid);
  combined_mu << dummy_mu_rr, dummy_mu_rd;
  MatrixXd combined_contact_jacobian_dense =
      MatrixXd::Zero((kNcRigidRigid + kNcRigidDeformable) * 3,
                     kNumRigidDofs + kNumDeformableDofs);
  combined_contact_jacobian_dense.topLeftCorner(
      3 * kNcRigidRigid, kNumRigidDofs) = dummy_contact_jacobian_rr;
  combined_contact_jacobian_dense.bottomRightCorner(
      3 * kNcRigidDeformable, kNumRigidDofs + kNumDeformableDofs) =
      dummy_contact_jacobian_rd;

  EXPECT_TRUE(CompareMatrices(combined_data.phi0(), combined_phi0));
  EXPECT_TRUE(CompareMatrices(combined_data.stiffness(), combined_stiffness));
  EXPECT_TRUE(CompareMatrices(combined_data.damping(), combined_damping));
  EXPECT_TRUE(CompareMatrices(combined_data.mu(), combined_mu));
  EXPECT_TRUE(CompareMatrices(MatrixXd(combined_data.contact_jacobian()),
                              combined_contact_jacobian_dense));
  EXPECT_EQ(combined_data.num_contacts(), kNcRigidDeformable + kNcRigidRigid);
  EXPECT_EQ(combined_data.num_dofs(), kNumRigidDofs + kNumDeformableDofs);
}
}  // namespace
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
