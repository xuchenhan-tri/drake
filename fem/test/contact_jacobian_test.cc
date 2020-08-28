#include "drake/fem/contact_jacobian.h"

#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "drake/fem/collision_object.h"
#include "drake/fem/half_space.h"

namespace drake {
namespace fem {
namespace {

class ContactJacobianTest : public ::testing::Test {
 public:
  /*
      The geometry setup viewing from the positive x-axis.
                         ls1
                          |
                q0        |        q4
                   x      |      x
                          |
                          |
               -----------+----------------- ls2
                          |
                q1        |        q3
                   x      |      x
                          |
   z-axis  ^
           |
           |
           +-------> y-axis
   */
  void SetUp() override {
    // Half space centered at origin with outward normal in the positive z-axis.
    auto ls1 = std::make_unique<HalfSpace<double>>(Vector3<double>(0, 0, 0),
                                                   Vector3<double>(0, 0, 1));
    // Half space centered at origin with outward normal in the positive y-axis.
    auto ls2 = std::make_unique<HalfSpace<double>>(Vector3<double>(0, 0, 0),
                                                   Vector3<double>(0, 1, 0));
    auto cb1 = std::make_unique<CollisionObject<double>>(std::move(ls1));
    auto cb2 = std::make_unique<CollisionObject<double>>(std::move(ls2));
    collision_objects_.emplace_back(std::move(cb1));
    collision_objects_.emplace_back(std::move(cb2));

    constexpr int nv = 4;
    q_.resize(3, nv);
    q_.col(0) = Vector3<double>(3, -1, 1);
    q_.col(1) = Vector3<double>(2, -1, -1);
    q_.col(2) = Vector3<double>(-1, 1, -1);
    q_.col(3) = Vector3<double>(0, 1, 1);
    contact_jacobian_ =
        std::make_unique<ContactJacobian<double>>(q_, collision_objects_);
  }

 protected:
  std::vector<std::unique_ptr<CollisionObject<double>>> collision_objects_;
  Matrix3X<double> q_;
  std::unique_ptr<ContactJacobian<double>> contact_jacobian_;
};

TEST_F(ContactJacobianTest, QueryContact) {
  Eigen::SparseMatrix<double> jacobian;
  VectorX<double> penetration_depth;
  contact_jacobian_->QueryContact(&jacobian, &penetration_depth);
  constexpr int nc = 4;
  constexpr int nv = 4;
  Matrix3X<double> analytic_normals(3, nc);
  // contact 0: q.col(0) vs. ls2.
  analytic_normals.col(0) = Vector3<double>(0, 1, 0);
  // contact 1: q.col(1) vs. ls1.
  analytic_normals.col(1) = Vector3<double>(0, 0, 1);
  // contact 2: q.col(1) vs. ls2.
  analytic_normals.col(2) = Vector3<double>(0, 1, 0);
  // contact 2: q.col(2) vs. ls1.
  analytic_normals.col(3) = Vector3<double>(0, 0, 1);
  // q.col(3) is not in contact with any collision object.

  // The calculated normals should be bitwise equal to the analytic normals.
  auto contact_normals = contact_jacobian_->get_normals();
  EXPECT_EQ(contact_normals, analytic_normals);

  VectorX<double> analytic_penetration(nc);
  for (int i = 0; i < nc; ++i) analytic_penetration(i) = -1.0;
  // The penetration depth is not bitwise equal to the analytic ones because a
  // dot product is required to get the penetration depth.
  EXPECT_NEAR((penetration_depth - analytic_penetration).norm(), 0.0,
              std::numeric_limits<double>::epsilon());

  // TODO(xuchenhan-tri) Need a way to test the Jacobian.
  if (0) {
    Eigen::SparseMatrix<double> analytic_jacobian(3 * nc, 3 * nv);
    std::vector<int> contact_indices = {0, 1, 2, 3};
    std::vector<int> vertex_indices = {0, 1, 1, 2};
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < nc; ++i) {
      triplets.emplace_back(3 * contact_indices[i], 3 * vertex_indices[i], 1.0);
      triplets.emplace_back(3 * contact_indices[i] + 1,
                            3 * vertex_indices[i] + 1, 1.0);
      triplets.emplace_back(3 * contact_indices[i] + 2,
                            3 * vertex_indices[i] + 2, 1.0);
    }
    analytic_jacobian.setFromTriplets(triplets.begin(), triplets.end());
    analytic_jacobian.makeCompressed();

    // We verify the analytic_jacobian is an exact bit by bit copy of jacobian.
    // Eigen does not offer SparseMatrix::operator==() and therefore we compare
    // the results by explicitly comparing the individual components of the CCS
    // format.
    Eigen::Map<VectorX<double>> analytic_jacobian_values(
        analytic_jacobian.valuePtr(), analytic_jacobian.nonZeros());
    Eigen::Map<VectorX<double>> jacobian_values(jacobian.valuePtr(),
                                                jacobian.nonZeros());
    Eigen::MatrixXd ad(analytic_jacobian);
    Eigen::MatrixXd cd(jacobian);
    std::cout << "analytic\n" << ad << std::endl;
    std::cout << "calculated\n" << cd << std::endl;
    EXPECT_EQ(analytic_jacobian_values, jacobian_values);

    Eigen::Map<VectorX<int>> analytic_jacobian_inner(
        analytic_jacobian.innerIndexPtr(), analytic_jacobian.innerSize());
    Eigen::Map<VectorX<int>> jacobian_inner(jacobian.innerIndexPtr(),
                                            jacobian.innerSize());
    EXPECT_EQ(analytic_jacobian_inner, jacobian_inner);

    Eigen::Map<VectorX<int>> analytic_jacobian_outer(
        analytic_jacobian.outerIndexPtr(), analytic_jacobian.outerSize());
    Eigen::Map<VectorX<int>> jacobian_outer(jacobian.outerIndexPtr(),
                                            jacobian.outerSize());
    EXPECT_EQ(analytic_jacobian_outer, jacobian_outer);
  }
}

}  // namespace
}  // namespace fem
}  // namespace drake
