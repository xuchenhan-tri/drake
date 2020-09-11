#include "drake/fem/query_contact.h"

#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "drake/fem/collision_object.h"
#include "drake/fem/half_space.h"

namespace drake {
namespace fem {
namespace {

class QueryContactTest : public ::testing::Test {
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
  }

 protected:
  std::vector<std::unique_ptr<CollisionObject<double>>> collision_objects_;
  Matrix3X<double> q_;
};

TEST_F(QueryContactTest, QueryContact) {
  Eigen::SparseMatrix<double> jacobian;
  VectorX<double> penetration_depth;
  std::vector<Vector3<double>> normals;
  QueryContact(q_, collision_objects_, &jacobian, &penetration_depth, &normals);
  constexpr int nc = 4;
  std::vector<Vector3<double>> analytic_normals(nc);
  // contact 0: q.col(0) vs. ls2.
  analytic_normals[0] = Vector3<double>(0, 1, 0);
  // contact 1: q.col(1) vs. ls1.
  analytic_normals[1] = Vector3<double>(0, 0, 1);
  // contact 2: q.col(1) vs. ls2.
  analytic_normals[2] = Vector3<double>(0, 1, 0);
  // contact 2: q.col(2) vs. ls1.
  analytic_normals[3] = Vector3<double>(0, 0, 1);
  // q.col(3) is not in contact with any collision object.

  // The calculated normals should be bitwise equal to the analytic normals.
  EXPECT_EQ(normals, analytic_normals);

  VectorX<double> analytic_penetration(nc);
  for (int i = 0; i < nc; ++i) analytic_penetration(i) = -1.0;
  // The penetration depth is not necessarily bitwise equal to the analytic ones because a
  // dot product is required to get the penetration depth.
  EXPECT_NEAR((penetration_depth - analytic_penetration).norm(), 0.0,
              std::numeric_limits<double>::epsilon());

  // Jc_blocks[i][j] is the jacobian block corresponding to i-th contact point and j-th vertex.
  MatrixX<double> dense_Jc(jacobian);
  std::vector<std::vector<Matrix3<double>>> Jc_blocks(4);
  for (int i = 0; i < 4; ++i){
      Jc_blocks[i].resize(4);
      for (int j = 0; j < 4; ++j){
         Jc_blocks[i][j] = dense_Jc.block<3,3>(3*i, 3*j);
      }
  }
    std::vector<int> contact_indices = {0, 1, 2, 3};
    std::vector<int> vertex_indices = {0, 1, 1, 2};
    for (int pair = 0; pair < nc; ++pair){
        auto& J = Jc_blocks[contact_indices[pair]][vertex_indices[pair]];
        // The Jacobian block should be a rotation matrix with the last row being the normal of the contact.
        Vector3<double> normal = J.row(2);
        EXPECT_EQ(normal, analytic_normals[pair]);
        EXPECT_TRUE(drake::math::RotationMatrix<double>::IsValid(J, std::numeric_limits<double>::epsilon()));
        // Set this non-zero block to zero.
        J.setZero();
    }
    // After setting the non-zero blocks to zero. All blocks should be zero.
    for (int pair = 0; pair < 4; ++pair){
        const auto &J = Jc_blocks[contact_indices[pair]][vertex_indices[pair]];
        EXPECT_EQ(J, Matrix3<double>::Zero());
    }
}

}  // namespace
}  // namespace fem
}  // namespace drake
