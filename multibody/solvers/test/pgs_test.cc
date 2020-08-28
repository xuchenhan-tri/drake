#include <memory>

#include <Eigen/SparseCore>
#include <gtest/gtest.h>

#include "drake/multibody/solvers/pgs_solver.h"
#include "drake/multibody/solvers/point_contact_data.h"
#include "drake/multibody/solvers/sparse_linear_operator.h"
#include "drake/multibody/solvers/system_dynamics_data.h"

namespace drake {
namespace multibody {
namespace solvers {
namespace {

using SparseMatrixd = Eigen::SparseMatrix<double>;
using SparseVectord = Eigen::SparseVector<double>;
using Eigen::VectorXd;
using Triplet = Eigen::Triplet<double>;

class PgsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    jacobian.resize(3, 3);
    {
      std::vector<Triplet> triplets;
      triplets.emplace_back(0, 0, 1.0);
      triplets.emplace_back(1, 1, 1.0);
      triplets.emplace_back(2, 2, 1.0);
      jacobian.setFromTriplets(triplets.begin(), triplets.end());
      jacobian.makeCompressed();
    }
    std::cout << "setup 1" << std::endl;
     Jc = std::make_unique<SparseLinearOperator<double>>("Jc", &jacobian);
    penetration_depth.resize(1);
      penetration_depth(0) = -1;
    v_free = -1.0 * VectorXd::Ones(3);
    tau = VectorXd::Zero(v_free.size());

    Minv_tmp.resize(3, 3);
    {
      std::vector<Triplet> triplets;
      triplets.emplace_back(0, 0, 12.0);
      triplets.emplace_back(1, 1, 4.0);
      triplets.emplace_back(2, 2, 3.0);
      Minv_tmp.setFromTriplets(triplets.begin(), triplets.end());
      Minv_tmp.makeCompressed();
    }
    Minv = std::make_unique<SparseLinearOperator<double>>("Minv", &Minv_tmp);
    dynamics_data = std::make_unique<SystemDynamicsData<double>>(Minv.get(), &v_free, &tau);
    stiffness = VectorXd::Zero(1);
    dissipation = VectorXd::Zero(1);
    mu = 1000.0 * VectorXd::Ones(1);
    point_data = std::make_unique<PointContactData<double>>(&penetration_depth, Jc.get(), &stiffness,
                                        &dissipation, &mu);

    pgs_ = std::make_unique<PgsSolver<double>>();
    pgs_->SetSystemDynamicsData(dynamics_data.get());
    pgs_->SetPointContactData(point_data.get());
  }

  std::unique_ptr<PgsSolver<double>> pgs_;
  VectorXd v_free;
  VectorXd tau;
  std::unique_ptr<SparseLinearOperator<double>> Minv;
    std::unique_ptr<SystemDynamicsData<double>> dynamics_data;
    std::unique_ptr<PointContactData<double>> point_data;
    VectorXd penetration_depth;
    std::unique_ptr<SparseLinearOperator<double>> Jc;
    SparseMatrixd jacobian;
    SparseMatrixd Minv_tmp;
    VectorXd stiffness;
    VectorXd dissipation;
    VectorXd mu;
};

TEST_F(PgsTest, Solve) {
  VectorXd v = -1.0 * VectorXd::Ones(3);
        std::cout << "solver " << std::endl;
  pgs_->SolveWithGuess(1, v);
  std::cout << pgs_->GetVelocities() << std::endl;
  EXPECT_EQ(1,2);
}
}  // namespace
}  // namespace solvers
}  // namespace multibody
}  // namespace drake

