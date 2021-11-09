#pragma once
#include <Eigen/Dense>

#include "drake/multibody/contact_solvers/supernodal_solver.h"

struct SpinFactorProduct {
  Eigen::MatrixXd w1;
  Eigen::VectorXd w0;
  SpinFactorProduct(int n) : w1(2, n), w0(n) {}

  SpinFactorProduct(const Eigen::VectorXd& d, int n) : SpinFactorProduct(n) {
    int offset = 0;
    for (int i = 0; i < n; i++) {
      w1.col(i) = d.segment(offset, 2);
      w0(i) = d(offset + 2);
      offset += 3;
    }
  }
};

struct SpinFactorElement {
  double w0;
  Eigen::VectorXd w1;
  SpinFactorElement(){};
  SpinFactorElement(const Eigen::VectorXd& x) {
    w0 = x(2);
    w1 = x.head(x.size() - 1);
  }
  SpinFactorElement(const SpinFactorProduct& x, int i)
      : w0(x.w0(i)), w1(x.w1.col(i)) {}
};

class SparseJacobian {
 public:
  SparseJacobian(const std::vector<drake::multibody::contact_solvers::internal::BlockMatrixTriplet>& blocks);
  Eigen::VectorXd Multiply(const Eigen::VectorXd& x) const;
  Eigen::VectorXd MultiplyByTranspose(const Eigen::VectorXd& x) const;

  void RescaleByFrictionCoefficient(const Eigen::VectorXd& x);

  std::vector<drake::multibody::contact_solvers::internal::BlockMatrixTriplet> blocks() { return blocks_; }
 private:
  std::vector<drake::multibody::contact_solvers::internal::BlockMatrixTriplet> blocks_;
  std::vector<int> row_offset_;
  std::vector<int> col_offset_;
  std::vector<int> row_block_sizes_;
  std::vector<int> col_block_sizes_;
  int num_col_blocks_ = 0;
  int num_row_blocks_ = 0;
};

class BlockDiagonalMatrix {
 public:
  BlockDiagonalMatrix(const std::vector<Eigen::MatrixXd>& blocks)
      : blocks_(blocks) {}
  Eigen::VectorXd Multiply(const Eigen::VectorXd& x) const;
  Eigen::VectorXd MultiplyByTranspose(const Eigen::VectorXd& x) const;

 private:
  std::vector<Eigen::MatrixXd> blocks_;
};

bool TakeStep(const SpinFactorProduct& wa, const SpinFactorProduct& da,
              SpinFactorProduct* ya, bool affine_step = false);

double NormInf(const SpinFactorProduct& x);

class NewtonDirection {
  using MatrixXd = Eigen::MatrixXd;
  using Matrix3d = Eigen::Matrix3d;
  using VectorXd = Eigen::VectorXd;
  using Vector3d = Eigen::Vector3d;

 public:
  NewtonDirection(const std::vector<drake::multibody::contact_solvers::internal::BlockMatrixTriplet>& Jtriplets,
                  const std::vector<Eigen::MatrixXd>& blocks_of_M,
                  int num_block_rows_of_J, const VectorXd& R,
                  int num_constraints)
      : num_constraints_(num_constraints),
        R_(num_constraints),
        W_(num_constraints),
        Rinv_(num_constraints),
        w_(num_constraints),
        kkt_solver_(num_block_rows_of_J, Jtriplets, blocks_of_M),
        M_(blocks_of_M),
        Jtriplets_(Jtriplets),
        J_(Jtriplets_) {
    SpinFactorProduct e(num_constraints);
    e.w0.setConstant(1);
    e.w1.setZero();
    int offset = 0;
    for (int i = 0; i < num_constraints_; i++) {
      R_.at(i) = R.segment(offset, 3);
      Rinv_.at(i) = R.segment(offset, 3).cwiseInverse();
      offset += 3;
    }
  }

  const std::vector<Eigen::MatrixXd>& QWQ() { return QWQ_; }

  void SetW(const SpinFactorProduct& w);
  void SolveNewtonSystem(double k, const VectorXd& vstar,
                         const VectorXd& v_hat, VectorXd* v);
  void FactorNewtonSystem();
  void CalculateNewtonDirection(double k, const VectorXd& vstar,
                                const VectorXd& v_hat, Eigen::VectorXd* d,
                                Eigen::VectorXd* v);

  const std::vector<Matrix3d>& Qwsqrt() const { return Qsqrt_; }

  int num_constraints_ = 1;
  std::vector<Eigen::Vector3d> R_;
  std::vector<Eigen::Matrix3d> W_;
  std::vector<Eigen::Vector3d> Rinv_;

  std::vector<Eigen::Matrix3d> Qsqrt_;
  std::vector<Eigen::MatrixXd> QWQ_;

  MatrixXd e_array_;
  bool newton_system_factored_ = false;
  SpinFactorProduct w_;
  drake::multibody::contact_solvers::internal::SuperNodalSolver kkt_solver_;
  const BlockDiagonalMatrix M_;
  std::vector<drake::multibody::contact_solvers::internal::BlockMatrixTriplet> Jtriplets_;
  const SparseJacobian J_;
};

struct GeodesicSolverInfo {
  int iterations;
  bool failed = false;
};

struct GeodesicSolverSolution {
  Eigen::VectorXd v;
  Eigen::VectorXd lambda;
  GeodesicSolverInfo info;
};

struct GeodesicSolverOptions {
  int verbosity = 0;
  int maximum_iterations = 50;
  double target_mu = 3e-7;
};

GeodesicSolverSolution GeodesicSolver(
    const GeodesicSolverSolution& w0,
    const std::vector<drake::multibody::contact_solvers::internal::BlockMatrixTriplet>& J,
    const std::vector<Eigen::MatrixXd>& M, const Eigen::VectorXd& R,
    int num_block_rows_of_J, int num_contacts, const Eigen::VectorXd& vstar,
    const Eigen::VectorXd& v_hat, 
    const GeodesicSolverOptions& options = GeodesicSolverOptions());

GeodesicSolverSolution GeodesicSolver(
    const SpinFactorProduct& w0,
    const std::vector<drake::multibody::contact_solvers::internal::BlockMatrixTriplet>& J,
    const std::vector<Eigen::MatrixXd>& M, const Eigen::VectorXd& R,
    int num_block_rows_of_J, int num_contacts, const Eigen::VectorXd& vstar,
    const Eigen::VectorXd& v_hat,
    const GeodesicSolverOptions& options = GeodesicSolverOptions());


GeodesicSolverSolution GeodesicSolver(
    const GeodesicSolverSolution& sol,
    const Eigen::VectorXd& friction_coefficient,
    const std::vector<drake::multibody::contact_solvers::internal::BlockMatrixTriplet>& J,
    const std::vector<Eigen::MatrixXd>& M, const Eigen::VectorXd& R,
    int num_block_rows_of_J, int num_contacts, const Eigen::VectorXd& vstar,
    const Eigen::VectorXd& v_hat,
    const GeodesicSolverOptions& options);


