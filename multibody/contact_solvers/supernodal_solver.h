#pragma once
#include <vector>

#include <Eigen/Dense>

#include "conex/kkt_solver.h"

namespace conex {

struct SolverData {
  std::vector<std::vector<int>> cliques;
  int num_vars;
  std::vector<int> order;
  std::vector<std::vector<int>> supernodes;
  std::vector<std::vector<int>> separators;
};

// MatrixBlock.first contains the data of a submatrix and
// MatrixBlock.second contains the columns numbers.  The
// rows are inferred.
using MatrixBlock = std::pair<Eigen::MatrixXd, std::vector<int>>;
using MatrixBlockData = std::pair<Eigen::MatrixXd, int>;
using MatrixBlocks = std::vector<MatrixBlock>;

using JacobianRowData = std::vector<MatrixBlockData>;

// A supernodal solver for the solving the symmetric positive definite system Tx
// = b,
// where T = M + J^T G J.  The matrices M and J are set by the
// constructor and the weight matrix G is set by SetWeightMatrix, which
// can be called multiple times on a constructed object.
//
// Example use case::
//
//  SolverNodalSolver solver( ... );
//  solver.SetWeightMatrix( ... );
//  solver.Factor();
//
//  // Solve Tx = b1.
//  x1 = solver.Solve(b1);
//  // Resolve Tx = b2.
//  x2 = solver.Solve(b2);
//
//  // Update weight matrix and refactor
//  solver.SetWeightMatrix( ... );
//  solver.Factor();
//  // Solve Tx = b1 with different weight matrix G.
//  x1 = solver.Solve(b1);
//
using BlockMatrixTriplet = std::tuple<int, int, Eigen::MatrixXd>;

struct SparsityData {
  SolverData data;
  std::vector<std::vector<int>> cliques_assembler;
};

class SuperNodalSolver {
 public:  // The i^{th} row of J is specified by the i^{th} entry of rows_of_J.
  // The argument num_vars should equal the number of columns of T.
  SuperNodalSolver(const MatrixBlocks& block_diagonal_M,
                   const std::vector<MatrixBlocks>& rows_of_J);

  SuperNodalSolver(int num_jacobian_row_blocks,
                   const std::vector<BlockMatrixTriplet>& jacobian_blocks,
                   const std::vector<Eigen::MatrixXd>& mass_matrices);

  void SetWeightMatrix(const std::vector<Eigen::MatrixXd>& block_diagonal_W);

  // Returns the matrix T (for debugging).
  Eigen::MatrixXd FullMatrix() {
    if (!matrix_ready_) {
      throw std::runtime_error(
          std::string("Call to FullMatrix() failed: weight matrix not set or "
                      "matrix already factored."));
    }
    return solver_.KKTMatrix();
  }

  // Computes the supernodal LLT factorization.
  void Factor();

  Eigen::MatrixXd Solve(const Eigen::VectorXd& b);
  void SolveInPlace(Eigen::VectorXd* b);

 private:
  class CliqueAssembler final : public LinearKKTAssemblerBase {
   public:
    virtual void SetDenseData() override;

    void SetWeightMatrixIndex(int start, int end) {
      weight_start_ = start;
      weight_end_ = end;
    }

    void SetWeightMatrixPointer(
        const std::vector<Eigen::MatrixXd>* weight_matrix) {
      weight_matrix_ = weight_matrix;
    }

    void AssignMassMatrix(int i, const Eigen::MatrixXd& A) {
      mass_matrix_position_.push_back(i);
      mass_matrix_.push_back(A);
    }
    int NumRows() { return row_data_.at(0).rows(); }

    void SetMatrixBlocks(const std::vector<Eigen::MatrixXd>& r) {
      row_data_ = r;
      temporaries_.resize(r.size());
      for (size_t j = 0; j < row_data_.size(); j++) {
        temporaries_.at(j).resize(r.at(j).rows(), r.at(j).cols());
      }
    }

   private:
    void BuildSubmatrixFromWeightedRow(const Eigen::MatrixXd& A);
    std::vector<Eigen::MatrixXd> row_data_;
    std::vector<int> mass_matrix_position_;
    std::vector<Eigen::MatrixXd> mass_matrix_;
    std::vector<Eigen::MatrixXd> temporaries_;
    const std::vector<Eigen::MatrixXd>* weight_matrix_;
    int weight_start_ = 0;
    int weight_end_ = 0;
  };

 private:
  void Initialize(const std::vector<std::vector<int>>& cliques,
                  const std::vector<std::vector<Eigen::MatrixXd>>& row_data);
  bool factorization_ready_ = false;
  bool matrix_ready_ = false;
  bool weight_matrix_ready_ = false;
  const MatrixBlocks mass_matrices_;
  std::vector<Eigen::MatrixXd> weight_matrices_;
  std::vector<std::vector<int>> cliques_;
  SparsityData clique_data_;
  Solver solver_;
  std::vector<CliqueAssembler*> clique_assemblers_;
  std::vector<CliqueAssembler> jacobian_assemblers_;
};

}  // namespace conex
