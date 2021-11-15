#include "conex/clique_ordering.h"
#include "conex/kkt_solver.h"
#include "conex/supernodal_solver.h"

#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/supernodal_solver.h"

namespace drake {
namespace multibody {
namespace fem {

using SolverData = conex::SolverData;
using SparsityData = conex::SparsityData;

class BlockSparseSolver {
 public:
  BlockSparseSolver(int num_vertices,
                    const std::vector<Eigen::Vector4i>& elements)
      : clique_data_(GetCliqueData(num_vertices, elements)),
        solver_(clique_data_.cliques_assembler, clique_data_.data.num_vars,
                clique_data_.data.order, clique_data_.data.supernodes,
                clique_data_.data.separators),
        element_clique_assemblers_(elements.size()),
        element_clique_assembler_ptrs_(elements.size()) {
    for (int i = 0; i < static_cast<int>(elements.size()); ++i) {
      element_clique_assemblers_[i].set_element(elements[i]);
      element_clique_assembler_ptrs_[i] = &element_clique_assemblers_[i];
    }
    solver_.Bind(element_clique_assembler_ptrs_);
  }

  /* Returns the dense matrix currently being factored/solved (for debugging).
   */
  MatrixX<double> MakeDenseMatrix() { return solver_.KKTMatrix(); }

  /* Computes the supernodal Cholesky factorization. */
  void Factor() {
    if (!matrix_ready_) {
      throw std::runtime_error(
          std::string("Call to Factor() failed: weight matrix not set."));
    }
    solver_.Factor();
    factorization_ready_ = true;
  }

  /* x = A\b. */
  VectorX<double> Solve(const VectorX<double>& b) {
    if (!factorization_ready_) {
      throw std::runtime_error(
          std::string("Call to Solve() failed: factorization not ready."));
    }
    MatrixX<double> x = b;
    Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> xmap(x.data(), b.rows(), 1);
    solver_.SolveInPlace(&xmap);
    return x;
  }

  /* Sets the tangent matrix for an element. */
  void SetElementTangentMatrix(int element,
                               Eigen::Matrix<double, 12, 12> data) {
    auto& assembler = element_clique_assemblers_[element];
    assembler.set_data(std::move(data));
  }

  void BuildMatrix() {
    solver_.AssembleFromCliques(element_clique_assembler_ptrs_);
    factorization_ready_ = false;
    matrix_ready_ = true;
  }

  void SetMatrixToZero() {
    for (auto& assembler : element_clique_assemblers_) {
      assembler.SetZero();
    }
    BuildMatrix();
  }

 private:
  class ElementCliqueAssembler final : public conex::LinearKKTAssemblerBase {
   public:
    void SetDenseData() override {
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          schur_complement_data.G.block<3, 3>(3 * i, 3 * j) =
              data_.block<3, 3>(3 * index_mapping_[i], 3 * index_mapping_[j]);
        }
      }
    }

    void set_element(const Eigen::Vector4i& element) {
      index_mapping_.resize(4);
      std::iota(index_mapping_.begin(), index_mapping_.end(), 0);
      sort(index_mapping_.begin(), index_mapping_.end(),
           [&](int i, int j) { return element(i) < element(j); });
    }

    void set_data(Eigen::Matrix<double, 12, 12> data) {
      data_ = std::move(data);
    }

    void SetZero() { data_.setZero(); }

   private:
    Eigen::Matrix<double, 12, 12> data_;
    // Mapping so that index within a single element is increasing.
    std::vector<int> index_mapping_;
  };

  void print(const std::vector<std::vector<int>>& vv) {
    for (const auto& v : vv) {
      print(v);
    }
  }

  void print(const std::vector<int>& v) {
    std::cout << "v = ";
    for (const auto& i : v) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }

  SparsityData GetCliqueData(int num_vertices,
                             const std::vector<Eigen::Vector4i>& elements) {
    using std::vector;
    SparsityData clique_data;
    auto& data = clique_data.data;

    size_t n = elements.size();
    vector<int> order(n);
    vector<vector<int>> supernodes(n);
    vector<vector<int>> separators(n);
    vector<vector<int>> cliques(n);
    clique_data.cliques_assembler.resize(n);
    conex::RootedTree tree(n);

    for (int e = 0; e < static_cast<int>(elements.size()); ++e) {
      for (int v = 0; v < 4; ++v) {
        cliques.at(e).push_back(elements[e](v));
      }
    }

    auto sort_cliques = [](std::vector<std::vector<int>>* v) {
      for (size_t i = 0; i < v->size(); i++) {
        std::sort(v->at(i).begin(), v->at(i).end());
      }
    };
    sort_cliques(&cliques);

    int largest_clique = 0;
    for (size_t i = 1; i < cliques.size(); i++) {
      if (cliques.at(i).size() > cliques.at(largest_clique).size()) {
        largest_clique = i;
      }
    }

    conex::GetCliqueEliminationOrder(cliques, largest_clique /*root*/, &order,
                                     &supernodes, &separators, &tree);
    std::cout << "supernodes:" << std::endl;
    print(supernodes);
    std::cout << "separators:" << std::endl;
    print(separators);
    std::cout << "order:" << std::endl;
    print(order);
    conex::FillIn(tree, num_vertices, order, &supernodes, &separators);
    
    std::cout << " ==============FILLIN==============" << std::endl;
    std::cout << "supernodes:" << std::endl;
    print(supernodes);
    std::cout << "separators:" << std::endl;
    print(separators);
    std::cout << "order:" << std::endl;
    print(order);
    std::cout << " =================" << std::endl;

    vector<vector<int>> supernodes_full(order.size());
    vector<vector<int>> separators_full(order.size());
    vector<vector<int>> cliques_full(order.size());
    for (size_t i = 0; i < order.size(); i++) {
      for (const auto& s : supernodes.at(i)) {
        for (int dim = 0; dim < 3; dim++) {
          supernodes_full.at(i).push_back(3 * s + dim);
          cliques_full.at(i).push_back(3 * s + dim);
        }
      }
      for (const auto& s : separators.at(i)) {
        for (int dim = 0; dim < 3; dim++) {
          separators_full.at(i).push_back(3 * s + dim);
          cliques_full.at(i).push_back(3 * s + dim);
        }
      }
      for (const auto& s : cliques.at(i)) {
        for (int dim = 0; dim < 3; dim++) {
          clique_data.cliques_assembler.at(i).push_back(3 * s + dim);
        }
      }
    }

    data.cliques = cliques_full;
    sort_cliques(&data.cliques);

    data.num_vars = 3 * num_vertices;
    data.supernodes = supernodes_full;
    data.order = order;
    data.separators = separators_full;
    return clique_data;
  }

  bool factorization_ready_ = false;
  bool matrix_ready_ = false;
  SparsityData clique_data_;
  conex::Solver solver_;
  std::vector<ElementCliqueAssembler> element_clique_assemblers_;
  std::vector<ElementCliqueAssembler*> element_clique_assembler_ptrs_;
};

}  // namespace fem
}  // namespace multibody
}  // namespace drake
