#pragma once

#include "drake/fem/fem_system.h"

#include <string>
#include <fstream>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {

template <typename T>
class FemSystem;

template <typename T>
class VtkParser {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(VtkParser)
  VtkParser(FemSystem<T>* fem_system) : fem_system_(fem_system) {}

  void Parse(const std::string& vtk_file) {
    std::ifstream fs;
    fs.open(vtk_file);

    std::string line;
    bool reading_points = false;
    bool reading_tets = false;
    int n_points = 0;
    int n_tets = 0;
    int position_count = 0;
    int indices_count = 0;
    Matrix3X<T>& positions = fem_system_->get_mutable_Q();
    std::vector<Vector4<int>> indices = fem_system_->get_mutable_indices();
    Vector3<T> position;
    Vector4<int> index;
    while (std::getline(fs, line)) {
      std::stringstream ss(line);
      if (static_cast<int>(line.size()) == 0) {
      } else if (line.substr(0, 6) == "POINTS") {
        reading_points = true;
        reading_tets = false;
        ss.ignore(128, ' ');  // Ignore "POINTS".
        ss >> n_points;
        positions.resize(3, n_points);
      } else if (line.substr(0, 5) == "CELLS") {
        reading_points = false;
        reading_tets = true;
        ss.ignore(128, ' ');  // Ignore "CELLS".
        ss >> n_tets;
        indices.resize(n_tets);
      } else if (line.substr(0, 10) == "CELL_TYPES") {
        reading_points = false;
        reading_tets = false;
      } else if (reading_points) {
        for (int i = 0; i < 3; ++i) ss >> position(i);
        positions.col(position_count++) = position;
      } else if (reading_tets) {
        int d;
        ss >> d;
        // Only tetrahedral mesh is supported.
        DRAKE_DEMAND(d == 4);
        ss.ignore(128, ' ');  // ignore "4"
        for (int i = 0; i < 4; i++) {
          ss >> index(i);
        }
        indices[indices_count++] = index;
      }
    }
    fs.close();
  }

 private:
  FemSystem<T>* fem_system_;
};
}  // namespace fem
}  // namespace drake
