#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/volume_mesh.h"

namespace drake {
namespace multibody {
namespace fixed_fem {

template <typename T>
geometry::VolumeMesh<T> ParseVtk(const std::string& vtk_file, const Vector3<T>& p_WB) {
  using VertexIndex = geometry::VolumeVertexIndex;
  using geometry::VolumeElement;
  using geometry::VolumeVertex;
  std::ifstream fs;
  fs.open(vtk_file);

  std::string line;
  bool reading_points = false;
  bool reading_tets = false;
  int n_points = 0;
  int n_tets = 0;
  Vector3<T> position;
  Vector4<int> index;
  std::vector<VolumeElement> elements;
  std::vector<VolumeVertex<T>> vertices;
  while (std::getline(fs, line)) {
    std::stringstream ss(line);
    if (static_cast<int>(line.size()) == 0) {
    } else if (line.substr(0, 6) == "POINTS") {
      reading_points = true;
      reading_tets = false;
      ss.ignore(128, ' ');  // Ignore "POINTS".
      ss >> n_points;
      vertices.reserve(n_points);
    } else if (line.substr(0, 5) == "CELLS") {
      reading_points = false;
      reading_tets = true;
      ss.ignore(128, ' ');  // Ignore "CELLS".
      ss >> n_tets;
      elements.reserve(n_tets);
    } else if (line.substr(0, 10) == "CELL_TYPES") {
      reading_points = false;
      reading_tets = false;
    } else if (reading_points) {
      for (int i = 0; i < 3; ++i) ss >> position(i);
      vertices.emplace_back(0.5 * position+p_WB);
    } else if (reading_tets) {
      int d;
      ss >> d;
      // Only tetrahedral mesh is supported.
      DRAKE_DEMAND(d == 4);
      ss.ignore(128, ' ');  // ignore "4"
      for (int i = 0; i < 4; i++) {
        ss >> index(i);
      }
      elements.emplace_back(VertexIndex(index(0)), VertexIndex(index(1)),
                            VertexIndex(index(2)), VertexIndex(index(3)));
    }
  }
  fs.close();
  return {std::move(elements), std::move(vertices)};
}
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
