#pragma once

#include <fstream>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/fem_system.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace fem {
template <typename T>
class ObjWriter final : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ObjWriter)
  ObjWriter(const FemSystem<T>& fem) {
    DRAKE_DEMAND(fem.is_finalized());
    this->DeclareInputPort("vertex_positions", systems::kVectorValued,
                           fem.get_num_position_dofs());
    // Generate boundary faces.
    const auto& tets = fem.get_indices();
    /* A typical tetrahedral element looks like:
         p2 *
            |
            |
         p3 *---* p0
           /
          /
      p1 *
      The indices order for a particular tetrahedral mesh has the order [p0, p1,
      p2, p3].
    */
    // Extract the indices of the four faces so that the right hand rule gives
    // the outward normal.
    const std::array<std::array<int, 3>, 4> local_indices{
        {{{1, 0, 2}}, {{3, 0, 1}}, {{3, 1, 2}}, {{2, 0, 3}}}};
    // Map from sorted face indices to actual indices.
    // E.g. If a tri-mesh has indices [7, 1, 12], the sorted indices would be
    // [1, 7, 12], and when inserted into dict, the mapping is [1, 7, 12] -> [7,
    // 1, 12].
    std::map<std::array<int, 3>, std::array<int, 3>> dict;
    for (const auto& tet : tets) {
      for (int f = 0; f < 4; ++f) {
        std::array<int, 3> face{tet(local_indices[f][0]),
                                tet(local_indices[f][1]),
                                tet(local_indices[f][2])};
        std::array<int, 3> face_sorted(face);
        std::sort(face_sorted.begin(), face_sorted.end());
        // If a face with the same vertices already exists: discards the new
        // face and the preexisiting face because they lay on top of each other
        // and thus are both internal faces.
        if (dict.find(face_sorted) != dict.end()) {
          dict.erase(face_sorted);
        } else {
          dict.insert(std::make_pair(face_sorted, face));
        }
      }
    }
    // Store the boundary faces in the member variable `faces_`.
    for (auto it = dict.begin(); it != dict.end(); ++it) {
      Vector3<int> f(it->second[0], it->second[1], it->second[2]);
      faces_.push_back(f);
    }
    this->DeclarePeriodicPublishEvent(fem.get_dt(), 0.0, &ObjWriter::WriteObj);
  }

  /** Write the boundary face tri-mesh to file. */
  void WriteObj(const systems::Context<T>& context) const {
    VectorX<T> input = this->get_input_port(0).Eval(context);
    const Matrix3X<T>& q =
        Eigen::Map<Matrix3X<T>>(input.data(), 3, input.size() / 3);
    std::ofstream fs;
    std::string filename = std::to_string(frame_++) + ".obj";
    fs.open(filename);
    DRAKE_DEMAND(fs.is_open());
    // Write vertex positions.
    for (int i = 0; i < q.cols(); ++i) {
      fs << "v";
      for (int d = 0; d < 3; d++) {
        fs << " " << q(d, i);
      }
      fs << "\n";
    }

    // Write tri-mesh.
    for (int i = 0; i < static_cast<int>(faces_.size()); ++i) {
      fs << "f";
      for (int n = 0; n < 3; n++) {
        fs << " " << faces_[i](n) + 1;
      }
      fs << "\n";
    }
    fs.close();
  }

 private:
  std::vector<Vector3<int>> faces_;
  mutable int frame_{0};
};
}  // namespace fem
}  // namespace drake
