#pragma once

#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {
template <typename T>
class MeshUtility {
public:
  static Matrix3X<T> AddRectangularBlockVertices(const int nx, const int ny,
                                                 const int nz, T h) {
    const int n_points = (nx + 1) * (ny + 1) * (nz + 1);
    Matrix3X<T> vertices(3, n_points);
    int position_count = 0;
    Vector3<T> position;
    for (int x = 0; x <= nx; ++x) {
      position(0) = h * x;
      for (int y = 0; y <= ny; ++y) {
        position(1) = h * y;
        for (int z = 0; z <= nz; ++z) {
          position(2) = h * z;
          vertices.col(position_count++) = position;
        }
      }
    }
    return vertices;
  }

  static std::vector<Vector4<int>> AddRectangularBlockMesh(const int nx,
                                                           const int ny,
                                                           const int nz) {
    std::vector<Vector4<int>> vertex_indices;
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
          // For each block, the 8 corners are numerated as:
          //     4*-----*7
          //     /|    /|
          //    / |   / |
          //  5*-----*6 |
          //   | 0*--|--*3
          //   | /   | /
          //   |/    |/
          //  1*-----*2

          //    j ^
          //      |
          //      |
          //      *-----> i
          //     /
          //    /
          //   k

          const int p0 = (i * (ny + 1) + j) * (nz + 1) + k;
          const int p1 = p0 + 1;
          const int p3 = ((i + 1) * (ny + 1) + j) * (nz + 1) + k;
          const int p2 = p3 + 1;
          const int p7 = ((i + 1) * (ny + 1) + (j + 1)) * (nz + 1) + k;
          const int p6 = p7 + 1;
          const int p4 = (i * (nx + 1) + (j + 1)) * (nz + 1) + k;
          const int p5 = p4 + 1;

          // Ensure that neighboring tetras are sharing faces, and within a
          // single tetrahedron, if the indices are ordered like [a,b,c,d], then
          // the normal given by right hand rule applied to the face [a,b,c]
          // points to the node d.
          if ((i + j + k) % 2 == 1) {
            vertex_indices.emplace_back(p2, p1, p6, p3);
            vertex_indices.emplace_back(p6, p3, p4, p7);
            vertex_indices.emplace_back(p4, p1, p6, p5);
            vertex_indices.emplace_back(p3, p1, p4, p0);
            vertex_indices.emplace_back(p6, p1, p4, p3);
          } else {
            vertex_indices.emplace_back(p0, p2, p5, p1);
            vertex_indices.emplace_back(p7, p2, p0, p3);
            vertex_indices.emplace_back(p5, p2, p7, p6);
            vertex_indices.emplace_back(p7, p0, p5, p4);
            vertex_indices.emplace_back(p0, p2, p7, p5);
          }
        }
      }
    }
    return vertex_indices;
  }
};
}  // namespace fem
}  // namespace drake
