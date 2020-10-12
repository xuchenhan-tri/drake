#pragma once

#include <array>
#include <string>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/nice_type_name.h"

namespace drake {
namespace fem {
template <typename T, int NaturalDimension>
class Quadrature {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Quadrature);

  using VectorD = Eigen::Matrix<T, NaturalDimension, 1>;

  virtual ~Quadrature() = default;

  /// The number of quadrature points for the quadrature rule.
  int num_points() const { return points_.size(); }

  /// The position in parent coordinate of the q-th quadrature point.
  const VectorD& get_point(int q) const {
    DRAKE_DEMAND(q >= 0);
    DRAKE_DEMAND(q < num_points());
    return points_[q];
  }

  /// The weight of the q-th quadrature point.
  T get_weight(int q) const {
    DRAKE_DEMAND(q >= 0);
    DRAKE_DEMAND(q < num_points());
    return weights_[q];
  }

 protected:
  Quadrature(std::vector<VectorD>&& points, std::vector<T>&& weights)
      : points_(points.begin(), points.end()),
        weights_(weights.begin(), weights.end()) {}

 private:
  std::vector<VectorD> points_;
  std::vector<T> weights_;
};

/** Calculates the Gaussian quadrature rule for 2D and 3D unit simplices
 (triangles and tetrahedrons up to cubic order as described in [Hammer, 1956].
The 2D unit triangle has vertices located at (0,0), (1,0) and (0,1). The 3D unit
tetrahedron has vertices located at (0,0,0), (1,0,0), (0,1,0) and (0,0,1).
 @tparam QuadratureOrder order of the quadrature rule. Must be 1, 2, or 3.
 @tparam NaturalDimension dimension of the unit simplex. Must be 2, or 3.

[Hammer, 1956] P.C. Hammer, O.P. Marlowe, and A.H. Stroud. Numerical integration
over simplexes and cones. Math. Tables Aids Comp. 10, 130-7, 1956.
 */
template <typename T, int QuadratureOrder, int NaturalDimension>
class SimplexGaussianQuadrature : public Quadrature<T, NaturalDimension> {
 public:
  using VectorD = typename Quadrature<T, NaturalDimension>::VectorD;
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SimplexGaussianQuadrature);
  SimplexGaussianQuadrature()
      : Quadrature<T, NaturalDimension>(GetPoints(), GetWeights()) {}

  std::vector<VectorD> GetPoints() {
    static_assert(1 <= QuadratureOrder && QuadratureOrder <= 3,
                  "Only linear, quadratic and cubic quadratures are supported");
    if constexpr (NaturalDimension == 2) {
      // For a unit triangle, area = 0.5.
      if constexpr (QuadratureOrder == 1) {
        // quadrature point location,  weight/area
        //  (1/3, 1/3)                     1.0
        std::vector<VectorD> points = {{1.0 / 3.0, 1.0 / 3.0}};
        return points;
      } else if constexpr (QuadratureOrder == 2) {
        // quadrature point location,  weight/area
        //  (1/6, 1/6)                     1/3
        //  (2/3, 1/6)                     1/3
        //  (1/6, 2/3)                     1/3
        std::vector<VectorD> points(3);
        points[0] = {1.0 / 6.0, 1.0 / 6.0};
        points[1] = {2.0 / 3.0, 1.0 / 6.0};
        points[2] = {1.0 / 6.0, 2.0 / 3.0};
        return points;
      } else if constexpr (QuadratureOrder == 3) {
        // quadrature point location,  weight/area
        //  (1/3, 1/3)                     -9/16
        //  (3/5, 1/5)                     25/48
        //  (1/5, 3/5)                     25/48
        //  (1/5, 1/5)                     25/48
        std::vector<VectorD> points(4);
        points[0] = {1.0 / 3.0, 1.0 / 3.0};
        points[1] = {0.6, 0.2};
        points[2] = {0.2, 0.6};
        points[3] = {0.2, 0.2};
        return points;
      } else {
        DRAKE_UNREACHABLE();
      }
    } else if constexpr (NaturalDimension == 3) {
      // For a unit tetrahedron, area = 1/6.
      if constexpr (QuadratureOrder == 1) {
        // quadrature point location,  weight/area
        //  (1/4, 1/4, 1/4)                1.0
        std::vector<VectorD> points = {{0.25, 0.25, 0.25}};
        return points;
      } else if constexpr (QuadratureOrder == 2) {
        // quadrature point location,  weight/area
        //  (a, b, b)                      1/4
        //  (b, a, b)                      1/4
        //  (b, b, a)                      1/4
        //  (b, b, b)                      1/4
        // where a = (1+3*sqrt(1/5))/4, b = (1-1/sqrt(1/5))/4.
        std::vector<VectorD> points(4);
        T a = (1.0 + 3.0 * std::sqrt(0.2)) / 4.0;
        T b = (1.0 - std::sqrt(0.2)) / 4.0;
        points[0] = {a, b, b};
        points[1] = {b, a, b};
        points[2] = {b, b, a};
        points[3] = {b, b, b};
        return points;
      } else if constexpr (QuadratureOrder == 3) {
        // quadrature point location,  weight/area
        //  (1/4, 1/4, 1/4)               -4/5
        //  (a, b, b)                      9/20
        //  (b, a, b)                      9/20
        //  (b, b, a)                      9/20
        //  (b, b, b)                      9/20
        // where a = 1/2, b = 1/6.
        std::vector<VectorD> points(5);
        T a = 0.5;
        T b = 1.0 / 6.0;
        points[0] = {0.25, 0.25, 0.25};
        points[1] = {a, b, b};
        points[2] = {b, a, b};
        points[3] = {b, b, a};
        points[4] = {b, b, b};
        return points;
      } else {
        DRAKE_UNREACHABLE();
      }
    } else {
      DRAKE_UNREACHABLE();
    }
  }

  std::vector<T> GetWeights() {
    static_assert(1 <= QuadratureOrder && QuadratureOrder <= 3,
                  "Only linear, quadratic and cubic quadratures are supported");
    if constexpr (NaturalDimension == 2) {
      // For a unit triangle, area = 0.5.
      if constexpr (QuadratureOrder == 1) {
        // quadrature point location,  weight/area
        //  (1/3, 1/3)                     1.0
        std::vector<T> weights = {0.5};
        return weights;
      } else if constexpr (QuadratureOrder == 2) {
        // quadrature point location,  weight/area
        //  (1/6, 1/6)                     1/3
        //  (2/3, 1/6)                     1/3
        //  (1/6, 2/3)                     1/3
        std::vector<T> weights = {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0};
        return weights;
      } else if constexpr (QuadratureOrder == 3) {
        // quadrature point location,  weight/area
        //  (1/3, 1/3)                     -9/16
        //  (3/5, 1/5)                     25/48
        //  (1/5, 3/5)                     25/48
        //  (1/5, 1/5)                     25/48
        std::vector<T> weights = {-9.0 / 32.0, 25.0 / 96.0, 25.0 / 96.0,
                                  25.0 / 96.0};
        return weights;
      } else {
        DRAKE_UNREACHABLE();
      }
    } else if constexpr (NaturalDimension == 3) {
      // For a unit tetrahedron, area = 1/6.
      if constexpr (QuadratureOrder == 1) {
        // quadrature point location,  weight/area
        //  (1/4, 1/4, 1/4)                1.0
        std::vector<T> weights = {1.0 / 6.0};
        return weights;
      } else if constexpr (QuadratureOrder == 2) {
        // quadrature point location,  weight/area
        //  (a, b, b)                      1/4
        //  (b, a, b)                      1/4
        //  (b, b, a)                      1/4
        //  (b, b, b)                      1/4
        // where a = (1+3*sqrt(1/5))/4, b = (1-1/sqrt(1/5))/4.
        std::vector<T> weights = {1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0,
                                  1.0 / 24.0};
        return weights;
      } else if constexpr (QuadratureOrder == 3) {
        // quadrature point location,  weight/area
        //  (1/4, 1/4, 1/4)               -4/5
        //  (a, b, b)                      9/20
        //  (b, a, b)                      9/20
        //  (b, b, a)                      9/20
        //  (b, b, b)                      9/20
        // where a = 1/2, b = 1/6.
        std::vector<T> weights = {-2.0 / 15.0, 3.0 / 40.0, 3.0 / 40.0,
                                  3.0 / 40.0, 3.0 / 40.0};
        return weights;
      } else {
        DRAKE_UNREACHABLE();
      }
    } else {
      DRAKE_UNREACHABLE();
    }
  }
};
}  // namespace fem
}  // namespace drake
