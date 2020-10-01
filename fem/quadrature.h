#pragma once

#include <array>
#include <string>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/nice_type_name.h"

namespace drake {
namespace fem {
template <typename T, int D>
class Quadrature {
 public:
  virtual ~Quadrature() = default;
  using VecType = Eigen::Matrix<T, D, 1>;
  /// The number of quadrature points for the quadrature rule.
  int num_points() const { return points_.size(); }

  /// The position in parent coordinate of the q-th quadrature point.
  const VecType& get_point(int q) const {
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
  void set_points(const std::vector<VecType>& points) { points_ = points; }

  void set_weights(const std::vector<T>& weights) { weights_ = weights; }

  // Helper to throw a specific exception when a given dimension or order is not
  // implemented.
  void ThrowNotImplemented(int N) const {
    throw std::runtime_error("Quadrature of type '" + NiceTypeName::Get(*this) +
                             " does not have an implementation in dimension " +
                             std::to_string(D) + " with order " +
                             std::to_string(N) + ".");
  }

 private:
  std::vector<VecType> points_;
  std::vector<T> weights_;
};

/** Calculates the Gaussian quadrature rule for 2D and 3D unit simplices
 (triangles and tetrahedrons up to cubic order. The 2D unit triangle has
 vertices located at (0,0), (1,0) and (0,1). The 3D unit tetrahedron has
 vertices located at (0,0,0), (1,0,0), (0,1,0) and (0,0,1).
 @tparam N order of the quadrature rule. Must be 1, 2, or 3.
 @tparam D dimension of the unit simplex. Must be 2, or 3.
 */
template <typename T, int N, int D>
class SimplexGaussianQuadrature : public Quadrature<T, D> {
 public:
  using VecType = typename Quadrature<T, D>::VecType;
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SimplexGaussianQuadrature);
  SimplexGaussianQuadrature() {
    // We only support linear, quadratic and cubic integration.
    if constexpr (N < 1 || N > 3) {
      this->ThrowNotImplemented(N);
    }
    // We only support triangles and tetrahedrons now.
    if constexpr (D < 2 || D > 3) {
      this->ThrowNotImplemented(N);
    }
    std::vector<VecType> points;
    std::vector<T> weights;
    if constexpr (D == 2) {
      // For a unit triangle, area = 0.5.
      if constexpr (N == 1) {
        // quadrature point location,  weight/area
        //  (1/3, 1/3)                     1.0
        points.resize(1, 1.0 / 3.0 * Vector2<T>::Ones(D));
        weights.resize(1, 0.5);
      } else if constexpr (N == 2) {
        // quadrature point location,  weight/area
        //  (1/6, 1/6)                     1/3
        //  (2/3, 1/6)                     1/3
        //  (1/6, 2/3)                     1/3
        points.resize(3);
        points[0] = (Vector2<T>() << 1.0 / 6.0, 1.0 / 6.0).finished();
        points[1] = (Vector2<T>() << 2.0 / 3.0, 1.0 / 6.0).finished();
        points[2] = (Vector2<T>() << 1.0 / 6.0, 2.0 / 3.0).finished();
        weights.resize(3, 1.0 / 6.0);
      } else if constexpr (N == 3) {
        // quadrature point location,  weight/area
        //  (1/3, 1/3)                     -9/16
        //  (3/5, 1/5)                     25/48
        //  (1/5, 3/5)                     25/48
        //  (1/5, 1/5)                     25/48
        points.resize(4);
        points[0] = (Vector2<T>() << 1.0 / 3.0, 1.0 / 3.0).finished();
        points[1] = (Vector2<T>() << 0.6, 0.2).finished();
        points[2] = (Vector2<T>() << 0.2, 0.6).finished();
        points[3] = (Vector2<T>() << 0.2, 0.2).finished();
        weights.resize(4);
        weights[0] = -9.0 / 32.0;
        for (int i = 1; i < 3; ++i) {
          weights[i] = 25.0 / 96.0;
        }
      } else {
        DRAKE_UNREACHABLE();
      }
    } else if constexpr (D == 3) {
      // For a unit tetrahedron, area = 1/6.
      if constexpr (N == 1) {
        // quadrature point location,  weight/area
        //  (1/4, 1/4, 1/4)                1.0
        points.resize(1, 0.25 * Vector3<T>::Ones(D));
        weights.resize(1, 1.0 / 6.0);
      } else if constexpr (N == 2) {
        // quadrature point location,  weight/area
        //  (a, b, b)                      1/4
        //  (b, a, b)                      1/4
        //  (b, b, a)                      1/4
        //  (b, b, b)                      1/4
        // where a = (1+3*sqrt(1/5))/4, b = (1-1/sqrt(1/5))/4.
        points.resize(4);
        T a = (1.0 + 3.0 * std::sqrt(0.2)) / 4.0;
        T b = (1.0 - std::sqrt(0.2)) / 4.0;
        points[0] = (Vector3<T>() << a, b, b).finished();
        points[1] = (Vector3<T>() << b, a, b).finished();
        points[2] = (Vector3<T>() << b, b, a).finished();
        points[3] = (Vector3<T>() << b, b, b).finished();
        weights.resize(4, 1.0 / 24.0);
      } else if constexpr (N == 3) {
        // quadrature point location,  weight/area
        //  (1/4, 1/4, 1/4)               -4/5
        //  (a, b, b)                      9/20
        //  (b, a, b)                      9/20
        //  (b, b, a)                      9/20
        //  (b, b, b)                      9/20
        // where a = 1/2, b = 1/6.
        points.resize(5);
        T a = 0.5;
        T b = 1.0 / 6.0;
        points[0] = (Vector3<T>() << 0.25, 0.25, 0.25).finished();
        points[1] = (Vector3<T>() << a, b, b).finished();
        points[2] = (Vector3<T>() << b, a, b).finished();
        points[3] = (Vector3<T>() << b, b, a).finished();
        points[4] = (Vector3<T>() << b, b, b).finished();

        weights.resize(5);
        weights[0] = -2.0 / 15.0;
        for (int i = 1; i < 5; ++i) {
          weights[i] = 3.0 / 40.0;
        }
      } else {
        DRAKE_UNREACHABLE();
      }
    } else {
      DRAKE_UNREACHABLE();
    }
    this->set_points(points);
    this->set_weights(weights);
  }
};
}  // namespace fem
}  // namespace drake
