#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "drake/fem/quadrature.h"

namespace drake {
namespace fem {
namespace {

using Eigen::Vector2d;
using Eigen::Vector3d;
class LinearTestFunction2D {
 public:
  static constexpr double a0 = 1.0;
  static constexpr double a1 = 3.0;
  static constexpr double a2 = -2.0;
  // Returns the value of function
  //     y(x₀, x₁) = a₀ + a₁ * x₀+ a₂ * x₁
  // evaluated at input x.
  static double Eval(const Vector2d& x) { return a0 + a1 * x(0) + a2 * x(1); }
};

class QuadraticTestFunction2D {
 public:
  static constexpr double a0 = 1.0;
  static constexpr double a1 = 3.0;
  static constexpr double a2 = -2.0;
  static constexpr double a3 = 7.2;
  static constexpr double a4 = -3.15;
  static constexpr double a5 = -0.82;
  // Returns the value of function
  //     y(x₀, x₁) = a₀ + a₁*x₀ + a₂*x₁ + a₃*x₀*x₁ + a₄*x₀² + a₅*x₁²
  // evaluated at input x.
  static double Eval(const Vector2d& x) {
    return a0 + a1 * x(0) + a2 * x(1) + a3 * x(0) * x(1) + a4 * x(0) * x(0) +
           a5 * x(1) * x(1);
  }
};

class LinearTestFunction3D {
 public:
  static constexpr double a0 = 1.0;
  static constexpr double a1 = 3.0;
  static constexpr double a2 = -2.0;
  static constexpr double a3 = -4.0;
  // Returns the value of function
  //     y(x₀, x₁, x₂) = a₀ + a₁ * x₀+ a₂ * x₁+ a₃ * x₂
  // evaluated at input x.
  static double Eval(const Vector3d& x) {
    return a0 + a1 * x(0) + a2 * x(1) + a3 * x(2);
  }
};

class QuadraticTestFunction3D {
 public:
  static constexpr double a0 = 1.0;
  static constexpr double a1 = 3.0;
  static constexpr double a2 = -2.0;
  static constexpr double a3 = -4.0;
  static constexpr double a4 = -2.7;
  static constexpr double a5 = -1.3;
  static constexpr double a6 = 7.0;
  static constexpr double a7 = -3.6;
  static constexpr double a8 = -2.1;
  static constexpr double a9 = 5.3;
  // Returns the value of function
  //     y(x₀, x₁, x₂) = a₀ + a₁ * x₀+ a₂ * x₁+ a₃ * x₂ + a₄ * x₀*x₁ + a₅ *
  //     x₀*x₂ + a₆ * x₁*x₂ + a₇*x₀²  + a₈*x₁² + a₉*x₂².
  // evaluated at input x.
  static double Eval(const Vector3d& x) {
    return a0 + a1 * x(0) + a2 * x(1) + a3 * x(2) + a4 * x(0) * x(1) +
           a5 * x(0) * x(2) + a6 * x(1) * x(2) + a7 * x(0) * x(0) +
           a8 * x(1) * x(1) + a9 * x(2) * x(2);
  }
};

class SimplexGaussianQuadratureTest : public ::testing::Test {
  void SetUp() override {
    linear_2d_quadrature_ =
        std::make_unique<SimplexGaussianQuadrature<double, 1, 2>>();
    quadratic_2d_quadrature_ =
        std::make_unique<SimplexGaussianQuadrature<double, 2, 2>>();
    linear_3d_quadrature_ =
        std::make_unique<SimplexGaussianQuadrature<double, 1, 3>>();
    quadratic_3d_quadrature_ =
        std::make_unique<SimplexGaussianQuadrature<double, 2, 3>>();
  }

 protected:
  std::unique_ptr<SimplexGaussianQuadrature<double, 1, 2>>
      linear_2d_quadrature_;
  std::unique_ptr<SimplexGaussianQuadrature<double, 2, 2>>
      quadratic_2d_quadrature_;
  std::unique_ptr<SimplexGaussianQuadrature<double, 1, 3>>
      linear_3d_quadrature_;
  std::unique_ptr<SimplexGaussianQuadrature<double, 2, 3>>
      quadratic_3d_quadrature_;
};

TEST_F(SimplexGaussianQuadratureTest, Linear2D) {
  // Linear Guassuan quadrature only needs 1 quadrature point.
  EXPECT_EQ(linear_2d_quadrature_->num_points(), 1);
  // Numerical integral of f = ∑ᵢ wᵢ f(xᵢ) where wᵢ is the weight of the i-th
  // quadrature point and xᵢ is the location of the i-th quadrature point.
  double numerical_integral =
      linear_2d_quadrature_->get_weight(0) *
      LinearTestFunction2D::Eval(linear_2d_quadrature_->get_point(0));
  // The integral of the monomial 1 and x over the unit triangle with end points
  // at (0,0), (1,0) and (0,1) is
  //     ∫₀¹∫₀¹⁻ˣ 1 dydx =  1/2.
  //     ∫₀¹∫₀¹⁻ʸ x dxdy =  1/6.
  // So the integral of f(x) = a₀ + a₁ * x₀ + a₂ * x₁ is equal to
  //     1/2 * a₀ + 1/6 * a₁ + 1/6 * a₂.
  double analytic_integral = 0.5 * LinearTestFunction2D::a0 +
                             1.0 / 6.0 * LinearTestFunction2D::a1 +
                             1.0 / 6.0 * LinearTestFunction2D::a2;
  EXPECT_NEAR(analytic_integral, numerical_integral,
              std::numeric_limits<double>::epsilon());
}

TEST_F(SimplexGaussianQuadratureTest, Linear3D) {
  // Linear Guassuan quadrature only needs 1 quadrature point.
  EXPECT_EQ(linear_3d_quadrature_->num_points(), 1);
  // Numerical integral of f = ∑ᵢ wᵢ f(xᵢ) where wᵢ is the weight of the i-th
  // quadrature point and xᵢ is the location of the i-th quadrature point.
  double numerical_integral =
      linear_3d_quadrature_->get_weight(0) *
      LinearTestFunction3D::Eval(linear_3d_quadrature_->get_point(0));
  // The integral of the monomial 1 and x over the unit tetrahedron with end
  // points at (0,0,0), (1,0,0), (0,1,0) and (0,0,1) is
  //     ∫₀¹∫₀¹⁻ˣ∫₀¹⁻ˣ⁻ʸ 1 dzdydx =  1/6.
  //     ∫₀¹∫₀¹⁻ˣ∫₀¹⁻ˣ⁻ʸ x dzdydx =  1/24.
  // So the integral of f(x) = a₀ + a₁*x₀ + a₂*x₁ + a₃*x₂ is equal to
  //     1/6 * a₀ + 1/24 * a₁ + 1/24 * a₂ + 1/24 * a₃.
  double analytic_integral = 1.0 / 6.0 * LinearTestFunction3D::a0 +
                             1.0 / 24.0 * LinearTestFunction3D::a1 +
                             1.0 / 24.0 * LinearTestFunction3D::a2 +
                             1.0 / 24.0 * LinearTestFunction3D::a3;
  EXPECT_NEAR(analytic_integral, numerical_integral,
              std::numeric_limits<double>::epsilon());
}

TEST_F(SimplexGaussianQuadratureTest, Quadratic2D) {
  // Quadratic Guassuan quadrature needs 3 quadrature point.
  EXPECT_EQ(quadratic_2d_quadrature_->num_points(), 3);
  // Numerical integral of f = ∑ᵢ wᵢ f(xᵢ) where wᵢ is the weight of the i-th
  // quadrature point and xᵢ is the location of the i-th quadrature point.
  double numerical_integral = 0;
  for (int i = 0; i < quadratic_2d_quadrature_->num_points(); ++i) {
    numerical_integral +=
        quadratic_2d_quadrature_->get_weight(i) *
        QuadraticTestFunction2D::Eval(quadratic_2d_quadrature_->get_point(i));
  }
  // The integral of the monomial 1, x, xy and x² over the unit triangle with
  // end points at (0,0), (1,0) and (0,1) are
  //     ∫₀¹∫₀¹⁻ˣ 1 dydx =  1/2.
  //     ∫₀¹∫₀¹⁻ˣ x dydx =  1/6.
  //     ∫₀¹∫₀¹⁻ʸ xy dxdy =  1/24.
  //     ∫₀¹∫₀¹⁻ʸ x² dxdy =  1/12.
  // So the integral of  f(x₀, x₁) = a₀ + a₁*x₀ + a₂*x₁ + a₃*x₀*x₁ + a₄*x₀² +
  // a₅*x₁² is equal to
  //     1/2 * a₀ + 1/6 * a₁ + 1/6 * a₂ + 1/24 * a₃ + 1/12 * a₄ + 1/12 * a₅.
  double analytic_integral = 0.5 * QuadraticTestFunction2D::a0 +
                             1.0 / 6.0 * QuadraticTestFunction2D::a1 +
                             1.0 / 6.0 * QuadraticTestFunction2D::a2 +
                             1.0 / 24.0 * QuadraticTestFunction2D::a3 +
                             1.0 / 12.0 * QuadraticTestFunction2D::a4 +
                             1.0 / 12.0 * QuadraticTestFunction2D::a5;
  EXPECT_NEAR(analytic_integral, numerical_integral,
              std::numeric_limits<double>::epsilon());
}

TEST_F(SimplexGaussianQuadratureTest, Quadratic3D) {
  // Quadratic Guassuan quadrature needs 3 quadrature point.
  EXPECT_EQ(quadratic_3d_quadrature_->num_points(), 4);
  // Numerical integral of f = ∑ᵢ wᵢ f(xᵢ) where wᵢ is the weight of the i-th
  // quadrature point and xᵢ is the location of the i-th quadrature point.
  double numerical_integral = 0;
  for (int i = 0; i < quadratic_3d_quadrature_->num_points(); ++i) {
    numerical_integral +=
        quadratic_3d_quadrature_->get_weight(i) *
        QuadraticTestFunction3D::Eval(quadratic_3d_quadrature_->get_point(i));
  }
  // The integral of the monomial 1, x, xy, and  x² over the unit tetrahedron
  // with end points at (0,0,0), (1,0,0), (0,1,0) and (0,0,1) is
  //     ∫₀¹∫₀¹⁻ˣ∫₀¹⁻ˣ⁻ʸ 1 dzdydx =  1/6.
  //     ∫₀¹∫₀¹⁻ˣ∫₀¹⁻ˣ⁻ʸ x dzdydx =  1/24.
  //     ∫₀¹∫₀¹⁻ˣ∫₀¹⁻ˣ⁻ʸ xy dzdydx =  1/120.
  //     ∫₀¹∫₀¹⁻ˣ∫₀¹⁻ˣ⁻ʸ x² dzdydx =  1/60.
  // So the integral of  f(x) = a₀ + a₁ * x₀+ a₂ * x₁+ a₃ * x₂  + a₄ * x₀*x₁ +
  // a₅ * x₀*x₂ + a₆ * x₁*x₂ + a₇*x₀²  + a₈*x₁² + a₉*x₂² is equal to
  //     1/6 * a₀ + 1/24 * a₁ + 1/24 * a₂ + 1/24 * a₃ + 1/120 * a₄ + 1/120 * a₅
  //     + 1/120 * a₆ + 1/60 * a₇ + 1/60 * a₈ + 1/60 * a₉
  double analytic_integral = 1.0 / 6.0 * QuadraticTestFunction3D::a0 +
                             1.0 / 24.0 * QuadraticTestFunction3D::a1 +
                             1.0 / 24.0 * QuadraticTestFunction3D::a2 +
                             1.0 / 24.0 * QuadraticTestFunction3D::a3 +
                             1.0 / 120.0 * QuadraticTestFunction3D::a4 +
                             1.0 / 120.0 * QuadraticTestFunction3D::a5 +
                             1.0 / 120.0 * QuadraticTestFunction3D::a6 +
                             1.0 / 60.0 * QuadraticTestFunction3D::a7 +
                             1.0 / 60.0 * QuadraticTestFunction3D::a8 +
                             1.0 / 60.0 * QuadraticTestFunction3D::a9;
  EXPECT_NEAR(analytic_integral, numerical_integral,
              std::numeric_limits<double>::epsilon());
}

// TODO(xuchenhan-tri): Add unit tests for cubic rules.

TEST_F(SimplexGaussianQuadratureTest, UnsupportedDimension) {
  EXPECT_THROW((std::make_unique<SimplexGaussianQuadrature<double, 1, 1>>()),
               std::runtime_error);
  EXPECT_THROW((std::make_unique<SimplexGaussianQuadrature<double, 1, 4>>()),
               std::runtime_error);
  EXPECT_THROW((std::make_unique<SimplexGaussianQuadrature<double, 2, 1>>()),
               std::runtime_error);
  EXPECT_THROW((std::make_unique<SimplexGaussianQuadrature<double, 2, 4>>()),
               std::runtime_error);
  EXPECT_THROW((std::make_unique<SimplexGaussianQuadrature<double, 3, 1>>()),
               std::runtime_error);
  EXPECT_THROW((std::make_unique<SimplexGaussianQuadrature<double, 3, 4>>()),
               std::runtime_error);
}

TEST_F(SimplexGaussianQuadratureTest, UnsupportedOrder) {
  EXPECT_THROW((std::make_unique<SimplexGaussianQuadrature<double, 0, 2>>()),
               std::runtime_error);
  EXPECT_THROW((std::make_unique<SimplexGaussianQuadrature<double, 4, 2>>()),
               std::runtime_error);
  EXPECT_THROW((std::make_unique<SimplexGaussianQuadrature<double, 0, 3>>()),
               std::runtime_error);
  EXPECT_THROW((std::make_unique<SimplexGaussianQuadrature<double, 4, 3>>()),
               std::runtime_error);
}
}  // namespace
}  // namespace fem
}  // namespace drake
