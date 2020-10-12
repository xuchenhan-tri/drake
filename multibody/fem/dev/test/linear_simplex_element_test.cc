#include "drake/multibody/fem/dev/linear_simplex_element.h"

#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "drake/multibody/fem/dev/quadrature.h"

namespace drake {
namespace fem {
namespace {

class LinearSimplexElementTest : public ::testing::Test {
 protected:
  void SetUp() override {
    SimplexGaussianQuadrature<double, 1, 2> tri_quadrature;
    SimplexGaussianQuadrature<double, 1, 3> tet_quadrature;
    tri_ = std::make_unique<LinearSimplexElement<double, 2>>(tri_quadrature);
    tet_ = std::make_unique<LinearSimplexElement<double, 3>>(tet_quadrature);
  }

  std::unique_ptr<LinearSimplexElement<double, 2>> tri_;
  std::unique_ptr<LinearSimplexElement<double, 3>> tet_;
};

TEST_F(LinearSimplexElementTest, ShapeFunction2D) {
  const auto& S = tri_->CalcShapeFunctions();
  // The first-order quadrature rule has one quadrature point.
  EXPECT_EQ(S.size(), 1);
  // There are three vertices in a triangle.
  EXPECT_EQ(S[0].size(), 3);
  // The shape function for the 3 nodes of the triangle are 1-x-y, x and y.
  // The only quadrature point is located at (1/3, 1/3).
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(S[0](i), 1.0 / 3.0, std::numeric_limits<double>::epsilon());
  }
}

TEST_F(LinearSimplexElementTest, ShapeFunction3D) {
  const auto& S = tet_->CalcShapeFunctions();
  // The first-order quadrature rule has one quadrature point.
  EXPECT_EQ(S.size(), 1);
  // There are four vertices in a tetrahedron.
  EXPECT_EQ(S[0].size(), 4);
  // The shape function for the 4 nodes of the tetrahedron are
  // 1-x-y-z, x, y and z. The only quadrature point is located at
  // (1/4, 1/4, 1/4).
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(S[0](i), 0.25, std::numeric_limits<double>::epsilon());
  }
}

TEST_F(LinearSimplexElementTest, ShapeFunctionDerivative2D) {
  const auto& S = tri_->CalcGradientInParentCoordinates();
  // The first-order quadrature rule has one quadrature point.
  EXPECT_EQ(S.size(), 1);
  // There are three vertices in a triangle.
  EXPECT_EQ(S[0].rows(), 3);
  // The dimension of the parent domain is two.
  EXPECT_EQ(S[0].cols(), 2);
  // The shape function for the 3 nodes of the triangle are 1-x-y, x and y. So
  // the derivatives with respect to x is [-1, 1, 0], and
  // the derivatives with respect to y is [-1, 0, 1].
  VectorX<double> x_deriv(3);
  x_deriv << -1, 1, 0;
  VectorX<double> y_deriv(3);
  y_deriv << -1, 0, 1;
  EXPECT_EQ(S[0].col(0), x_deriv);
  EXPECT_EQ(S[0].col(1), y_deriv);
}

TEST_F(LinearSimplexElementTest, ShapeFunctionDerivative3D) {
  const auto& S = tet_->CalcGradientInParentCoordinates();
  // The first-order quadrature rule has one quadrature point.
  EXPECT_EQ(S.size(), 1);
  // There are four vertices in a tetrahedron.
  EXPECT_EQ(S[0].rows(), 4);
  // The dimension of the parent domain is three.
  EXPECT_EQ(S[0].cols(), 3);
  // The shape function for the 3 nodes of the triangle are 1-x-y-z, x, y and z.
  // So the derivatives with respect to x is [-1, 1, 0, 0],
  // the derivative with respect to y is [-1, 0, 1, 0],
  // and the derivative with respect to z is [-1, 0, 0, 1].
  VectorX<double> x_deriv(4);
  x_deriv << -1, 1, 0, 0;
  VectorX<double> y_deriv(4);
  y_deriv << -1, 0, 1, 0;
  VectorX<double> z_deriv(4);
  z_deriv << -1, 0, 0, 1;
  EXPECT_EQ(S[0].col(0), x_deriv);
  EXPECT_EQ(S[0].col(1), y_deriv);
  EXPECT_EQ(S[0].col(2), z_deriv);
}

}  // namespace
}  // namespace fem
}  // namespace drake
