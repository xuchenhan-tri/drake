#pragma once

#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/dev/quadrature.h"

namespace drake {
namespace fem {

/** IsoparametricElement is a class that helps evaluate shape functions
    and their derivatives at quadrature locations. The shape function
    `S` located at a vertex `a` maps parent domain to a scalar. The reference
    position `X` as well as `u`, the function we are solving for, are
    interpolated from the shape function, i,e:

        X(ξ) = Sₐ(ξ)Xₐ, and
        u(ξ) = Sₐ(ξ)uₐ,

    where ξ ∈ ℝᵈ and d is the natural dimension (dimension of the parent
    domain), which may be different from the dimensions of X and u (e.g. 2D
    membrane or shell element in 3D dynamics simulation).
    @tparam T The scalar type for ξ, X and u.
    @tparam NaturalDim The dimension of the parent domain.
*/
template <typename T, int NaturalDim>
class IsoparametricElement {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(IsoparametricElement);
  /** Constructs an isoparametric element that performs calculations at the
      quadrature points specified by the input `quadrature`.
  */
  explicit IsoparametricElement(const Quadrature<T, NaturalDim>& quadrature)
      : quadrature_(quadrature) {}
  virtual ~IsoparametricElement() = default;

  /** The number of nodes in the element. E.g. 3 for linear triangles, 9 for
      quadratic quadrilaterals and 4 for linear tetrahedrons.
  */
  virtual int num_nodes() const = 0;

  /** The number of quadrature points in the element. */
  int num_quads() const { return quadrature_.num_points(); }

  const Quadrature<T, NaturalDim>& get_quadrature() const {
    return quadrature_;
  }

  /** Computes the shape function vector
         S(ξ) = [S₀(ξ); S₁(ξ); ... Sₐ(ξ); ... Sₙ₋₁(ξ)]
      at each quadrature point specified by `quadrature`. The a-th
      component of S(ξ) corresponds to the shape function Sₐ(ξ) for node a.
      @returns S Vector of size equal to the number of quadrature points in
      `quadrature_`. The q-th entry of the output contains vector S(ξ), of size
      num_nodes(), evaluated at the q-th quadrature point.
  */
  virtual const std::vector<VectorX<T>>& CalcShapeFunctions() const = 0;

  /** Computes dS/dξ, a matrix of size num_nodes() by NaturalDim evaluated at
      each quadrature point specified by `quadrature_`.
      As defined in `CalcShapeFunctions()`, S(ξ) is a vector of size
      num_nodes(). Therefore it's gradient in the natural coordinates is a
      matrix of size num_nodes() by NaturalDim.
      @returns dSdxi The gradient of the shape function with respect to the
      parent coordinates evaluated at each quadrature point. dSdxi is a vector
      of size equal to num_quads(). The q-th entry contains dS/dxi evaluated at
      the q-th quadrature point.
  */
  virtual const std::vector<MatrixX<T>>& CalcGradientInParentCoordinates()
      const = 0;

  /** Computes dx/dξ, a matrix of size nsd by NaturalDim at each quadrature
      point specified by `quadrature_`, where nsd is the "number of spatial
      dimensions".
      @param xa Spatial coordinates for each element node. xa should be a
      matrix of size nsd x num_nodes(),
      @returns jacobian The Jacobian of the spatial coordinates with respect to
      parent coordinates evaluated at each quadrature point. 'jacobian' is
      represented by a vector of size equal to num_quads(). The q-th entry
      contains the dx/dxi evaluated at the q-th quadrature point.
   */
  std::vector<MatrixX<T>> CalcElementJacobian(
      const Eigen::Ref<const MatrixX<T>>& xa) const;

  /** Computes dξ/dx, a matrix of size NaturalDim by nsd, at each quadrature
      point specified by `quadrature_`. As defined in `CalcElementJacobian()`,
      nsd is the "number of spatial dimensions.
      @param xa Spatial coordinates for each element node. xa should be a matrix
      of size nsd x num_nodes().
      @returns The gradient of the shape function with respect to the spatial
      coordinates evaluated at each quadrature point represented by a vector of
      size equal to num_quad(). The q-th entry contains
      dξ/dx evaluated at the q-th quadrature point.
  */
  std::vector<MatrixX<T>> CalcElementJacobianInverse(
      const Eigen::Ref<const MatrixX<T>>& xa) const;

  /** Alternative signature for computing dξ/dx when the element Jacobians are
      available.
      @param jacobian A vector of size num_quads() with the q-th entry
      containing the element Jacobian evaluated at the q-th quadrature point.
   */
  std::vector<MatrixX<T>> CalcElementJacobianInverse(
      const std::vector<MatrixX<T>>& jacobian) const;

  /** Interpolates scalar nodal values ua into u(ξ) at each quadrature point.
      @param ua The value of function u at each element node. `ua` must be a
      vector of size num_nodes(),
  */
  std::vector<T> InterpolateScalar(
      const Eigen::Ref<const VectorX<T>>& ua) const;

  /** Interpolates vector nodal values ua into u(ξ) at each quadrature point.
      @param ua The value of function u at each element node. `ua` must be a
      matrix of size nsd-by-num_nodes(), where nsd is the number of spatial
      dimensions.
  */
  std::vector<VectorX<T>> InterpolateVector(
      const Eigen::Ref<const MatrixX<T>>& ua) const;

 private:
  // The quadrature at which to evaluate various quantities of this element.
  Quadrature<T, NaturalDim> quadrature_;
};
}  // namespace fem
}  // namespace drake
