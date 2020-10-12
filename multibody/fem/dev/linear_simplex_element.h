#pragma once

#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/dev/isoparametric_element.h"

namespace drake {
namespace fem {
template <typename T, int NaturalDim>
class LinearSimplexElement : public IsoparametricElement<T, NaturalDim> {
 public:
  explicit LinearSimplexElement(const Quadrature<T, NaturalDim>& quadrature)
      : IsoparametricElement<T, NaturalDim>(quadrature) {
    S_ = CalcShapeFunctionsHelper(quadrature);
    dSdxi_ = CalcGradientInParentCoordinatesHelper();
  }

  int num_nodes() const final { return NaturalDim + 1; }

  const std::vector<VectorX<T>>& CalcShapeFunctions() const { return S_; }

  const std::vector<MatrixX<T>>& CalcGradientInParentCoordinates() const {
    return dSdxi_;
  }

 private:
  std::vector<VectorX<T>> CalcShapeFunctionsHelper(
      const Quadrature<T, NaturalDim>& quadrature) const;

  std::vector<MatrixX<T>> CalcGradientInParentCoordinatesHelper() const;

  // Shape functions evaluated at quadrature points specified by quadrature_.
  std::vector<VectorX<T>> S_;
  // Shape function derivatives evaluated at quadrature points specified by
  // quadrature_.
  std::vector<MatrixX<T>> dSdxi_;
};
}  // namespace fem
}  // namespace drake
