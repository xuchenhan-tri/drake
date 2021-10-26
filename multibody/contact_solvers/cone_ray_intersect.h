#pragma once

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

constexpr double kEpsilonSoftNorm = 1.0e-7;

// Computes s(g) = μgₙ − ‖gₜ‖ₛ + ε = 0
template <typename T, typename g_t>
T SoftConeDistance(const T& mu, const T& epsilon_soft,
                   const Eigen::MatrixBase<g_t>& g) {
  static_assert(is_eigen_vector_of<g_t, T>::value,
                "g must be templated on the same scalar type of mu.");
  static_assert(g_t::SizeAtCompileTime == 3,
                "g must be a fixed size vector of size 3.");
  using std::sqrt;
  const T& gn = g(2);
  const Vector2<T> gt = g.template head<2>();
  const T gr_squared = gt.squaredNorm();

  const T gr_soft = sqrt(gr_squared + epsilon_soft * epsilon_soft);

  // N.B. Notice the "soft" definition of s(gamma) so that gradient and
  // Hessian are defined everywhere including gamma = 0.
  // N.B. Notice we add epsilon_soft so that s > 0 enforces gn > 0.
  // N.B. The region s >= 0 IS convex, even close to zero.
  return mu * gn - gr_soft + epsilon_soft;
}

// This method searches for the intersection of the ray
//   g(α) = g + α dg
// with the "soft" cone defined by:
//   s(α) = μgₙ(α) − ‖gₜ(α)‖ₛ + ε = 0
// where gₙ and gₜ are the (scalar) normal component and is the (vector ℝ²)
// tangential componets of g, respectively. We define the "soft" norm as
//   ‖v‖ₛ = sqrt(‖v‖²+ε²)
// The soft cone has the property that derivatives of s(g) with respect to g
// are well defined even at g = 0.
// Appends intersections into alpha_set. There might be zero, one or two
// intersections between a ray and a cone. Only alpha in ℝ₊₊ are reported.
// @param[out] a1 If there is at least one intersection, the value α₁ of the
// first root, otherwise it is left unmodified.
// @param[out] a2 If there are two intersections, the value α₂ of the
// second root, otherwise it is left unmodified. This method guarantees α₁ < α₂.
// @returns the number of intersections.
template <typename T, typename g_t, typename dg_t>
int CalcRayConeIntersection(const T& mu, const T& epsilon_soft,
                            const Eigen::MatrixBase<g_t>& g,
                            const Eigen::MatrixBase<dg_t>& dg,
                            Vector2<T>* intersections) {
  static_assert(is_eigen_vector_of<g_t, T>::value,
                "g must be templated on the same scalar type of mu.");
  static_assert(is_eigen_vector_of<dg_t, T>::value,
                "dg must be templated on the same scalar type of mu.");
  static_assert(g_t::SizeAtCompileTime == 3,
                "g must be a fixed size vector of size 3.");
  static_assert(dg_t::SizeAtCompileTime == 3,
                "dg must be a fixed size vector of size 3.");
  DRAKE_DEMAND(intersections != nullptr);

  using std::abs;
  using std::max;
  using std::sqrt;
  using std::swap;

  // This method finds alpha values in (0, +infty) where the ray defined by
  // r(alpha) = g + alpha * dg intersects the walls of the friction cone defined
  // by friction coefficient mu.

  const Vector3<T> gp = g + dg;

  // We use this as a scale to normalize all coefficients in the resulting
  // quadrating to avoid round-off errors.
  const T scale = max(g.norm(), gp.norm());

  if (scale < std::numeric_limits<double>::epsilon())
    return 0;  // we report no intersections.

  // We define the shortnames:
  // n = gₙ/mu
  // t = gₜ
  // Therefore we are looking for the zero of the "smooth" cone:
  //   s = n - ‖t‖ₛ + ε
  // using the smooth norm ‖t‖ₛ = sqrt(‖t‖² + ε²).
  // Along the direction dg, we have:
  //   n(α) = n + α dn
  //   t(α) = t + α dt
  // Then we find the roots of:
  //   n(α) + ε = ‖t(α)‖ₛ
  // We square it to get rid of the square root in the norm of t:
  //   (n(α) + ε)² = ‖t(α)‖ₛ² = ‖t(α)‖² + ε²
  // and rearrange to get a quadratic equation in α:
  //   aα²+bα+c=0
  // with:
  //   a = dn²−dt²
  //   b = 2(ndn+εdn−t⋅dt)
  //   c = n²-t²+2εn

  // Handy short names used in the derivation above. Scaled to avoid round-off
  // errors.
  const T n = mu * g(2) / scale;
  const T dn = mu * dg(2) / scale;
  const Vector2<T> t = g.template head<2>() / scale;
  const Vector2<T> dt = dg.template head<2>() / scale;
  const double e = epsilon_soft / scale;

  // Helper to append only roots that are on the positive n side.
  // Returns the number of appended roots, either 0 or 1.
  auto is_valid_root = [&n, &dn](const T& alpha) {
    if (alpha > 0 && n + alpha * dn > 0.0) {
      return true;
    }
    return false;
  };

  // Compute quadratic coefficients, they are all scaled with scale.
  const T a = dn * dn - dt.squaredNorm();
  const T b = 2.0 * (n * dn + e * dn - t.dot(dt));
  const T c = n * n - t.squaredNorm() + 2.0 * e * n;

  // Quadratic's discriminant.
  const T Delta = b * b - 4 * a * c;

  // Cases:
  // N.B. We squared the original equations and therefore we are essentially
  // "mirroring" the cone and solutions with dn < 0 are included. Therefore we
  // must double check that n(α) >=0 (or that s(α) = 0).
  //
  // 1. Delta < 0 ==> no intersection.
  // N.B. From now on we assume Delta >= 0.
  // 2. a != 0 ==> At most two roots. We pick the positive ones.
  // 3. a = 0, b != 0 ==> Ray tangent to the cone's wall.
  //    i) n(α) >= ==> one valid root.
  //    ii) n(α) < 0 ==> ray intersects mirror image.
  //    N.B. You can verify this without computing the error prone quadratic by
  //    verifying that dg is parallel to the cone.
  // 4. a = 0, b = 0:
  //    i) c != 0 ==> no intersections.
  //    ii) c == 0 ==> g, dg are zero and we couldn't even get here.
  // N.B. You can verify this without computing the error prone quadratic by
  //    verifying that s(g)=0 and s(gp)=0.

  // Case 1. Ray completely outside the cone.
  if (Delta < 0.0) return 0;

  // Case 2, a != 0.
  if (abs(a) > std::numeric_limits<double>::epsilon()) {
    const T sqrt_Delta = sqrt(Delta);

    // To avoid loss of significance, when 4ac is relatively small compared
    // to b² (i.e. the square root of the discriminant is close to b), we use
    // Vieta's formula (α₁α₂ = c / a) to compute the second root given we
    // computed the first root without precision lost. This guarantees the
    // numerical stability of the method.
    const T numerator = -0.5 * (b + (b > 0.0 ? sqrt_Delta : -sqrt_Delta));
    T a1 = numerator / a;
    // When b = sqrt_Delta = 0, we'd have a division by zero.
    T a2 = abs(numerator) > std::numeric_limits<double>::epsilon()
               ? c / numerator
               : -1.0;  // Invalid root, it won't get reported.

    // If both roots are valid, we want to report them in order.
    if (a1 > a2) swap(a1, a2);

    // We add root if we are in the positive cone.
    int num_roots = 0;
    if (is_valid_root(a1)) {
      (*intersections)[num_roots++] = a1;
    }

    // If both roots are equal (to machine precision) we are done and return (we
    // do not report repeated roots.)
    if (abs(a1 - a2) < std::numeric_limits<double>::epsilon()) return num_roots;

    // At this point a2 != a1 and we only add it to the set if on the positive
    // side of the cone.
    if (is_valid_root(a2)) {
      (*intersections)[num_roots++] = a2;
    }
    return num_roots;
  }

  // Case 3, a = 0, b != 0 ==> Ray tangent to the cone's wall.
  // 3. a = 0, b != 0 ==> Ray tangent to the cone's wall.
  //    i) n(α) >= ==> one valid root.
  //    ii) n(α) < 0 ==> ray intersects mirror image.
  // If we are here, then abs(a) is zero to machine precision. Therefore we
  // solve for the single root:
  if (abs(b) > std::numeric_limits<double>::epsilon()) {
    const T a1 = -c / b;
    if (is_valid_root(a1)) {
      (*intersections)[0] = a1;
      return 1;
    }
  }

  // Case 4. If we are here, then a = b = 0.
  // If c = 0, the entire ray lies on the wall of the cone. For our purposes,
  // this is not considered and intersection and there are no roots to report.
  // If c > 0, then the ray definitely does not intersect the cone.
  return 0;
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake