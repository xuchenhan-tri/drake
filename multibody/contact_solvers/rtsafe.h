#pragma once

#include <functional>

#include "drake/common/drake_assert.h"
#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

// Exactly copied from NR.
template <typename T>
T RtSafe(const std::function<T(const T&, std::optional<T*>)>& function, T xl,
         T xh, double xacc, const T& x_guess, bool check_interval,
         int* num_iters = nullptr) {
  using std::abs;
  using std::max;
  using std::min;
  using std::swap;
  constexpr int kMaxIterations = 100;
  DRAKE_THROW_UNLESS(xacc > 0);

  if (check_interval) {
    DRAKE_THROW_UNLESS(xh > xl);
    DRAKE_THROW_UNLESS(xl <= x_guess && x_guess <= xh);

    const T fl = function(xl, std::nullopt);
    const T fh = function(xh, std::nullopt);

    DRAKE_THROW_UNLESS(fl * fh <= 0);

    if (fl == 0) return xl;
    if (fh == 0) return xh;

    // Re-orient the search so that f(xl) < 0.
    if (fl > 0) swap(xl, xh);
  }

  T rts = x_guess;
  T dxold = (xh - xl);
  T dx = dxold;
  T df;
  T f = function(rts, &df);
  for (int k = 1; k <= kMaxIterations; ++k) {
    // If the function is already too small, return.
    //if (abs(f) < 10 * std::numeric_limits<double>::epsilon()) {
    //  if (num_iters) *num_iters = k;
    //  return rts;
    //}

    if (((rts - xh) * df - f) * ((rts - xl) * df - f) > 0.0 ||
        abs(2.0 * f) > abs(dxold * df)) {
      dxold = dx;
      dx = 0.5 * (xh - xl);
      rts = xl + dx;
      DRAKE_LOGGER_DEBUG(
          "Bisect. k = {:d}. x = {:12.6g}. [xl, xh] = [{:12.8g}, {:12.8g}].", k,
          rts, xl, xh);
      if (xl == rts) {
        if (num_iters) *num_iters = k;
        return rts;
      }
    } else {
      dxold = dx;
      dx = f / df;
      T temp = rts;
      rts -= dx;
      DRAKE_LOGGER_DEBUG(
          "Newton. k = {:d}. x = {:12.6g}. [xl, xh] = [{:12.8g}, {:12.8g}].", k,
          rts, xl, xh);
      if (temp == rts) {
        if (num_iters) *num_iters = k;
        return rts;
      }
    }
    if (abs(dx) < xacc) {
      if (num_iters) *num_iters = k;
      return rts;
    }
    f = function(rts, &df);
    if (f < 0.0)
      xl = rts;
    else
      xh = rts;
  }

  // If here, then RtSafe did not converge.
  throw std::runtime_error(
      fmt::format("RtSafe did not converge.\n"
                  "|x - x_prev| = {}. |xh-xl| = {}",
                  abs(dx), abs(xh - xl)));
};

template <typename T>
T RtSafe(const std::function<T(const T&, std::optional<T*>)>& function, T xl,
         T xh, double xacc, int* num_iters = nullptr) {
  const T x_guess = 0.5 * (xl + xh);
  const bool check_bracket = true;
  return RtSafe<T>(function, xl, xh, xacc, x_guess, check_bracket, num_iters);
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
