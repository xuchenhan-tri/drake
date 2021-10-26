#pragma once

#include <chrono>

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

class Timer {
 public:
  Timer() { Reset(); }

  // Resets the timer.
  void Reset() { start_ = clock::now(); }

  // Returns the elapsed time, in seconds, since the last
  // call to Reset().
  double Elapsed() {
    clock::time_point end = clock::now();
    return std::chrono::duration<double>(end - start_).count();
  }

 private:
  using clock = std::chrono::steady_clock;
  clock::time_point start_;
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
