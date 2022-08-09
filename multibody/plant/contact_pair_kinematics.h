#pragma once

#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace multibody {
namespace internal {

template <typename T>
struct ContactPairKinematics {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ContactPairKinematics);

  // Struct to store the block contribution from a given clique to the contact
  // Jacobian for a contact pair.
  struct JacobianCliqueBlock {
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(JacobianCliqueBlock);

    JacobianCliqueBlock(int clique_in, Matrix3X<T> J_in)
        : clique(clique_in), J(std::move(J_in)) {}

    // Index of the clique for this block.
    int clique;

    // J.cols() must equal the number of generalized velocities for
    // the corresponding clique.
    Matrix3X<T> J;
  };

  ContactPairKinematics(T phi_in, std::vector<JacobianCliqueBlock> jacobian_in,
                        math::RotationMatrix<T> R_WC_in)
      : phi(std::move(phi_in)),
        jacobian(std::move(jacobian_in)),
        R_WC(std::move(R_WC_in)) {}

  // Signed distance for the given pair. Defined negative for overlapping
  // bodies.
  T phi{};

  // TODO(amcastro-tri): consider using absl::InlinedVector since here we know
  // this has a size of at most 2.
  // Jacobian for a discrete contact pair stored as individual blocks for each
  // of the cliques participating in the contact. Only one or two cliques can
  // participate in a given contact.
  std::vector<JacobianCliqueBlock> jacobian;

  // Rotation matrix to re-express between contact frame C and world frame W.
  math::RotationMatrix<T> R_WC;
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake
