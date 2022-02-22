#pragma once

#include "drake/common/identifier.h"
#include "drake/common/type_safe_index.h"

namespace drake {
namespace multibody {
namespace fem {

/** Index used to identify element by index among FEM elements. */
using ElementIndex = TypeSafeIndex<class ElementTag>;

/** Index used to identify FEM models. */
using ModelId = Identifier<class ModelTag>;

/** Index used to identify node by index among FEM nodes. */
using NodeIndex = TypeSafeIndex<class NodeTag>;

/** Index used to identify degrees of freedom (Dof) by index among FEM Dofs. */
using DofIndex = TypeSafeIndex<class DofTag>;

/** Index into a vector of deformable bodies. */
using DeformableBodyIndex = TypeSafeIndex<class DeformableBodyTag>;

}  // namespace fem
}  // namespace multibody
}  // namespace drake
