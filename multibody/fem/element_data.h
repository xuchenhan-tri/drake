#pragma once

#include <memory>

#include "drake/common/drake_copyable.h"

namespace drake {
namespace multibody {
namespace fem {

/** Abstract class for per element FEM data.
 @tparam_nonsymbolic_scalar */
class ElementData {
 public:
  virtual ~ElementData() = default;

  /* Returns the number of elements that this ElementData has data for. */
  virtual int size() const = 0;

  virtual std::unique_ptr<ElementData> Clone() const = 0;

 protected:
  ElementData() = default;
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ElementData);
};

}  // namespace fem
}  // namespace multibody
}  // namespace drake
