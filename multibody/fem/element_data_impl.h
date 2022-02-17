#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/multibody/fem/element_data.h"
#include "drake/multibody/fem/fem_indexes.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* ElementDataImpl implements ElementData for a particular type of FEM element
 data. 
 @tparam Data The type of per-element data stored in this ElementData. */
template <typename Data>
class ElementDataImpl : public ElementData {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ElementDataImpl);

  explicit ElementDataImpl(int num_elements) : element_data_(num_elements) {}

  int size() const final { return element_data_.size(); }

  std::unique_ptr<ElementData> Clone() const final {
    return std::make_unique<ElementDataImpl<Data>>(*this);
  }

  void set_data(ElementIndex index, Data data) {
    DRAKE_ASSERT(0 <= index && index < size());
    element_data_[index] = std::move(data);
  }

  const Data& get_data(ElementIndex index) const {
    DRAKE_ASSERT(0 <= index && index < size());
    return element_data_[index];
  }

 private:
  std::vector<Data> element_data_;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
