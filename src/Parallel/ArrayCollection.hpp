// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>

#include "Domain/Structure/ElementId.hpp"

namespace Parallel {
template <typename ReceiveTag, typename ReceiveDataType, size_t Dim>
struct CollectionMessage {
  typename ReceiveTag::temporal_id instance{};
  ElementId<Dim> element_id{};
  ReceiveDataType data{};
  bool enable_if_disabled{false};
};

template <size_t Dim>
class DgElementArrayMember {
 public:
 private:
  P
};

template <size_t Dim, typename ArrayElement>
class ElementCollection {
 public:
  /// \brief Threaded action that receives neighbor data and calls the next
  /// iterable action on the specified element.
  template <typename ReceiveTag, typename ReceiveDataType>
  void receive_data(CollectionMessage<ReceiveTag, ReceiveDataType, Dim> data) {
    // (void)Parallel::charmxx::RegisterReceiveData<ParallelComponent,
    // ReceiveTag,
    //                                              false>::registrar;
    const ElementId<Dim> element_id = data.element_id;  // TODO: I don't know
                                                        // if I need this.
    array_elements_.at(element_id).receive_data(std::move(data));
  }

 private:
  std::unordered_map<ElementId<Dim>, ArrayElement> array_elements_{};
};

// template <typename ArrayId, typename ArrayElement>
// class ArrayCollection {
//  public:

//  private:
//   std::unordered_map<ArrayId, ArrayElement> array_elements_{};
// };

}  // namespace Parallel
