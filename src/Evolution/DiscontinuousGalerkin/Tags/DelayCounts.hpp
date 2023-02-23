// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"

namespace evolution::dg::Tags {
struct DelayCount : db::SimpleTag {
  using type = size_t;
};
}  // namespace evolution::dg::Tags
