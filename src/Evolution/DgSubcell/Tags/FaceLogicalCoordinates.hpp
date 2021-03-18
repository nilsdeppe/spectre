// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

namespace evolution::dg::subcell::Tags {
/// The logical coordinates on the subcell faces.
///
/// This is an `optional` and a simple tag because we only need to store this
/// when we are using the subcell solver. When switching back to DG we can free
/// the memory by setting the `optional` to `std::nullopt`. The memory does need
/// to be persistent on the subcells (at least to an extent) because when we
/// send data to our neighbor, we need to send the normal vector (unnormalized
/// in the case that we are evolving the spacetime) to our neighbor.
template <size_t Dim>
struct FaceLogicalCoordinates : db::SimpleTag {
  using type = std::optional<tnsr::I<DataVector, Dim, Frame::Logical>>;
};
}  // namespace evolution::dg::subcell::Tags
