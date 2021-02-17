// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"

/// \cond
template <typename X, typename Symm, typename IndexList>
class Tensor;
template <typename TagsList>
class Variables;
/// \endcond

namespace evolution::dg::subcell::Tags {
/// Mark a tag as the reconstructed solution on the subcells.
template <typename Tag>
struct Reconstructed : db::PrefixTag, db::SimpleTag {
  static_assert(
      tt::is_a_v<Tensor, typename Tag::type>,
      "A reconstructed solution tag must be either a Tensor or a Variables.");
  using type = typename Tag::type;
  using tag = Tag;
};

/// \cond
template <typename TagList>
struct Reconstructed<::Tags::Variables<TagList>> : db::PrefixTag,
                                                   db::SimpleTag {
 private:
  using wrapped_tags_list = db::wrap_tags_in<Reconstructed, TagList>;

 public:
  using tag = ::Tags::Variables<TagList>;
  using type = Variables<wrapped_tags_list>;
};
/// \endcond
}  // namespace evolution::dg::subcell::Tags
