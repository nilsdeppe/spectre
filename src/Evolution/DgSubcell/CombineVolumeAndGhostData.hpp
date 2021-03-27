// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/Variables.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Index;
namespace evolution::dg::subcell {
class NeighborData;
template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
class FixedHashMap;
}  // namespace evolution::dg::subcell
/// \endcond

namespace evolution::dg::subcell {
namespace detail {
void combine_volume_and_ghost_data_impl(
    const gsl::not_null<gsl::span<double>*> volume_and_ghost_subcell_vars,
    const gsl::span<const double>& transposed_volume_vars,
    const size_t number_of_points_in_first_dim,
    const size_t number_of_points_in_orthogonal_dims,
    const size_t number_of_ghost_points,
    const gsl::span<const double>& lower_neighbor_vars,
    const gsl::span<const double>& upper_neighbor_vars) noexcept;
}  // namespace detail

/*!
 * \brief Combine the volume and ghost data in the fastest varying/first
 * dimension.
 *
 * The `transposed_volume_vars` must have been transposed so that the fastest
 * varying dimension is the one we are doing reconstruction in. The `dimension`
 * is the dimension we are doing reconstruction in.
 */
template <size_t Dim, typename TagList>
void combine_volume_and_ghost_data(
    const gsl::not_null<Variables<TagList>*> volume_and_ghost_subcell_vars,
    const Variables<TagList>& transposed_volume_vars, const size_t dimension,
    const Index<Dim>& subcell_extents, const size_t number_of_ghost_points,
    const gsl::span<const double>& lower_neighbor_vars,
    const gsl::span<const double>& upper_neighbor_vars) noexcept {
  // TODO: we should have a version that allows combining only a subset of the
  // volume vars into the volume+ghost.
  auto vars_span = gsl::make_span(volume_and_ghost_subcell_vars->data(),
                                  volume_and_ghost_subcell_vars->size());
  return detail::combine_volume_and_ghost_data_impl(
      make_not_null(&vars_span),
      gsl::make_span(transposed_volume_vars.data(),
                     transposed_volume_vars.size()),
      subcell_extents[dimension],
      subcell_extents.slice_away(dimension).product(), number_of_ghost_points,
      lower_neighbor_vars, upper_neighbor_vars);
}
}  // namespace evolution::dg::subcell
