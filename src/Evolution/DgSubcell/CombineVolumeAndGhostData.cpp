// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/CombineVolumeAndGhostData.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell::detail {
void combine_volume_and_ghost_data_impl(
    const gsl::not_null<gsl::span<double>*> volume_and_ghost_subcell_vars,
    const gsl::span<const double>& transposed_volume_vars,
    const size_t number_of_points_in_first_dim,
    const size_t number_of_points_in_orthogonal_dims,
    const size_t number_of_ghost_points,
    const gsl::span<const double>& lower_neighbor_vars,
    const gsl::span<const double>& upper_neighbor_vars) noexcept {
  const size_t num_points_with_ghost_cells =
      number_of_points_in_first_dim + 2 * number_of_ghost_points;

  ASSERT(
      transposed_volume_vars.size() % (number_of_points_in_first_dim *
                                       number_of_points_in_orthogonal_dims) ==
          0,
      "The transposed_volume_vars must have an exact multiple of the total "
      "number of volume points. We have "
          << (number_of_points_in_first_dim *
              number_of_points_in_orthogonal_dims)
          << " volume points and transposed_volume_vars is of size "
          << transposed_volume_vars.size());
  ASSERT(
      volume_and_ghost_subcell_vars->size() %
              (num_points_with_ghost_cells *
               number_of_points_in_orthogonal_dims) ==
          0,
      "The volume_and_ghost_subcell_vars must have an exact multiple of the "
      "total number of volume (with ghost) points. We have "
          << (num_points_with_ghost_cells * number_of_points_in_orthogonal_dims)
          << " volume (with ghost) points and volume_and_ghost_subcell_vars is "
             "of size "
          << volume_and_ghost_subcell_vars->size());
  ASSERT(
      lower_neighbor_vars.size() %
              (number_of_ghost_points * number_of_points_in_orthogonal_dims) ==
          0,
      "The lower_neighbor_vars size "
          << lower_neighbor_vars.size()
          << " must be a multiple of the number of ghost points times the "
             "number of grid points in the directions not being reconstructed ("
          << (number_of_ghost_points * number_of_points_in_orthogonal_dims)
          << ")");
  ASSERT(
      upper_neighbor_vars.size() %
              (number_of_ghost_points * number_of_points_in_orthogonal_dims) ==
          0,
      "The upper_neighbor_vars size "
          << upper_neighbor_vars.size()
          << " must be a multiple of the number of ghost points times the "
             "number of grid points in the directions not being reconstructed ("
          << (number_of_ghost_points * number_of_points_in_orthogonal_dims)
          << ")");
  ASSERT(upper_neighbor_vars.size() / (number_of_ghost_points *
                                       number_of_points_in_orthogonal_dims) ==
             lower_neighbor_vars.size() /
                 (number_of_ghost_points * number_of_points_in_orthogonal_dims),
         "The number of components differ for the lower ("
             << (lower_neighbor_vars.size() /
                 (number_of_ghost_points * number_of_points_in_orthogonal_dims))
             << ") and upper ("
             << (upper_neighbor_vars.size() /
                 (number_of_ghost_points * number_of_points_in_orthogonal_dims))
             << ") neighbor data");
  ASSERT(
      upper_neighbor_vars.size() /
              (number_of_ghost_points * number_of_points_in_orthogonal_dims) ==
          transposed_volume_vars.size() / (number_of_points_in_first_dim *
                                           number_of_points_in_orthogonal_dims),
      "Number of components differ for the neighbor data ("
          << (upper_neighbor_vars.size() /
              (number_of_ghost_points * number_of_points_in_orthogonal_dims))
          << ") and the volume data: "
          << transposed_volume_vars.size() /
                 (number_of_points_in_first_dim *
                  number_of_points_in_orthogonal_dims));

  // Easiest way to deal with reconstruction in multiple dimensions:
  // 1. transpose so dim we are reconstructing in is varying fastest
  // 2. copy over ghost cells
  // 3. do reconstruction in fastest vary dimension.
  // Issue is with when sending not the exact vars we will reconstruct
  // because
  //
  // I think the best thing is:
  // 1. compute whatever we want to construct in the volume
  // 2. for each logical direction:
  //      compute what we want to reconstruct on the ghost cells;
  //      do reconstruction;
  //      compute fluxes on faces;
  //      compute boundary correction on faces;
  //      compute time derivative;
  //
  // How to split this up:
  // - transposition of reconstruction Variables in volume
  // - copying volume variables into volume+ghost buffer
  // - reconstruction can then be done component-by-component
  // - can then "easily" compute extra variables for fluxes and boundary
  //   corrections

  const size_t number_of_slices =
      transposed_volume_vars.size() / number_of_points_in_first_dim;
  // Loop over each slice in the direction that we are reconstructing.
  for (size_t slice_index = 0; slice_index < number_of_slices; ++slice_index) {
    const size_t start_volume_and_ghost =
        slice_index * num_points_with_ghost_cells;
    const size_t start_volume = slice_index * number_of_points_in_first_dim;
    const size_t neighbor_offset = number_of_ghost_points * slice_index;
    // Copy over lower ghost points, then volume, then upper ghost points
    for (size_t i = 0; i < number_of_ghost_points; ++i) {
      (*volume_and_ghost_subcell_vars)[start_volume_and_ghost + i] =
          lower_neighbor_vars[neighbor_offset + i];
    }
    std::copy(transposed_volume_vars.data() + start_volume,
              transposed_volume_vars.data() + start_volume +
                  number_of_points_in_first_dim,
              volume_and_ghost_subcell_vars->data() + start_volume_and_ghost +
                  number_of_ghost_points);
    const size_t volume_and_ghost_offset = start_volume_and_ghost +
                                           number_of_ghost_points +
                                           number_of_points_in_first_dim;
    for (size_t i = 0; i < number_of_ghost_points; ++i) {
      (*volume_and_ghost_subcell_vars)[volume_and_ghost_offset + i] =
          upper_neighbor_vars[neighbor_offset + i];
    }
  }
}
}  // namespace evolution::dg::subcell::detail
