// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/OptionalHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace fd {
/// @{
/*!
 * \brief Computes a high-order boundary correction $G$ at the FD interface.
 *
 * The correction to the second-order boundary correction is given by
 *
 * \f{align*}{
 *  G=G^{(2)}-G^{(4)}+G^{(6)}-G^{(8)}+G^{(10)},
 * \f}
 *
 * where
 *
 *\f{align*}{
 * G^{(4)}_{j+1/2}&=\frac{1}{6}\left(G_j -2 G^{(2)} +
 *                         G_{j+1}\right), \\
 * G^{(6)}_{j+1/2}&=\frac{1}{180}\left(G_{j-1} - 9 G_j + 16 G^{(2)}
 *                         -9 G_{j+1} + G_{j+2}\right), \\
 * G^{(8)}_{j+1/2}&=\frac{1}{2100}\left(G_{j-2} - \frac{25}{3} G_{j-1}
 *                         + 50 G_j - \frac{256}{3} G^{(2)} + 50 G_{j+1}
 *                         - \frac{25}{3} G_{j+2} +G_{j+3}\right), \\
 * G^{(10)}_{j+1/2}&=\frac{1}{17640}
 *                         \left(G_{j-3} - \frac{49}{5} G_{j-2}
 *                     + 49 G_{j-1} - 245 G_j + \frac{2048}{5} G^{(2)}\right.
 *                         \nonumber \\
 *                       &\left.- 245 G_{j+1}+ 49 G_{j+2} - \frac{49}{5} G_{j+3}
 *                         + G_{j+4}\right),
 * \f}
 *
 * where
 *
 * \f{align*}{
 *  G_{j} &= F^i_j n_i^{j+1/2}, \\
 *  G_{j\pm1} &= F^i_{j\pm1} n_i^{j+1/2}, \\
 *  G_{j\pm2} &= F^i_{j\pm2} n_i^{j+1/2}, \\
 *  G_{j\pm3} &= F^i_{j\pm3} n_i^{j+1/2}, \\
 *  G_{j\pm4} &= F^i_{j\pm4} n_i^{j+1/2}.
 * \f}
 *
 * This is a generalization of the correction presented in \cite CHEN2016604.
 *
 * This high-order flux can be fed into a flux limiter, e.g. to guarantee
 * positivity.
 *
 * \note This implementation should be profiled and optimized.
 *
 * \warning This documentation is for the general case. In the restricted
 * Cartesian case we use the cell-centered flux as opposed to `G^{(4)}`, which
 * differs by a minus sign. This amounts to a minus sign change in front of the
 * $G^{(k)}$ terms in computing $G$ for $k>2$, and also a sign change in front
 * of $G^{(2)}$ in all $G^{(k)}$ for $k>2$.
 */
template <DerivativeOrder DerivOrder, size_t Dim, typename... EvolvedVarsTags>
void cartesian_high_order_fluxes_using_nodes(
    const gsl::not_null<
        std::array<Variables<tmpl::list<EvolvedVarsTags...>>, Dim>*>
        high_order_boundary_corrections_in_logical_direction,

    const std::array<Variables<tmpl::list<EvolvedVarsTags...>>, Dim>&
        second_order_boundary_corrections_in_logical_direction,
    const Variables<tmpl::list<
        ::Tags::Flux<EvolvedVarsTags, tmpl::size_t<Dim>, Frame::Inertial>...>>&
        cell_centered_inertial_flux,
    const DirectionMap<
        Dim, Variables<tmpl::list<::Tags::Flux<
                 EvolvedVarsTags, tmpl::size_t<Dim>, Frame::Inertial>...>>>&
        ghost_cell_inertial_flux,
    const Mesh<Dim>& subcell_mesh, const size_t number_of_ghost_cells,
    [[maybe_unused]] const std::array<gsl::span<std::uint8_t>, Dim>&
        reconstruction_order = {}) {
  using std::min;
  constexpr int max_correction_order = 10;
  static_assert(fd::fd_order(DerivOrder) <= max_correction_order);
  constexpr int deriv_order_int = fd::fd_order(DerivOrder);
  constexpr size_t stencil_size =
      deriv_order_int < 0 ? 8 : (static_cast<size_t>(deriv_order_int) - 2);
  const size_t correction_width =
      min((static_cast<size_t>(deriv_order_int) / 2) - 1,
          min(number_of_ghost_cells, stencil_size / 2));
  ASSERT(correction_width <= number_of_ghost_cells,
         "The width of the derivative correction ("
             << correction_width
             << ") must be less than or equal to the number of ghost cells "
             << number_of_ghost_cells);
  ASSERT(alg::all_of(reconstruction_order,
                     [](const auto& t) { return not t.empty(); }) or
             deriv_order_int > 0,
         "For adaptive derivative orders the reconstruction_order must be set");
  for (size_t dim = 0; dim < Dim; ++dim) {
    gsl::at(*high_order_boundary_corrections_in_logical_direction, dim)
        .initialize(
            gsl::at(second_order_boundary_corrections_in_logical_direction, dim)
                .number_of_grid_points());
  }

  // Reconstruction order is always first-varying fastest since we don't
  // transpose that back to {x,y,z} ordering.
  Index<Dim> reconstruction_extents = subcell_mesh.extents();
  reconstruction_extents[0] += 2;

  const auto impl = [&cell_centered_inertial_flux, &ghost_cell_inertial_flux,
                     &high_order_boundary_corrections_in_logical_direction,
                     number_of_ghost_cells,
                     &second_order_boundary_corrections_in_logical_direction,
                     &subcell_mesh, &correction_width, &reconstruction_order,
                     &reconstruction_extents](auto tag_v, auto dim_v) {
    (void)reconstruction_extents;
    using tag = decltype(tag_v);
    constexpr size_t dim = decltype(dim_v)::value;

    auto& high_order_var_correction =
        get<tag>((*high_order_boundary_corrections_in_logical_direction)[dim]);
    const auto& second_order_var_correction =
        get<tag>(second_order_boundary_corrections_in_logical_direction[dim]);
    const auto& recons_order = reconstruction_order[dim];
    const auto& cell_centered_flux =
        get<::Tags::Flux<tag, tmpl::size_t<Dim>, Frame::Inertial>>(
            cell_centered_inertial_flux);
    const auto& lower_neighbor_cell_centered_flux =
        get<::Tags::Flux<tag, tmpl::size_t<Dim>, Frame::Inertial>>(
            ghost_cell_inertial_flux.at(Direction<Dim>{dim, Side::Lower}));
    const auto& upper_neighbor_cell_centered_flux =
        get<::Tags::Flux<tag, tmpl::size_t<Dim>, Frame::Inertial>>(
            ghost_cell_inertial_flux.at(Direction<Dim>{dim, Side::Upper}));
    using FluxTensor = std::decay_t<decltype(cell_centered_flux)>;
    const auto& subcell_extents = subcell_mesh.extents();
    auto subcell_face_extents = subcell_extents;
    ++subcell_face_extents[dim];
    auto neighbor_extents = subcell_extents;
    neighbor_extents[dim] = number_of_ghost_cells;
    const size_t number_of_components = second_order_var_correction.size();
    for (size_t storage_index = 0; storage_index < number_of_components;
         ++storage_index) {
      const auto flux_multi_index = prepend(
          second_order_var_correction.get_tensor_index(storage_index), dim);
      const size_t flux_storage_index =
          FluxTensor::get_storage_index(flux_multi_index);
      // Loop over each face
      for (size_t k = 0; k < (Dim == 3 ? subcell_face_extents[2] : 1); ++k) {
        for (size_t j = 0; j < (Dim >= 2 ? subcell_face_extents[1] : 1); ++j) {
          for (size_t i = 0; i < subcell_face_extents[0]; ++i) {
            const Index<Dim> face_index = [i, j, k]() -> Index<Dim> {
              if constexpr (Dim == 3) {
                return Index<Dim>{i, j, k};
              } else if constexpr (Dim == 2) {
                (void)k;
                return Index<Dim>{i, j};
              } else {
                (void)k, (void)j;
                return Index<Dim>{i};
              }
            }();
            const size_t face_storage_index =
                collapsed_index(face_index, subcell_face_extents);
            Index<Dim> neighbor_index{};
            for (size_t l = 0; l < Dim; ++l) {
              if (l != dim) {
                neighbor_index[l] = face_index[l];
              }
            }

            double& correction =
                high_order_var_correction[storage_index][face_storage_index] =
                    0.0;

            std::array<double, stencil_size> cell_centered_fluxes_for_stencil{};
            // fill if we have to retrieve from lower neighbor
            size_t stencil_index = 0;
            for (int grid_index = static_cast<int>(face_index[dim]) -
                                  static_cast<int>(correction_width);
                 grid_index < static_cast<int>(face_index[dim]) +
                                  static_cast<int>(correction_width);
                 ++grid_index, ++stencil_index) {
              if (grid_index < 0) {
                neighbor_index[dim] = static_cast<size_t>(
                    static_cast<int>(number_of_ghost_cells) + grid_index);
                gsl::at(cell_centered_fluxes_for_stencil, stencil_index) =
                    lower_neighbor_cell_centered_flux[flux_storage_index]
                                                     [collapsed_index(
                                                         neighbor_index,
                                                         neighbor_extents)];
              } else if (grid_index >= static_cast<int>(subcell_extents[dim])) {
                neighbor_index[dim] = static_cast<size_t>(
                    grid_index - static_cast<int>(subcell_extents[dim]));
                gsl::at(cell_centered_fluxes_for_stencil, stencil_index) =
                    upper_neighbor_cell_centered_flux[flux_storage_index]
                                                     [collapsed_index(
                                                         neighbor_index,
                                                         neighbor_extents)];
              } else {
                Index<Dim> volume_index = face_index;
                volume_index[dim] = static_cast<size_t>(grid_index);
                gsl::at(cell_centered_fluxes_for_stencil, stencil_index) =
                    cell_centered_flux[flux_storage_index][collapsed_index(
                        volume_index, subcell_extents)];
              }
            }

            size_t lower_neighbor_index = std::numeric_limits<size_t>::max();
            size_t upper_neighbor_index = std::numeric_limits<size_t>::max();
            if constexpr (deriv_order_int < 0) {
              Index<Dim> lower_n{};
              Index<Dim> upper_n{};
              if constexpr (dim == 0) {
                if constexpr (Dim == 1) {
                  lower_n = Index<Dim>{i};
                  upper_n = Index<Dim>{i + 1};
                } else if constexpr (Dim == 2) {
                  lower_n = Index<Dim>{i, j};
                  upper_n = Index<Dim>{i + 1, j};
                } else if constexpr (Dim == 3) {
                  lower_n = Index<Dim>{i, j, k};
                  upper_n = Index<Dim>{i + 1, j, k};
                }
              } else if constexpr (dim == 1) {
                if constexpr (Dim == 2) {
                  lower_n = Index<Dim>{j, i};
                  upper_n = Index<Dim>{j + 1, i};
                } else if constexpr (Dim == 3) {
                  lower_n = Index<Dim>{j, k, i};
                  upper_n = Index<Dim>{j + 1, k, i};
                }
              } else if constexpr (dim == 2) {
                if constexpr (Dim == 3) {
                  lower_n = Index<Dim>{k, i, j};
                  upper_n = Index<Dim>{k + 1, i, j};
                }
              }
              lower_neighbor_index =
                  collapsed_index(lower_n, reconstruction_extents);
              upper_neighbor_index =
                  collapsed_index(upper_n, reconstruction_extents);
            }

            if (deriv_order_int >= 10 or
                (deriv_order_int < 0 and
                 min(recons_order[lower_neighbor_index],
                     recons_order[upper_neighbor_index]) >= 9)) {
              correction -=
                  5.6689342403628117913e-5 *
                  (gsl::at(cell_centered_fluxes_for_stencil,
                           correction_width - 4) +
                   gsl::at(cell_centered_fluxes_for_stencil,
                           correction_width + 3) -
                   9.8 * (gsl::at(cell_centered_fluxes_for_stencil,
                                  correction_width - 3) +
                          gsl::at(cell_centered_fluxes_for_stencil,
                                  correction_width + 2)) +
                   49.0 * (gsl::at(cell_centered_fluxes_for_stencil,
                                   correction_width - 2) +
                           gsl::at(cell_centered_fluxes_for_stencil,
                                   correction_width + 1)) -
                   245.0 * (gsl::at(cell_centered_fluxes_for_stencil,
                                    correction_width - 1) +
                            gsl::at(cell_centered_fluxes_for_stencil,
                                    correction_width)) -
                   409.6 * second_order_var_correction[storage_index]
                                                      [face_storage_index]);
            }
            if (deriv_order_int >= 8 or
                (deriv_order_int < 0 and
                 min(recons_order[lower_neighbor_index],
                     recons_order[upper_neighbor_index]) >= 7)) {
              correction +=
                  4.7619047619047619047e-4 *
                  (gsl::at(cell_centered_fluxes_for_stencil,
                           correction_width - 3) +
                   gsl::at(cell_centered_fluxes_for_stencil,
                           correction_width + 2) -
                   8.3333333333333333333 *
                       (gsl::at(cell_centered_fluxes_for_stencil,
                                correction_width - 2) +
                        gsl::at(cell_centered_fluxes_for_stencil,
                                correction_width + 1)) +
                   50.0 * (gsl::at(cell_centered_fluxes_for_stencil,
                                   correction_width - 1) +
                           gsl::at(cell_centered_fluxes_for_stencil,
                                   correction_width)) +
                   85.333333333333333333 *
                       second_order_var_correction[storage_index]
                                                  [face_storage_index]);
            }
            if (deriv_order_int >= 6 or
                (deriv_order_int < 0 and
                 min(recons_order[lower_neighbor_index],
                     recons_order[upper_neighbor_index]) >=
                     (DerivOrder == DerivativeOrder::
                                       OneHigherThanReconsButFiveToFourMnd
                          ? 6
                          : 5))) {
              correction -=
                  5.5555555555555555555e-3 *
                  (gsl::at(cell_centered_fluxes_for_stencil,
                           correction_width - 2) +
                   gsl::at(cell_centered_fluxes_for_stencil,
                           correction_width + 1) -
                   9.0 * (gsl::at(cell_centered_fluxes_for_stencil,
                                  correction_width - 1) +
                          gsl::at(cell_centered_fluxes_for_stencil,
                                  correction_width)) -
                   16.0 * second_order_var_correction[storage_index]
                                                     [face_storage_index]);
            }
            if (deriv_order_int >= 4 or
                (deriv_order_int < 0 and
                 min(recons_order[lower_neighbor_index],
                     recons_order[upper_neighbor_index]) >= 3)) {
              correction +=
                  0.166666666666666666 *
                  (gsl::at(cell_centered_fluxes_for_stencil,
                           correction_width - 1) +
                   gsl::at(cell_centered_fluxes_for_stencil, correction_width) +
                   2.0 * second_order_var_correction[storage_index]
                                                    [face_storage_index]);
            }

            // Add second-order correction last
            correction +=
                second_order_var_correction[storage_index][face_storage_index];
          }
        }
      }
    }
  };

  EXPAND_PACK_LEFT_TO_RIGHT(
      impl(EvolvedVarsTags{}, std::integral_constant<size_t, 0>{}));
  if constexpr (Dim > 1) {
    EXPAND_PACK_LEFT_TO_RIGHT(
        impl(EvolvedVarsTags{}, std::integral_constant<size_t, 1>{}));
    if constexpr (Dim > 2) {
      EXPAND_PACK_LEFT_TO_RIGHT(
          impl(EvolvedVarsTags{}, std::integral_constant<size_t, 2>{}));
    }
  }
}

/// @{
/*!
 * \brief Computes a high-order boundary correction using only midpoint
 * (face-centered) values.
 *
 * Unlike `cartesian_high_order_fluxes_using_nodes` which uses both face and
 * cell-centered (node) values (MND scheme), this function uses only
 * face-centered values (MD scheme). The high-order flux at face $j+1/2$ is
 *
 * \f{align*}{
 *  G_{j+1/2} &= d_0 \hat{F}_{j+1/2}
 *    + d_1\left(\hat{F}_{j-1/2} + \hat{F}_{j+3/2}\right)
 *    + d_2\left(\hat{F}_{j-3/2} + \hat{F}_{j+5/2}\right)
 *    + d_3\left(\hat{F}_{j-5/2} + \hat{F}_{j+7/2}\right)
 *    + d_4\left(\hat{F}_{j-7/2} + \hat{F}_{j+9/2}\right),
 * \f}
 *
 * where the coefficients are:
 *
 * | Order | \f$d_0\f$ | \f$d_1\f$ | \f$d_2\f$ | \f$d_3\f$ | \f$d_4\f$ |
 * |-------|-----------|-----------|-----------|-----------|-----------|
 * | 2     | 1         | 0         | 0         | 0         | 0         |
 * | 4     | 13/12     | -1/24     | 0         | 0         | 0         |
 * | 6     | 1067/960  | -29/480   | 3/640     | 0         | 0         |
 * | 8     | 30251/26880 | -7621/107520 | 159/17920 | -5/7168 | 0     |
 * | 10    | 5851067/5160960 | -100027/1290240 | 31471/2580480 |
 *   -425/258048 | 35/294912 |
 *
 * These are derived from the midpoint-to-node differencing (MD) derivative
 * coefficients of \cite Nonomura20138 via the relation \f$d_k = \sum_{m=k}^N
 * a_m\f$.
 *
 * \warning The `ghost_cell_inertial_flux` for this function contains
 * second-order boundary corrections from neighboring elements evaluated at
 * ghost face positions, NOT cell-centered fluxes. For order $p$, the number
 * of ghost faces per direction is $p/2 - 1$.
 */
template <DerivativeOrder DerivOrder, size_t Dim, typename... EvolvedVarsTags>
void cartesian_high_order_fluxes(
    const gsl::not_null<
        std::array<Variables<tmpl::list<EvolvedVarsTags...>>, Dim>*>
        high_order_boundary_corrections_in_logical_direction,

    const std::array<Variables<tmpl::list<EvolvedVarsTags...>>, Dim>&
        second_order_boundary_corrections_in_logical_direction,
    const DirectionMap<
        Dim, Variables<tmpl::list<::Tags::Flux<
                 EvolvedVarsTags, tmpl::size_t<Dim>, Frame::Inertial>...>>>&
        ghost_cell_inertial_flux,
    const Mesh<Dim>& subcell_mesh, const size_t number_of_ghost_cells,
    [[maybe_unused]] const std::array<gsl::span<std::uint8_t>, Dim>&
        reconstruction_order = {}) {
  constexpr int deriv_order_int = fd::fd_order(DerivOrder);
  static_assert(deriv_order_int >= 2 and deriv_order_int <= 10,
                "Only orders 2, 4, 6, 8, 10 are supported for MD fluxes");
  // correction_width is the number of extra faces on each side of the center
  // face. E.g. order 4 -> cw=1, order 10 -> cw=4.
  const size_t correction_width =
      (static_cast<size_t>(deriv_order_int) / 2) - 1;
  ASSERT(correction_width <= number_of_ghost_cells,
         "The width of the derivative correction ("
             << correction_width
             << ") must be less than or equal to the number of ghost cells "
             << number_of_ghost_cells);
  // The stencil has 2*correction_width + 1 entries (the center face plus
  // correction_width faces on each side).
  constexpr size_t stencil_size =
      deriv_order_int <= 2 ? 1 : (static_cast<size_t>(deriv_order_int) - 1);

  for (size_t dim = 0; dim < Dim; ++dim) {
    gsl::at(*high_order_boundary_corrections_in_logical_direction, dim)
        .initialize(
            gsl::at(second_order_boundary_corrections_in_logical_direction, dim)
                .number_of_grid_points());
  }

  // For order 2, the result is just the second-order correction (no ghost data
  // needed).
  if constexpr (deriv_order_int <= 2) {
    for (size_t dim = 0; dim < Dim; ++dim) {
      gsl::at(*high_order_boundary_corrections_in_logical_direction, dim) =
          gsl::at(second_order_boundary_corrections_in_logical_direction, dim);
    }
    return;
  }

  const auto impl = [&ghost_cell_inertial_flux,
                     &high_order_boundary_corrections_in_logical_direction,
                     number_of_ghost_cells,
                     &second_order_boundary_corrections_in_logical_direction,
                     &subcell_mesh, correction_width](auto tag_v, auto dim_v) {
    using tag = decltype(tag_v);
    constexpr size_t dim = decltype(dim_v)::value;

    auto& high_order_var_correction =
        get<tag>((*high_order_boundary_corrections_in_logical_direction)[dim]);
    const auto& second_order_var_correction =
        get<tag>(second_order_boundary_corrections_in_logical_direction[dim]);
    const auto& lower_neighbor_face_flux =
        get<::Tags::Flux<tag, tmpl::size_t<Dim>, Frame::Inertial>>(
            ghost_cell_inertial_flux.at(Direction<Dim>{dim, Side::Lower}));
    const auto& upper_neighbor_face_flux =
        get<::Tags::Flux<tag, tmpl::size_t<Dim>, Frame::Inertial>>(
            ghost_cell_inertial_flux.at(Direction<Dim>{dim, Side::Upper}));
    using FluxTensor = std::decay_t<decltype(lower_neighbor_face_flux)>;
    const auto& subcell_extents = subcell_mesh.extents();
    auto subcell_face_extents = subcell_extents;
    ++subcell_face_extents[dim];
    // Ghost extents: correction_width faces in the dim direction, same in
    // transverse directions.
    auto neighbor_extents = subcell_extents;
    neighbor_extents[dim] = number_of_ghost_cells;
    const size_t number_of_components = second_order_var_correction.size();
    for (size_t storage_index = 0; storage_index < number_of_components;
         ++storage_index) {
      const auto flux_multi_index = prepend(
          second_order_var_correction.get_tensor_index(storage_index), dim);
      const size_t flux_storage_index =
          FluxTensor::get_storage_index(flux_multi_index);
      // Loop over each face
      for (size_t k = 0; k < (Dim == 3 ? subcell_face_extents[2] : 1); ++k) {
        for (size_t j = 0; j < (Dim >= 2 ? subcell_face_extents[1] : 1); ++j) {
          for (size_t i = 0; i < subcell_face_extents[0]; ++i) {
            const Index<Dim> face_index = [i, j, k]() -> Index<Dim> {
              if constexpr (Dim == 3) {
                return Index<Dim>{i, j, k};
              } else if constexpr (Dim == 2) {
                (void)k;
                return Index<Dim>{i, j};
              } else {
                (void)k, (void)j;
                return Index<Dim>{i};
              }
            }();
            const size_t face_storage_index =
                collapsed_index(face_index, subcell_face_extents);

            // Build stencil of face values centered on the current face.
            // stencil[correction_width] is the center face (j+1/2).
            // stencil[correction_width - m] is the face at j+1/2 - m
            //   (i.e. face_index[dim] - m).
            // stencil[correction_width + m] is the face at j+1/2 + m
            //   (i.e. face_index[dim] + m).
            std::array<double, stencil_size> face_stencil{};
            for (int m = -static_cast<int>(correction_width);
                 m <= static_cast<int>(correction_width); ++m) {
              const int face_idx = static_cast<int>(face_index[dim]) + m;
              const auto stencil_idx =
                  static_cast<size_t>(m + static_cast<int>(correction_width));
              if (face_idx < 0) {
                // Lower neighbor ghost face
                Index<Dim> neighbor_index{};
                for (size_t l = 0; l < Dim; ++l) {
                  if (l != dim) {
                    neighbor_index[l] = face_index[l];
                  }
                }
                neighbor_index[dim] = static_cast<size_t>(
                    static_cast<int>(number_of_ghost_cells) + face_idx);
                gsl::at(face_stencil, stencil_idx) =
                    lower_neighbor_face_flux[flux_storage_index]
                                            [collapsed_index(neighbor_index,
                                                             neighbor_extents)];
              } else if (face_idx > static_cast<int>(subcell_extents[dim])) {
                // Upper neighbor ghost face
                Index<Dim> neighbor_index{};
                for (size_t l = 0; l < Dim; ++l) {
                  if (l != dim) {
                    neighbor_index[l] = face_index[l];
                  }
                }
                neighbor_index[dim] = static_cast<size_t>(
                    face_idx - static_cast<int>(subcell_extents[dim]) - 1);
                gsl::at(face_stencil, stencil_idx) =
                    upper_neighbor_face_flux[flux_storage_index]
                                            [collapsed_index(neighbor_index,
                                                             neighbor_extents)];
              } else {
                // Interior face: read from second_order correction
                // (which stores -n_i F^i at each face)
                Index<Dim> interior_face_index = face_index;
                interior_face_index[dim] = static_cast<size_t>(face_idx);
                gsl::at(face_stencil, stencil_idx) =
                    second_order_var_correction[storage_index][collapsed_index(
                        interior_face_index, subcell_face_extents)];
              }
            }

            // Apply MD coefficients.
            double& correction =
                high_order_var_correction[storage_index][face_storage_index];
            // d_0 * center
            // + d_1 * (stencil[cw-1] + stencil[cw+1])
            // + d_2 * (stencil[cw-2] + stencil[cw+2])
            // + ...
            if constexpr (deriv_order_int == 4) {
              // d_0 = 13/12, d_1 = -1/24
              correction = 1.0833333333333333333 *
                               gsl::at(face_stencil, correction_width) +
                           (-0.041666666666666666667) *
                               (gsl::at(face_stencil, correction_width - 1) +
                                gsl::at(face_stencil, correction_width + 1));
            } else if constexpr (deriv_order_int == 6) {
              // d_0 = 1067/960, d_1 = -29/480, d_2 = 3/640
              correction =
                  1.1114583333333333333 *
                      gsl::at(face_stencil, correction_width) +
                  (-0.060416666666666666667) *
                      (gsl::at(face_stencil, correction_width - 1) +
                       gsl::at(face_stencil, correction_width + 1)) +
                  0.0046875 * (gsl::at(face_stencil, correction_width - 2) +
                               gsl::at(face_stencil, correction_width + 2));
            } else if constexpr (deriv_order_int == 8) {
              // d_0 = 30251/26880, d_1 = -7621/107520,
              // d_2 = 159/17920, d_3 = -5/7168
              correction = 1.1254092261904762307 *
                               gsl::at(face_stencil, correction_width) +
                           (-0.070879836309523810978) *
                               (gsl::at(face_stencil, correction_width - 1) +
                                gsl::at(face_stencil, correction_width + 1)) +
                           0.0088727678571428568455 *
                               (gsl::at(face_stencil, correction_width - 2) +
                                gsl::at(face_stencil, correction_width + 2)) +
                           (-6.9754464285714287263e-4) *
                               (gsl::at(face_stencil, correction_width - 3) +
                                gsl::at(face_stencil, correction_width + 3));
            } else if constexpr (deriv_order_int == 10) {
              // d_0 = 5851067/5160960, d_1 = -100027/1290240,
              // d_2 = 31471/2580480, d_3 = -425/258048,
              // d_4 = 35/294912
              correction = 1.1337167891245039097 *
                               gsl::at(face_stencil, correction_width) +
                           (-0.077525886656746034742) *
                               (gsl::at(face_stencil, correction_width - 1) +
                                gsl::at(face_stencil, correction_width + 1)) +
                           0.012195793030753968728 *
                               (gsl::at(face_stencil, correction_width - 2) +
                                gsl::at(face_stencil, correction_width + 2)) +
                           (-1.6469804067460317495e-3) *
                               (gsl::at(face_stencil, correction_width - 3) +
                                gsl::at(face_stencil, correction_width + 3)) +
                           1.1867947048611110961e-4 *
                               (gsl::at(face_stencil, correction_width - 4) +
                                gsl::at(face_stencil, correction_width + 4));
            }
          }
        }
      }
    }
  };

  EXPAND_PACK_LEFT_TO_RIGHT(
      impl(EvolvedVarsTags{}, std::integral_constant<size_t, 0>{}));
  if constexpr (Dim > 1) {
    EXPAND_PACK_LEFT_TO_RIGHT(
        impl(EvolvedVarsTags{}, std::integral_constant<size_t, 1>{}));
    if constexpr (Dim > 2) {
      EXPAND_PACK_LEFT_TO_RIGHT(
          impl(EvolvedVarsTags{}, std::integral_constant<size_t, 2>{}));
    }
  }
}

template <size_t Dim, typename... EvolvedVarsTags>
void cartesian_high_order_fluxes(
    const gsl::not_null<
        std::array<Variables<tmpl::list<EvolvedVarsTags...>>, Dim>*>
        high_order_boundary_corrections_in_logical_direction,

    const std::array<Variables<tmpl::list<EvolvedVarsTags...>>, Dim>&
        second_order_boundary_corrections_in_logical_direction,
    const DirectionMap<
        Dim, Variables<tmpl::list<::Tags::Flux<
                 EvolvedVarsTags, tmpl::size_t<Dim>, Frame::Inertial>...>>>&
        ghost_cell_inertial_flux,
    const Mesh<Dim>& subcell_mesh, const size_t number_of_ghost_cells,
    const DerivativeOrder derivative_order,
    [[maybe_unused]] const std::array<gsl::span<std::uint8_t>, Dim>&
        reconstruction_order = {}) {
  switch (derivative_order) {
    case DerivativeOrder::Two:
      cartesian_high_order_fluxes<DerivativeOrder::Two>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          ghost_cell_inertial_flux, subcell_mesh, number_of_ghost_cells,
          reconstruction_order);
      break;
    case DerivativeOrder::FourMd:
      cartesian_high_order_fluxes<DerivativeOrder::FourMd>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          ghost_cell_inertial_flux, subcell_mesh, number_of_ghost_cells,
          reconstruction_order);
      break;
    case DerivativeOrder::SixMd:
      cartesian_high_order_fluxes<DerivativeOrder::SixMd>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          ghost_cell_inertial_flux, subcell_mesh, number_of_ghost_cells,
          reconstruction_order);
      break;
    case DerivativeOrder::EightMd:
      cartesian_high_order_fluxes<DerivativeOrder::EightMd>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          ghost_cell_inertial_flux, subcell_mesh, number_of_ghost_cells,
          reconstruction_order);
      break;
    case DerivativeOrder::TenMd:
      cartesian_high_order_fluxes<DerivativeOrder::TenMd>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          ghost_cell_inertial_flux, subcell_mesh, number_of_ghost_cells,
          reconstruction_order);
      break;
    default:
      ERROR("Unsupported derivative order for midpoint-only fluxes: "
            << derivative_order
            << ". Use cartesian_high_order_fluxes_using_nodes for MND orders.");
  };
}
/// @}

template <size_t Dim, typename... EvolvedVarsTags>
void cartesian_high_order_fluxes_using_nodes(
    const gsl::not_null<
        std::array<Variables<tmpl::list<EvolvedVarsTags...>>, Dim>*>
        high_order_boundary_corrections_in_logical_direction,

    const std::array<Variables<tmpl::list<EvolvedVarsTags...>>, Dim>&
        second_order_boundary_corrections_in_logical_direction,
    const Variables<tmpl::list<
        ::Tags::Flux<EvolvedVarsTags, tmpl::size_t<Dim>, Frame::Inertial>...>>&
        cell_centered_inertial_flux,
    const DirectionMap<
        Dim, Variables<tmpl::list<::Tags::Flux<
                 EvolvedVarsTags, tmpl::size_t<Dim>, Frame::Inertial>...>>>&
        ghost_cell_inertial_flux,
    const Mesh<Dim>& subcell_mesh, const size_t number_of_ghost_cells,
    const DerivativeOrder derivative_order,
    [[maybe_unused]] const std::array<gsl::span<std::uint8_t>, Dim>&
        reconstruction_order = {}) {
  switch (derivative_order) {
    case DerivativeOrder::OneHigherThanReconsMnd:
      cartesian_high_order_fluxes_using_nodes<
          DerivativeOrder::OneHigherThanReconsMnd>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          cell_centered_inertial_flux, ghost_cell_inertial_flux, subcell_mesh,
          number_of_ghost_cells, reconstruction_order);
      break;
    case DerivativeOrder::OneHigherThanReconsButFiveToFourMnd:
      cartesian_high_order_fluxes_using_nodes<
          DerivativeOrder::OneHigherThanReconsButFiveToFourMnd>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          cell_centered_inertial_flux, ghost_cell_inertial_flux, subcell_mesh,
          number_of_ghost_cells, reconstruction_order);
      break;
    case DerivativeOrder::Two:
      cartesian_high_order_fluxes_using_nodes<DerivativeOrder::Two>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          cell_centered_inertial_flux, ghost_cell_inertial_flux, subcell_mesh,
          number_of_ghost_cells, reconstruction_order);
      break;
    case DerivativeOrder::FourMnd:
      cartesian_high_order_fluxes_using_nodes<DerivativeOrder::FourMnd>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          cell_centered_inertial_flux, ghost_cell_inertial_flux, subcell_mesh,
          number_of_ghost_cells, reconstruction_order);
      break;
    case DerivativeOrder::SixMnd:
      cartesian_high_order_fluxes_using_nodes<DerivativeOrder::SixMnd>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          cell_centered_inertial_flux, ghost_cell_inertial_flux, subcell_mesh,
          number_of_ghost_cells, reconstruction_order);
      break;
    case DerivativeOrder::EightMnd:
      cartesian_high_order_fluxes_using_nodes<DerivativeOrder::EightMnd>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          cell_centered_inertial_flux, ghost_cell_inertial_flux, subcell_mesh,
          number_of_ghost_cells, reconstruction_order);
      break;
    case DerivativeOrder::TenMnd:
      cartesian_high_order_fluxes_using_nodes<DerivativeOrder::TenMnd>(
          high_order_boundary_corrections_in_logical_direction,
          second_order_boundary_corrections_in_logical_direction,
          cell_centered_inertial_flux, ghost_cell_inertial_flux, subcell_mesh,
          number_of_ghost_cells, reconstruction_order);
      break;
    case DerivativeOrder::FourMd:
    case DerivativeOrder::SixMd:
    case DerivativeOrder::EightMd:
    case DerivativeOrder::TenMd:
      ERROR(
          "Midpoint-only (Md) derivative orders must use "
          "cartesian_high_order_fluxes, not "
          "cartesian_high_order_fluxes_using_nodes. Got: "
          << derivative_order);
    default:
      ERROR("Unsupported correction order " << derivative_order);
  };
}
/// @}

/*!
 * \brief Fill the `flux_neighbor_data` with pointers into the
 * `all_ghost_data`.
 *
 * The `all_ghost_data` is stored in the tag
 * `evolution::dg::subcell::Tags::GhostDataForReconstruction`, and the
 * `ghost_zone_size` should come from the FD reconstructor.
 */
template <size_t Dim, typename FluxesTags>
void set_cartesian_neighbor_cell_centered_fluxes(
    const gsl::not_null<DirectionMap<Dim, Variables<FluxesTags>>*>
        flux_neighbor_data,
    const DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const Mesh<Dim>& subcell_mesh, const size_t ghost_zone_size,
    const size_t number_of_rdmp_values_in_ghost_data) {
  for (const auto& [direction_id, ghost_data] : all_ghost_data) {
    const size_t neighbor_flux_size =
        subcell_mesh.number_of_grid_points() /
        subcell_mesh.extents(direction_id.direction().dimension()) *
        ghost_zone_size *
        Variables<FluxesTags>::number_of_independent_components;
    const DataVector& neighbor_data =
        ghost_data.neighbor_ghost_data_for_reconstruction();
    (*flux_neighbor_data)[direction_id.direction()].set_data_ref(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<double*>(std::next(
            neighbor_data.data(),
            static_cast<std::ptrdiff_t>(neighbor_data.size() -
                                        number_of_rdmp_values_in_ghost_data -
                                        neighbor_flux_size))),
        neighbor_flux_size);
  }
}

/*!
 * \brief Computes the high-order Cartesian flux corrections if necessary.
 *
 * The `cell_centered_fluxes` is stored in the tag
 * `evolution::dg::subcell::Tags::CellCenteredFlux`, `fd_derivative_order` is
 * from `evolution::dg::subcell::Tags::SubcellOptions`
 * (`.finite_difference_derivative_order()`), the `all_ghost_data`
 * is stored in the tag
 * `evolution::dg::subcell::Tags::GhostDataForReconstruction`, the
 * `ghost_zone_size` should come from the FD reconstructor.
 *
 * By default we assume no RDMP data is in the `ghost_data` buffer. In the
 * future we will want to update how we store the data in order to eliminate
 * more memory allocations and copies, in which case that value will be
 * non-zero.
 *
 * \note `high_order_corrections` must either not have a value or have all
 * elements be of the same size as
 * `second_order_boundary_corrections[0].number_of_grid_points()`, where we've
 * assumed `second_order_boundary_corrections` is the same in all directions.
 */
template <size_t Dim, typename... EvolvedVarsTags,
          typename FluxesTags = tmpl::list<::Tags::Flux<
              EvolvedVarsTags, tmpl::size_t<Dim>, Frame::Inertial>...>>
void cartesian_high_order_flux_corrections(
    const gsl::not_null<std::optional<
        std::array<Variables<tmpl::list<EvolvedVarsTags...>>, Dim>>*>
        high_order_corrections,

    const std::optional<Variables<FluxesTags>>& cell_centered_fluxes,
    const std::array<Variables<tmpl::list<EvolvedVarsTags...>>, Dim>&
        second_order_boundary_corrections,
    const fd::DerivativeOrder& fd_derivative_order,
    const DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const Mesh<Dim>& subcell_mesh, const size_t ghost_zone_size,
    [[maybe_unused]] const std::array<gsl::span<std::uint8_t>, Dim>&
        reconstruction_order = {},
    const size_t number_of_rdmp_values_in_ghost_data = 0) {
  const bool is_md = ::fd::is_md_order(fd_derivative_order);
  if (is_md) {
    ASSERT(not cell_centered_fluxes.has_value(),
           "Cell-centered fluxes should not be computed when using "
           "midpoint-only (Md) derivative orders. The extra computation "
           "is wasted. Derivative order: "
               << fd_derivative_order);
    ASSERT(alg::all_of(
               second_order_boundary_corrections,
               [expected_size = second_order_boundary_corrections[0]
                                    .number_of_grid_points()](const auto& e) {
                 return e.number_of_grid_points() == expected_size;
               }),
           "All second-order boundary corrections must be of the same size, "
               << second_order_boundary_corrections[0].number_of_grid_points());
    if (not high_order_corrections->has_value()) {
      (*high_order_corrections) =
          make_array<Dim>(Variables<tmpl::list<EvolvedVarsTags...>>{
              second_order_boundary_corrections[0].number_of_grid_points()});
    }
    ASSERT(high_order_corrections->has_value() and
               alg::all_of(high_order_corrections->value(),
                           [expected_size =
                                second_order_boundary_corrections[0]
                                    .number_of_grid_points()](const auto& e) {
                             return e.number_of_grid_points() == expected_size;
                           }),
           "The high_order_corrections must all have size "
               << second_order_boundary_corrections[0].number_of_grid_points());
    const size_t md_ghost_zone_size =
        (static_cast<size_t>(fd_order(fd_derivative_order)) / 2) - 1;
    DirectionMap<Dim, Variables<FluxesTags>> face_ghost_data{};
    set_cartesian_neighbor_cell_centered_fluxes(
        make_not_null(&face_ghost_data), all_ghost_data, subcell_mesh,
        md_ghost_zone_size, number_of_rdmp_values_in_ghost_data);
    cartesian_high_order_fluxes(
        make_not_null(&(high_order_corrections->value())),
        second_order_boundary_corrections, face_ghost_data, subcell_mesh,
        md_ghost_zone_size, fd_derivative_order, reconstruction_order);
  } else if (cell_centered_fluxes.has_value()) {
    ASSERT(alg::all_of(
               second_order_boundary_corrections,
               [expected_size = second_order_boundary_corrections[0]
                                    .number_of_grid_points()](const auto& e) {
                 return e.number_of_grid_points() == expected_size;
               }),
           "All second-order boundary corrections must be of the same size, "
               << second_order_boundary_corrections[0].number_of_grid_points());
    if (fd_derivative_order != DerivativeOrder::Two) {
      if (not high_order_corrections->has_value()) {
        (*high_order_corrections) =
            make_array<Dim>(Variables<tmpl::list<EvolvedVarsTags...>>{
                second_order_boundary_corrections[0].number_of_grid_points()});
      }
      ASSERT(
          high_order_corrections->has_value() and
              alg::all_of(high_order_corrections->value(),
                          [expected_size =
                               second_order_boundary_corrections[0]
                                   .number_of_grid_points()](const auto& e) {
                            return e.number_of_grid_points() == expected_size;
                          }),
          "The high_order_corrections must all have size "
              << second_order_boundary_corrections[0].number_of_grid_points());
      DirectionMap<Dim, Variables<FluxesTags>> flux_neighbor_data{};
      set_cartesian_neighbor_cell_centered_fluxes(
          make_not_null(&flux_neighbor_data), all_ghost_data, subcell_mesh,
          ghost_zone_size, number_of_rdmp_values_in_ghost_data);

      cartesian_high_order_fluxes_using_nodes(
          make_not_null(&(high_order_corrections->value())),
          second_order_boundary_corrections, cell_centered_fluxes.value(),
          flux_neighbor_data, subcell_mesh, ghost_zone_size,
          fd_derivative_order, reconstruction_order);
    }
  }
}
}  // namespace fd
