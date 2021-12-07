// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/AderDg/Divergence.hpp"
#include "Evolution/AderDg/Matrices.hpp"
#include "Evolution/AderDg/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace AderDg {
/*!
 * \ingroup AderDgGroup
 * \brief Performs the Picard iterations to compute the spacetime predictor
 * solution.
 */
template <typename Metavariables, typename DbTagsList>
void perform_picard_iterations(
    const gsl::not_null<db::DataBox<DbTagsList>*> box) {
  using system = typename Metavariables::system;
  using variables_tag = typename system::variables_tag;
  using fluxes_tag = db::split_tag<
      db::add_tag_prefix<::Tags::Flux, variables_tag,
                         tmpl::size_t<system::volume_dim>, Frame::Inertial>>;
  using sources_tag =
      db::wrap_tags_in<::Tags::Source, typename system::sourced_variables>;
  using computer = typename system::compute_time_derivative;
  constexpr size_t volume_dim = system::volume_dim;

  size_t iteration_number = 0;
  const size_t max_iterations = 12;

  const auto& mesh = db::get<::Tags::Mesh<volume_dim>>(*box);
  const size_t number_temporal_grid_points =
      db::get<Tags::TemporalGridPoints>(box);
  const size_t number_spatial_grid_points = mesh.number_of_grid_points();
  const size_t number_spacetime_grid_points =
      number_spatial_grid_points * number_temporal_grid_points;

  // Create allocation for flux divergence and logical partial derivs of fluxes
  Variables<db::wrap_tags_in<::Tags::div, fluxes_tag>> div_fluxes(
      number_spacetime_grid_points);
  std::array<db::item_type<fluxes_tag>, volume_dim>
      logical_partial_derivs_of_fluxes_buffer{};

  while (true) {
    // Compute volume fluxes, volume non-conservative products, and volume
    // sources.
    db::mutate_apply<fluxes_tag, typename system::volume_fluxes::argument_tags>(
        typename system::volume_fluxes{}, make_not_null(&box));
    db::mutate_apply<sources_tag,
                     typename system::volume_sources::argument_tags>(
        typename system::volume_sources{}, make_not_null(&box));

    // Compute the divergence of the volume fluxes
    // TODO(nils): This assumes a time-independent coordinate transformation
    divergence_time_independent_jacobian(
        make_not_null(&div_fluxes), &logical_partial_derivs_of_fluxes_buffer,
        mesh, number_temporal_grid_points, db::get<fluxes_tag>(box),
        inverse_jacobian);

    // Combine div F, non-conservative products, and sources
    db::mutate<tmpl::list<sources_tag>>(
        box,
        [&div_fluxes](const auto ptr_sources) { *ptr_sources -= div_fluxes; });

    // Multiply by inverse temporal matrix for ADER-DG. This is done by
    // transposing from memory ordering xyztn to tnxyz, applying the matrix,
    // then transposing the result back to xyztn ordering.
    const auto& predictor_bulk = db::get<sources_tag>(*box);
    const size_t chunk_size = mesh.number_of_grid_points();
    const size_t number_of_chunks = predictor_bulk.size() / chunk_size;
    transpose(make_not_null(&div_fluxes), predictor_bulk, chunk_size,
              number_of_chunks);
    // TODO(nils): decide if we should generally support arbitrary quadrature
    const Matrix& ader_dg_matrix = AderDg::predictor_inverse_temporal_matrix<
        Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>(
        number_temporal_grid_points);
    db::mutate<tmpl::list<sources_tag>>(
        box, [number_of_chunks, chunk_size](const auto ptr_predictor_bulk) {
          dgemm_<true>('N', 'N', number_temporal_grid_points,
                       predictor_bulk.size() / number_temporal_grid_points,
                       number_temporal_grid_points, 1.0, ader_dg_matrix.data(),
                       number_temporal_grid_points, div_fluxes.data(),
                       number_temporal_grid_points, 0.0,
                       ptr_predictor_bulk->data(), number_temporal_grid_points);
        });
    transpose(make_not_null(&div_fluxes), predictor_bulk, number_of_chunks,
              chunk_size);

    // Add initial value to  predictor solution.
    // TODO(nils): if we are not using Gauss-Lobatto quadrature need to add
    // different correction.
    db::mutate<tmpl::list<sources_tag>>(
        box,
        [number_temporal_grid_points, number_spatial_grid_points](
            const auto ptr_predictor_bulk, const auto& initial_data) {
          DataVector spacetime_view{};
          DataVector initial_data_view{};
          for (size_t var = 0;
               var < initial_data.number_of_independent_components; ++var) {
            initial_data_view.set_data_ref(
                const_cast<std::remove_cv_t<decltype(initial_data)>>(
                    initial_data)
                        .data() +
                    var * number_spatial_grid_points,
                number_spatial_grid_points);
            for (size_t i = 0; i < number_temporal_grid_points; ++i) {
              spacetime_view.set_data_ref(
                  ptr_predictor_bulk->data() + i * number_spatial_grid_points +
                      var * number_spacetime_grid_points,
                  number_spatial_grid_points);
              spacetime_view += initial_data_view;
            }
          }
        },
        db::get<Tags::Corrector<variables_tag>>(*box));

    // Swap the source (which holds the updated predictor solution) with the old
    // predictor solution
    db::mutate<tmpl::list<variables_tag, sources_tag>>(
        box, [](const auto ptr_predictor_bulk_u,
                const auto ptr_predictor_bulk_source) {
          swap(*ptr_predictor_bulk_u, *ptr_predictor_bulk_source);
        });

    // Check if we converged
    iteration_number++;
    if (iteration_number >= max_iterations) {
    }
  }
}
}  // namespace AderDg
