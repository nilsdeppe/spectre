// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/AderDg/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace AderDg {
template <typename DivFluxesTagList, typename DerivativeTagList,
          typename FluxesTagList, size_t Dim, typename DerivativeFrame>
void divergence_time_independent_jacobian(
    const gsl::not_null<Variables<DivFluxesTagList>*> div_fluxes,
    const std::array<Variables<DerivativeTagList>, Dim>&
        logical_partial_derivs_of_fluxes_buffer,
    const Mesh<Dim>& mesh, const size_t number_temporal_pts,
    const Variables<FluxesTagList>& fluxes,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        spatial_inverse_jacobian) {
  logical_partial_derivatives<FluxesTagList>(
      logical_partial_derivs_of_fluxes_buffer, fluxes, mesh);

  if (UNLIKELY(div_fluxes->number_of_grid_points() !=
               fluxes.number_of_grid_points())) {
    div_fluxes->initialize(fluxes.number_of_grid_points());
  }

  const size_t number_spatial_pts = mesh.number_of_grid_points();
  tmpl::for_each<FluxesTagList>([&div_fluxes, &spatial_inverse_jacobian,
                                 &logical_partial_derivs_of_fluxes_buffer,
                                 number_temporal_pts,
                                 number_spatial_pts](auto tag) {
    using flux_tag = tmpl::type_from<decltype(tag)>;
    using div_flux_tag = ::Tags::div<flux_tag>;

    using first_index =
        tmpl::front<typename db::item_type<flux_tag>::index_list>;
    static_assert(
        cpp17::is_same_v<typename first_index::Frame, DerivativeFrame> and
            first_index::ul == UpLo::Up,
        "First index of tensor cannot be contracted with derivative "
        "because either it is in the wrong frame or it has the wrong "
        "valence");

    auto& divergence_of_flux = get<div_flux_tag>(*div_fluxes) = 0.0;
    DataVector div_flux_view{};
    DataVector logical_deriv_flux_view{};
    for (auto div_component_it = divergence_of_flux.begin();
         div_component_it != divergence_of_flux.end(); ++div_component_it) {
      const auto div_flux_indices =
          divergence_of_flux.get_tensor_index(div_component_it);
      for (size_t i0 = 0; i0 < Dim; ++i0) {
        const auto flux_indices = prepend(div_flux_indices, i0);
        // loop over time slices for this component
        for (size_t i = 0; i < number_temporal_pts; ++i) {
          div_flux_view.set_data_ref(
              div_component_it->data() + i * number_spatial_pts,
              number_spatial_pts);

          for (size_t d = 0; d < Dim; ++d) {
            logical_deriv_flux_view.set_data_ref(
                get<flux_tag>(
                    gsl::at(*logical_partial_derivs_of_fluxes_buffer, d))
                        .get(flux_indices)
                        .data() +
                    i * number_spatial_pts,
                number_spatial_pts);
            div_flux_view +=
                spatial_inverse_jacobian.get(d, i0) * logical_deriv_flux_view;
          }
        }
      }
    }
  });
}
}  // namespace AderDg
