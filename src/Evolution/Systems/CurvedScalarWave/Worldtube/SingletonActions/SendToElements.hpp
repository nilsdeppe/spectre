// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube::Actions {
/*!
 * \brief Sends the regular field to each element abutting the worldtube,
 * evaluated at the grid coordinates of each face
 *
 * \details h-refinement could be accounted for by sending the
 * coefficients of the internal solution directly and have each element evaluate
 * it for themselves.
 */
template <typename Metavariables>
struct SendToElements {
  static constexpr size_t Dim = Metavariables::volume_dim;
  using psi_tag = CurvedScalarWave::Tags::Psi;
  using dt_psi_tag = ::Tags::dt<CurvedScalarWave::Tags::Psi>;
  using di_psi_tag = ::Tags::deriv<CurvedScalarWave::Tags::Psi,
                                   tmpl::size_t<Dim>, Frame::Grid>;
  using tags_to_send = tmpl::list<psi_tag, dt_psi_tag, di_psi_tag>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& element_proxies = Parallel::get_parallel_component<
        typename Metavariables::dg_element_array>(cache);
    const auto& faces_grid_coords =
        get<Tags::ElementFacesGridCoordinates<Dim>>(box);

    for (const auto& [element_id, grid_coords] : faces_grid_coords) {
      const size_t grid_size = get<0>(grid_coords).size();
      Variables<tags_to_send> vars_to_send(grid_size);
      get(get<psi_tag>(vars_to_send)) = get<Tags::PsiMonopole>(box);
      get(get<dt_psi_tag>(vars_to_send)) =
          get<::Tags::dt<Tags::PsiMonopole>>(box);
      for (size_t i = 0; i < Dim; ++i) {
        // at 0th order the spatial derivative is just zero
        get<di_psi_tag>(vars_to_send).get(i) = 0.;
      }
      Parallel::receive_data<Tags::RegularFieldInbox<Dim>>(
          element_proxies[element_id], db::get<::Tags::TimeStepId>(box),
          std::move(vars_to_send));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace CurvedScalarWave::Worldtube::Actions