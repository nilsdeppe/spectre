// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace NewtonianEuler {
namespace Actions {
/// \ingroup InitializationGroup
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds:
///   * Metavariables::source_term_tag
///
/// - Removes: nothing
/// - Modifies: nothing
struct InitializeSourceTerm {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                typename db::DataBox<DbTagsList>::simple_item_tags,
                Initialization::Tags::InitialTime>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags =
        db::AddSimpleTags<typename Metavariables::source_term_tag>;
    using compute_tags = db::AddComputeTags<>;
    auto source_term =
        Parallel::get<typename Metavariables::initial_data_tag>(cache)
            .source_term();
    return std::make_tuple(
        Initialization::merge_into_databox<InitializeSourceTerm, simple_tags,
                                           compute_tags>(
            std::move(box), std::move(source_term)));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                typename db::DataBox<DbTagsList>::simple_item_tags,
                Initialization::Tags::InitialTime>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Could not find dependency 'Initialization::Tags::InitialTime' in "
        "DataBox");
  }
};

}  // namespace Actions
}  // namespace NewtonianEuler
