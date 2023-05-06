// Distributed under the MIT License.
// See LICENSE.txt for details.

template <class...>
struct td;

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel::Actions {
struct SetTerminateOnElement;
}  // namespace Parallel::Actions
namespace Parallel::Tags {
struct ElementLocationsPointerBase;
struct ThreadPoolPtrBase;
}  // namespace Parallel::Tags

namespace Events {
/// \ingroup EventsAndTriggersGroup
/// Sets the termination flag for the code to exit.
class Completion : public Event {
 public:
  /// \cond
  explicit Completion(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Completion);  // NOLINT
  /// \endcond

  using compute_tags_for_observation_box = tmpl::list<>;
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Sets the termination flag for the code to exit."};

  Completion() = default;

  using argument_tags = tmpl::list<::Tags::DataBox>;

  template <typename Metavariables, typename ArrayIndex, typename Component,
            typename DbTagsList>
  void operator()(const db::DataBox<DbTagsList>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const Component* const /*meta*/) const {
    if constexpr (db::tag_is_retrievable_v<Parallel::Tags::ThreadPoolPtrBase,
                                           db::DataBox<DbTagsList>>) {
      auto& thread_pool = db::get<Parallel::Tags::ThreadPoolPtrBase>(box);

      thread_pool->data()->at(array_index).set_terminate(true);
      thread_pool->increment_terminated_elements();
      // ERROR("We do not yet know how to handle thread pools :(");
    } else if constexpr (db::tag_is_retrievable_v<
                             Parallel::Tags::ElementLocationsPointerBase,
                             db::DataBox<DbTagsList>>) {
      Parallel::local_synchronous_action<
          Parallel::Actions::SetTerminateOnElement>(
          Parallel::get_parallel_component<Component>(cache),
          make_not_null(&cache), array_index);
    } else {
      (void)box;
      auto al_gore = Parallel::local(
          Parallel::get_parallel_component<Component>(cache)[array_index]);
      al_gore->set_terminate(true);
    }
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }
};
}  // namespace Events
