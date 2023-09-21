// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

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

  using argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const Component* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    // auto al_gore = Parallel::local(
    //     Parallel::get_parallel_component<Component>(cache)[array_index]);
    // al_gore->set_terminate(true);
    ERROR("Not yet implemented");
    (void)cache, (void)array_index;
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
