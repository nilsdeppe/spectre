// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <pup.h>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Options/Options.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class TimeDelta;
namespace Tags {
struct DataBox;
template <typename Tag>
struct Next;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace ChangeSlabSize_detail {
struct NewSlabSizeInbox
    : public Parallel::InboxInserters::MemberInsert<NewSlabSizeInbox> {
  using temporal_id = int64_t;
  using type = std::map<temporal_id, std::unordered_multiset<double>>;
};

// This inbox doesn't receive any data, it just counts messages (using
// the size of the multiset).  Whenever a message is sent to the
// NewSlabSizeInbox, another message is sent here synchronously, so
// the count here is the number of messages we expect in the
// NewSlabSizeInbox.
struct NumberOfExpectedMessagesInbox
    : public Parallel::InboxInserters::MemberInsert<
          NumberOfExpectedMessagesInbox> {
  using temporal_id = int64_t;
  using NoData = std::tuple<>;
  using type = std::map<temporal_id,
                        std::unordered_multiset<NoData, boost::hash<NoData>>>;
};

struct StoreNewSlabSize {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(const db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const int64_t slab_number,
                    const double slab_size) {
    Parallel::receive_data<ChangeSlabSize_detail::NewSlabSizeInbox>(
        *Parallel::local(Parallel::get_parallel_component<ParallelComponent>(
            cache)[array_index]),
        slab_number, slab_size);
  }
};
}  // namespace ChangeSlabSize_detail

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// Adjust the slab size based on previous executions of
/// Events::ChangeSlabSize
///
/// Uses:
/// - GlobalCache:
///   - Tags::TimeStepperBase
/// - DataBox:
///   - Tags::HistoryEvolvedVariables
///   - Tags::TimeStep
///   - Tags::TimeStepId
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Tags::Next<Tags::TimeStepId>
///   - Tags::TimeStep
///   - Tags::TimeStepId
struct ChangeSlabSize {
  using inbox_tags =
      tmpl::list<ChangeSlabSize_detail::NumberOfExpectedMessagesInbox,
                 ChangeSlabSize_detail::NewSlabSizeInbox>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    if (not time_step_id.is_at_slab_boundary()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    auto& message_count_inbox =
        tuples::get<ChangeSlabSize_detail::NumberOfExpectedMessagesInbox>(
            inboxes);
    if (message_count_inbox.empty() or
        message_count_inbox.begin()->first != time_step_id.slab_number()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    auto& new_slab_size_inbox =
        tuples::get<ChangeSlabSize_detail::NewSlabSizeInbox>(inboxes);

    const auto slab_number = time_step_id.slab_number();
    const auto number_of_changes = [&slab_number](const auto& inbox) -> size_t {
      if (inbox.empty()) {
        return 0;
      }
      if (inbox.begin()->first == slab_number) {
        return inbox.begin()->second.size();
      }
      ASSERT(inbox.begin()->first >= slab_number,
             "Received data for a change at slab " << inbox.begin()->first
             << " but it is already slab " << slab_number);
      return 0;
    };

    const size_t expected_messages = number_of_changes(message_count_inbox);
    const size_t received_messages = number_of_changes(new_slab_size_inbox);
    ASSERT(expected_messages >= received_messages,
           "Received " << received_messages << " size change messages at slab "
                       << slab_number << ", but only expected "
                       << expected_messages);
    if (expected_messages != received_messages) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    message_count_inbox.erase(message_count_inbox.begin());

    const double new_slab_size =
        *alg::min_element(new_slab_size_inbox.begin()->second);
    new_slab_size_inbox.erase(new_slab_size_inbox.begin());

    const TimeStepper& time_stepper = db::get<::Tags::TimeStepper<>>(box);

    // Sometimes time steppers need to run with a fixed step size.
    // This is generally at the start of an evolution when the history
    // is in an unusual state.
    if (not time_stepper.can_change_step_size(
            time_step_id, db::get<::Tags::HistoryEvolvedVariables<>>(box))) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const auto& current_step = db::get<::Tags::TimeStep>(box);
    const auto& current_slab = current_step.slab();

    const auto new_slab =
        current_step.is_positive()
            ? current_slab.with_duration_from_start(new_slab_size)
            : current_slab.with_duration_to_end(new_slab_size);

    const auto new_step = current_step.with_slab(new_slab);

    // We are at a slab boundary, so the substep is 0.
    const auto new_time_step_id =
        TimeStepId(time_step_id.time_runs_forward(), time_step_id.slab_number(),
                   time_step_id.step_time().with_slab(new_slab));

    const auto new_next_time_step_id =
        time_stepper.next_time_id(new_time_step_id, new_step);

    db::mutate<::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
               ::Tags::Next<::Tags::TimeStep>, ::Tags::TimeStepId,
               ::Tags::AdaptiveSteppingDiagnostics>(
        make_not_null(&box),
        [&new_next_time_step_id, &new_step, &new_time_step_id](
            const gsl::not_null<TimeStepId*> next_time_step_id,
            const gsl::not_null<TimeDelta*> time_step,
            const gsl::not_null<TimeDelta*> next_time_step,
            const gsl::not_null<TimeStepId*> local_time_step_id,
            const gsl::not_null<AdaptiveSteppingDiagnostics*> diags) {
          *next_time_step_id = new_next_time_step_id;
          *time_step = new_step;
          *next_time_step = new_step;
          *local_time_step_id = new_time_step_id;
          ++diags->number_of_slab_size_changes;
        });

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions

namespace Events {
/// \ingroup TimeGroup
/// %Trigger a slab size change.
///
/// The new size will be the minimum suggested by any of the provided
/// step choosers on any element.  This requires a global reduction,
/// so it is possible to delay the change until a later slab to avoid
/// a global synchronization.  The actual change is carried out by
/// Actions::ChangeSlabSize.
///
/// When running with global time-stepping, the slab size and step
/// size are the same, so this adjusts the step size used by the time
/// integration.  With local time-stepping this controls the interval
/// between times when the sequences of steps on all elements are
/// forced to align.
class ChangeSlabSize : public Event {
  using ReductionData = Parallel::ReductionData<
      Parallel::ReductionDatum<int64_t, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<double, funcl::Min<>>>;

 public:
  /// \cond
  explicit ChangeSlabSize(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ChangeSlabSize);  // NOLINT
  /// \endcond

  struct StepChoosers {
    static constexpr Options::String help = "Limits on slab size";
    using type =
        std::vector<std::unique_ptr<StepChooser<StepChooserUse::Slab>>>;
    static size_t lower_bound_on_size() { return 1; }
  };

  struct DelayChange {
    static constexpr Options::String help = "Slabs to wait before changing";
    using type = uint64_t;
  };

  using options = tmpl::list<StepChoosers, DelayChange>;
  static constexpr Options::String help =
      "Trigger a slab size change chosen by the provided step choosers.\n"
      "The actual changing of the slab size can be delayed until a later\n"
      "slab to improve parallelization.";

  ChangeSlabSize() = default;
  ChangeSlabSize(std::vector<std::unique_ptr<StepChooser<StepChooserUse::Slab>>>
                     step_choosers,
                 const uint64_t delay_change)
      : step_choosers_(std::move(step_choosers)), delay_change_(delay_change) {}

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags = tmpl::list<::Tags::TimeStepId, ::Tags::DataBox>;

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const TimeStepId& time_step_id,
                  const db::DataBox<DbTags>& box_for_step_choosers,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/) const {
    const auto next_changable_slab = time_step_id.is_at_slab_boundary()
                                         ? time_step_id.slab_number()
                                         : time_step_id.slab_number() + 1;
    const auto slab_to_change =
        next_changable_slab + static_cast<int64_t>(delay_change_);

    double desired_slab_size = std::numeric_limits<double>::infinity();
    bool synchronization_required = false;
    for (const auto& step_chooser : step_choosers_) {
      desired_slab_size = std::min(
          desired_slab_size,
          step_chooser
              ->desired_step(time_step_id.step_time().slab().duration().value(),
                             box_for_step_choosers)
              .first);
      // We must synchronize if any step chooser requires it, not just
      // the limiting one, because choosers requiring synchronization
      // can be limiting on some processors and not others.
      if (not synchronization_required) {
        synchronization_required = step_chooser->uses_local_data();
      }
    }

    // const auto& component_proxy =
    //     Parallel::get_parallel_component<ParallelComponent>(cache);
    // const auto& self_proxy = component_proxy[array_index];
    // // This message is sent synchronously, so it is guaranteed to
    // // arrive before the ChangeSlabSize action is called.
    // Parallel::receive_data<
    //     ChangeSlabSize_detail::NumberOfExpectedMessagesInbox>(
    //     *Parallel::local(self_proxy), slab_to_change,
    //     ChangeSlabSize_detail::NumberOfExpectedMessagesInbox::NoData{});
    // if (synchronization_required) {
    //   Parallel::contribute_to_reduction<
    //       ChangeSlabSize_detail::StoreNewSlabSize>(
    //       ReductionData(slab_to_change, desired_slab_size), self_proxy,
    //       component_proxy);
    // } else {
    //   Parallel::receive_data<ChangeSlabSize_detail::NewSlabSizeInbox>(
    //       *Parallel::local(self_proxy), slab_to_change, desired_slab_size);
    // }
    ERROR("Not yet implemented");
    (void)slab_to_change, (void)cache, (void)array_index;
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override {
    // This depends on the chosen StepChoosers, but they don't have a
    // way to report this information so we just return true to be
    // safe.
    return true;
  }

  template <typename F>
  void for_each_step_chooser(F&& f) const {
    for (const auto& step_chooser : step_choosers_) {
      f(*step_chooser);
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | step_choosers_;
    p | delay_change_;
  }

 private:
  std::vector<std::unique_ptr<StepChooser<StepChooserUse::Slab>>>
      step_choosers_;
  uint64_t delay_change_ = std::numeric_limits<uint64_t>::max();
};
}  // namespace Events
