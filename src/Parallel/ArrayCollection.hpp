#include <brigand/algorithms/transform.hpp>
#include <iostream>
#include <stdexcept>
#include "DataStructures/DataBox/Tag.hpp"
#include "Parallel/Invoke.hpp"

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <cmath>
#include <mutex>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroupDeclarations.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Spinlock.hpp"
#include "Parallel/Tags/ArrayIndex.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"

namespace Parallel {
namespace detail {
// template <typename Metavariable, bool = false>
// constexpr bool use_z_order_distribution  = false;

// template <typename Metavariables>
// constexpr bool use_z_order_distribution<
//     Metavariables, Metavariables::use_z_order_distribution> = false;

CREATE_HAS_STATIC_MEMBER_VARIABLE(use_z_order_distribution)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(use_z_order_distribution)
CREATE_HAS_STATIC_MEMBER_VARIABLE(local_time_stepping)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(local_time_stepping)
}  // namespace detail

template <size_t Dim, class Metavariables, class PhaseDepActionList>
struct DgElementCollection;

template <typename ReceiveTag, typename ReceiveDataType, size_t Dim>
struct CollectionMessage {
  typename ReceiveTag::temporal_id instance{};
  ElementId<Dim> element_id{};
  ReceiveDataType data{};
  bool enable_if_disabled{false};
};

/// This is effective the "distributed object"
template <size_t Dim, typename Metavariables, typename PhaseDepActionList,
          typename SimpleTagsFromOptions>
class DgElementArrayMember;

namespace Tags {
struct ElementCollectionBase : db::BaseTag {};

template <size_t Dim, class Metavariables, class PhaseDepActionList,
          typename SimpleTagsFromOptions>
struct ElementCollection : db::SimpleTag, ElementCollectionBase {
  using type = std::unordered_map<
      ElementId<Dim>,
      DgElementArrayMember<Dim, Metavariables, PhaseDepActionList,
                           SimpleTagsFromOptions>>;
};
template <size_t Dim>
struct ElementLocations : db::SimpleTag {
  using type = std::unordered_map<ElementId<Dim>, size_t>;
};

struct ElementLocationsPointerBase : db::BaseTag {};

template <size_t Dim>
struct ElementLocationsPointer : db::SimpleTag, ElementLocationsPointerBase {
  using type = std::unordered_map<ElementId<Dim>, size_t>*;
};

struct NumberOfElementsTerminated : db::SimpleTag {
  using type = size_t;
};
}  // namespace Tags


template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
class DgElementArrayMember<Dim, Metavariables,
                           tmpl::list<PhaseDepActionListsPack...>,
                           SimpleTagsFromOptions> {
 public:
  using ParallelComponent =
      DgElementCollection<Dim, Metavariables,
                          tmpl::list<PhaseDepActionListsPack...>>;

  /// List of Actions in the order that generates the DataBox types
  using all_actions_list = tmpl::flatten<
      tmpl::list<typename PhaseDepActionListsPack::action_list...>>;
  /// The metavariables class passed to the Algorithm
  using metavariables = Metavariables;
  /// List off all the Tags that can be received into the Inbox
  using inbox_tags_list = Parallel::get_inbox_tags<all_actions_list>;
  using phase_dependent_action_lists = tmpl::list<PhaseDepActionListsPack...>;
  using all_cache_tags = get_const_global_cache_tags<metavariables>;

  using databox_type = db::compute_databox_type<tmpl::flatten<tmpl::list<
      Tags::MetavariablesImpl<metavariables>,
      Tags::ArrayIndexImpl<ElementId<Dim>>, Tags::ElementLocationsPointer<Dim>,
      Tags::GlobalCacheProxy<metavariables>, SimpleTagsFromOptions,
      Tags::GlobalCacheImplCompute<metavariables>,
      Tags::ResourceInfoReference<metavariables>,
      db::wrap_tags_in<Tags::FromGlobalCache, all_cache_tags>,
      Algorithm_detail::get_pdal_simple_tags<phase_dependent_action_lists>,
      Algorithm_detail::get_pdal_compute_tags<phase_dependent_action_lists>>>>;

  using inbox_type = tuples::tagged_tuple_from_typelist<inbox_tags_list>;

  DgElementArrayMember() = default;

  template <class... InitializationTags>
  DgElementArrayMember(
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache_proxy,
      tuples::TaggedTuple<InitializationTags...> initialization_items,
      ElementId<Dim> element_id,
      gsl::not_null<std::unordered_map<ElementId<Dim>, size_t>*>
          element_locations,
      gsl::not_null<size_t*> number_of_elements_terminated,
      gsl::not_null<Parallel::NodeLock*> nodegroup_lock);

  /// \cond
  ~DgElementArrayMember() = default;

  DgElementArrayMember(const DgElementArrayMember& /*unused*/) = default;
  DgElementArrayMember& operator=(const DgElementArrayMember& /*unused*/) =
      default;
  DgElementArrayMember(DgElementArrayMember&& /*unused*/) = default;
  DgElementArrayMember& operator=(DgElementArrayMember&& /*unused*/) = default;
  /// \endcond

  /// Start execution of the phase-dependent action list in `next_phase`. If
  /// `next_phase` has already been visited, execution will resume at the point
  /// where the previous execution of the same phase left off.
  void start_phase(Parallel::Phase next_phase);

  /// Get the current phase
  Phase phase() const { return phase_; }

  /// Tell the Algorithm it should no longer execute the algorithm. This does
  /// not mean that the execution of the program is terminated, but only that
  /// the algorithm has terminated. An algorithm can be restarted by passing
  /// `true` as the second argument to the `receive_data` method or by calling
  /// perform_algorithm(true).
  void set_terminate(const bool terminate) {
    std::lock_guard nodegroup_lock(*nodegroup_lock_);
    if (not terminate_ and terminate) {
      ++(*number_of_elements_terminated_);
    } else if (terminate_ and not terminate) {
      --(*number_of_elements_terminated_);
    } else {
      ASSERT(terminate_ == terminate,
             "The DG element with id "
                 << element_id_ << " currently has termination status "
                 << terminate_ << " and is being set to " << terminate
                 << ". This is an internal inconsistency problem.");
    }
    terminate_ = terminate;
  }

  /// Check if an algorithm should continue being evaluated
  bool get_terminate() const { return terminate_; }

  size_t algorithm_step() const { return algorithm_step_; }

  const auto& databox() const { return box_; }

  /// Start evaluating the algorithm until it is stopped by an action.
  void perform_algorithm();

  template <typename ThisAction, typename PhaseIndex, typename DataBoxIndex>
  bool invoke_iterable_action();

  /// Print the expanded type aliases
  std::string print_types() const;

  /// Print the current state of the algorithm
  std::string print_state() const;

  /// Print the current contents of the inboxes
  std::string print_inbox() const;

  /// Print the current contents of the DataBox
  std::string print_databox() const;

  /// Get read access to all the inboxes
  // const auto& get_inboxes() const { return inboxes_; }

  void receive_data();

  auto& inboxes() { return inboxes_; }
  const auto& inboxes() const { return inboxes_; }

  /// The inbox_lock_ only locks the inbox, nothing else. The inbox is unsafe
  /// to access without the lock.
  ///
  /// This should always be managed by `std::unique_lock` or `std::lock_guard`.
  auto& inbox_lock() { return inbox_lock_; }

  /// Locks the element, except for the inbox, which is guarded by the
  /// inbox_lock_.
  ///
  /// This should always be managed by `std::unique_lock` or `std::lock_guard`.
  auto& element_lock() { return element_lock_; }

  void pup(PUP::er& p) {
    p | global_cache_proxy_;
    p | performing_action_;
    p | phase_;
    p | phase_bookmarks_;
    p | algorithm_step_;
    // TODO: what to do with the locks?
    p | terminate_;
    p | halt_algorithm_until_next_phase_;
    p | deadlock_analysis_next_iterable_action_;
    p | element_id_;
  }

 private:
  size_t number_of_actions_in_phase(Parallel::Phase phase) const;

  // After catching an exception, shutdown the simulation
  void initiate_shutdown(const std::exception& exception);

  template <typename PhaseDepActions, size_t... Is>
  bool iterate_over_actions(std::index_sequence<Is...> /*meta*/);

  Parallel::CProxy_GlobalCache<Metavariables> global_cache_proxy_;
  bool performing_action_ = false;
  Parallel::Phase phase_{Parallel::Phase::Initialization};
  std::unordered_map<Parallel::Phase, size_t> phase_bookmarks_{};
  std::size_t algorithm_step_ = 0;
  Parallel::NodeLock inbox_lock_;
  Parallel::NodeLock element_lock_;
  // TODO: These Spinlocks are fast but non-copyable and non-movable because
  // they use std::atomic
  // Parallel::Spinlock inbox_lock_;
  // Parallel::Spinlock element_lock_;

  size_t* number_of_elements_terminated_{0};
  Parallel::NodeLock* nodegroup_lock_{nullptr};

  bool terminate_{true};
  bool halt_algorithm_until_next_phase_{false};

  // Records the name of the next action to be called so that during deadlock
  // analysis we can print this out.
  std::string deadlock_analysis_next_iterable_action_{};

  databox_type box_;
  inbox_type inboxes_{};
  ElementId<Dim> element_id_;
};

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
template <class... InitializationTags>
DgElementArrayMember<Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
                     SimpleTagsFromOptions>::
    DgElementArrayMember(
        const Parallel::CProxy_GlobalCache<Metavariables>& global_cache_proxy,
        tuples::TaggedTuple<InitializationTags...> initialization_items,
        ElementId<Dim> element_id,
        const gsl::not_null<std::unordered_map<ElementId<Dim>, size_t>*>
            element_locations,
        const gsl::not_null<size_t*> number_of_elements_terminated,
        const gsl::not_null<Parallel::NodeLock*> nodegroup_lock) {
  (void)initialization_items;  // avoid potential compiler warnings if unused
  global_cache_proxy_ = global_cache_proxy;
  element_id_ = std::move(element_id);
  number_of_elements_terminated_ = number_of_elements_terminated;
  nodegroup_lock_ = nodegroup_lock;
  ::Initialization::mutate_assign<
      tmpl::list<Tags::ArrayIndex, Tags::GlobalCacheProxy<Metavariables>,
                 Tags::ElementLocationsPointer<Dim>, InitializationTags...>>(
      make_not_null(&box_), element_id_, global_cache_proxy_,
      element_locations.get(),
      std::move(get<InitializationTags>(initialization_items))...);
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
void DgElementArrayMember<
    Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
    SimpleTagsFromOptions>::start_phase(const Parallel::Phase next_phase) {
  try {
    // terminate should be true since we exited a phase previously.
    if (not get_terminate() and not halt_algorithm_until_next_phase_) {
      ERROR(
          "An algorithm must always be set to terminate at the beginning of a "
          "phase. Since this is not the case the previous phase did not end "
          "correctly. The previous phase is: "
          << phase_ << " and the next phase is: " << next_phase
          << ", The termination flag is: " << get_terminate()
          << ", and the halt flag is: " << halt_algorithm_until_next_phase_);
    }
    // set terminate to true if there are no actions in this PDAL
    set_terminate(number_of_actions_in_phase(next_phase) == 0);

    // Ideally, we'd set the bookmarks as we are leaving a phase, but there is
    // no 'clean-up' code that we run when departing a phase, so instead we set
    // the bookmark for the previous phase (still stored in `phase_` at this
    // point), before we update the member variable `phase_`.
    // Then, after updating `phase_`, we check if we've ever stored a bookmark
    // for the new phase previously. If so, we start from where we left off,
    // otherwise, start from the beginning of the action list.
    phase_bookmarks_[phase_] = algorithm_step_;
    phase_ = next_phase;
    if (phase_bookmarks_.count(phase_) != 0) {
      algorithm_step_ = phase_bookmarks_.at(phase_);
    } else {
      algorithm_step_ = 0;
    }
    halt_algorithm_until_next_phase_ = false;
    perform_algorithm();
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
void DgElementArrayMember<Dim, Metavariables,
                          tmpl::list<PhaseDepActionListsPack...>,
                          SimpleTagsFromOptions>::perform_algorithm() {
  try {
    if (performing_action_ or get_terminate() or
        halt_algorithm_until_next_phase_) {
      return;
    }
    const auto invoke_for_phase = [this](auto phase_dep_v) {
      using PhaseDep = decltype(phase_dep_v);
      constexpr Parallel::Phase phase = PhaseDep::phase;
      using actions_list = typename PhaseDep::action_list;
      if (phase_ == phase) {
        while (
            tmpl::size<actions_list>::value > 0 and not get_terminate() and
            not halt_algorithm_until_next_phase_ and
            iterate_over_actions<PhaseDep>(
                std::make_index_sequence<tmpl::size<actions_list>::value>{})) {
        }
        tmpl::for_each<actions_list>([this](auto action_v) {
          using action = tmpl::type_from<decltype(action_v)>;
          if (algorithm_step_ == tmpl::index_of<actions_list, action>::value) {
            deadlock_analysis_next_iterable_action_ =
                pretty_type::name<action>();
          }
        });
      }
    };
    // Loop over all phases, once the current phase is found we perform the
    // algorithm in that phase until we are no longer able to because we are
    // waiting on data to be sent or because the algorithm has been marked as
    // terminated.
    EXPAND_PACK_LEFT_TO_RIGHT(invoke_for_phase(PhaseDepActionListsPack{}));
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
std::string
DgElementArrayMember<Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
                     SimpleTagsFromOptions>::print_databox() const {
  std::ostringstream os;
  os << "box_:\n" << box_;
  return os.str();
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
template <typename PhaseDepActions, size_t... Is>
bool DgElementArrayMember<Dim, Metavariables,
                          tmpl::list<PhaseDepActionListsPack...>,
                          SimpleTagsFromOptions>::
    iterate_over_actions(const std::index_sequence<Is...> /*meta*/) {
  bool take_next_action = true;
  const auto helper = [this, &take_next_action](auto iteration) {
    constexpr size_t iter = decltype(iteration)::value;
    if (not(take_next_action and not terminate_ and
            not halt_algorithm_until_next_phase_ and algorithm_step_ == iter)) {
      return;
    }
    using actions_list = typename PhaseDepActions::action_list;
    using this_action = tmpl::at_c<actions_list, iter>;

    constexpr size_t phase_index =
        tmpl::index_of<phase_dependent_action_lists, PhaseDepActions>::value;
    performing_action_ = true;
    ++algorithm_step_;
    // While the overhead from using the local entry method to enable
    // profiling is fairly small (<2%), we still avoid it when we aren't
    // tracing.
    // #ifdef SPECTRE_CHARM_PROJECTIONS
    //     if constexpr (Parallel::is_array<parallel_component>::value) {
    //       if (not this->thisProxy[array_index_]
    //                   .template invoke_iterable_action<
    //                       this_action, std::integral_constant<size_t,
    //                       phase_index>, std::integral_constant<size_t,
    //                       iter>>()) {
    //         take_next_action = false;
    //         --algorithm_step_;
    //       }
    //     } else {
    // #endif  // SPECTRE_CHARM_PROJECTIONS
    if (not invoke_iterable_action<this_action,
                                   std::integral_constant<size_t, phase_index>,
                                   std::integral_constant<size_t, iter>>()) {
      take_next_action = false;
      --algorithm_step_;
    }
    // #ifdef SPECTRE_CHARM_PROJECTIONS
    //     }
    // #endif  // SPECTRE_CHARM_PROJECTIONS
    performing_action_ = false;
    // Wrap counter if necessary
    if (algorithm_step_ >= tmpl::size<actions_list>::value) {
      algorithm_step_ = 0;
    }
  };
  // In case of no Actions avoid compiler warning.
  (void)helper;
  // This is a template for loop for Is
  EXPAND_PACK_LEFT_TO_RIGHT(helper(std::integral_constant<size_t, Is>{}));
  return take_next_action;
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
template <typename ThisAction, typename PhaseIndex, typename DataBoxIndex>
bool DgElementArrayMember<Dim, Metavariables,
                          tmpl::list<PhaseDepActionListsPack...>,
                          SimpleTagsFromOptions>::invoke_iterable_action() {
  using phase_dep_action =
      tmpl::at_c<phase_dependent_action_lists, PhaseIndex::value>;
  using actions_list = typename phase_dep_action::action_list;

#ifdef SPECTRE_CHARM_PROJECTIONS
  if constexpr (Parallel::is_array<parallel_component>::value) {
    (void)Parallel::charmxx::RegisterInvokeIterableAction<
        ParallelComponent, ThisAction, PhaseIndex, DataBoxIndex>::registrar;
  }
#endif  // SPECTRE_CHARM_PROJECTIONS

  const auto& [requested_execution, next_action_step] = ThisAction::apply(
      box_, inboxes_, *Parallel::local_branch(global_cache_proxy_),
      std::as_const(element_id_), actions_list{},
      std::add_pointer_t<ParallelComponent>{});

  if (next_action_step.has_value()) {
    ASSERT(
        AlgorithmExecution::Retry != requested_execution,
        "Switching actions on Retry doesn't make sense. Specify std::nullopt "
        "as the second argument of the iterable action return type");
    algorithm_step_ = next_action_step.value();
  }

  switch (requested_execution) {
    case AlgorithmExecution::Continue:
      return true;
    case AlgorithmExecution::Retry:
      return false;
    case AlgorithmExecution::Pause:
      terminate_ = true;
      return true;
    case AlgorithmExecution::Halt:
      halt_algorithm_until_next_phase_ = true;
      terminate_ = true;
      return true;
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("No case for a Parallel::AlgorithmExecution with integral value "
            << static_cast<std::underlying_type_t<AlgorithmExecution>>(
                   requested_execution)
            << "\n");
      // LCOV_EXCL_STOP
  }
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
size_t DgElementArrayMember<
    Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
    SimpleTagsFromOptions>::number_of_actions_in_phase(const Parallel::Phase
                                                           phase) const {
  size_t number_of_actions = 0;
  const auto helper = [&number_of_actions, phase](auto pdal_v) {
    if (pdal_v.phase == phase) {
      number_of_actions = pdal_v.number_of_actions;
    }
  };
  EXPAND_PACK_LEFT_TO_RIGHT(helper(PhaseDepActionListsPack{}));
  return number_of_actions;
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
void DgElementArrayMember<
    Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
    SimpleTagsFromOptions>::initiate_shutdown(const std::exception& exception) {
  // In order to make it so that we can later run other actions for cleanup
  // (e.g. dumping data) we need to make sure that we enable running actions
  // again
  performing_action_ = false;
  // Send message to `Main` that we received an exception and set termination.
  auto* global_cache = Parallel::local_branch(global_cache_proxy_);
  if (UNLIKELY(global_cache == nullptr)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    CkError(
        "Global cache pointer is null. This is an internal inconsistency "
        "error. Please file an issue.");
    sys::abort("");
  }
  auto main_proxy = global_cache->get_main_proxy();
  if (UNLIKELY(not main_proxy.has_value())) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    CkError(
        "The main proxy has not been set in the global cache when terminating "
        "the component. This is an internal inconsistency error. Please file "
        "an issue.");
    sys::abort("");
  }
  const std::string message = MakeString{}
                              << "Message: " << exception.what() << "\nType: "
                              << pretty_type::get_runtime_type_name(exception);
  main_proxy.value().add_exception_message(message);
  set_terminate(true);
}

namespace Actions {
/// Always invoked on the local component.
///
/// In the future we can use this to try and elide Charm++ calls when the
/// receiver is local.

struct SetTerminateOnElement {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, size_t Dim>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const size_t& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> node_lock,
                    const ElementId<Dim>& element_to_terminate) {
    auto& element = [&node_lock, &box, &element_to_terminate ]() -> auto& {
      std::lock_guard node_guard(*node_lock);
      return db::mutate<Tags::ElementCollectionBase>(
          make_not_null(&box),
          [&element_to_terminate](const auto element_collection_ptr) -> auto& {
            try {
              return element_collection_ptr->at(element_to_terminate);
            } catch (const std::out_of_range&) {
              ERROR("Could not find element with ID " << element_to_terminate
                                                      << " on node");
            }
          });
    }
    ();
    element.set_terminate(true);

    std::lock_guard node_guard(*node_lock);
    const size_t number_of_elements = db::mutate<Tags::ElementCollectionBase>(
        make_not_null(&box),
        [&element_to_terminate](const auto element_collection_ptr) {
          try {
            return element_collection_ptr->size();
          } catch (const std::out_of_range&) {
            ERROR("Could not find element with ID " << element_to_terminate
                                                    << " on node");
          }
        });
    if (db::get<Tags::NumberOfElementsTerminated>(box) == number_of_elements) {
      auto* local_branch = Parallel::local_branch(
          Parallel::get_parallel_component<ParallelComponent>(cache));
      ASSERT(local_branch != nullptr,
             "The local branch is nullptr, which is inconsistent");
      local_branch->set_terminate(true);
    }
  }
};

/// \brief Receive data for a specific element.
struct ReceiveDataForElement {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, typename ReceiveData,
            typename ReceiveTag, size_t Dim>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> node_lock,
                    const ReceiveTag& /*meta*/,
                    const ElementId<Dim>& element_to_execute_on,
                    typename ReceiveTag::temporal_id instance,
                    ReceiveData receive_data) {
    auto& element = [&node_lock, &box, &element_to_execute_on ]() -> auto& {
      std::lock_guard node_guard(*node_lock);
      return db::mutate<Tags::ElementCollectionBase>(
          make_not_null(&box),
          [&element_to_execute_on](const auto element_collection_ptr) -> auto& {
            try {
              return element_collection_ptr->at(element_to_execute_on);
            } catch (const std::out_of_range&) {
              ERROR("Could not find element with ID " << element_to_execute_on
                                                      << " on node");
            }
          });
    }
    ();
    // {
    //   std::lock_guard inbox_lock(element.inbox_lock());
    //   ReceiveTag::insert_into_inbox(
    //       make_not_null(&tuples::get<ReceiveTag>(element.inboxes())),
    //       instance,
    //       std::move(receive_data));
    // }
    {
      // TODO: we should really do a `try_lock` and then have an integer
      // guarded by the inbox lock that counts the number of missed messages.
      std::lock_guard element_lock(element.element_lock());
      ReceiveTag::insert_into_inbox(
          make_not_null(&tuples::get<ReceiveTag>(element.inboxes())), instance,
          std::move(receive_data));
      element.perform_algorithm();
    }
  }
};

/// \brief Starts the current nodegroup phase on the element to execute.
struct StartPhaseOnElement {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, size_t Dim>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> node_lock,
                    const ElementId<Dim>& element_to_execute) {
    const Phase current_phase =
        Parallel::local_branch(
            Parallel::get_parallel_component<ParallelComponent>(cache))
            ->phase();
    auto& element = [&node_lock, &box, &element_to_execute ]() -> auto& {
      std::lock_guard node_guard(*node_lock);
      return db::mutate<Tags::ElementCollectionBase>(
          make_not_null(&box),
          [&element_to_execute](const auto element_collection_ptr) -> auto& {
            try {
              return element_collection_ptr->at(element_to_execute);
            } catch (const std::out_of_range&) {
              ERROR("Could not find element with ID " << element_to_execute
                                                      << " on node");
            }
          });
    }
    ();
    const std::lock_guard element_lock(element.element_lock());
    element.start_phase(current_phase);
  }
};

/// \brief Starts the next phase on the nodegroup and calls
/// `StartPhaseOnElement` for each element on the node.
struct StartPhaseOnNodegroup {
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t my_node = Parallel::my_node<size_t>(cache);
    auto proxy_to_this_node =
        Parallel::get_parallel_component<ParallelComponent>(cache)[my_node];
    for (const auto& [element_id, element] :
         db::get<Tags::ElementCollectionBase>(box)) {
      Parallel::threaded_action<StartPhaseOnElement>(proxy_to_this_node,
                                                     element_id);
    }
    return {Parallel::AlgorithmExecution::Halt, std::nullopt};
  }
};

struct SpawnInitializeElementsInCollection {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double /*unused_but_we_needed_to_reduce_something*/) {
    auto my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    db::mutate<Tags::ElementCollectionBase>(
        make_not_null(&box), [&my_proxy](const auto element_collection_ptr) {
          for (auto& [element_id, element] : *element_collection_ptr) {
            Parallel::threaded_action<StartPhaseOnElement>(my_proxy,
                                                           element_id);
          }
        });
  }
};

template <size_t Dim, class Metavariables, class PhaseDepActionList,
          typename SimpleTagsFromOptions>
struct InitializeElementCollection {  // TODO: CreateElementCollection
  using simple_tags =
      tmpl::list<Tags::ElementCollection<Dim, Metavariables, PhaseDepActionList,
                                         SimpleTagsFromOptions>,
                 Tags::ElementLocations<Dim>, Tags::NumberOfElementsTerminated>;
  using compute_tags = tmpl::list<>;

  using return_tag_list = tmpl::append<simple_tags, compute_tags>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& local_cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const std::unordered_set<size_t> procs_to_ignore{};

    const auto& domain = Parallel::get<domain::Tags::Domain<Dim>>(local_cache);
    const auto& initial_refinement_levels =
        get<domain::Tags::InitialRefinementLevels<Dim>>(box);
    const auto& initial_extents = get<domain::Tags::InitialExtents<Dim>>(box);
    const auto& quadrature = get<evolution::dg::Tags::Quadrature>(box);

    bool use_z_order_distribution = true;
    if constexpr (detail::has_use_z_order_distribution_v<Metavariables>) {
      use_z_order_distribution = Metavariables::use_z_order_distribution;
    }

    bool local_time_stepping = false;
    if constexpr (detail::has_local_time_stepping_v<Metavariables>) {
      local_time_stepping = Metavariables::local_time_stepping;
    }

    const size_t number_of_procs =
        Parallel::number_of_procs<size_t>(local_cache);
    const size_t number_of_nodes =
        Parallel::number_of_nodes<size_t>(local_cache);
    const size_t num_of_procs_to_use = number_of_procs - procs_to_ignore.size();

    const auto& blocks = domain.blocks();

    const std::unordered_map<ElementId<Dim>, double> element_costs =
        domain::get_element_costs(
            blocks, initial_refinement_levels, initial_extents,
            local_time_stepping
                ? domain::ElementWeight::NumGridPointsAndGridSpacing
                : domain::ElementWeight::NumGridPoints,
            quadrature);
    const domain::BlockZCurveProcDistribution<Dim> element_distribution{
        element_costs,   num_of_procs_to_use, blocks, initial_refinement_levels,
        initial_extents, procs_to_ignore};

    // Will be used to print domain diagnostic info
    std::vector<size_t> elements_per_core(number_of_procs, 0_st);
    std::vector<size_t> elements_per_node(number_of_nodes, 0_st);
    std::vector<size_t> grid_points_per_core(number_of_procs, 0_st);
    std::vector<size_t> grid_points_per_node(number_of_nodes, 0_st);

    size_t which_proc = 0;

    const size_t total_num_elements = [&blocks, &initial_refinement_levels]() {
      size_t result = 0;
      for (const auto& block : blocks) {
        const auto& initial_ref_levs = initial_refinement_levels[block.id()];
        for (const size_t ref_lev : initial_ref_levs) {
          result += two_to_the(ref_lev);
        }
      }
      return result;
    }();
    std::vector<ElementId<Dim>> my_elements{};
    my_elements.reserve(
        static_cast<size_t>(std::round(static_cast<double>(total_num_elements) /
                                       static_cast<double>(number_of_nodes))));
    std::unordered_map<ElementId<Dim>, size_t> node_of_elements{};
    const size_t my_node = Parallel::my_node<size_t>(local_cache);

    for (const auto& block : blocks) {
      const auto& initial_ref_levs = initial_refinement_levels[block.id()];
      const size_t grid_points_per_element = alg::accumulate(
          initial_extents[block.id()], 1_st, std::multiplies<>());

      const std::vector<ElementId<Dim>> element_ids =
          initial_element_ids(block.id(), initial_ref_levs);

      if (use_z_order_distribution) {
        for (const auto& element_id : element_ids) {
          const size_t target_proc =
              element_distribution.get_proc_for_element(element_id);
          const size_t target_node =
              Parallel::node_of<size_t>(target_proc, local_cache);
          node_of_elements.insert(std::pair{element_id, target_node});
          if (target_node == my_node) {
            my_elements.push_back(element_id);
          }
          ++elements_per_core[target_proc];
          ++elements_per_node[target_node];
          grid_points_per_core[target_proc] += grid_points_per_element;
          grid_points_per_node[target_node] += grid_points_per_element;
        }
      } else {
        for (size_t i = 0; i < element_ids.size(); ++i) {
          while (procs_to_ignore.find(which_proc) != procs_to_ignore.end()) {
            which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
          }

          const size_t target_node =
              Parallel::node_of<size_t>(which_proc, local_cache);
          node_of_elements.insert(std::pair{element_ids[i], target_node});
          if (target_node == my_node) {
            my_elements.push_back(ElementId<Dim>(element_ids[i]));
          }
          ++elements_per_core[which_proc];
          ++elements_per_node[target_node];
          grid_points_per_core[which_proc] += grid_points_per_element;
          grid_points_per_node[target_node] += grid_points_per_element;

          which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
        }
      }
    }

    tuples::tagged_tuple_from_typelist<SimpleTagsFromOptions>
        initialization_items = db::copy_items<SimpleTagsFromOptions>(box);

    const gsl::not_null<Parallel::NodeLock*> node_lock = make_not_null(
        &Parallel::local_branch(
             Parallel::get_parallel_component<ParallelComponent>(local_cache))
             ->get_node_lock());
    db::mutate<Tags::ElementLocations<Dim>,
               Tags::ElementCollection<Dim, Metavariables, PhaseDepActionList,
                                       SimpleTagsFromOptions>,
               Tags::NumberOfElementsTerminated>(
        make_not_null(&box),
        [&local_cache, &initialization_items, &my_elements, &node_of_elements,
         &node_lock](
            const auto element_locations_ptr, const auto collection_ptr,
            const gsl::not_null<size_t*> number_of_elements_terminated) {
          const auto serialized_initialization_items =
              serialize(initialization_items);
          *element_locations_ptr = std::move(node_of_elements);
          for (const auto& element_id : my_elements) {
            if (not collection_ptr
                        ->insert(std::pair{
                            element_id,
                            DgElementArrayMember<Dim, Metavariables,
                                                 PhaseDepActionList,
                                                 SimpleTagsFromOptions>{
                                local_cache.get_this_proxy(),
                                deserialize<tuples::tagged_tuple_from_typelist<
                                    SimpleTagsFromOptions>>(
                                    serialized_initialization_items.data()),
                                element_id, element_locations_ptr,
                                number_of_elements_terminated, node_lock}})
                        .second) {
              ERROR("Failed to insert element with ID: " << element_id);
            }
            (*number_of_elements_terminated) = 0;
            if (collection_ptr->at(element_id).get_terminate() == true) {
              ++(*number_of_elements_terminated);
            }
          }
        });

    Parallel::contribute_to_reduction<SpawnInitializeElementsInCollection>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<int, funcl::AssertEqual<>>>{0},
        Parallel::get_parallel_component<ParallelComponent>(
            local_cache)[my_node],
        Parallel::get_parallel_component<ParallelComponent>(local_cache));

    // Parallel::printf(
    //     "\n%s\n", domain::diagnostic_info(
    //                   domain, local_cache, elements_per_core,
    //                   elements_per_node, grid_points_per_core,
    //                   grid_points_per_node));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions

/// \brief Metafunction that turns the PhaseActions into the right one for
/// each phase for the nodegroup.
template <typename OnePhaseActions>
struct TransformPdalToStartPhaseOnNodegroup {
  using type = tmpl::conditional_t<
      OnePhaseActions::phase == Parallel::Phase::Initialization, tmpl::list<>,
      Parallel::PhaseActions<OnePhaseActions::phase,
                             tmpl::list<Actions::StartPhaseOnNodegroup>>>;
};

template <class...>
struct td;

template <size_t Dim, class Metavariables, class PhaseDepActionList>
struct DgElementCollection {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using metavariables = Metavariables;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<PhaseDepActionList>>;

  // td<tmpl::flatten<tmpl::transform<
  //     PhaseDepActionList, TransformPdalToStartPhaseOnNodegroup<tmpl::_1>>>>
  //     aoeu;

  using phase_dependent_action_list =
      tmpl::append<tmpl::list<
          Parallel::PhaseActions<
              Parallel::Phase::Initialization,
              tmpl::list<Actions::InitializeElementCollection<
                             Dim, Metavariables, PhaseDepActionList,
                             simple_tags_from_options>,
                         Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<Parallel::Phase::InitializeTimeStepperHistory,
                                 tmpl::list<Actions::StartPhaseOnNodegroup>>,

          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<Actions::StartPhaseOnNodegroup>>,

          Parallel::PhaseActions<Parallel::Phase::Evolve,
                                 tmpl::list<Actions::StartPhaseOnNodegroup>>>

                   // tmpl::flatten<tmpl::transform<
                   //       PhaseDepActionList,
                   //       TransformPdalToStartPhaseOnNodegroup<tmpl::_1>>>
                   >;
  using const_global_cache_tags = tmpl::remove_duplicates<tmpl::append<
      tmpl::list<::domain::Tags::Domain<Dim>>,
      typename Parallel::detail::get_const_global_cache_tags_from_pdal<
          PhaseDepActionList>::type>>;
  using mutable_global_cache_tags =
      typename Parallel::detail::get_mutable_global_cache_tags_from_pdal<
          PhaseDepActionList>::type;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    Parallel::printf("%s\n", next_phase);
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<DgElementCollection>(local_cache)
        .start_phase(next_phase);
  }
};

// template <size_t Dim>
// class ElementCollection {
//  public:
//   ElementCollection();

//   /// \brief Threaded action that receives neighbor data and calls the next
//   /// iterable action on the specified element.
//   template <typename ReceiveTag, typename ReceiveDataType>
//   void receive_data(CollectionMessage<ReceiveTag, ReceiveDataType, Dim> data)
//   {
//     // (void)Parallel::charmxx::RegisterReceiveData<ParallelComponent,
//     // ReceiveTag,
//     //                                              false>::registrar;
//     const ElementId<Dim> element_id = data.element_id;  // TODO: I don't know
//                                                         // if I need this.
//     try {
//       array_elements_.at(element_id).receive_data(std::move(data));
//     } catch (...) {
//       // TODO: catch for real.
//     }
//   }

//  private:
//   Parallel::CProxy_GlobalCache<metavariables> global_cache_proxy_{};

//   std::unordered_map<ElementId<Dim>, DgElementArrayMember<Dim>>
//       array_elements_{};
// };

// template <typename ArrayId, typename ArrayElement>
// class ArrayCollection {
//  public:

//  private:
//   std::unordered_map<ArrayId, ArrayElement> array_elements_{};
// };

}  // namespace Parallel
