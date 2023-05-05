// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/DiagnosticInfo.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroupDeclarations.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
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
#include "Utilities/TypeTraits/IsStdArray.hpp"

namespace OptionTags {
struct WaitTime {
  using type = size_t;
  static constexpr Options::String help = {"The wait time in microseconds"};
};
}

namespace Tags {
struct WaitTime : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::WaitTime>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t wait_time) {
    return wait_time;
  }
};
}  // namespace Tags

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

struct PassComponentThisPointer {};

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

/// TODO: make into input file option
constexpr size_t max_num_evaluations = 1;

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

struct InboxesLockPtr : db::SimpleTag {
  using type = Parallel::NodeLock*;
  // using type = std::mutex*;
};

/// \brief The elements that have messages locally queued on the core.
///
/// The outer vector is the size of the number of cores on a node. The inner
/// vector has a fixed size set at runtime. Each element is attempted to be
/// invoked up to `N` times, where the second element in the pair is a counter
/// for the number of times an invocation has been attempted.
template <size_t Dim>
struct ElementsToEvaluate : db::SimpleTag {
  using type = std::vector<std::vector<ElementId<Dim>>>;
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
      Tags::InboxesLockPtr, Tags::GlobalCacheProxy<metavariables>,
      SimpleTagsFromOptions, Tags::GlobalCacheImplCompute<metavariables>,
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
    if (p.isUnpacking()) {
      my_node_ = Parallel::my_node<size_t>(
          *Parallel::local_branch(global_cache_proxy_));
    }
  }

 private:
  size_t number_of_actions_in_phase(Parallel::Phase phase) const;

  // After catching an exception, shutdown the simulation
  void initiate_shutdown(const std::exception& exception);

  template <typename PhaseDepActions, size_t... Is>
  bool iterate_over_actions(std::index_sequence<Is...> /*meta*/);

  // std::mutex inbox_lock_{};
  // std::mutex element_lock_{};

  Parallel::NodeLock inbox_lock_{};
  Parallel::NodeLock element_lock_{};

  Parallel::CProxy_GlobalCache<Metavariables> global_cache_proxy_;
  bool performing_action_ = false;
  Parallel::Phase phase_{Parallel::Phase::Initialization};
  std::unordered_map<Parallel::Phase, size_t> phase_bookmarks_{};
  std::size_t algorithm_step_ = 0;

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
  size_t my_node_{std::numeric_limits<size_t>::max()};
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
  my_node_ =
      Parallel::my_node<size_t>(*Parallel::local_branch(global_cache_proxy_));
  number_of_elements_terminated_ = number_of_elements_terminated;
  nodegroup_lock_ = nodegroup_lock;
  ::Initialization::mutate_assign<
      tmpl::list<Tags::ArrayIndex, Tags::GlobalCacheProxy<Metavariables>,
                 Tags::ElementLocationsPointer<Dim>, Tags::InboxesLockPtr,
                 InitializationTags...>>(
      make_not_null(&box_), element_id_, global_cache_proxy_,
      element_locations.get(), &inbox_lock_,
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

  // If anything got copied we need to update the lock location.
  //
  // We could probably fix this by being _extremely_ careful.
  //
  // TODO: do this for simple & threaded actions too?
  db::mutate<Tags::InboxesLockPtr>(make_not_null(&box_),
                                   [this](const auto inbox_lock_ptr) {
                                     *inbox_lock_ptr = &(this->inbox_lock_);
                                   });

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
  using return_type = void;

  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, size_t Dim>
  static return_type apply(
      db::DataBox<DbTagList>& box,
      const gsl::not_null<Parallel::NodeLock*> node_lock,
      const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
      const ElementId<Dim>& my_element_id) {
    auto& element = [&box, &cache, &my_element_id, &node_lock ]() -> auto& {
      std::lock_guard node_guard(*node_lock);
      return db::mutate<Tags::ElementCollectionBase,
                        Tags::NumberOfElementsTerminated>(
          make_not_null(&box),
          [&cache, &my_element_id ](const auto element_collection_ptr,
                                    const auto num_terminated_ptr) -> auto& {
            try {
              ++(*num_terminated_ptr);
              if (*num_terminated_ptr == element_collection_ptr->size()) {
                auto* local_branch = Parallel::local_branch(
                    Parallel::get_parallel_component<ParallelComponent>(
                        *cache));
                ASSERT(local_branch != nullptr,
                       "The local branch is nullptr, which is inconsistent");
                local_branch->set_terminate(true);
              }
              return element_collection_ptr->at(my_element_id);
            } catch (const std::out_of_range&) {
              ERROR("Could not find element with ID " << my_element_id
                                                      << " on node");
            }
          });
    }
    ();
    // Note: since this is a local synchronous action running on the element
    // that called it, we already have the element locked and so setting
    // terminate is threadsafe.
    element.set_terminate(true);
  }
};

/// \brief Receive data for a specific element.
template <bool StartPhase = false>
struct ReceiveDataForElement {
  /// \brief Entry method called when receiving data from another node.
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, typename ReceiveData,
            typename ReceiveTag, size_t Dim, typename DistObject>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    DistObject* dist_object, const ReceiveTag& /*meta*/,
                    const ElementId<Dim>& element_to_execute_on,
                    typename ReceiveTag::temporal_id instance,
                    ReceiveData receive_data) {
    ERROR(
        "The multi-node code hasn't been tested. It should work, but be aware "
        "that I haven't tried yet.");
    std::vector<ElementId<Dim>>* core_queue = nullptr;
    const size_t my_proc = Parallel::my_proc<size_t>(cache);
    using ElementCollection =
        std::decay_t<decltype(db::get<Tags::ElementCollectionBase>(box))>;
    ElementCollection* element_collection = nullptr;
    ASSERT(
        dist_object->evil_ptr0 != nullptr and dist_object->evil_ptr1 != nullptr,
        "The evil pointers were not set!");
    element_collection =
        static_cast<ElementCollection*>(dist_object->evil_ptr0);
    core_queue =
        std::addressof((*static_cast<std::vector<std::vector<ElementId<Dim>>>*>(
            dist_object->evil_ptr1))[my_proc]);
    {
      auto& element = element_collection->at(element_to_execute_on);
      std::lock_guard inbox_lock(element.inbox_lock());
      ReceiveTag::insert_into_inbox(
          make_not_null(&tuples::get<ReceiveTag>(element.inboxes())), instance,
          std::move(receive_data));
    }

    apply_impl<ParallelComponent>(cache, element_to_execute_on,
                                  make_not_null(core_queue),
                                  make_not_null(element_collection));
  }

  /// \brief Entry method call when receiving from same node.
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, size_t Dim,
            typename DistObject>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    DistObject* dist_object,
                    const ElementId<Dim>& element_to_execute_on) {
    // std::vector<ElementId<Dim>>* core_queue = nullptr;
    // const size_t my_proc = Parallel::my_proc<size_t>(cache);
    using ElementCollection =
        std::decay_t<decltype(db::get<Tags::ElementCollectionBase>(box))>;
    ElementCollection* element_collection = nullptr;
    ASSERT(
        dist_object->evil_ptr0 != nullptr and dist_object->evil_ptr1 != nullptr,
        "The evil pointers were not set!");
    element_collection =
        static_cast<ElementCollection*>(dist_object->evil_ptr0);
    const thread_local size_t my_node = Parallel::my_node<size_t>(cache);
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);

    if constexpr (StartPhase) {
      const Phase current_phase =
          Parallel::local_branch(
              Parallel::get_parallel_component<ParallelComponent>(cache))
              ->phase();
      auto& element = element_collection->at(element_to_execute_on);
      const std::lock_guard element_lock(element.element_lock());
      element.start_phase(current_phase);
    } else {
      auto& element = element_collection->at(element_to_execute_on);
      std::unique_lock element_lock(element.element_lock(), std::defer_lock);
      if (element_lock.try_lock()) {
        element.perform_algorithm();
      } else {
        Parallel::threaded_action<Parallel::Actions::ReceiveDataForElement<>>(
            my_proxy[my_node], element_to_execute_on);
        return;
        // core_queue->push_back(element_to_execute_on);
      }
    }

    // core_queue =
    //     std::addressof((*static_cast<std::vector<std::vector<
    // ElementId<Dim>>>*>(
    //         dist_object->evil_ptr1))[my_proc]);
    // apply_impl<ParallelComponent>(cache, element_to_execute_on,
    //                               make_not_null(core_queue),
    //                               make_not_null(element_collection));
  }

 private:
  template <typename ParallelComponent, typename Metavariables,
            typename ElementCollection, size_t Dim>
  static void apply_impl(
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_to_execute_on,
      const gsl::not_null<std::vector<ElementId<Dim>>*> core_queue,
      const gsl::not_null<ElementCollection*> element_collection) {
    const size_t my_node = Parallel::my_node<size_t>(cache);
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);

    if constexpr (StartPhase) {
      const Phase current_phase =
          Parallel::local_branch(
              Parallel::get_parallel_component<ParallelComponent>(cache))
              ->phase();
      auto& element = element_collection->at(element_to_execute_on);
      const std::lock_guard element_lock(element.element_lock());
      element.start_phase(current_phase);
    } else {
      auto& element = element_collection->at(element_to_execute_on);
      std::unique_lock element_lock(element.element_lock(), std::defer_lock);
      if (element_lock.try_lock()) {
        element.perform_algorithm();
      } else {
        Parallel::threaded_action<Parallel::Actions::ReceiveDataForElement<>>(
            my_proxy[my_node], element_to_execute_on);
        return;
        // core_queue->push_back(element_to_execute_on);
      }
    }
    const size_t max_iters = 5;
    for (size_t iter = 0; iter < max_iters; ++iter) {
      for (size_t i = 0; i < core_queue->size(); ++i) {
        const auto& id = (*core_queue)[i];
        // NOTE: this is only safe if we aren't inserting. If we were to start
        // inserting randomly, then we need a lock for insertions or use a
        // lock-free map. For example, folly's AtomicUnorderedInsertMap would
        // work. This allows concurrent insert and read. You can't remove
        // elements, but that's okay because what we can do is just have the
        // mapped value be something like node_id_and_data, or we can always
        // default-construct the data if the element is no longer on our node.
        auto& element = element_collection->at(id);
        std::unique_lock element_lock(element.element_lock(), std::defer_lock);
        if (element_lock.try_lock()) {
          element.perform_algorithm();
          core_queue->erase(
              std::next(core_queue->begin(), static_cast<std::ptrdiff_t>(i)));
          // Move back one because the loop will increment.
          --i;
          element_lock.unlock();
        }
      }
    }
    if (not core_queue->empty()) {
      for (size_t i = 0; i < core_queue->size(); ++i) {
        Parallel::threaded_action<Parallel::Actions::ReceiveDataForElement<>>(
            my_proxy[my_node], (*core_queue)[i]);
      }
      core_queue->clear();
    }
  }
};

/// \brief A local synchronous action where data is communicated to neighbor
/// elements in the most optimal way possible.
struct SendDataToElement {
  using return_type = void;

  template <typename ParallelComponent, typename DbTagList, size_t Dim,
            typename ReceiveTag, typename ReceiveData, typename Metavariables>
  static return_type apply(
      db::DataBox<DbTagList>& box,
      const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
      const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
      const ReceiveTag& /*meta*/, const ElementId<Dim>& element_to_execute_on,
      typename ReceiveTag::temporal_id instance, ReceiveData&& receive_data) {
    using ElementType =
        typename std::decay_t<decltype(db::get<Tags::ElementCollectionBase>(
            box))>::mapped_type;
    using ElementCollection =
        std::decay_t<decltype(db::get<Tags::ElementCollectionBase>(box))>;

    const size_t my_node = Parallel::my_node<size_t>(*cache);
    const size_t my_proc = Parallel::my_proc<size_t>(*cache);
    auto* dist_object = Parallel::local_branch(
        Parallel::get_parallel_component<ParallelComponent>(*cache));

    ElementType* element =
        std::addressof(static_cast<ElementCollection*>(dist_object->evil_ptr0)
                           ->at(element_to_execute_on));
    std::vector<ElementId<Dim>>* core_queue = nullptr;
    core_queue =
        std::addressof((*static_cast<std::vector<std::vector<ElementId<Dim>>>*>(
            dist_object->evil_ptr1))[my_proc]);
    const size_t node_of_element =
        static_cast<std::unordered_map<ElementId<Dim>, size_t>*>(
            dist_object->evil_ptr2)
            ->at(element_to_execute_on);
    auto& my_proxy =
        Parallel::get_parallel_component<ParallelComponent>(*cache);
    if (node_of_element == my_node) {
      size_t count = 0;
      if constexpr (tt::is_std_array_v<typename ReceiveTag::type>) {
        // Scope so that we minimize how long we lock the inbox.
        count = ReceiveTag::insert_into_inbox(
            make_not_null(&tuples::get<ReceiveTag>(element->inboxes())),
            instance, std::forward<ReceiveData>(receive_data));
      } else {
        // Scope so that we minimize how long we lock the inbox.
        std::lock_guard inbox_lock(element->inbox_lock());
        count = ReceiveTag::insert_into_inbox(
            make_not_null(&tuples::get<ReceiveTag>(element->inboxes())),
            instance, std::forward<ReceiveData>(receive_data));
      }
      if (count == 2 * Dim) {
        Parallel::threaded_action<Parallel::Actions::ReceiveDataForElement<>>(
            my_proxy[node_of_element], element_to_execute_on);
      }
    } else {
      Parallel::threaded_action<Parallel::Actions::ReceiveDataForElement<>>(
          my_proxy[node_of_element], ReceiveTag{}, element_to_execute_on,
          instance, std::forward<ReceiveData>(receive_data));
    }
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
      Parallel::threaded_action<ReceiveDataForElement<true>>(proxy_to_this_node,
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
            Parallel::threaded_action<ReceiveDataForElement<true>>(my_proxy,
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
                 Tags::ElementLocations<Dim>, Tags::NumberOfElementsTerminated,
                 Tags::ElementsToEvaluate<Dim>>;
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
          ::initial_element_ids(block.id(), initial_ref_levs);

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

    auto* dist_object = Parallel::local_branch(
        Parallel::get_parallel_component<ParallelComponent>(local_cache));
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
         &node_lock, dist_object](
            const auto element_locations_ptr, const auto collection_ptr,
            const gsl::not_null<size_t*> number_of_elements_terminated) {
          dist_object->evil_ptr0 =
              static_cast<void*>(collection_ptr.get());
          const auto serialized_initialization_items =
              serialize(initialization_items);
          *element_locations_ptr = std::move(node_of_elements);
          dist_object->evil_ptr2 =
              static_cast<void*>(element_locations_ptr.get());
          for (const auto& element_id : my_elements) {
            if (not collection_ptr
                        ->emplace(
                            std::piecewise_construct,
                            std::forward_as_tuple(element_id),
                            std::forward_as_tuple(
                                local_cache.get_this_proxy(),
                                deserialize<tuples::tagged_tuple_from_typelist<
                                    SimpleTagsFromOptions>>(
                                    serialized_initialization_items.data()),
                                element_id, element_locations_ptr,
                                number_of_elements_terminated, node_lock))
                        .second) {
              ERROR("Failed to insert element with ID: " << element_id);
            }
            (*number_of_elements_terminated) = 0;
            if (collection_ptr->at(element_id).get_terminate() == true) {
              ++(*number_of_elements_terminated);
            }
          }
        });

    const size_t number_of_cores_on_node =
        Parallel::procs_on_node<size_t>(my_node, local_cache);
    db::mutate<Tags::ElementsToEvaluate<Dim>>(
        make_not_null(&box), [number_of_cores_on_node, dist_object](
                                 const auto elements_to_evaluate_ptr) {
          elements_to_evaluate_ptr->resize(number_of_cores_on_node);
          for (auto& elements_to_evaluate_on_core : *elements_to_evaluate_ptr) {
            elements_to_evaluate_on_core.reserve(max_num_evaluations);
          }
          dist_object->evil_ptr1 =
              static_cast<void*>(std::addressof((*elements_to_evaluate_ptr)));
        });

    Parallel::contribute_to_reduction<SpawnInitializeElementsInCollection>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<int, funcl::AssertEqual<>>>{0},
        Parallel::get_parallel_component<ParallelComponent>(
            local_cache)[my_node],
        Parallel::get_parallel_component<ParallelComponent>(local_cache));

    if (my_node == 0) {
      Parallel::printf("\n%s\n", domain::diagnostic_info(
                                     domain, local_cache, elements_per_core,
                                     elements_per_node, grid_points_per_core,
                                     grid_points_per_node));
    }

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
struct DgElementCollection : PassComponentThisPointer {
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

          Parallel::PhaseActions<Parallel::Phase::ImportInitialData,
                                 tmpl::list<Actions::StartPhaseOnNodegroup>>,

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
      tmpl::list<::domain::Tags::Domain<Dim>, ::Tags::WaitTime>,
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
}  // namespace Parallel
