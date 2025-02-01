// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <limits>
#include <memory>

#include "ControlSystem/Tags/IsActiveMap.hpp"
#include "ControlSystem/Tags/OptionTags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct TimeStep;
struct TimeStepId;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace control_system {
/*!
 * \brief Holds options used to control when to start disabling the rotation
 * control system in a BNS simulation.
 */
struct DisableRotationWhen {
  struct DisableAtSeparation {
    using type = double;
    static type lower_bound() { return 2.0; }
    static constexpr Options::String help{
        "The separation at which we start turning off rotation."};
  };
  struct RotationDecayTimescale {
    using type = double;
    static type lower_bound() { return 5.0; }
    static type upper_bound() { return 200.0; }
    static constexpr Options::String help{
        "The timescale in code units over which grid rotation is disabled. A "
        "reasonable value is 20M-50M"};
  };

  using options = tmpl::list<DisableAtSeparation, RotationDecayTimescale>;
  static constexpr Options::String help{
      "Constrols the separation at which the rotation control system is "
      "disabled and the rotation function of time starts settling to a "
      "constant."};

  void pup(PUP::er& p);

  double disable_at_separation{std::numeric_limits<double>::signaling_NaN()};
  double rotation_decay_timescale{std::numeric_limits<double>::signaling_NaN()};
};

namespace OptionTags {
struct DisableRotationWhen {
  static constexpr Options::String help =
      "Options for controlling how the rotation control system stops as the "
      "two neutron stars merge.";
  using type = control_system::DisableRotationWhen;
  using group = control_system::OptionTags::ControlSystemGroup;
};
}  // namespace OptionTags

namespace Tags {
struct DisableRotationWhen : db::SimpleTag {
  using type = control_system::DisableRotationWhen;
  using option_tags = tmpl::list<OptionTags::DisableRotationWhen>;

  static constexpr bool pass_metavariables = false;
  static control_system::DisableRotationWhen create_from_options(
      const control_system::DisableRotationWhen& disable_rotation_when);
};
}  // namespace Tags

namespace Actions {
/*!
 * \brief Checks if the Rotation function of time has been updated because the
 * separation between the neutron star grid centers is small enough.
 *
 *
 */
struct SwitchGridRotationToSettle {
  struct UpdateRotationToSettle {
    static void apply(
        gsl::not_null<std::unordered_map<
            std::string,
            std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
            f_of_t_list,
        const std::string& function_of_time_name,
        const std::array<DataVector, 3>& initial_func_and_derivs,
        double match_time, double decay_time);
  };

  struct DisableControlSystem {
    static void apply(
        gsl::not_null<std::unordered_map<std::string, bool>*> is_active_map,
        const std::string& control_system_name);
  };

  using const_global_cache_tags =
      tmpl::list<control_system::Tags::DisableRotationWhen>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<3>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    if (not time_step_id.is_at_slab_boundary()) {
      ERROR(
          "Expected to be at a Slab boundary when changing the Rotation "
          "function of time to a SettleToConstant.");
    }
    if (not is_zeroth_element(db::get<domain::Tags::Element<3>>(box).id())) {
      // Only one element needs to modify the control system and function of
      // time.
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const auto& bns_rotation_control =
        db::get<control_system::Tags::DisableRotationWhen>(box);

    const double time = db::get<::Tags::Time>(box);
    const domain::FunctionsOfTime::FunctionOfTime& grid_centers_fot =
        *db::get<domain::Tags::FunctionsOfTime>(box).at("GridCenters");
    const DataVector grid_centers = grid_centers_fot.func(time)[0];
    const double separation = sqrt(square(grid_centers[0] - grid_centers[3]) +
                                   square(grid_centers[1] - grid_centers[4]) +
                                   square(grid_centers[2] - grid_centers[5]));
    if (separation > bns_rotation_control.disable_at_separation) {
      ERROR(
          "Disabling the rotation control system should happen when the "
          "separation is less than or equal to "
          << bns_rotation_control.disable_at_separation
          << " but the separation at time " << time << " is calculated to be "
          << separation);
    }

    const domain::FunctionsOfTime::FunctionOfTime& rotation_fot =
        *db::get<domain::Tags::FunctionsOfTime>(box).at("Rotation");
    const std::array<DataVector, 3> current_func_and_derivs =
        rotation_fot.func_and_2_derivs(time);
    Parallel::mutate<domain::Tags::FunctionsOfTime, UpdateRotationToSettle>(
        cache, std::string{"Rotation"}, current_func_and_derivs, time,
        bns_rotation_control.rotation_decay_timescale);
    Parallel::mutate<control_system::Tags::IsActiveMap, DisableControlSystem>(
        cache, std::string{"Rotation"});
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace control_system
