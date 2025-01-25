// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/Tags/ObjectCenter.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
/// \endcond

namespace control_system::ControlErrors {
/*!
 * \brief Needs some dox
 */
struct GridCenters : tt::ConformsTo<protocols::ControlError> {
  using object_centers =
      domain::object_list<domain::ObjectLabel::A, domain::ObjectLabel::B>;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Computes the control error for the grid centers of two objects. "
      "This should not take any options."};

  GridCenters(const Options::Context& context = {});

  void pup(PUP::er& p);

  template <typename Metavariables, typename... TupleTags>
  DataVector operator()(const ::TimescaleTuner<true>& /*unused*/,
                        const Parallel::GlobalCache<Metavariables>& cache,
                        const double time,
                        const std::string& function_of_time_name,
                        const tuples::TaggedTuple<TupleTags...>& measurements) {
    using grid_center_A =
        control_system::QueueTags::Center<::domain::ObjectLabel::A,
                                          Frame::Grid>;
    using grid_center_B =
        control_system::QueueTags::Center<::domain::ObjectLabel::B,
                                          Frame::Grid>;

    const auto& measured_grid_position_of_A = get<grid_center_A>(measurements);
    const auto& measured_grid_position_of_B = get<grid_center_B>(measurements);

    return impl(get<domain::Tags::FunctionsOfTime>(cache)
                    .at(function_of_time_name)
                    ->func(time)[0],
                measured_grid_position_of_A, measured_grid_position_of_B);
  }

 private:
  static DataVector impl(const DataVector& fot_positions_dv,
                         const DataVector& measured_grid_position_of_A,
                         const DataVector& measured_grid_position_of_B);
};
}  // namespace control_system::ControlErrors
