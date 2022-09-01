// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename T>
class Variables;
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
/// \endcond

namespace ScalarAdvection::subcell {
/*!
 * \brief Returns \f$U\f$, the variables needed for reconstruction.
 *
 * This mutator is passed to
 * `evolution::dg::subcell::Actions::SendDataForReconstruction`.
 */
class GhostVariables {
 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<tmpl::list<ScalarAdvection::Tags::U>>>;

  static Variables<tmpl::list<ScalarAdvection::Tags::U>> apply(
      const Variables<tmpl::list<ScalarAdvection::Tags::U>>& vars);
};
}  // namespace ScalarAdvection::subcell
