// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"

namespace NewtonianEuler {
namespace Sources {

/*!
 * \brief Used to mark that the initial data do not require source
 * terms in the evolution equations.
 */
struct NoSource {
  using sourced_variables = tmpl::list<>;
};
}  // namespace Sources
}  // namespace NewtonianEuler
