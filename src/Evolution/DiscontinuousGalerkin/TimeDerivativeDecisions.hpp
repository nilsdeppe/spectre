// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace evolution::dg {
/*!
 * \brief Runtime control over time derivative work done.
 *
 * - `skip_flux_divergence_calculation` if `true` then we assume the
 *   flux  divergence is identically zero.
 */
template <size_t Dim>
struct TimeDerivativeDecisions {
  bool compute_flux_divergence;
};
}  // namespace evolution::dg
