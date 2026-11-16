// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace grmhd::ValenciaDivClean {
/*!
 * \brief Compute \f$B^m v_m\f$ and \f$\alpha b_i / W\f$.
 *
 * The lapse times the spatial part of the comoving magnetic field one-form
 * divided by the Lorentz factor is
 *
 * \f{align*}
 * \frac{\alpha b_i}{W} = \alpha \left(\frac{B_i}{W^2} + v_i B^m v_m\right)
 * \f}
 *
 * where \f$B^i\f$ is the Eulerian magnetic field, \f$v^i\f$ the spatial
 * velocity, \f$W\f$ the Lorentz factor, and \f$\alpha\f$ the lapse.
 *
 * These are the two quantities multiplying \f$\partial_i \tilde{B}^i\f$ in the
 * non-conservative Powell terms of the \f$\tilde{\tau}\f$ and
 * \f$\tilde{S}_i\f$ evolution equations, see `grmhd::ValenciaDivClean::System`.
 * They are computed as temporaries by `TimeDerivativeTerms`, and this function
 * is used to compute them where those temporaries are not available, e.g.
 * in boundary conditions and on the faces of the finite-difference grid.
 */
void magnetic_field_dot_spatial_velocity_and_lapse_times_b_over_w(
    gsl::not_null<Scalar<DataVector>*> magnetic_field_dot_spatial_velocity,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> lapse_b_over_w,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3, Frame::Inertial>& spatial_velocity_one_form,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);
}  // namespace grmhd::ValenciaDivClean
