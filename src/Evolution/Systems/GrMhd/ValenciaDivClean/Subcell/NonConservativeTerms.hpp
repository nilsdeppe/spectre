// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <functional>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Index;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief Add the non-conservative Powell and GLM terms of the ValenciaDivClean
 * system to the time derivatives on the finite-difference grid.
 *
 * The non-conservative terms are documented in
 * `grmhd::ValenciaDivClean::System`. On the finite-difference grid they are
 * evaluated at the cell centers as
 *
 * \f{align*}
 * \partial_t {\tilde B}^j &\mathrel{-}= \alpha v^j
 *   \left(\partial_i {\tilde B}^i\right) \\
 * \partial_t {\tilde S}_j &\mathrel{-}= \frac{\alpha b_j}{W}
 *   \left(\partial_i {\tilde B}^i\right) \\
 * \partial_t {\tilde \tau} &\mathrel{-}= \alpha B^m v_m
 *   \left(\partial_i {\tilde B}^i\right)
 *   + {\tilde B}^i \left(\partial_i {\tilde \Phi}\right) \\
 * \partial_t {\tilde \Phi} &\mathrel{-}= \alpha v^i
 *   \left(\partial_i {\tilde \Phi}\right)
 * \f}
 *
 * where all quantities multiplying the derivatives are cell-centered values.
 * The derivatives are second-order differences of the values on the cell
 * faces, written in conservative form:
 *
 * \f{align*}
 * \partial_i {\tilde B}^i &= \frac{1}{J} \sum_{\hat{k}}
 *   \partial_{\hat{k}} \left(J \frac{\partial \xi^{\hat{k}}}{\partial x^i}
 *   {\tilde B}^i\right), &
 * \partial_i {\tilde \Phi} &= \frac{1}{J} \sum_{\hat{k}}
 *   \partial_{\hat{k}} \left(J \frac{\partial \xi^{\hat{k}}}{\partial x^i}
 *   {\tilde \Phi}\right),
 * \f}
 *
 * where \f$J\f$ is the determinant of the Jacobian from logical to inertial
 * coordinates.
 *
 * \param dt_tilde_b,dt_tilde_phi,dt_tilde_tau,dt_tilde_s the cell-centered
 * time derivatives the terms are added to
 * \param face_tilde_b,face_tilde_phi for each logical dimension \f$\hat{k}\f$,
 * \f${\tilde B}^i\f$ and \f${\tilde \Phi}\f$ on the cell faces normal to that
 * dimension, typically the average of the two reconstructed states
 * \param face_conormal for each logical dimension \f$\hat{k}\f$,
 * \f$J \partial \xi^{\hat{k}} / \partial x^i\f$ on the cell faces normal to
 * that dimension
 * \param
 * tilde_b,lapse,spatial_velocity,lorentz_factor,magnetic_field,spatial_metric
 * cell-centered values
 * \param det_inv_jacobian the cell-centered determinant of the inverse Jacobian
 * \f$1/J\f$
 * \param one_over_delta_xi the inverse logical grid spacing
 * \param subcell_extents the extents of the cell-centered mesh
 * \param number_of_computational_dimensions the number of dimensions that are
 * not Cartoon dimensions. Faces are only needed in these dimensions.
 * \param add_flux_divergence function `f(result, flux, dim)` that adds the
 * finite-difference derivative in logical dimension `dim` of the face-centered
 * `flux` of a vector to the cell-centered `result`, e.g.
 * `evolution::dg::subcell::add_cartesian_flux_divergence()`, or
 * `evolution::dg::subcell::add_cartoon_cartesian_flux_divergence()` when using
 * Cartoon dimensions. It is used for \f$\partial_i {\tilde B}^i\f$. The
 * gradient \f$\partial_i {\tilde \Phi}\f$ of a scalar has no contributions
 * from the Cartoon dimensions, so it always uses
 * `evolution::dg::subcell::add_cartesian_flux_divergence()`.
 */
void add_non_conservative_terms(
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> dt_tilde_b,
    gsl::not_null<Scalar<DataVector>*> dt_tilde_phi,
    gsl::not_null<Scalar<DataVector>*> dt_tilde_tau,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> dt_tilde_s,
    const std::array<tnsr::I<DataVector, 3, Frame::Inertial>, 3>& face_tilde_b,
    const std::array<Scalar<DataVector>, 3>& face_tilde_phi,
    const std::array<tnsr::i<DataVector, 3, Frame::Inertial>, 3>& face_conormal,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const DataVector& det_inv_jacobian,
    const std::array<double, 3>& one_over_delta_xi,
    const Index<3>& subcell_extents, size_t number_of_computational_dimensions,
    const std::function<void(gsl::not_null<DataVector*>, const DataVector&,
                             size_t)>& add_flux_divergence);
}  // namespace grmhd::ValenciaDivClean::subcell
