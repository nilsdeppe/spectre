// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
/// Tags for the data that each element sends to its neighbors in order to
/// compute the non-conservative boundary terms, see
/// `add_non_conservative_boundary_terms()`.
namespace NonConservativeTags {
/// The lapse times the spatial velocity, \f$\alpha v^i\f$
struct LapseTimesSpatialVelocity : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};
/// The normal component of the lapse times the spatial velocity,
/// \f$\alpha v^i n_i\f$
struct NormalDotLapseTimesSpatialVelocity : db::SimpleTag {
  using type = Scalar<DataVector>;
};
/// The outward-directed unit normal covector \f$n_i\f$ of the element that
/// packaged the data
struct NormalCovector : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};
/// The lapse times the magnetic field contracted with the spatial velocity,
/// \f$\alpha B^m v_m\f$
struct LapseTimesMagneticFieldDotSpatialVelocity : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace NonConservativeTags

/// The `dg_package_field_tags` needed for the non-conservative boundary terms.
/// The boundary corrections append these to their own package fields.
///
/// Every term added by `add_non_conservative_boundary_terms()` is proportional
/// to at least one of these fields, so setting all of them to zero on both
/// sides of an interface disables the non-conservative boundary terms. The
/// finite-difference code uses this to keep its interface fluxes conservative.
using non_conservative_package_field_tags =
    tmpl::list<NonConservativeTags::LapseTimesSpatialVelocity,
               NonConservativeTags::NormalDotLapseTimesSpatialVelocity,
               NonConservativeTags::NormalCovector,
               NonConservativeTags::LapseTimesMagneticFieldDotSpatialVelocity,
               Tags::LapseTimesbOverW>;

/// Fill the `non_conservative_package_field_tags`.
void package_non_conservative_data(
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_lapse_times_spatial_velocity,
    gsl::not_null<Scalar<DataVector>*>
        packaged_normal_dot_lapse_times_spatial_velocity,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_covector,
    gsl::not_null<Scalar<DataVector>*>
        packaged_lapse_times_magnetic_field_dot_spatial_velocity,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_lapse_b_over_w,
    const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lapse_b_over_w,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector);

/*!
 * \brief Add the boundary terms of the non-conservative Powell and GLM terms
 * of the ValenciaDivClean system to the boundary corrections.
 *
 * The non-conservative volume terms are documented in
 * `grmhd::ValenciaDivClean::System`. Denoting the outward-directed unit normal
 * covector of the interior element by \f$n^\mathrm{int}_i\f$ and the one of
 * the exterior (neighboring) element by
 * \f$n^\mathrm{ext}_i=-n^\mathrm{int}_i\f$, the terms added to the boundary
 * corrections \f$G\f$ are
 *
 * \f{align*}
 * G({\tilde \Phi}) &\mathrel{+}= \frac{1}{4}
 *   \left({\tilde \Phi}_\mathrm{ext} - {\tilde \Phi}_\mathrm{int}\right)
 *   \left[(\alpha v^i n_i)_\mathrm{int} + (\alpha v^i n_i)_\mathrm{ext}
 *   \right] \\
 * G({\tilde \tau}) &\mathrel{+}= \frac{1}{4}
 *   \left({\tilde \Phi}_\mathrm{ext} - {\tilde \Phi}_\mathrm{int}\right)
 *   \left[({\tilde B}^i n_i)_\mathrm{int} + ({\tilde B}^i n_i)_\mathrm{ext}
 *   \right] + \frac{1}{4}
 *   \left[({\tilde B}^i n_i)_\mathrm{int} - ({\tilde B}^i n_i)_\mathrm{ext}
 *   \right]
 *   \left[(\alpha B^m v_m)_\mathrm{int} + (\alpha B^m v_m)_\mathrm{ext}
 *   \right] \\
 * G({\tilde B}^i) &\mathrel{+}= \frac{1}{4}
 *   \left({\tilde B}^j_\mathrm{ext} - {\tilde B}^j_\mathrm{int}\right)
 *   \left[(n_j \alpha v^i)_\mathrm{int} + (n_j \alpha v^i)_\mathrm{ext}
 *   \right] \\
 * G({\tilde S}_i) &\mathrel{+}= \frac{1}{4}
 *   \left({\tilde B}^j_\mathrm{ext} - {\tilde B}^j_\mathrm{int}\right)
 *   \left[\left(n_j \frac{\alpha b_i}{W}\right)_\mathrm{int} +
 *   \left(n_j \frac{\alpha b_i}{W}\right)_\mathrm{ext}\right]
 * \f}
 *
 * The same terms are added in the strong and in the weak form. All terms
 * change sign when the interior and exterior states are swapped.
 *
 * \note The last term in \f$G({\tilde \tau})\f$ does not vanish when the
 * interior and exterior states are identical: since
 * \f$n^\mathrm{ext}_i=-n^\mathrm{int}_i\f$ it then is \f$\alpha B^m v_m
 * {\tilde B}^i n_i\f$, which cancels the magnetic field term of the
 * \f${\tilde \tau}\f$ flux normal to the interface.
 */
void add_non_conservative_boundary_terms(
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_phi_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        lapse_times_spatial_velocity_int,
    const Scalar<DataVector>& normal_dot_lapse_times_spatial_velocity_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector_int,
    const Scalar<DataVector>&
        lapse_times_magnetic_field_dot_spatial_velocity_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lapse_b_over_w_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataVector>& tilde_phi_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        lapse_times_spatial_velocity_ext,
    const Scalar<DataVector>& normal_dot_lapse_times_spatial_velocity_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector_ext,
    const Scalar<DataVector>&
        lapse_times_magnetic_field_dot_spatial_velocity_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lapse_b_over_w_ext);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
