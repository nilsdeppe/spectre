// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/NonConservativeTerms.hpp"

#include <array>
#include <cstddef>
#include <functional>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/LapseTimesbOverW.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::subcell {
namespace {
struct DivTildeB : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct GradTildePhi : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};
}  // namespace

void add_non_conservative_terms(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> dt_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> dt_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> dt_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> dt_tilde_s,
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
    const Index<3>& subcell_extents,
    const size_t number_of_computational_dimensions,
    const std::function<void(gsl::not_null<DataVector*>, const DataVector&,
                             size_t)>& add_flux_divergence) {
  Variables<tmpl::list<
      DivTildeB, GradTildePhi,
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>,
      hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
      Tags::LapseTimesbOverW>>
      buffer{get(lapse).size(), 0.0};
  DataVector& div_tilde_b = get(get<DivTildeB>(buffer));
  auto& grad_tilde_phi = get<GradTildePhi>(buffer);

  // Compute the divergence of TildeB and the gradient of TildePhi from the
  // values on the faces.
  DataVector face_flux{};
  for (size_t dim = 0; dim < number_of_computational_dimensions; ++dim) {
    const auto& conormal = gsl::at(face_conormal, dim);
    const auto& face_b = gsl::at(face_tilde_b, dim);
    const DataVector& face_phi = get(gsl::at(face_tilde_phi, dim));

    face_flux = face_b.get(0) * conormal.get(0);
    for (size_t j = 1; j < 3; ++j) {
      face_flux += face_b.get(j) * conormal.get(j);
    }
    add_flux_divergence(make_not_null(&div_tilde_b), face_flux, dim);

    for (size_t i = 0; i < 3; ++i) {
      face_flux = face_phi * conormal.get(i);
      evolution::dg::subcell::add_cartesian_flux_divergence(
          make_not_null(&grad_tilde_phi.get(i)),
          gsl::at(one_over_delta_xi, dim), det_inv_jacobian, face_flux,
          subcell_extents, dim);
    }
  }

  auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>>(
          buffer);
  raise_or_lower_index(make_not_null(&spatial_velocity_one_form),
                       spatial_velocity, spatial_metric);
  auto& magnetic_field_dot_spatial_velocity =
      get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(buffer);
  auto& lapse_b_over_w = get<Tags::LapseTimesbOverW>(buffer);
  magnetic_field_dot_spatial_velocity_and_lapse_times_b_over_w(
      make_not_null(&magnetic_field_dot_spatial_velocity),
      make_not_null(&lapse_b_over_w), lapse, spatial_velocity_one_form,
      lorentz_factor, magnetic_field, spatial_metric);

  get(*dt_tilde_tau) -=
      get(lapse) * get(magnetic_field_dot_spatial_velocity) * div_tilde_b;
  for (size_t i = 0; i < 3; ++i) {
    dt_tilde_b->get(i) -= get(lapse) * spatial_velocity.get(i) * div_tilde_b;
    dt_tilde_s->get(i) -= lapse_b_over_w.get(i) * div_tilde_b;
    get(*dt_tilde_tau) -= tilde_b.get(i) * grad_tilde_phi.get(i);
    get(*dt_tilde_phi) -=
        get(lapse) * spatial_velocity.get(i) * grad_tilde_phi.get(i);
  }
}
}  // namespace grmhd::ValenciaDivClean::subcell
