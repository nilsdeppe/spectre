// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/LapseTimesbOverW.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/NonConservativeTerms.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

// Test the non-conservative terms on the FD grid using fields that are linear
// in the coordinates, for which the second-order derivatives are exact, on a
// grid that is stretched by a different factor in each dimension.
SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Subcell.NonConservativeTerms",
                  "[Unit][GrMhd]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 0.9);
  std::uniform_real_distribution<> dist_one_to_two(1.1, 1.9);

  const Index<3> subcell_extents{5, 4, 6};
  const size_t num_pts = subcell_extents.product();
  const std::array<double, 3> stretch{{2.0, 3.0, 0.5}};
  const double det_jacobian = stretch[0] * stretch[1] * stretch[2];
  const DataVector det_inv_jacobian{num_pts, 1.0 / det_jacobian};
  std::array<double, 3> one_over_delta_xi{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(one_over_delta_xi, d) =
        0.5 * static_cast<double>(subcell_extents[d]);
  }

  // TildeB^j = b0^j + db^j_k x^k and TildePhi = phi0 + dphi_k x^k
  const std::array<double, 3> b0{{0.3, -0.7, 1.3}};
  const std::array<std::array<double, 3>, 3> db{
      {{{0.4, -0.2, 0.9}}, {{1.7, 0.6, -0.5}}, {{-0.8, 0.25, -1.4}}}};
  const double div_tilde_b = db[0][0] + db[1][1] + db[2][2];
  const double phi0 = 0.45;
  const std::array<double, 3> dphi{{1.2, -0.6, 0.35}};

  const auto inertial_coords =
      [&stretch](const tnsr::I<DataVector, 3, Frame::ElementLogical>& xi) {
        tnsr::I<DataVector, 3, Frame::Inertial> x{};
        for (size_t d = 0; d < 3; ++d) {
          x.get(d) = gsl::at(stretch, d) * xi.get(d);
        }
        return x;
      };
  const auto tilde_b_at =
      [&b0, &db](const tnsr::I<DataVector, 3, Frame::Inertial>& x) {
        tnsr::I<DataVector, 3, Frame::Inertial> result{};
        for (size_t j = 0; j < 3; ++j) {
          result.get(j) = gsl::at(b0, j) +
                          gsl::at(gsl::at(db, j), 0) * get<0>(x) +
                          gsl::at(gsl::at(db, j), 1) * get<1>(x) +
                          gsl::at(gsl::at(db, j), 2) * get<2>(x);
        }
        return result;
      };
  const auto tilde_phi_at =
      [&phi0, &dphi](const tnsr::I<DataVector, 3, Frame::Inertial>& x) {
        return Scalar<DataVector>{DataVector{phi0 + dphi[0] * get<0>(x) +
                                             dphi[1] * get<1>(x) +
                                             dphi[2] * get<2>(x)}};
      };

  const Mesh<3> subcell_mesh{subcell_extents.indices(),
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  const auto tilde_b =
      tilde_b_at(inertial_coords(logical_coordinates(subcell_mesh)));

  std::array<tnsr::I<DataVector, 3, Frame::Inertial>, 3> face_tilde_b{};
  std::array<Scalar<DataVector>, 3> face_tilde_phi{};
  std::array<tnsr::i<DataVector, 3, Frame::Inertial>, 3> face_conormal{};
  for (size_t d = 0; d < 3; ++d) {
    auto face_extents = subcell_extents.indices();
    ++gsl::at(face_extents, d);
    auto quadrature = make_array<3>(Spectral::Quadrature::CellCentered);
    gsl::at(quadrature, d) = Spectral::Quadrature::FaceCentered;
    const Mesh<3> face_mesh{face_extents,
                            make_array<3>(Spectral::Basis::FiniteDifference),
                            quadrature};
    const auto face_coords = inertial_coords(logical_coordinates(face_mesh));
    gsl::at(face_tilde_b, d) = tilde_b_at(face_coords);
    gsl::at(face_tilde_phi, d) = tilde_phi_at(face_coords);
    // J d xi^d / d x^j
    gsl::at(face_conormal, d) = tnsr::i<DataVector, 3, Frame::Inertial>{
        face_mesh.number_of_grid_points(), 0.0};
    gsl::at(face_conormal, d).get(d) = det_jacobian / gsl::at(stretch, d);
  }

  // Random cell-centered quantities multiplying the derivatives
  const DataVector used_for_size{num_pts};
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto spatial_velocity =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto lorentz_factor = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), make_not_null(&dist_one_to_two), used_for_size);
  const auto magnetic_field =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  auto spatial_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) += 2.0;
  }

  const auto initial_dt_tilde_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto initial_dt_tilde_phi = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto initial_dt_tilde_tau = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto initial_dt_tilde_s =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  auto dt_tilde_b = initial_dt_tilde_b;
  auto dt_tilde_phi = initial_dt_tilde_phi;
  auto dt_tilde_tau = initial_dt_tilde_tau;
  auto dt_tilde_s = initial_dt_tilde_s;

  grmhd::ValenciaDivClean::subcell::add_non_conservative_terms(
      make_not_null(&dt_tilde_b), make_not_null(&dt_tilde_phi),
      make_not_null(&dt_tilde_tau), make_not_null(&dt_tilde_s), face_tilde_b,
      face_tilde_phi, face_conormal, tilde_b, lapse, spatial_velocity,
      lorentz_factor, magnetic_field, spatial_metric, det_inv_jacobian,
      one_over_delta_xi, subcell_extents, 3,
      [&det_inv_jacobian, &one_over_delta_xi, &subcell_extents](
          const gsl::not_null<DataVector*> result, const DataVector& flux,
          const size_t dim) {
        evolution::dg::subcell::add_cartesian_flux_divergence(
            result, gsl::at(one_over_delta_xi, dim), det_inv_jacobian, flux,
            subcell_extents, dim);
      });

  Scalar<DataVector> magnetic_field_dot_spatial_velocity{};
  tnsr::i<DataVector, 3, Frame::Inertial> lapse_b_over_w{};
  grmhd::ValenciaDivClean::
      magnetic_field_dot_spatial_velocity_and_lapse_times_b_over_w(
          make_not_null(&magnetic_field_dot_spatial_velocity),
          make_not_null(&lapse_b_over_w), lapse,
          raise_or_lower_index(spatial_velocity, spatial_metric),
          lorentz_factor, magnetic_field, spatial_metric);

  auto expected_dt_tilde_b = initial_dt_tilde_b;
  auto expected_dt_tilde_phi = initial_dt_tilde_phi;
  auto expected_dt_tilde_tau = initial_dt_tilde_tau;
  auto expected_dt_tilde_s = initial_dt_tilde_s;
  get(expected_dt_tilde_tau) -=
      get(lapse) * get(magnetic_field_dot_spatial_velocity) * div_tilde_b;
  for (size_t i = 0; i < 3; ++i) {
    expected_dt_tilde_b.get(i) -=
        get(lapse) * spatial_velocity.get(i) * div_tilde_b;
    expected_dt_tilde_s.get(i) -= lapse_b_over_w.get(i) * div_tilde_b;
    get(expected_dt_tilde_tau) -= tilde_b.get(i) * gsl::at(dphi, i);
    get(expected_dt_tilde_phi) -=
        get(lapse) * spatial_velocity.get(i) * gsl::at(dphi, i);
  }

  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_tilde_b, expected_dt_tilde_b, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_tilde_phi, expected_dt_tilde_phi,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_tilde_tau, expected_dt_tilde_tau,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_tilde_s, expected_dt_tilde_s, custom_approx);
}
