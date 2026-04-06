// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_set>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/HighOrderFluxCorrection.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace {
struct Scalar0 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Vector0 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
void test_using_nodes(const fd::DerivativeOrder correction_order) {
  CAPTURE(correction_order);
  CAPTURE(Dim);
  const size_t max_degree =
      correction_order == fd::DerivativeOrder::OneHigherThanRecons
          ? 6
          : (correction_order ==
                     fd::DerivativeOrder::OneHigherThanReconsButFiveToFour
                 ? 4
                 : static_cast<size_t>(correction_order));
  const size_t points_per_dimension = static_cast<size_t>(max_degree) + 2;
  const size_t stencil_width = max_degree + 1;
  const size_t number_of_ghost_points = (stencil_width - 1) / 2 + 1;
  CAPTURE(points_per_dimension);

  using FluxTags = tmpl::list<Scalar0, Vector0<Dim>>;
  using Scalar0Flux = ::Tags::Flux<Scalar0, tmpl::size_t<Dim>, Frame::Inertial>;
  using Vector0Flux =
      ::Tags::Flux<Vector0<Dim>, tmpl::size_t<Dim>, Frame::Inertial>;
  using FluxVars =
      Variables<db::wrap_tags_in<::Tags::Flux, FluxTags, tmpl::size_t<Dim>,
                                 Frame::Inertial>>;
  using CorrectionVars = Variables<FluxTags>;

  const Mesh<Dim> mesh{points_per_dimension, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < Dim; ++i) {
    logical_coords.get(i) += 4.0 * static_cast<double>(i);
  }

  // Compute polynomial on cell centers in FD cluster of points
  const auto set_polynomial = Overloader{
      [max_degree](const gsl::not_null<FluxVars*> vars_ptr,
                   const auto& local_logical_coords) {
        (void)max_degree;
        for (size_t storage_index = 0;
             storage_index < get<Scalar0Flux>(*vars_ptr).size();
             ++storage_index) {
          get<Scalar0Flux>(*vars_ptr)[storage_index] = 0.0;
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Scalar0Flux>(*vars_ptr)[storage_index] +=
                  pow(local_logical_coords.get(i), degree);
            }
          }
        }
        for (size_t storage_index = 0;
             storage_index < get<Vector0Flux>(*vars_ptr).size();
             ++storage_index) {
          get<Vector0Flux>(*vars_ptr)[storage_index] =
              1.0 + 0.3 * static_cast<double>(storage_index);
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Vector0Flux>(*vars_ptr)[storage_index] +=
                  pow(local_logical_coords.get(i), degree);
            }
          }
        }
      },
      [max_degree](const gsl::not_null<CorrectionVars*> vars_ptr,
                   const auto& local_logical_coords) {
        (void)max_degree;
        for (size_t storage_index = 0;
             storage_index < get<Scalar0>(*vars_ptr).size(); ++storage_index) {
          get<Scalar0>(*vars_ptr)[storage_index] = 0.0;
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Scalar0>(*vars_ptr)[storage_index] +=
                  pow(local_logical_coords.get(i), degree);
            }
          }
        }
        for (size_t storage_index = 0;
             storage_index < get<Vector0<Dim>>(*vars_ptr).size();
             ++storage_index) {
          get<Vector0<Dim>>(*vars_ptr)[storage_index] =
              100.0 + 11.0 * static_cast<double>(storage_index);
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Vector0<Dim>>(*vars_ptr)[storage_index] +=
                  pow(local_logical_coords.get(i), degree);
            }
          }
        }
      }};
  const auto set_polynomial_divergence =
      [max_degree](const gsl::not_null<CorrectionVars*> d_vars_ptr,
                   const auto& local_logical_coords) {
        (void)max_degree;
        get(get<Scalar0>(*d_vars_ptr)) = 0.0;
        for (size_t i = 0; i < Dim; ++i) {
          // constant deriv is zero
          get<Vector0<Dim>>(*d_vars_ptr).get(i) = 0.0;
        }
        // Compute divergence
        for (size_t deriv_dim = 0; deriv_dim < Dim; ++deriv_dim) {
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            get(get<Scalar0>(*d_vars_ptr)) +=
                degree * pow(local_logical_coords.get(deriv_dim), degree - 1);
            for (size_t i = 0; i < Dim; ++i) {
              get<Vector0<Dim>>(*d_vars_ptr).get(i) +=
                  degree * pow(local_logical_coords.get(deriv_dim), degree - 1);
            }
          }
        }
      };
  std::optional<FluxVars> volume_vars(mesh.number_of_grid_points());
  set_polynomial(&(volume_vars.value()), logical_coords);

  CorrectionVars expected_divergence(mesh.number_of_grid_points());
  set_polynomial_divergence(&expected_divergence, logical_coords);

  // Compute the polynomial at the cell center for the neighbor data that we
  // "received".
  //
  // We do this by computing the solution in our entire neighbor, then using
  // slice_data to get the subset of points that are needed.
  DirectionMap<Dim, FluxVars> neighbor_data{};
  DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>
      reconstruction_ghost_data{};

  for (const auto& direction : Direction<Dim>::all_directions()) {
    auto neighbor_logical_coords = logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    FluxVars neighbor_vars(mesh.number_of_grid_points(), 0.0);
    set_polynomial(&neighbor_vars, neighbor_logical_coords);

    const auto sliced_data = evolution::dg::subcell::slice_data(
        neighbor_vars, mesh.extents(), number_of_ghost_points,
        std::unordered_set{direction.opposite()}, 0, {});
    CAPTURE(number_of_ghost_points);
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    REQUIRE(sliced_data.at(direction.opposite()).size() %
                FluxVars::number_of_independent_components ==
            0);
    neighbor_data[direction].initialize(
        sliced_data.at(direction.opposite()).size() /
        FluxVars::number_of_independent_components);
    std::copy(sliced_data.at(direction.opposite()).begin(),
              sliced_data.at(direction.opposite()).end(),
              neighbor_data[direction].data());

    const DirectionalId<Dim> mortar_id{direction, ElementId<Dim>{0}};
    reconstruction_ghost_data[mortar_id] = evolution::dg::subcell::GhostData{1};
    reconstruction_ghost_data[mortar_id]
        .neighbor_ghost_data_for_reconstruction() =
        DataVector{sliced_data.at(direction.opposite()).size()};
    std::copy(sliced_data.at(direction.opposite()).begin(),
              sliced_data.at(direction.opposite()).end(),
              reconstruction_ghost_data[mortar_id]
                  .neighbor_ghost_data_for_reconstruction()
                  .data());
  }

  std::array<CorrectionVars, Dim> second_order_corrections{};
  for (size_t i = 0; i < Dim; ++i) {
    // Compare to analytic solution on the faces.
    const auto basis = make_array<Dim>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<Dim>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<Dim>(points_per_dimension);
    gsl::at(extents, i) = points_per_dimension + 1;
    gsl::at(quadrature, i) = Spectral::Quadrature::FaceCentered;
    const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
    auto face_logical_coords = logical_coordinates(face_centered_mesh);
    for (size_t j = 1; j < Dim; ++j) {
      face_logical_coords.get(j) += 4.0 * static_cast<double>(j);
    }
    gsl::at(second_order_corrections, i)
        .initialize(face_centered_mesh.number_of_grid_points());
    set_polynomial(make_not_null(&gsl::at(second_order_corrections, i)),
                   face_logical_coords);
    // We use n_i F^i in the code, so need to negate to get sign to agree.
    gsl::at(second_order_corrections, i) *= -1.0;
  }

  std::array<std::vector<std::uint8_t>, Dim> reconstruction_order_storage{};
  std::array<gsl::span<std::uint8_t>, Dim> reconstruction_order{};
  if (correction_order == fd::DerivativeOrder::OneHigherThanRecons or
      correction_order ==
          fd::DerivativeOrder::OneHigherThanReconsButFiveToFour) {
    Index<Dim> recons_extents = mesh.extents();
    recons_extents[0] += 2;
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(reconstruction_order_storage, i) =
          std::vector<std::uint8_t>(recons_extents.product(), 5);
      gsl::at(reconstruction_order, i) =
          gsl::span(gsl::at(reconstruction_order_storage, i).data(),
                    gsl::at(reconstruction_order_storage, i).size());
    }
  }

  std::optional<std::array<CorrectionVars, Dim>> high_order_corrections{};
  ::fd::cartesian_high_order_flux_corrections(
      make_not_null(&high_order_corrections),

      volume_vars, second_order_corrections, correction_order,
      reconstruction_ghost_data, mesh, number_of_ghost_points,
      reconstruction_order);

  // Now compute the Cartesian derivative of the high_order_corrections to
  // verify that it is computed sufficiently accurately.
  const DataVector inv_jacobian{mesh.number_of_grid_points(), 1.0};
  CorrectionVars flux_divergence{mesh.number_of_grid_points(), 0.0};
  for (size_t d = 0; d < Dim; ++d) {
    const auto& corrections_in_dim =
        high_order_corrections.has_value()
            ? gsl::at(high_order_corrections.value(), d)
            : gsl::at(second_order_corrections, d);
    // Note: assumes isotropic mesh
    const double one_over_delta_xi =
        -1.0 / (logical_coords.get(0)[1] - logical_coords.get(0)[0]);
    evolution::dg::subcell::add_cartesian_flux_divergence(
        make_not_null(&get(get<Scalar0>(flux_divergence))), one_over_delta_xi,
        inv_jacobian, get(get<Scalar0>(corrections_in_dim)), mesh.extents(), d);
    for (size_t i = 0; i < Dim; ++i) {
      evolution::dg::subcell::add_cartesian_flux_divergence(
          make_not_null(&get<Vector0<Dim>>(flux_divergence).get(i)),
          one_over_delta_xi, inv_jacobian,
          get<Vector0<Dim>>(corrections_in_dim).get(i), mesh.extents(), d);
    }
  }

  // With high-order corrections roundoff can accumulate.
  Approx custom_approx = Approx::custom().epsilon(5.e-12);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Scalar0>(flux_divergence),
                               get<Scalar0>(expected_divergence),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Vector0<Dim>>(flux_divergence),
                               get<Vector0<Dim>>(expected_divergence),
                               custom_approx);

  // Test assertions
#ifdef SPECTRE_DEBUG
  if (correction_order != fd::DerivativeOrder::Two) {
    std::optional<std::array<CorrectionVars, Dim>>
        high_order_corrections_assert = make_array<Dim>(CorrectionVars{
            second_order_corrections[0].number_of_grid_points()});
    high_order_corrections_assert.value()[0].initialize(
        second_order_corrections[0].number_of_grid_points() * 2);
    CHECK_THROWS_WITH(
        ::fd::cartesian_high_order_flux_corrections(
            make_not_null(&high_order_corrections_assert), volume_vars,
            second_order_corrections, correction_order,
            reconstruction_ghost_data, mesh, number_of_ghost_points),
        Catch::Matchers::ContainsSubstring(
            "The high_order_corrections must all have size"));
  }
  if constexpr (Dim > 1) {
    auto second_order_corrections_copy = second_order_corrections;
    second_order_corrections_copy[0].initialize(
        second_order_corrections_copy[0].number_of_grid_points() * 2);
    CHECK_THROWS_WITH(
        ::fd::cartesian_high_order_flux_corrections(
            make_not_null(&high_order_corrections), volume_vars,
            second_order_corrections_copy, correction_order,
            reconstruction_ghost_data, mesh, number_of_ghost_points),
        Catch::Matchers::ContainsSubstring(
            "All second-order boundary corrections must be of the same size"));
  }
#endif  // SPECTRE_DEBUG
}

template <size_t Dim>
void test(const fd::DerivativeOrder correction_order) {
  CAPTURE(correction_order);
  CAPTURE(Dim);
  const auto deriv_order = static_cast<size_t>(fd::fd_order(correction_order));
  // The MD stencil of order p is exact for polynomials of degree p-1.
  const size_t max_degree = deriv_order - 1;
  // Number of ghost face layers per direction: order/2 - 1
  const size_t number_of_ghost_faces = (deriv_order / 2) - 1;
  // Need enough points so the stencil fits
  const size_t points_per_dimension = max_degree + 2;
  CAPTURE(points_per_dimension);

  using FluxTags = tmpl::list<Scalar0, Vector0<Dim>>;
  using Scalar0Flux = ::Tags::Flux<Scalar0, tmpl::size_t<Dim>, Frame::Inertial>;
  using Vector0Flux =
      ::Tags::Flux<Vector0<Dim>, tmpl::size_t<Dim>, Frame::Inertial>;
  using FluxVars =
      Variables<db::wrap_tags_in<::Tags::Flux, FluxTags, tmpl::size_t<Dim>,
                                 Frame::Inertial>>;
  using CorrectionVars = Variables<FluxTags>;

  const Mesh<Dim> mesh{points_per_dimension, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < Dim; ++i) {
    logical_coords.get(i) += 4.0 * static_cast<double>(i);
  }

  // Polynomial for the boundary correction. Both the interior face corrections
  // and the ghost face data must use the same polynomial. For each component
  // (Scalar0 and Vector0<Dim>), the value at position x is:
  //   offset(storage_index) + sum_{degree=1}^{max_degree} sum_{i} x_i^degree
  //
  // We use the same offsets for both CorrectionVars and FluxVars (via mapping
  // storage_index -> flux_storage_index) to ensure consistency.
  const auto set_polynomial_on_corrections =
      [max_degree](const gsl::not_null<CorrectionVars*> vars_ptr,
                   const auto& local_logical_coords) {
        (void)max_degree;
        for (size_t storage_index = 0;
             storage_index < get<Scalar0>(*vars_ptr).size(); ++storage_index) {
          get<Scalar0>(*vars_ptr)[storage_index] = 0.0;
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Scalar0>(*vars_ptr)[storage_index] +=
                  pow(local_logical_coords.get(i), degree);
            }
          }
        }
        for (size_t storage_index = 0;
             storage_index < get<Vector0<Dim>>(*vars_ptr).size();
             ++storage_index) {
          get<Vector0<Dim>>(*vars_ptr)[storage_index] =
              1.0 + (0.3 * static_cast<double>(storage_index));
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Vector0<Dim>>(*vars_ptr)[storage_index] +=
                  pow(local_logical_coords.get(i), degree);
            }
          }
        }
      };
  // Fill the ghost FluxVars so that the flux_storage_index component
  // for each evolved var and storage_index matches the correction polynomial.
  // This requires mapping correction (tag, storage_index) ->
  // flux (FluxTag, flux_storage_index), and writing the same polynomial
  // into the flux's flux_storage_index slot.
  const auto set_polynomial_on_ghost_fluxes =
      [max_degree](const gsl::not_null<FluxVars*> vars_ptr,
                   const auto& local_logical_coords, const size_t dim) {
        (void)max_degree;
        // For Scalar0: flux_multi_index = (dim,), flux has 1 component per
        // spatial dim. We need the dim-th component to match the correction's
        // polynomial (which has offset 0).
        auto& scalar_flux = get<Scalar0Flux>(*vars_ptr);
        // Zero everything first
        for (size_t si = 0; si < scalar_flux.size(); ++si) {
          scalar_flux[si] = 0.0;
        }
        // Fill the dim-th component with the correction polynomial.
        // For Scalar0, correction storage_index=0, tensor_index=().
        // flux_multi_index = prepend((), dim) = (dim,).
        // flux_storage_index = get_storage_index((dim,)) = dim
        // (for a tnsr::I, storage layout is component order).
        scalar_flux.get(dim) = 0.0;
        for (size_t degree = 1; degree <= max_degree; ++degree) {
          for (size_t i = 0; i < Dim; ++i) {
            scalar_flux.get(dim) += pow(local_logical_coords.get(i), degree);
          }
        }

        // For Vector0<Dim>: correction is tnsr::I<DV,Dim>
        //   storage_index -> tensor_index = (idx_i,)
        //   flux_multi_index = prepend((idx_i,), dim) = (dim, idx_i)
        //   flux = tnsr::IJ, flux.get(dim, idx_i) is the right component.
        auto& vector_flux = get<Vector0Flux>(*vars_ptr);
        for (size_t si = 0; si < vector_flux.size(); ++si) {
          vector_flux[si] = 0.0;
        }
        for (size_t idx_i = 0; idx_i < Dim; ++idx_i) {
          // The correction polynomial for Vector0's storage_index
          // corresponding to tensor_index (idx_i) has offset:
          //   1.0 + 0.3 * storage_index
          // where storage_index =
          // Vector0<Dim>::type::get_storage_index({idx_i})
          const size_t corr_storage_index =
              tnsr::I<DataVector, Dim, Frame::Inertial>::get_storage_index(
                  std::array<size_t, 1>{{idx_i}});
          vector_flux.get(dim, idx_i) =
              1.0 + (0.3 * static_cast<double>(corr_storage_index));
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              vector_flux.get(dim, idx_i) +=
                  pow(local_logical_coords.get(i), degree);
            }
          }
        }
      };
  const auto set_polynomial_divergence =
      [max_degree](const gsl::not_null<CorrectionVars*> d_vars_ptr,
                   const auto& local_logical_coords) {
        (void)max_degree;
        get(get<Scalar0>(*d_vars_ptr)) = 0.0;
        for (size_t i = 0; i < Dim; ++i) {
          get<Vector0<Dim>>(*d_vars_ptr).get(i) = 0.0;
        }
        for (size_t deriv_dim = 0; deriv_dim < Dim; ++deriv_dim) {
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            get(get<Scalar0>(*d_vars_ptr)) +=
                degree * pow(local_logical_coords.get(deriv_dim), degree - 1);
            for (size_t i = 0; i < Dim; ++i) {
              get<Vector0<Dim>>(*d_vars_ptr).get(i) +=
                  degree * pow(local_logical_coords.get(deriv_dim), degree - 1);
            }
          }
        }
      };

  CorrectionVars expected_divergence(mesh.number_of_grid_points());
  set_polynomial_divergence(&expected_divergence, logical_coords);

  // Compute second-order boundary corrections at face-centered positions
  // (these are -n_i F^i at each face).
  std::array<CorrectionVars, Dim> second_order_corrections{};
  for (size_t d = 0; d < Dim; ++d) {
    const auto basis = make_array<Dim>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<Dim>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<Dim>(points_per_dimension);
    gsl::at(extents, d) = points_per_dimension + 1;
    gsl::at(quadrature, d) = Spectral::Quadrature::FaceCentered;
    const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
    auto face_logical_coords = logical_coordinates(face_centered_mesh);
    for (size_t j = 1; j < Dim; ++j) {
      face_logical_coords.get(j) += 4.0 * static_cast<double>(j);
    }
    gsl::at(second_order_corrections, d)
        .initialize(face_centered_mesh.number_of_grid_points());
    set_polynomial_on_corrections(
        make_not_null(&gsl::at(second_order_corrections, d)),
        face_logical_coords);
    // We use n_i F^i in the code, so need to negate to get sign to agree.
    gsl::at(second_order_corrections, d) *= -1.0;
  }

  // Compute ghost face data for each direction.
  // The ghost data stores the boundary correction at ghost face positions in
  // the neighbor domain. It is stored in FluxVars format where the
  // flux_storage_index component for each correction component holds the
  // correction polynomial value.
  DirectionMap<Dim, FluxVars> neighbor_face_data{};
  if (number_of_ghost_faces > 0) {
    for (const auto& direction : Direction<Dim>::all_directions()) {
      const size_t d = direction.dimension();
      // Create a mesh for the ghost faces in the neighbor domain.
      // The ghost region has number_of_ghost_faces faces in the normal
      // direction and the same number of points in the transverse directions.
      auto ghost_extents = make_array<Dim>(points_per_dimension);
      gsl::at(ghost_extents, d) = number_of_ghost_faces;
      const auto ghost_basis =
          make_array<Dim>(Spectral::Basis::FiniteDifference);
      const auto ghost_quadrature =
          make_array<Dim>(Spectral::Quadrature::CellCentered);

      // The ghost face coordinates: neighbor domain is shifted by sign*2.0
      // in the d-th logical coordinate.
      // For the face-centered quadrature in direction d, the full neighbor has
      // points_per_dimension + 1 faces. We need the faces closest to our
      // boundary (excluding the shared boundary face).
      //
      // For the upper neighbor (direction.sign() > 0), the ghost faces are
      // the first number_of_ghost_faces interior faces of the neighbor (faces
      // at indices 1..number_of_ghost_faces when counting from lower boundary).
      // For the lower neighbor, it's the last number_of_ghost_faces interior
      // faces.
      //
      // We build the coordinates directly.
      const Mesh<Dim> ghost_mesh{ghost_extents, ghost_basis, ghost_quadrature};
      // Get the cell-centered coordinates in the transverse directions
      auto ghost_logical_coords = logical_coordinates(ghost_mesh);
      for (size_t j = 1; j < Dim; ++j) {
        ghost_logical_coords.get(j) += 4.0 * static_cast<double>(j);
      }

      // Overwrite the d-th coordinate with the actual ghost face positions.
      // Our element's faces span from -1 to +1 in logical coords with spacing
      // delta_face = 2/N where N = points_per_dimension.
      //
      // For upper neighbor: ghost face with index k in the ghost data is at
      //   1 + (k+1)*delta_face (k=0 is closest to boundary).
      // For lower neighbor: ghost face with index k is at
      //   -1 - (number_of_ghost_faces - k)*delta_face
      //   (k=0 is furthest from boundary, k=number_of_ghost_faces-1 is
      //   closest).
      const double delta_face = 2.0 / static_cast<double>(points_per_dimension);
      // Stride for dimension d in the ghost mesh (column-major)
      size_t stride_d = 1;
      for (size_t l = 0; l < d; ++l) {
        stride_d *= gsl::at(ghost_extents, l);
      }
      // Face positions in the unshifted logical coords are at
      // -1.0, -1.0 + delta, ..., 1.0. We add the same shift as the
      // volume/face coordinates: +4.0*d for dimension d.
      const double coord_shift = 4.0 * static_cast<double>(d);
      const size_t total_ghost_points = ghost_mesh.number_of_grid_points();
      for (size_t linear_idx = 0; linear_idx < total_ghost_points;
           ++linear_idx) {
        // Extract the d-th index from the linear index
        const size_t ghost_idx =
            (linear_idx / stride_d) % number_of_ghost_faces;
        double face_coord = 0.0;
        if (direction.sign() > 0.0) {
          face_coord = 1.0 + (static_cast<double>(ghost_idx + 1) * delta_face) +
                       coord_shift;
        } else {
          face_coord = -1.0 -
                       (static_cast<double>(number_of_ghost_faces - ghost_idx) *
                        delta_face) +
                       coord_shift;
        }
        ghost_logical_coords.get(d)[linear_idx] = face_coord;
      }

      neighbor_face_data[direction].initialize(
          ghost_mesh.number_of_grid_points());
      set_polynomial_on_ghost_fluxes(
          make_not_null(&neighbor_face_data[direction]), ghost_logical_coords,
          d);
      // Negate to match interior sign convention (-n_i F^i).
      neighbor_face_data[direction] *= -1.0;
    }
  }  // if number_of_ghost_faces > 0

  // Call the MD high-order flux function
  std::array<CorrectionVars, Dim> high_order_corrections{};
  ::fd::cartesian_high_order_fluxes(
      make_not_null(&high_order_corrections), second_order_corrections,
      neighbor_face_data, mesh, number_of_ghost_faces, correction_order);

  // Compute the Cartesian derivative of the high_order_corrections to
  // verify that it is computed sufficiently accurately.
  const DataVector inv_jacobian{mesh.number_of_grid_points(), 1.0};
  CorrectionVars flux_divergence{mesh.number_of_grid_points(), 0.0};
  for (size_t d = 0; d < Dim; ++d) {
    const auto& corrections_in_dim = gsl::at(high_order_corrections, d);
    // Note: assumes isotropic mesh
    const double one_over_delta_xi =
        -1.0 / (logical_coords.get(0)[1] - logical_coords.get(0)[0]);
    evolution::dg::subcell::add_cartesian_flux_divergence(
        make_not_null(&get(get<Scalar0>(flux_divergence))), one_over_delta_xi,
        inv_jacobian, get(get<Scalar0>(corrections_in_dim)), mesh.extents(), d);
    for (size_t i = 0; i < Dim; ++i) {
      evolution::dg::subcell::add_cartesian_flux_divergence(
          make_not_null(&get<Vector0<Dim>>(flux_divergence).get(i)),
          one_over_delta_xi, inv_jacobian,
          get<Vector0<Dim>>(corrections_in_dim).get(i), mesh.extents(), d);
    }
  }

  const Approx custom_approx = Approx::custom().epsilon(5.e-12);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Scalar0>(flux_divergence),
                               get<Scalar0>(expected_divergence),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Vector0<Dim>>(flux_divergence),
                               get<Vector0<Dim>>(expected_divergence),
                               custom_approx);
}

SPECTRE_TEST_CASE("Unit.FiniteDifference.CartesianHighOrderFluxCorrection",
                  "[Unit][NumericalAlgorithms]") {
  using DO = fd::DerivativeOrder;
  for (const fd::DerivativeOrder correction_order :
       {DO::Two, DO::FourMnd, DO::SixMnd, DO::EightMnd, DO::TenMnd,
        DO::OneHigherThanRecons, DO::OneHigherThanReconsButFiveToFour}) {
    test_using_nodes<1>(correction_order);
    test_using_nodes<2>(correction_order);
    test_using_nodes<3>(correction_order);
  }
  for (const fd::DerivativeOrder correction_order :
       {DO::Two, DO::FourMd, DO::SixMd, DO::EightMd, DO::TenMd}) {
    test<1>(correction_order);
    test<2>(correction_order);
    test<3>(correction_order);
  }
}
}  // namespace
