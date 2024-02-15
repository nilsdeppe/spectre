// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "NumericalAlgorithms/LinearSolver/Lapack.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/WrapText.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

void compute_delta_integral_logical(
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<DataVector*> integrand_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& phi,
    const Scalar<DataVector>& pi,
    const tnsr::I<DataVector, 1, Frame::ElementLogical>& logical,
    const Scalar<DataVector>& det_jacobian) {
  ASSERT(get(*delta).size() == get<0>(logical).size(),
         "Radius and delta must be of same size. Radius is: "
             << get<0>(logical).size()
             << " and delta is: " << get(*delta).size());
  ASSERT(integrand_buffer->size() == get<0>(logical).size(), "uh oh");

  *integrand_buffer =
      -M_PI * (square(get(pi)) + square(get(phi))) * get(det_jacobian);
  std::array<std::reference_wrapper<const Matrix>, 1> matrices{
      {std::cref(Spectral::integration_matrix(mesh_of_one_element))}};

  apply_matrices(make_not_null(&get(*delta)), matrices, *integrand_buffer,
                 mesh_of_one_element.extents());

  const size_t pts_per_element = mesh_of_one_element.number_of_grid_points();
  DataVector view{};
  for (size_t grid_index = pts_per_element; grid_index < get(pi).size();
       grid_index += pts_per_element) {
    view.set_data_ref(&get(*delta)[grid_index], pts_per_element);
    view += get(*delta)[grid_index - 1];
  }
}

const tnsr::I<DataVector, 1, Frame::ElementLogical> coordinate_transformation(
    const size_t number_of_elements, const size_t number_of_grid_points) {
  tnsr::I<DataVector, 1, Frame::ElementLogical> coordinates{};
}

void compute_mass_integral(
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Matrix*> matrix_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& phi,
    const Scalar<DataVector>& pi, const Scalar<DataVector>& det_jacobian,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const size_t spacetime_dim) {
  const size_t pts_per_element = mesh_of_one_element.number_of_grid_points();
  const size_t number_of_grids = get(pi).size() / pts_per_element;
  const Matrix& integration_matrix =
      Spectral::integration_matrix(mesh_of_one_element);
  for (size_t grid = 0; grid < number_of_grids; ++grid) {
    DataVector view{&get(*mass)[grid * pts_per_element], pts_per_element};
    for (size_t i = 0; i < pts_per_element; ++i) {
      view[i] = 0.0;
      for (size_t k = 0; k < pts_per_element; ++k) {
        const double common = integration_matrix(i, k) * M_PI *
                              get(det_jacobian)[k] *
                              (square(get(pi)[k]) + square(get(phi)[k]));
        view[i] += pow(get<0>(radius)[k], spacetime_dim - 2) * common;
        matrix_buffer->operator()(i, k) =
            (i == k ? 1.0 : 0.0) + 2.0 * get<0>(radius)[k] * common;
      }
    }
    // Solve the linear system A m = b for m (the mass)
    lapack::general_matrix_linear_solve(make_not_null(&view), matrix_buffer);
  }

  // Loop through each CG grid and add offset from previous grid integration
  // Should we change the name of this Datavector, since we have used view
  // before.
  DataVector view{};
  for (size_t grid_index = pts_per_element; grid_index < get(pi).size();
       grid_index += pts_per_element) {
    view.set_data_ref(&get(*mass)[grid_index - pts_per_element],
                      pts_per_element);
    view += get(*mass)[grid_index - 1];
  }
}

void compute_metric_function_a_from_mass(
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    const Scalar<DataVector>& mass,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const size_t spacetime_dim) {
  DataVector view_a{&get(*metric_function_a)[1],
                    get(*metric_function_a).size() - 1};
  const DataVector view_mass{&const_cast<double&>(get(mass)[1]),  // NOLINT
                             get(mass).size() - 1};
  const DataVector view_radius{
      &const_cast<double&>(get<0>(radius)[1]),  // NOLINT
      get<0>(radius).size() - 1};

  get(*metric_function_a)[0] = 1.0;
  view_a = 1.0 - 2.0 * view_mass / pow(view_radius, spacetime_dim - 3);
}

void compute_seg_ids(std::vector<SegmentId>& segids,
                     const size_t number_of_elements,
                     const size_t refinement_level) {
  std::cout << segids << "\n\n";
  for (size_t element_index = 0; element_index < number_of_elements;
       element_index += 1) {
    SegmentId segid{refinement_level, element_index};
    std::cout << segid << "\n\n";
    segids.push_back(segid);
  }
}
// void compute_time_derivatives_first_order(
//     const gsl::not_null<Scalar<DataVector>*> dt_psi,
//     const gsl::not_null<Scalar<DataVector>*> dt_phi,
//     const gsl::not_null<Scalar<DataVector>*> dt_pi,
//     const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& psi,
//     const Scalar<DataVector>& phi, const Scalar<DataVector>& pi,
//     const Scalar<DataVector>& metric_function_a,
//     const Scalar<DataVector>& metric_function_delta, const double gamma2,
//     const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
//     const InverseJacobian<DataVector, 1, Frame::Logical, Frame::Inertial>&
//         inverse_jacobian,
//     const InverseJacobian<DataVector, 1, Frame::Logical, Frame::Inertial>&
//         inverse_jacobian_divided_by_radius,
//     const size_t spacetime_dim)  {
//   std::array<std::reference_wrapper<const Matrix>, 1> logical_diff_matrices{
//       {std::cref(Spectral::differentiation_matrix(mesh_of_one_element))}};
//   // Assemble dt_pi, using dt_psi as a buffer
//   {
//     // Use dt_phi as buffer for A exp^{-delta}
//     get(*dt_phi) = get(metric_function_a) * exp(-get(metric_function_delta));
//     const Scalar<DataVector>& metric_terms = *dt_phi;

//     // Compute second term, dr(A e^{-delta} Phi)
//     get(*dt_psi) = get(metric_terms) * get(phi);
//     apply_matrices(make_not_null(&get(*dt_pi)), logical_diff_matrices,
//                    get(*dt_psi), mesh_of_one_element.extents());
//     get(*dt_pi) *= get<0, 0>(inverse_jacobian);

//     apply_matrices(make_not_null(&get(*dt_psi)), logical_diff_matrices,
//                    get(psi), mesh_of_one_element.extents());
//     get(*dt_pi) += (spacetime_dim - 2.0) * get(metric_terms) *
//                    get<0, 0>(inverse_jacobian_divided_by_radius) *
//                    get(*dt_psi);

//     // Compute dt_phi
//     get(*dt_psi) = get(metric_terms) * get(pi) + gamma2 * get(psi);
//   }
//   apply_matrices(make_not_null(&get(*dt_phi)), logical_diff_matrices,
//                  get(*dt_psi), mesh_of_one_element.extents());
//   get(*dt_phi) *= get<0, 0>(inverse_jacobian);
//   get(*dt_phi) -= gamma2 * get(phi);

//   // Compute dt_psi, need to remove constraint damping term that was added
//   // while compute dt_phi
//   get(*dt_psi) -= gamma2 * get(psi);

//   // Apply boundary conditions at outer boundary.
//   const size_t outer_boundary_index = get(psi).size() - 1;
//   get(*dt_psi)[outer_boundary_index] =
//       -get(psi)[outer_boundary_index] / get<0>(radius)[outer_boundary_index]
//       - get(phi)[outer_boundary_index];

//   // Phi boundary condition
//   double logical_d_phi_at_boundary = 0.0;
//   for (size_t i = 0,
//               last_mesh_location =
//                   get(phi).size() -
//                   mesh_of_one_element.number_of_grid_points();
//        i < mesh_of_one_element.number_of_grid_points(); ++i) {
//     logical_d_phi_at_boundary +=
//         logical_diff_matrices[0].get()(
//             mesh_of_one_element.number_of_grid_points(), i) *
//         get(phi)[last_mesh_location + i];
//   }
//   get(*dt_phi)[outer_boundary_index] =
//       get(psi)[outer_boundary_index] /
//           square(get<0>(radius)[outer_boundary_index]) -
//       get(phi)[outer_boundary_index] / get<0>(radius)[outer_boundary_index] +
//       get<0, 0>(inverse_jacobian)[outer_boundary_index] *
//           logical_d_phi_at_boundary;

//   // Pi boundary condition
// }

int main(int argc, char** argv) {
  const size_t refinement_level = 2;
  const size_t number_of_elements = 4;
  const Mesh<1> mesh_of_one_element{8, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};
  Scalar<DataVector> delta{mesh_of_one_element.number_of_grid_points() *
                           number_of_elements};
  Scalar<DataVector> jacobian{
      mesh_of_one_element.number_of_grid_points() * number_of_elements, 0.0};
  DataVector integrand_buffer{mesh_of_one_element.number_of_grid_points() *
                              number_of_elements};
  const tnsr::I<DataVector, 1, Frame::ElementLogical> logical_coord{
      logical_coordinates(mesh_of_one_element)};

  const std::array<DataVector, 1> logical_coords_array{logical_coord.get(0)};

  std::vector<SegmentId> segids = {};
  compute_seg_ids(segids, number_of_elements, refinement_level);

  tnsr::I<DataVector, 1, Frame::ElementLogical> coordinate_values{
      mesh_of_one_element.number_of_grid_points() * number_of_elements, 0.0};

  for (size_t element_index = 0; element_index < number_of_elements;
       element_index += 1) {
    double lower = segids[element_index].endpoint(Side::Lower);
    double upper = segids[element_index].endpoint(Side::Upper);
    domain::CoordinateMaps::Affine affine_map(-1, 1, lower, upper);

    DataVector jacobian_of_element{
        std::next(get(jacobian).data(),
                  element_index * mesh_of_one_element.number_of_grid_points()),
        mesh_of_one_element.number_of_grid_points()};
    jacobian_of_element = affine_map.jacobian(logical_coords_array)[0];

    for (size_t coord_index = 0;
         coord_index < mesh_of_one_element.number_of_grid_points();
         coord_index += 1) {
      std::array<double, 1> point{};
      std::array<double, 1> point_to_check{logical_coord[0][coord_index]};
      point = affine_map(point_to_check);
      coordinate_values.get(
          0)[element_index * mesh_of_one_element.number_of_grid_points() +
             coord_index] = point[0];
    }
  }

  std::cout << coordinate_values << "\n\n";

  const Scalar<DataVector> pi{
      DataVector{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}};
  const Scalar<DataVector> phi{
      DataVector{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}};

  compute_delta_integral_logical(&delta, &integrand_buffer, mesh_of_one_element,
                                 phi, pi, coordinate_values, jacobian);
  std::cout << 'next' << std::setprecision(7) << delta << "\n\n";

  return 0;
  boost::program_options::options_description desc(wrap_text(
      "Spherical gravitational collapse using one-sided Legendre polynomials "
      "at r=0 to analytically regularize the evolution equations. The metric "
      "used is:\n\n"
      "  ds^2 = -A exp(-2delta)dt^2 + (1/A) dr^2 + r^{n-2}d Omega^{n-2}\n\n"
      "where n is the number of spacetime dimensions, and A and delta are "
      "metric functions depending on space and time. This form of the metric "
      "is an Schwarzschild-like coordinates and so the event horizon can never "
      "be reach. Instead, A goes to zero where the event horizon is and so "
      "some finite cutoff must be chosen. A smaller cutoff means more "
      "accurately determining the horizon location (and thus the mass of the "
      "black hole), but also a longer simulation time."
      "\n\nOptions",
      79));
  desc.add_options()("help,h,", "show this help message")(
      "input_file", boost::program_options::value<std::string>()->required(),
      "input file to use for evolution");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u or vars.count("input_file") == 0u) {
    Parallel::printf("%s\n", desc);
    return 0;
  }
}
