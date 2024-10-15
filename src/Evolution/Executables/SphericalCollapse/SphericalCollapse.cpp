// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need Boost MultiArray because it is used internally by ODEINT
#include "DataStructures/BoostMultiArray.hpp"

#include <algorithm>
#include <boost/numeric/odeint.hpp>
#include <boost/program_options.hpp>
#include <cstddef>
#include <iostream>
#include <limits>

#include <iterator>
#include <limits>
#include <string>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "NumericalAlgorithms/LinearSolver/Lapack.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Auto.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/WrapText.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty

extern "C" void CkRegisterMainModule(void) {}

namespace Tags {
struct Amplitude : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {"The amplitude of the scalar wave."};
};
struct Width : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {
      "The width of the exp(-(r-center)^p/width^p) factor of the Gaussian "
      "scalar wave."};
};
struct ExponentP : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {
      "The exponent of the exp(-(r-center)^p/width^p) factor of the Gaussian "
      "scalar wave."};
};
struct ExponentQ : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {
      "The exponent of the r^(2q) factor of the Gaussian scalar wave."};
};
struct Center : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {
      "The radial center of the scalar wave."};
};

struct Gamma2 : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {
      "The constraint damping parameter, gamma2."};
};

struct SpacetimeDimensions : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {
      "The number of spacetime dimensions. Should generally be an integer, but "
      "investigating non-integer values may be interesting."};
};
struct HorizonFinderTolerance : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {
      "The value that A needs to reach for us to decide that a black "
      "hole/horizon has formed. 0.01 is a reasonable value."};
};

struct InnerRefinementLevel : db::SimpleTag {
  using type = size_t;
  static constexpr Options::String help = {"The refinement level at r=0."};
};
struct OuterRefinementLevel : db::SimpleTag {
  using type = size_t;
  static constexpr Options::String help = {
      "The refinement level at the outer boundary."};
};
struct PointsPerElement : db::SimpleTag {
  using type = size_t;
  static constexpr Options::String help = {
      "The number of grid points per element."};
};
struct FilterAlpha : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {
      "The alpha used in the exponential filter."};
};

struct FilterHalfPower : db::SimpleTag {
  using type = size_t;
  static constexpr Options::String help = {
      "The half power of the exponential filter."};
};

struct OuterBoundaryRadius : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {
      "The radius of the outer boundary."};
};
struct FinalTime : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {
      "The time at which the simulation is ended if no black hole formed."};
};
struct CflFactor : db::SimpleTag {
  using type = double;
  static constexpr Options::String help = {
      "The CFL factor used for time integration.."};
};

struct VolumeDataDirectory : db::SimpleTag {
  using type = std::string;
  static constexpr Options::String help = {
      "The name of the directory into which volume data is written."};
};
struct VolumeDataOutputFrequency : db::SimpleTag {
  using type = size_t;
  static constexpr Options::String help = {
      "How many time steps to output data."};
};
struct TimePrintFrequency : db::SimpleTag {
  struct DoNotPrintTimeInfo {};
  using type = Options::Auto<size_t, DoNotPrintTimeInfo>;
  static constexpr Options::String help = {
      "How many time steps to print the current time and time step to screen."};
};

struct UseFlatSpace : db::SimpleTag {
  using type = bool;
  static constexpr Options::String help = {
      "Use a flat spacetime instead of a dynamic one."};
};

using options_list =
    tmpl::list<Amplitude, Width, ExponentP, ExponentQ, Center, Gamma2,
               SpacetimeDimensions, HorizonFinderTolerance,
               InnerRefinementLevel, OuterRefinementLevel, PointsPerElement,
               FilterAlpha, FilterHalfPower, OuterBoundaryRadius, FinalTime,
               CflFactor, VolumeDataDirectory, VolumeDataOutputFrequency,
               TimePrintFrequency, UseFlatSpace>;
}  // namespace Tags

using options_list = Tags::options_list;

/*
 * \brief A very simple ElementId so we are not limited by the refinement
 * levels that are realistic for a 3d code. In 1d you can afford much higher
 * refinement.
 */
struct ElementId1d {
  size_t block_id;
  SegmentId segment_id;
};

static bool use_flat_space = false;

Scalar<DataVector> differential_eq_for_A(
    const Scalar<DataVector>& phi, const Scalar<DataVector>& pi,
    const Scalar<DataVector>& A,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const double spacetime_dim) {
  Scalar<DataVector> diff_eq{get<0>(radius).size()};
  get(diff_eq)[0] = 0.0;
  for (size_t i = 1; i < get<0>(radius).size(); i++) {
    get(diff_eq)[i] =
        ((spacetime_dim - 3) / get<0>(radius))[i] * (1 - get(A)[i]) -
        2 * M_PI * get<0>(radius)[i] * get(A)[i] *
            (square(get(pi)[i]) + square(get(phi)[i]));

    if (use_flat_space) {
      get(diff_eq)[i] = 0.0;
    }
  }
  return diff_eq;
}

Scalar<DataVector> differential_eq_for_delta(
    const Scalar<DataVector>& phi, const Scalar<DataVector>& pi,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius) {
  Scalar<DataVector> diff_eq{-4 * M_PI * get<0>(radius) *
                             (square(get(pi)) + square(get(phi)))};
  if (use_flat_space) {
    get(diff_eq) = 0.0;
  }
  return diff_eq;
}

void compute_delta_integral_logical(
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<DataVector*> integrand_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& phi,
    const Scalar<DataVector>& pi, const Scalar<DataVector>& det_jacobian,
    const tnsr::I<DataVector, 1, Frame::Inertial>& /*radius*/,
    const double outer_boundary_radius) {
  (*integrand_buffer) = -M_PI * (square(get(pi)) + square(get(phi))) *
                        get(det_jacobian) * square(outer_boundary_radius);

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

  if (use_flat_space) {
    get(*delta) = 0.0;
  }
}

void compute_mass_integral(
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Matrix*> matrix_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& phi,
    const Scalar<DataVector>& pi, const Scalar<DataVector>& det_jacobian,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const double spacetime_dim, const double outer_boundary_radius) {
  const size_t pts_per_element = mesh_of_one_element.number_of_grid_points();
  const size_t number_of_grids = get(pi).size() / pts_per_element;
  const Matrix& integration_matrix =
      Spectral::integration_matrix(mesh_of_one_element);
  for (size_t grid = 0; grid < number_of_grids; ++grid) {
    DataVector view{&get(*mass)[grid * pts_per_element], pts_per_element};
    const double boundary_condition =
        grid == 0 ? 0.0 : get(*mass)[grid * pts_per_element - 1];
    for (size_t i = 0; i < pts_per_element; ++i) {
      view[i] = boundary_condition;
      for (size_t k = 0; k < pts_per_element; ++k) {
        const size_t index = k + grid * pts_per_element;
        const double sigma =
            0.5 * M_PI * (square(get(pi)[index]) + square(get(phi)[index])) *
            square(outer_boundary_radius);
        view[i] += integration_matrix(i, k) * 0.5 * sigma *
                   get(det_jacobian)[index] *
                   pow(get<0>(radius)[index], spacetime_dim - 3);
        matrix_buffer->operator()(i, k) =
            (i == k ? 1.0 : 0.0) +
            integration_matrix(i, k) * sigma * get(det_jacobian)[index];
      }
    }
    // Solve the linear system A m = b for m (the mass)
    lapack::general_matrix_linear_solve(make_not_null(&view), matrix_buffer);
  }

  if (use_flat_space) {
    get(*mass) = 0.0;
  }
}

void compute_metric_function_a_from_mass(
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    const Scalar<DataVector>& mass,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const double spacetime_dim) {
  DataVector view_a{&get(*metric_function_a)[1],
                    get(*metric_function_a).size() - 1};
  const DataVector view_mass{&const_cast<double&>(get(mass)[1]),  // NOLINT
                             get(mass).size() - 1};
  const DataVector view_radius{
      &const_cast<double&>(get<0>(radius)[1]),  // NOLINT
      get<0>(radius).size() - 1};

  get(*metric_function_a)[0] = 1.0;
  view_a = 1.0 - 2.0 * view_mass / pow(view_radius, spacetime_dim - 3);
  for (size_t i = 1; i < get(mass).size(); ++i) {
    get(*metric_function_a)[i] =
        1.0 - 2.0 * get(mass)[i] / pow(get<0>(radius)[i], spacetime_dim - 3.0);
  }
}

/*
 * \brief Create `ElementId`s in a non-uniform manner from
 * `inner_refinement_level` to the `outer_refinement_level`.
 *
 * The outer half of the elements are at the `outer_refinement_level`, then
 * we go self-similarly inwards. This means the outer index of each refinement
 * level is always `2^(outer_refinement_level - 1)`.
 */
std::vector<ElementId1d> compute_element_ids(
    const size_t inner_refinement_level, const size_t outer_refinement_level) {
  std::vector<ElementId1d> element_ids;
  const size_t block_id = 0;
  for (size_t j = inner_refinement_level; j >= outer_refinement_level; j--) {
    for (size_t element_index = (j == inner_refinement_level
                                     ? 0
                                     : two_to_the(outer_refinement_level - 1));
         element_index < two_to_the(outer_refinement_level); element_index++) {
      element_ids.emplace_back(block_id, SegmentId{j, element_index});
    }
  }
  return element_ids;
}

void compute_time_derivatives_first_order_2(
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const gsl::not_null<Scalar<DataVector>*> dt_phi_tilde,
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& phi_tilde, const Scalar<DataVector>& pi,
    const Scalar<DataVector>& phi, const Scalar<DataVector>& metric_function_a,
    const Scalar<DataVector>& metric_function_delta, const double gamma2,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const Scalar<DataVector>& det_inverse_jacobian, const double spacetime_dim,
    const double outer_boundary_radius,
    const std::array<std::reference_wrapper<const Matrix>, 1>&
        filter_matrices) {
  Scalar<DataVector> diff_eq_A =
      differential_eq_for_A(phi, pi, metric_function_a, radius, spacetime_dim);
  Scalar<DataVector> diff_eq_delta = differential_eq_for_delta(phi, pi, radius);
  const size_t number_of_elements =
      get(pi).size() / mesh_of_one_element.number_of_grid_points();
  Scalar<DataVector> buffer1{mesh_of_one_element.number_of_grid_points() *
                             number_of_elements};
  Scalar<DataVector> buffer2{mesh_of_one_element.number_of_grid_points() *
                             number_of_elements};
  Scalar<DataVector> buffer4{mesh_of_one_element.number_of_grid_points() *
                             number_of_elements};

  std::array<std::reference_wrapper<const Matrix>, 1> logical_diff_matrices{
      {std::cref(Spectral::differentiation_matrix(mesh_of_one_element))}};

  // compute dt_psi
  get(buffer1) = get(metric_function_a) * exp(-get(metric_function_delta));
  get(*dt_psi) = get(buffer1) * get(pi);

  // compute 2nd term of dt_pi
  apply_matrices(make_not_null(&get(*dt_pi)), logical_diff_matrices,
                 get(phi_tilde), mesh_of_one_element.extents());
  get(*dt_pi) *= (4.0 * get<0>(radius) / square(outer_boundary_radius));
  get(*dt_pi) *= get(det_inverse_jacobian) * get(buffer1);
  get(*dt_pi) += (get(diff_eq_A) * exp(-get(metric_function_delta)) -
                  get(diff_eq_delta) * get(buffer1)) *
                 *get(phi_tilde);
  get(*dt_pi) *= 4.0 * get<0>(radius);

  // compute 1st term of dt_pi
  get(*dt_pi) += 4.0 * (spacetime_dim - 1.0) * get(buffer1) *
                 get(phi_tilde);  // adding in the first term
                                  // of the expansion

  // compute dt_phi_tilde
  get(buffer2) = get(buffer1) * get(pi) + gamma2 * get(psi);

  apply_matrices(make_not_null(&get(*dt_phi_tilde)), logical_diff_matrices,
                 get(buffer2), mesh_of_one_element.extents());
  get(*dt_phi_tilde) *=
      get(det_inverse_jacobian);  //  * (1.0 / square(outer_boundary_radius));
  get(*dt_phi_tilde) *= (1.0 / square(outer_boundary_radius));

  get(*dt_phi_tilde) -= gamma2 * get(phi_tilde);

  {
    DataVector pre_filter_data{get(*dt_psi)};
    apply_matrices(make_not_null(&get(*dt_psi)), filter_matrices,
                   pre_filter_data, mesh_of_one_element.extents());
    pre_filter_data = get(*dt_pi);
    apply_matrices(make_not_null(&get(*dt_pi)), filter_matrices,
                   pre_filter_data, mesh_of_one_element.extents());
    pre_filter_data = get(*dt_phi_tilde);
    apply_matrices(make_not_null(&get(*dt_phi_tilde)), filter_matrices,
                   pre_filter_data, mesh_of_one_element.extents());
  }

  for (size_t element = mesh_of_one_element.number_of_grid_points();
       element <
       number_of_elements * mesh_of_one_element.number_of_grid_points() - 1;
       element = element + mesh_of_one_element.number_of_grid_points()) {
    const double lower_jacobian =
        (square(outer_boundary_radius) / (4.0 * get<0>(radius)[element - 1])) /
        get(det_inverse_jacobian)[element - 1];
    const double upper_jacobian =
        (square(outer_boundary_radius) / (4.0 * get<0>(radius)[element])) /
        get(det_inverse_jacobian)[element];

    // CG
    //
    // Note: we assume the weights are the same on both sides. This is true
    // for uniform p-refinement.
    //
    // However, we do still need to weight by the Jacobians on the two sides.
    get(*dt_psi)[element] = (upper_jacobian * get(*dt_psi)[element] +
                             lower_jacobian * get(*dt_psi)[element - 1]) /
                            (lower_jacobian + upper_jacobian);
    get(*dt_psi)[element - 1] = get(*dt_psi)[element];
    get(*dt_phi_tilde)[element] =
        (upper_jacobian * get(*dt_phi_tilde)[element] +
         lower_jacobian * get(*dt_phi_tilde)[element - 1]) /
        (lower_jacobian + upper_jacobian);
    get(*dt_phi_tilde)[element - 1] = get(*dt_phi_tilde)[element];
    get(*dt_pi)[element] = (upper_jacobian * get(*dt_pi)[element] +
                            lower_jacobian * get(*dt_pi)[element - 1]) /
                           (lower_jacobian + upper_jacobian);
    get(*dt_pi)[element - 1] = get(*dt_pi)[element];
  }
  const size_t outer_boundary_index = get(psi).size() - 1;
  get(*dt_pi)[outer_boundary_index] =
      (-get(*dt_phi_tilde)[outer_boundary_index] * 4 *
           get<0>(radius)[outer_boundary_index] -
       get(*dt_psi)[outer_boundary_index] /
           get<0>(radius)[outer_boundary_index]) /
      (get(metric_function_a)[outer_boundary_index] *
       exp(-get(metric_function_delta)[outer_boundary_index]));
}

double compute_adaptive_step_size(
    const Scalar<DataVector>& delta,
    const Scalar<DataVector>& metric_function_a,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const double CFL_safety_factor) {
  double min_adapted_dt = 1.0e300;

  for (size_t i = 0; i < get(delta).size() - 1; i++) {
    double dt = 0.0;
    if ((get<0>(radius)[i + 1] - get<0>(radius)[i]) >
        1.0e-14 * get<0>(radius)[i]) {
      dt = (get<0>(radius)[i + 1] - get<0>(radius)[i]) * exp(get(delta)[i]) /
           get(metric_function_a)[i];
      if (dt < min_adapted_dt) {
        min_adapted_dt = dt;
      }
    }
  }

  min_adapted_dt = CFL_safety_factor * min_adapted_dt;
  return min_adapted_dt;
}

void write_data_hd5file(const std::vector<ElementVolumeData>& volume_data,
                        const observers::ObservationId& observation_id,
                        const double lower_r, const double upper_r,
                        const double time) {
  const std::string h5_file_name{"VolumeDataForFields"};
  const std::string input_source{""};
  const std::string subfile_path{"/ElementData"};
  const uint32_t version_number = 0;
  h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name + ".h5"s, true,
                                                input_source};
  auto& volume_file =
      h5_file.try_insert<h5::VolumeData>(subfile_path, version_number);

  // Just write an invalid domain for now.
  domain::creators::Interval interval{std::array{lower_r}, std::array{upper_r},
                                      std::array{0_st}, std::array{10_st}};
  Domain<1> domain = interval.create_domain();
  const auto serialized_domain = serialize(domain);
  volume_file.write_volume_data(observation_id.hash(), time, volume_data,
                                serialized_domain);
}

void create_data_for_file(
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const Mesh<1>& mesh_of_one_element,
    const std::vector<ElementId1d>& element_ids,
    const std::array<DataVector, 3>& vars,
    const gsl::not_null<DataVector*> integrand_buffer,
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    const double /*gamma2*/, const Scalar<DataVector>& det_jacobian,
    const gsl::not_null<Matrix*> matrix_buffer, const double spacetime_dim,
    const double outer_boundary_radius, const size_t step_number,
    const double time, const std::string& volume_data_directory) {
  const Scalar<DataVector> temp_phi{vars[1] * 4 * get<0>(radius)};
  const Scalar<DataVector> temp_pi{vars[2]};
  compute_delta_integral_logical(delta, integrand_buffer, mesh_of_one_element,
                                 temp_phi, temp_pi, det_jacobian, radius,
                                 outer_boundary_radius);
  compute_mass_integral(mass, matrix_buffer, mesh_of_one_element, temp_phi,
                        temp_pi, det_jacobian, radius, spacetime_dim,
                        outer_boundary_radius);
  compute_metric_function_a_from_mass(metric_function_a, *mass, radius,
                                      spacetime_dim);
  std::stringstream data_to_write{};
  data_to_write
      << std::setprecision(18) << std::scientific << "# Time: " << time
      << "\n# 0 radius\n# 1 psi\n# 2 phi\n# 3 phi_tilde\n# 4 pi\n# 5 delta\n"
      << "# 6 mass\n# 7 A\n# 8 dt_psi\n# 9 dt_phi_tilde\n# 10 dt_pi\n";
  for (size_t i = 0; i < get<0>(radius).size(); ++i) {
    data_to_write << std::setprecision(18) << get<0>(radius)[i] << ' '
                  << vars[0][i] << ' ' << get(temp_phi)[i] << ' ' << vars[1][i]
                  << ' ' << get(temp_pi)[i] << ' ' << get(*delta)[i] << ' '
                  << get(*mass)[i] << ' ' << get(*metric_function_a)[i] << "\n";
  }
  std::ofstream out_file{volume_data_directory + "/Step" +
                         std::to_string(step_number) + ".txt"};
  out_file << data_to_write.str();
  out_file.close();
  return;

  std::vector<ElementVolumeData> volume_data;
  for (size_t element_i = 0; element_i < element_ids.size(); ++element_i) {
    const ElementId<1>& element_id{
        element_ids[element_i].block_id,
        std::array{element_ids[element_i].segment_id}};
    const size_t grid_i =
        element_i * mesh_of_one_element.number_of_grid_points();
    std::vector<TensorComponent> Data;
    const auto add_variable = [&Data](const std::string& name,
                                      const DataVector& variable) {
      Data.emplace_back(name, variable);
    };
    const std::string in_name1{"Psi"};
    const std::string in_name2{"Phi"};
    const std::string in_name3{"Pi"};
    const std::string in_name4{"Mass"};
    const std::string in_name5{"A"};
    const std::string in_name6{"Delta"};
    const DataVector Psi_per_element{
        &const_cast<double&>(vars[0][grid_i]),  // NOLINT
        mesh_of_one_element.number_of_grid_points()};
    const DataVector Phi_per_element{
        &const_cast<double&>(vars[1][grid_i]),  // NOLINT
        mesh_of_one_element.number_of_grid_points()};
    const DataVector Pi_per_element{
        &const_cast<double&>(vars[2][grid_i]),  // NOLINT
        mesh_of_one_element.number_of_grid_points()};
    const DataVector Mass_per_element{
        &const_cast<double&>(get(*mass)[grid_i]),  // NOLINT
        mesh_of_one_element.number_of_grid_points()};
    const DataVector A_per_element{
        &const_cast<double&>(get(*metric_function_a)[grid_i]),  // NOLINT
        mesh_of_one_element.number_of_grid_points()};
    const DataVector Delta_per_element{
        &const_cast<double&>(get(*delta)[grid_i]),  // NOLINT
        mesh_of_one_element.number_of_grid_points()};
    const DataVector radius_in_element{
        &const_cast<double&>(get<0>(radius)[grid_i]),  // NOLINT
        mesh_of_one_element.number_of_grid_points()};
    add_variable("Radius", radius_in_element);
    add_variable(in_name1, Psi_per_element);
    add_variable(in_name2, Phi_per_element);
    add_variable(in_name3, Pi_per_element);
    add_variable(in_name4, Mass_per_element);
    add_variable(in_name5, A_per_element);
    add_variable(in_name6, Delta_per_element);

    volume_data.emplace_back(element_id, Data, mesh_of_one_element);
  }

  const std::string tag{"ElementData"};
  // Use the step_number since we can end up taking very small time steps as a
  // BH forms.
  write_data_hd5file(
      volume_data,
      observers::ObservationId{static_cast<double>(step_number), tag}, 0.0,
      get<0>(radius)[get<0>(radius).size() - 1], time);
}

std::optional<double> find_min_A(
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const double horizon_tolerance) {
  for (size_t index = 0; index < get(*metric_function_a).size(); index++) {
    if (abs(get(*metric_function_a)[index]) < horizon_tolerance) {
      return get<0>(radius)[index];
    }
  }
  return std::nullopt;
}

std::array<DataVector, 3> integrate_fields_in_time(
    const gsl::not_null<DataVector*> integrand_buffer,
    const Scalar<DataVector>& det_jacobian,
    const gsl::not_null<Matrix*> matrix_buffer,
    const Mesh<1>& mesh_of_one_element,
    const std::vector<ElementId1d>& element_ids, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& phi_tilde, const Scalar<DataVector>& pi,
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const Scalar<DataVector>& det_inverse_jacobian, const db::Access& box,
    const std::array<std::reference_wrapper<const Matrix>, 1>&
        filter_matrices) {
  using Vars = std::array<DataVector, 3>;
  const size_t observation_frequency =
      get<Tags::VolumeDataOutputFrequency>(box);
  const std::string& volume_data_directory =
      get<Tags::VolumeDataDirectory>(box);
  const std::optional<size_t>& time_print_frequency =
      get<Tags::TimePrintFrequency>(box);

  Vars vars{get(psi), get(phi_tilde), get(pi)};

  using StateDopri5 = boost::numeric::odeint::runge_kutta_dopri5<Vars>;
  StateDopri5 st{};
  std::vector<double> times;

  double time = 0.0;
  double dt = 0.0;
  const bool filter_evolved_vars = false;
  size_t step = 0;
  std::optional<double> black_hole_radius{};

  using std::abs;
  while (abs(time) < get<Tags::FinalTime>(box)) {
    auto system = [&mesh_of_one_element, &metric_function_a, &delta, &mass,
                   &radius, &det_inverse_jacobian, &integrand_buffer,
                   &det_jacobian, &matrix_buffer, &box, &filter_matrices,
                   &element_ids, filter_evolved_vars](
                      const Vars& local_vars, Vars& local_dvars,
                      [[maybe_unused]] const double current_time) {
      (void)filter_evolved_vars;  // silence compiler warning

      Scalar<DataVector> temp_psi{
          const_cast<DataVector&>(local_vars[0]).data(),  // NOLINT
          local_vars[0].size()};
      Scalar<DataVector> temp_phi_tilde{
          const_cast<DataVector&>(local_vars[1]).data(),  // NOLINT
          local_vars[1].size()};
      Scalar<DataVector> temp_pi{
          const_cast<DataVector&>(local_vars[2]).data(),  // NOLINT
          local_vars[2].size()};

      if (filter_evolved_vars) {
        DataVector no_filter{get(temp_psi)};
        apply_matrices(make_not_null(&get(temp_psi)), filter_matrices,
                       no_filter, mesh_of_one_element.extents());
        no_filter = get(temp_pi);
        apply_matrices(make_not_null(&get(temp_pi)), filter_matrices, no_filter,
                       mesh_of_one_element.extents());
        no_filter = get(temp_phi_tilde);
        apply_matrices(make_not_null(&get(temp_phi_tilde)), filter_matrices,
                       no_filter, mesh_of_one_element.extents());
        for (size_t i = 1; i < element_ids.size(); ++i) {
          get(temp_psi)[i * mesh_of_one_element.number_of_grid_points()] =
              0.5 *
              (get(temp_psi)[i * mesh_of_one_element.number_of_grid_points()] +
               get(temp_psi)[i * mesh_of_one_element.number_of_grid_points() -
                             1]);
          get(temp_psi)[i * mesh_of_one_element.number_of_grid_points() - 1] =
              get(temp_psi)[i * mesh_of_one_element.number_of_grid_points()];
          get(temp_phi_tilde)[i * mesh_of_one_element.number_of_grid_points()] =
              0.5 * (get(temp_phi_tilde)[i * mesh_of_one_element
                                                 .number_of_grid_points()] +
                     get(temp_phi_tilde)
                         [i * mesh_of_one_element.number_of_grid_points() - 1]);
          get(temp_phi_tilde)[i * mesh_of_one_element.number_of_grid_points() -
                              1] =
              get(temp_phi_tilde)[i *
                                  mesh_of_one_element.number_of_grid_points()];
          get(temp_pi)[i * mesh_of_one_element.number_of_grid_points()] =
              0.5 *
              (get(temp_pi)[i * mesh_of_one_element.number_of_grid_points()] +
               get(temp_pi)[i * mesh_of_one_element.number_of_grid_points() -
                            1]);
          get(temp_pi)[i * mesh_of_one_element.number_of_grid_points() - 1] =
              get(temp_pi)[i * mesh_of_one_element.number_of_grid_points()];
        }
      }

      Scalar<DataVector> temp_phi{get(temp_phi_tilde) * 4 * get<0>(radius)};

      compute_delta_integral_logical(
          delta, integrand_buffer, mesh_of_one_element, temp_phi, temp_pi,
          det_jacobian, radius, get<Tags::OuterBoundaryRadius>(box));
      compute_mass_integral(mass, matrix_buffer, mesh_of_one_element, temp_phi,
                            temp_pi, det_jacobian, radius,
                            get<Tags::SpacetimeDimensions>(box),
                            get<Tags::OuterBoundaryRadius>(box));
      compute_metric_function_a_from_mass(metric_function_a, *mass, radius,
                                          get<Tags::SpacetimeDimensions>(box));

      const auto size = get(temp_psi).size();
      Scalar<DataVector> temp_dtpsi{size, 0.0};
      Scalar<DataVector> temp_dtphi_tilde{size, 0.0};
      Scalar<DataVector> temp_dtpi{size, 0.0};
      compute_time_derivatives_first_order_2(
          make_not_null(&temp_dtpsi), make_not_null(&temp_dtphi_tilde),
          make_not_null(&temp_dtpi), mesh_of_one_element, temp_psi,
          temp_phi_tilde, temp_pi, temp_phi, *metric_function_a, *delta,
          get<Tags::Gamma2>(box), radius, det_inverse_jacobian,
          get<Tags::SpacetimeDimensions>(box),
          get<Tags::OuterBoundaryRadius>(box), filter_matrices);

      local_dvars[0] = get(temp_dtpsi);
      local_dvars[1] = get(temp_dtphi_tilde);
      local_dvars[2] = get(temp_dtpi);
    };

    if (not use_flat_space) {
      black_hole_radius =
          find_min_A(metric_function_a, radius,
                     get<Tags::HorizonFinderTolerance>(box));
    }
    if (step % observation_frequency == 0 or black_hole_radius.has_value()) {
      create_data_for_file(radius, mesh_of_one_element, element_ids, vars,
                           integrand_buffer, mass, delta, metric_function_a,
                           get<Tags::Gamma2>(box), det_jacobian, matrix_buffer,
                           get<Tags::SpacetimeDimensions>(box),
                           get<Tags::OuterBoundaryRadius>(box), step, time,
                           volume_data_directory);
      if (black_hole_radius.has_value()) {
        std::cout << "Found black hole!!\nRadius: " << black_hole_radius.value()
                  << "\nTime: " << time << "\n";
        return vars;
      }
    }
    if (time_print_frequency.has_value() and
        (step % time_print_frequency.value() == 0)) {
      std::cout << "time: " << time << " step: " << step << " dt: " << dt
                << "\n";
    }

    dt = compute_adaptive_step_size(*delta, *metric_function_a, radius,
                                    get<Tags::CflFactor>(box));
    st.do_step(system, vars, time, dt);

    time = time + dt;
    ++step;
  }
  return vars;
}

void run(const db::Access& box) {
  domain::creators::register_derived_with_charm();
  const std::vector<ElementId1d> element_ids =
      compute_element_ids(get<Tags::InnerRefinementLevel>(box),
                          get<Tags::OuterRefinementLevel>(box));
  const size_t number_of_elements = element_ids.size();
  const Mesh<1> mesh_of_one_element{get<Tags::PointsPerElement>(box),
                                    Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};

  Scalar<DataVector> delta{mesh_of_one_element.number_of_grid_points() *
                           number_of_elements};
  Scalar<DataVector> dt_psi{mesh_of_one_element.number_of_grid_points() *
                            number_of_elements};
  Scalar<DataVector> dt_phi{mesh_of_one_element.number_of_grid_points() *
                            number_of_elements};
  Scalar<DataVector> dt_phi_tilde{mesh_of_one_element.number_of_grid_points() *
                                  number_of_elements};
  Scalar<DataVector> dt_pi{mesh_of_one_element.number_of_grid_points() *
                           number_of_elements};
  const tnsr::I<DataVector, 1, Frame::ElementLogical>
      logical_coords_one_element{logical_coordinates(mesh_of_one_element)};

  std::vector<domain::CoordinateMap<Frame::ElementLogical, Frame::Grid,
                                    domain::CoordinateMaps::Affine,
                                    domain::CoordinateMaps::Interval>>
      coordinate_maps{element_ids.size()};

  // const std::optional<double> singularity{-1.002499999999999};
  const std::optional<double> singularity{};
  const domain::CoordinateMaps::Distribution distribution =
      singularity.has_value()
          ? domain::CoordinateMaps::Distribution::Logarithmic
          : domain::CoordinateMaps::Distribution::Linear;
  const domain::CoordinateMaps::Interval interval_map(
      -1, 1, -1.0, 1.0, distribution, singularity);
  for (size_t element_index = 0; element_index < number_of_elements;
       element_index += 1) {
    const double lower =
        element_ids[element_index].segment_id.endpoint(Side::Lower);
    const double upper =
        element_ids[element_index].segment_id.endpoint(Side::Upper);
    coordinate_maps[element_index] =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
            domain::CoordinateMaps::Affine{-1.0, 1.0, lower, upper},
            interval_map);
  }
  tnsr::I<DataVector, 1, Frame::Grid> grid_coords{
      mesh_of_one_element.number_of_grid_points() * number_of_elements};
  Jacobian<DataVector, 1, Frame::ElementLogical, Frame::Grid> jacobian{
      mesh_of_one_element.number_of_grid_points() * number_of_elements};
  InverseJacobian<DataVector, 1, Frame::ElementLogical, Frame::Grid>
      inv_jacobian{mesh_of_one_element.number_of_grid_points() *
                   number_of_elements};
  for (size_t element_index = 0; element_index < number_of_elements;
       element_index += 1) {
    tnsr::I<DataVector, 1, Frame::Grid> grid_coords_this_element{
        std::next(
            get<0>(grid_coords).data(),
            static_cast<std::ptrdiff_t>(
                element_index * mesh_of_one_element.number_of_grid_points())),
        mesh_of_one_element.number_of_grid_points()};
    grid_coords_this_element =
        coordinate_maps[element_index](logical_coords_one_element);

    Jacobian<DataVector, 1, Frame::ElementLogical, Frame::Grid>
        jacobian_this_element{};
    get<0, 0>(jacobian_this_element)
        .set_data_ref(
            std::next(get<0, 0>(jacobian).data(),
                      static_cast<std::ptrdiff_t>(
                          element_index *
                          mesh_of_one_element.number_of_grid_points())),
            mesh_of_one_element.number_of_grid_points());
    jacobian_this_element =
        coordinate_maps[element_index].jacobian(logical_coords_one_element);

    InverseJacobian<DataVector, 1, Frame::ElementLogical, Frame::Grid>
        inv_jacobian_this_element{};
    get<0, 0>(inv_jacobian_this_element)
        .set_data_ref(
            std::next(get<0, 0>(inv_jacobian).data(),
                      static_cast<std::ptrdiff_t>(
                          element_index *
                          mesh_of_one_element.number_of_grid_points())),
            mesh_of_one_element.number_of_grid_points());
    inv_jacobian_this_element =
        coordinate_maps[element_index].inv_jacobian(logical_coords_one_element);
  }
  const Scalar<DataVector> det_jacobian = determinant(jacobian);
  const Scalar<DataVector> det_inv_jacobian = determinant(inv_jacobian);

  Scalar<DataVector> mass{mesh_of_one_element.number_of_grid_points() *
                          number_of_elements};
  Scalar<DataVector> buffer{mesh_of_one_element.number_of_grid_points() *
                            number_of_elements};

  Scalar<DataVector> metric_function_a{
      mesh_of_one_element.number_of_grid_points() * number_of_elements};

  const tnsr::I<DataVector, 1, Frame::Inertial> radius{
      {sqrt((get<0>(grid_coords) + 1.0) * 0.5) *
       get<Tags::OuterBoundaryRadius>(box)}};

  const double amplitude = get<Tags::Amplitude>(box);
  const double width = get<Tags::Width>(box);
  const double p = get<Tags::ExponentP>(box);
  const double q = get<Tags::ExponentQ>(box);
  const double center = get<Tags::Center>(box);

  const Scalar<DataVector> psi{
      amplitude * pow(get<0>(radius), 2 * q) *
      exp(-pow(get<0>(radius) - center, p) / pow(width, p))};
  const Scalar<DataVector> phi{
      -amplitude * (pow(get<0>(radius), 2 * q - 1)) *
      (p * pow(get<0>(radius) - center, p) - 2 * q * pow(width, p)) *
      exp(-pow(get<0>(radius) - center, p) / pow(width, p)) / pow(width, p)};
  const Scalar<DataVector> phi_tilde{
      -amplitude * 0.25 * (pow(get<0>(radius), 2 * q - 2)) *
      (p * pow(get<0>(radius) - center, p) - 2 * q * pow(width, p)) *
      exp(-pow(get<0>(radius) - center, p) / pow(width, p)) / pow(width, p)};
  const Scalar<DataVector> pi{-get<0>(radius) * get(phi)};

  Matrix matrix_buffer{mesh_of_one_element.number_of_grid_points(),
                       mesh_of_one_element.number_of_grid_points()};

  const long unsigned int FilterIndex = 0;
  Filters::Exponential<FilterIndex> exponential_filter =
      Filters::Exponential<FilterIndex>(get<Tags::FilterAlpha>(box),
                                        get<Tags::FilterHalfPower>(box), true,
                                        std::nullopt);
  const Matrix& filter_matrix =
      exponential_filter.filter_matrix(mesh_of_one_element);
  const std::array<std::reference_wrapper<const Matrix>, 1> filter_matrices{
      {std::cref(filter_matrix)}};
  DataVector integrand_buffer{mesh_of_one_element.number_of_grid_points() *
                              number_of_elements};
  compute_delta_integral_logical(&delta, &integrand_buffer, mesh_of_one_element,
                                 phi, pi, det_jacobian, radius,
                                 get<Tags::OuterBoundaryRadius>(box));
  compute_mass_integral(
      &mass, &matrix_buffer, mesh_of_one_element, phi, pi, det_jacobian, radius,
      get<Tags::SpacetimeDimensions>(box), get<Tags::OuterBoundaryRadius>(box));
  compute_metric_function_a_from_mass(&metric_function_a, mass, radius,
                                      get<Tags::SpacetimeDimensions>(box));
  get(delta) = 0.0;
  get(metric_function_a) = 1.0;
  compute_time_derivatives_first_order_2(
      &dt_psi, &dt_phi_tilde, &dt_pi, mesh_of_one_element, psi, phi_tilde, pi,
      phi, metric_function_a, delta, get<Tags::Gamma2>(box), radius,
      det_inv_jacobian, get<Tags::SpacetimeDimensions>(box),
      get<Tags::OuterBoundaryRadius>(box), filter_matrices);

  std::ofstream out_file{"./Data/output.txt"};
  out_file << "# 0 radius\n# 1 psi\n# 2 phi\n# 3 phi_tilde\n# 4 pi\n# 5 delta\n"
           << "# 6 mass\n# 7 A\n# 8 dt_psi\n# 9 dt_phi_tilde\n# 10 dt_pi\n"
           << "# 11 det_jacobian\n";
  for (size_t i = 0; i < get<0>(radius).size(); ++i) {
    out_file << std::setprecision(18) << get<0>(radius)[i] << ' ' << get(psi)[i]
             << ' ' << get(phi)[i] << ' ' << get(phi_tilde)[i] << ' '
             << get(pi)[i] << ' ' << get(delta)[i] << ' ' << get(mass)[i] << ' '
             << get(metric_function_a)[i] << ' ' << get(dt_psi)[i] << ' '
             << get(dt_phi_tilde)[i] << ' ' << get(dt_pi)[i] << ' '
             << get(det_inv_jacobian)[i] << "\n";
  }
  out_file.close();

  std::array<DataVector, 3> evaluated_vars = integrate_fields_in_time(
      &integrand_buffer, det_jacobian, &matrix_buffer, mesh_of_one_element,
      element_ids, psi, phi_tilde, pi, &mass, &delta, &metric_function_a,
      radius, det_inv_jacobian, box, filter_matrices);
}

int main(int argc, char** argv) {
  Options::Parser<tmpl::remove<options_list, Options::Tags::InputSource>>
      option_parser(
          "Input file options for studying spherical gravitational collapse.");

  boost::program_options::options_description desc(wrap_text(
      "Spherical gravitational collapse using one-sided Legendre polynomials "
      "at r=0 to analytically regularize the evolution equations. The metric "
      "used is:\n\n"
      " ds^2 = -A exp(-2delta)dt^2 + (1/A) dr^2 + r^{n-2}d Omega^{n-2}\n\n"
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
      "input-file", boost::program_options::value<std::string>()->required(),
      "input file to use for evolution");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u or vars.count("input-file") == 0u) {
    Parallel::printf("%s\n%s", desc, option_parser.help());
    return 1;
  }

  // Parse out options.
  option_parser.parse_file(vars["input-file"].as<std::string>());
  const auto options =
      option_parser.template apply<options_list>([](auto... args) {
        return db::create<options_list>(std::move(args)...);
      });

  use_flat_space = get<Tags::UseFlatSpace>(options);

  run(options);
}
