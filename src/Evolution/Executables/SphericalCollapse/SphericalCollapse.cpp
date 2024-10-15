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
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/WrapText.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty

extern "C" void CkRegisterMainModule(void) {}

constexpr bool use_flat_space = false;

Scalar<DataVector> differential_eq_for_A(
    const Scalar<DataVector>& phi, const Scalar<DataVector>& pi,
    const Scalar<DataVector>& A,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const size_t spacetime_dim) {
  Scalar<DataVector> diff_eq{get<0>(radius).size()};
  get(diff_eq)[0] = 0.0;
  for (size_t i = 1; i < get<0>(radius).size(); i++) {
    get(diff_eq)[i] =
        ((spacetime_dim - 3) / get<0>(radius))[i] * (1 - get(A)[i]) -
        2 * M_PI * get<0>(radius)[i] * get(A)[i] *
            (square(get(pi)[i]) + square(get(phi)[i]));

    if constexpr (use_flat_space) {
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
  if constexpr (use_flat_space) {
    get(diff_eq) = 0.0;
  }
  return diff_eq;
}

void compute_delta_integral_logical(
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<DataVector*> integrand_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& phi,
    const Scalar<DataVector>& pi, const Scalar<DataVector>& det_jacobian,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const double one_sided_jacobi_boundary) {
  (*integrand_buffer) = -M_PI * (square(get(pi)) + square(get(phi))) *
                        get(det_jacobian) * square(one_sided_jacobi_boundary);

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

  if constexpr (use_flat_space) {
    get(*delta) = 0.0;
  }
}

void compute_mass_integral(
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Matrix*> matrix_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& phi,
    const Scalar<DataVector>& pi, const Scalar<DataVector>& det_jacobian,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const size_t spacetime_dim, const double one_sided_jacobi_boundary) {
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
            square(one_sided_jacobi_boundary);
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

  if constexpr (use_flat_space) {
    get(*mass) = 0.0;
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
std::vector<ElementId<1>> compute_element_ids2(
    const size_t /*number_of_elements*/, const size_t /*refinement_level*/) {
  const size_t inner_refinement_level = 11;
  const size_t outer_refinement_level = 4;
  std::vector<ElementId<1>> element_ids;
  const size_t block_id = 0;
  for (size_t j = inner_refinement_level; j >= outer_refinement_level; j--) {
    for (size_t element_index = (j == inner_refinement_level
                                     ? 0
                                     : two_to_the(outer_refinement_level - 1));
         element_index < two_to_the(outer_refinement_level); element_index++) {
      ElementId element_id(block_id, std::array{SegmentId{j, element_index}});
      element_ids.push_back(element_id);
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
    const Scalar<DataVector>& det_inverse_jacobian, const size_t spacetime_dim,
    const double one_sided_jacobi_boundary,
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
  get(*dt_pi) *= (4.0 * get<0>(radius) / square(one_sided_jacobi_boundary));
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
  get(*dt_phi_tilde) *= get(
      det_inverse_jacobian);  //  * (1.0 / square(one_sided_jacobi_boundary));
  get(*dt_phi_tilde) *= (1.0 / square(one_sided_jacobi_boundary));

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
    const double lower_jacobian = (square(one_sided_jacobi_boundary) /
                                   (4.0 * get<0>(radius)[element - 1])) /
                                  get(det_inverse_jacobian)[element - 1];
    const double upper_jacobian =
        (square(one_sided_jacobi_boundary) / (4.0 * get<0>(radius)[element])) /
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

void write_data_hd5file(std::vector<ElementVolumeData>& volume_data,
                        const observers::ObservationId& observation_id,
                        const double lower_r, const double upper_r,
                        const size_t refinement_level, const size_t grd_pts) {
  const std::string h5_file_name{"VolumeDataForFields"};
  const std::string input_source{""};
  const std::string subfile_path{"/ElementData"};
  const uint32_t version_number = 0;
  h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name + ".h5"s, true,
                                                input_source};
  auto& volume_file =
      h5_file.try_insert<h5::VolumeData>(subfile_path, version_number);

  domain::creators::Interval interval{std::array{lower_r}, std::array{upper_r},
                                      std::array{refinement_level},
                                      std::array{grd_pts}};
  Domain<1> domain = interval.create_domain();
  const auto serialized_domain = serialize(domain);
  volume_file.write_volume_data(observation_id.hash(), observation_id.value(),
                                std::move(volume_data), serialized_domain);
}

std::vector<ElementVolumeData> create_data_for_file(
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const Mesh<1>& mesh_of_one_element,
    const std::vector<ElementId<1>>& element_ids,
    const std::array<DataVector, 3> vars,
    const gsl::not_null<DataVector*> integrand_buffer,
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    const double /*gamma2*/, const Scalar<DataVector>& det_jacobian,
    const gsl::not_null<Matrix*> matrix_buffer, const size_t spacetime_dim,
    const double one_sided_jacobi_boundary, const size_t step_number) {
  std::vector<ElementVolumeData> VolumeData;
  const Scalar<DataVector> temp_phi{vars[1] * 4 * get<0>(radius)};
  const Scalar<DataVector> temp_pi{vars[2]};
  compute_delta_integral_logical(delta, integrand_buffer, mesh_of_one_element,
                                 temp_phi, temp_pi, det_jacobian, radius,
                                 one_sided_jacobi_boundary);
  compute_mass_integral(mass, matrix_buffer, mesh_of_one_element, temp_phi,
                        temp_pi, det_jacobian, radius, spacetime_dim,
                        one_sided_jacobi_boundary);
  compute_metric_function_a_from_mass(metric_function_a, *mass, radius,
                                      spacetime_dim);

  std::ofstream out_file{"./Data/output" + std::to_string(step_number) +
                         ".txt"};
  out_file << "# 0 radius\n# 1 psi\n# 2 phi\n# 3 phi_tilde\n# 4 pi\n# 5 delta\n"
           << "# 6 mass\n# 7 A\n# 8 dt_psi\n# 9 dt_phi_tilde\n# 10 dt_pi\n";
  for (size_t i = 0; i < get<0>(radius).size(); ++i) {
    out_file << std::setprecision(18) << get<0>(radius)[i] << ' ' << vars[0][i]
             << ' ' << get(temp_phi)[i] << ' ' << vars[1][i] << ' '
             << get(temp_pi)[i] << ' ' << get(*delta)[i] << ' ' << get(*mass)[i]
             << ' ' << get(*metric_function_a)[i] << "\n";
  }
  out_file.close();

  for (size_t element_i = 0; element_i < element_ids.size(); ++element_i) {
    const ElementId<1>& element_id = element_ids[element_i];
    const size_t grid_i =
        element_i * mesh_of_one_element.number_of_grid_points();
    std::vector<TensorComponent> Data;
    const auto add_variable = [&Data](const std::string& name,
                                      const DataVector& variable) {
      Data.push_back(TensorComponent(name, variable));
    };
    const std::string in_name1{"Psi"};
    const std::string in_name2{"Phi"};
    const std::string in_name3{"Pi"};
    const std::string in_name4{"Mass"};
    const std::string in_name5{"A"};
    const std::string in_name6{"Delta"};
    const DataVector Psi_per_element{
        &const_cast<double&>(vars[0][grid_i]),
        mesh_of_one_element.number_of_grid_points()};
    const DataVector Phi_per_element{
        &const_cast<double&>(vars[1][grid_i]),
        mesh_of_one_element.number_of_grid_points()};
    const DataVector Pi_per_element{
        &const_cast<double&>(vars[2][grid_i]),
        mesh_of_one_element.number_of_grid_points()};
    const DataVector Mass_per_element{
        &const_cast<double&>(get(*mass)[grid_i]),
        mesh_of_one_element.number_of_grid_points()};
    const DataVector A_per_element{
        &const_cast<double&>(get(*metric_function_a)[grid_i]),
        mesh_of_one_element.number_of_grid_points()};
    const DataVector Delta_per_element{
        &const_cast<double&>(get(*delta)[grid_i]),
        mesh_of_one_element.number_of_grid_points()};
    const DataVector radius_in_element{
        &const_cast<double&>(get<0>(radius)[grid_i]),
        mesh_of_one_element.number_of_grid_points()};
    add_variable("Radius", radius_in_element);
    add_variable(in_name1, Psi_per_element);
    add_variable(in_name2, Phi_per_element);
    add_variable(in_name3, Pi_per_element);
    add_variable(in_name4, Mass_per_element);
    add_variable(in_name5, A_per_element);
    add_variable(in_name6, Delta_per_element);

    VolumeData.emplace_back(element_id, Data, mesh_of_one_element);
  }
  return VolumeData;
}

bool find_min_A(const gsl::not_null<Scalar<DataVector>*> metric_function_a,
                const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
                const double epsilon, double time) {
  for (size_t index = 0; index < get(*metric_function_a).size(); index++) {
    if (abs(get(*metric_function_a)[index]) < epsilon) {
      std::cout << "A at min: " << get(*metric_function_a)[index] << "\n";
      std::cout << "Radius: " << get<0>(radius)[index] << "\n";
      std::cout << "here"
                << "\n";
      return true;
    }
  }
  return false;
}

std::array<DataVector, 3> integrate_fields_in_time(
    const gsl::not_null<DataVector*> integrand_buffer,
    const Scalar<DataVector>& det_jacobian,
    const gsl::not_null<Matrix*> matrix_buffer,
    const Mesh<1>& mesh_of_one_element,
    const std::vector<ElementId<1>>& element_ids, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& phi_tilde, const Scalar<DataVector>& pi,
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    const double gamma2, const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const Scalar<DataVector>& det_inverse_jacobian, const size_t spacetime_dim,
    const double one_sided_jacobi_boundary, const size_t refinement_level,
    bool found_black_hole,
    const std::array<std::reference_wrapper<const Matrix>, 1>& filter_matrices,
    const double end_time) {
  using Vars = std::array<DataVector, 3>;
  const size_t observation_frequency = 100;

  Vars vars{get(psi), get(phi_tilde), get(pi)};

  using StateDopri5 = boost::numeric::odeint::runge_kutta_dopri5<Vars>;
  StateDopri5 st{};
  std::vector<double> times;
  const double epsilon = 0.01;
  double time = 0.0;
  double dt = 0.0;
  const double CFL = 0.25;
  const bool filter_evolved_vars = false;
  size_t step = 0;

  using std::abs;
  while (abs(time) < end_time) {
    auto system = [&mesh_of_one_element, &metric_function_a, &delta, &mass,
                   &radius, &det_inverse_jacobian, &gamma2, &spacetime_dim,
                   &integrand_buffer, &det_jacobian, &matrix_buffer,
                   &one_sided_jacobi_boundary, &filter_matrices, &element_ids,
                   filter_evolved_vars](
                      const Vars& local_vars, Vars& local_dvars,
                      [[maybe_unused]] const double current_time) {
      Scalar<DataVector> temp_psi{const_cast<DataVector&>(local_vars[0]).data(),
                                  local_vars[0].size()};
      Scalar<DataVector> temp_phi_tilde{
          const_cast<DataVector&>(local_vars[1]).data(), local_vars[1].size()};
      Scalar<DataVector> temp_pi{const_cast<DataVector&>(local_vars[2]).data(),
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
          det_jacobian, radius, one_sided_jacobi_boundary);
      compute_mass_integral(mass, matrix_buffer, mesh_of_one_element, temp_phi,
                            temp_pi, det_jacobian, radius, spacetime_dim,
                            one_sided_jacobi_boundary);
      compute_metric_function_a_from_mass(metric_function_a, *mass, radius,
                                          spacetime_dim);

      const auto size = get(temp_psi).size();
      Scalar<DataVector> temp_dtpsi{size, 0.0};
      Scalar<DataVector> temp_dtphi_tilde{size, 0.0};
      Scalar<DataVector> temp_dtpi{size, 0.0};
      compute_time_derivatives_first_order_2(
          make_not_null(&temp_dtpsi), make_not_null(&temp_dtphi_tilde),
          make_not_null(&temp_dtpi), mesh_of_one_element, temp_psi,
          temp_phi_tilde, temp_pi, temp_phi, *metric_function_a, *delta, gamma2,
          radius, det_inverse_jacobian, spacetime_dim,
          one_sided_jacobi_boundary, filter_matrices);

      local_dvars[0] = get(temp_dtpsi);
      local_dvars[1] = get(temp_dtphi_tilde);
      local_dvars[2] = get(temp_dtpi);
    };

    if constexpr (use_flat_space) {
      found_black_hole = false;
    } else {
      found_black_hole =
          find_min_A(metric_function_a, radius, epsilon, time);
    }
    if (step % observation_frequency == 0 or found_black_hole) {
      std::string tag{"ElementData"};
      const observers::ObservationId obs_id =
          observers::ObservationId(time, tag);
      std::vector<ElementVolumeData> VolumeData = create_data_for_file(
          radius, mesh_of_one_element, element_ids, vars, integrand_buffer,
          mass, delta, metric_function_a, gamma2, det_jacobian, matrix_buffer,
          spacetime_dim, one_sided_jacobi_boundary, step);

      std::cout << "time: " << time << " step: " << step << " dt: " << dt
                << "\n";

      write_data_hd5file(VolumeData, obs_id, 0.0, one_sided_jacobi_boundary,
                         refinement_level,
                         mesh_of_one_element.number_of_grid_points());
    }
    if (found_black_hole) {
      std::cout << "Found a black hole!!\n"
                << "time: " << time << "\n"
                << "Number of Steps: " << step << "\n";
      return vars;
    }

    dt = compute_adaptive_step_size(*delta, *metric_function_a, radius, CFL);
    st.do_step(system, vars, time, dt);

    time = time + dt;
    step = step + 1;
  }
  return vars;
}

void run(const size_t refinement_level, const size_t points_per_element,
         const double amp, const double one_sided_jacobi_boundary,
         const double outer_boundary_radius, const double time) {
  domain::creators::register_derived_with_charm();
  const std::vector<ElementId<1>> element_ids =
      compute_element_ids2(two_to_the(refinement_level), refinement_level);
  const size_t number_of_elements = element_ids.size();
  const Mesh<1> mesh_of_one_element{points_per_element,
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
        element_ids[element_index].segment_id(0).endpoint(Side::Lower);
    const double upper =
        element_ids[element_index].segment_id(0).endpoint(Side::Upper);
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

  const double width = 1.0;
  const double p = 2;
  const double q = 2;
  const double center = 0.0;

  Scalar<DataVector> mass{mesh_of_one_element.number_of_grid_points() *
                          number_of_elements};
  Scalar<DataVector> buffer{mesh_of_one_element.number_of_grid_points() *
                            number_of_elements};

  Scalar<DataVector> metric_function_a{
      mesh_of_one_element.number_of_grid_points() * number_of_elements};

  const auto radius = [&grid_coords, &mesh_of_one_element,
                       &one_sided_jacobi_boundary]()
      -> tnsr::I<DataVector, 1, Frame::Inertial> {
    DataVector rad = get<0>(grid_coords);
    rad = sqrt((rad + 1.0) * 0.5) * one_sided_jacobi_boundary;
    return tnsr::I<DataVector, 1, Frame::Inertial>{{rad}};
  }();

  const Scalar<DataVector> psi{
      amp * pow(get<0>(radius), 2 * q) *
      exp(-pow(get<0>(radius) - center, p) / pow(width, p))};
  const Scalar<DataVector> phi{
      -amp * (pow(get<0>(radius), 2 * q - 1)) *
      (p * pow(get<0>(radius) - center, p) - 2 * q * pow(width, p)) *
      exp(-pow(get<0>(radius) - center, p) / pow(width, p)) / pow(width, p)};
  const Scalar<DataVector> phi_tilde{
      -amp * 0.25 * (pow(get<0>(radius), 2 * q - 2)) *
      (p * pow(get<0>(radius) - center, p) - 2 * q * pow(width, p)) *
      exp(-pow(get<0>(radius) - center, p) / pow(width, p)) / pow(width, p)};
  const Scalar<DataVector> pi{-get<0>(radius) * get(phi)};

  Matrix matrix_buffer{mesh_of_one_element.number_of_grid_points(),
                       mesh_of_one_element.number_of_grid_points()};

  const size_t spacetime_dim = 4;

  bool found_black_hole = false;
  const double gamma2 = 0.0;
  [[maybe_unused]] const size_t last_point = get(mass).size() - 1;

  const double alpha = 128.0;
  const unsigned half_power = 64;  // to remove lower mode.
  const long unsigned int FilterIndex = 0;
  Filters::Exponential<FilterIndex> exponential_filter =
      Filters::Exponential<FilterIndex>(alpha, half_power, true, std::nullopt);
  const Matrix& filter_matrix =
      exponential_filter.filter_matrix(mesh_of_one_element);
  const std::array<std::reference_wrapper<const Matrix>, 1> filter_matrices{
      {std::cref(filter_matrix)}};
  DataVector integrand_buffer{mesh_of_one_element.number_of_grid_points() *
                              number_of_elements};
  compute_delta_integral_logical(&delta, &integrand_buffer, mesh_of_one_element,
                                 phi, pi, det_jacobian, radius,
                                 one_sided_jacobi_boundary);
  compute_mass_integral(&mass, &matrix_buffer, mesh_of_one_element, phi, pi,
                        det_jacobian, radius, spacetime_dim,
                        one_sided_jacobi_boundary);
  compute_metric_function_a_from_mass(&metric_function_a, mass, radius,
                                      spacetime_dim);
  get(delta) = 0.0;
  get(metric_function_a) = 1.0;
  compute_time_derivatives_first_order_2(
      &dt_psi, &dt_phi_tilde, &dt_pi, mesh_of_one_element, psi, phi_tilde, pi,
      phi, metric_function_a, delta, gamma2, radius, det_inv_jacobian,
      spacetime_dim, one_sided_jacobi_boundary, filter_matrices);

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
      gamma2, radius, det_inv_jacobian, spacetime_dim,
      one_sided_jacobi_boundary, refinement_level, found_black_hole,
      filter_matrices, time);
}

int main(int argc, char** argv) {
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
      "input file to use for evolution")(
      "ref", boost::program_options::value<size_t>()->required(),
      "Refinement level")("points",
                          boost::program_options::value<size_t>()->required(),
                          "Points per element")(
      "amplitude", boost::program_options::value<double>()->required(),
      "Initial Amplitude")("outer-boundary",
                           boost::program_options::value<double>()->required(),
                           "Radius of the outer boundary")(
      "time", boost::program_options::value<double>()->required(), "Time T");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u or vars.count("input-file") == 0u or
      vars.count("ref") == 0u or vars.count("points") == 0u or
      vars.count("amplitude") == 0u or vars.count("outer-boundary") == 0u or
      vars.count("time") == 0u) {
    Parallel::printf("%s\n", desc);
    return 0;
  }

  run(vars["ref"].as<size_t>(), vars["points"].as<size_t>(),
      vars["amplitude"].as<double>(), vars["outer-boundary"].as<double>(),
      vars["outer-boundary"].as<double>(), vars["time"].as<double>());
}
