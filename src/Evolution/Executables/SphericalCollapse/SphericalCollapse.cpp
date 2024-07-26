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
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
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
std::vector<SegmentId> compute_seg_ids(const size_t number_of_elements,
                                       const size_t refinement_level) {
  std::vector<SegmentId> segids{};
  segids.reserve(number_of_elements);
  for (size_t element_index = 0; element_index < number_of_elements;
       element_index += 1) {
    SegmentId segid{refinement_level, element_index};
    segids.push_back(segid);
  }
  return segids;
}
std::vector<ElementId<1>> compute_element_ids(const size_t number_of_elements,
                                              const size_t refinement_level) {
  std::vector<ElementId<1>> ElementIds;
  const size_t block_id = 0;
  const std::vector<SegmentId> segids =
      compute_seg_ids(number_of_elements, refinement_level);
  for (size_t i = 0; i < number_of_elements; i++) {
    std::array<SegmentId, 1> segid{{segids[i]}};
    ElementId element_id(block_id, segid);
    ElementIds.push_back(element_id);
  }
  return ElementIds;
}

void compute_delta_integral_logical(
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<DataVector*> integrand_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& phi,
    const Scalar<DataVector>& pi, const Scalar<DataVector>& det_jacobian,
    const double R_0, const Scalar<DataVector>& det_jacobian_tau) {
  *integrand_buffer = -M_PI * (square(get(pi)) + square(get(phi))) *
                      get(det_jacobian) * square(R_0) * get(det_jacobian_tau);
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

void compute_mass_integral(
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Matrix*> matrix_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& phi,
    const Scalar<DataVector>& pi, const Scalar<DataVector>& det_jacobian,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const size_t spacetime_dim, const double R_0,
    const Scalar<DataVector>& det_jacobian_tau) {
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
        const double sigma = square(R_0) * 0.5 * M_PI *
                             (square(get(pi)[index]) + square(get(phi)[index]));
        view[i] += integration_matrix(i, k) * 0.5 * sigma *
                   get(det_jacobian)[index] * get(det_jacobian_tau)[index] *
                   pow(get<0>(radius)[index], spacetime_dim - 3);
        matrix_buffer->operator()(i, k) =
            (i == k ? 1.0 : 0.0) + integration_matrix(i, k) * sigma *
                                       get(det_jacobian)[index] *
                                       get(det_jacobian_tau)[index];
      }
    }
    // Solve the linear system A m = b for m (the mass)
    lapack::general_matrix_linear_solve(make_not_null(&view), matrix_buffer);
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

void compute_time_derivatives_first_order(
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const gsl::not_null<Scalar<DataVector>*> dt_phi,
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const gsl::not_null<Scalar<DataVector>*> buffer3,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& phi, const Scalar<DataVector>& pi,
    const Scalar<DataVector>& metric_function_a,
    const Scalar<DataVector>& metric_function_delta, const double gamma2,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const Scalar<DataVector>& det_inverse_jacobian, const size_t spacetime_dim,
    const double R_0, const Scalar<DataVector>& det_inv_jacobian_tau) {
  const double number_of_elements =
      get(pi).size() / mesh_of_one_element.number_of_grid_points();
  Scalar<DataVector> buffer1{mesh_of_one_element.number_of_grid_points() *
                             number_of_elements};
  Scalar<DataVector> buffer2{mesh_of_one_element.number_of_grid_points() *
                             number_of_elements};

  std::array<std::reference_wrapper<const Matrix>, 1> logical_diff_matrices{
      {std::cref(Spectral::differentiation_matrix(mesh_of_one_element))}};
  {
    get(buffer1) = get(metric_function_a) * exp(-get(metric_function_delta));
    get(*dt_psi) = get(buffer1) * get(pi);

    // Compute second term, dr(A e^{-delta} Phi)
    get(buffer2) = get(buffer1) * get(phi);
    apply_matrices(make_not_null(&get(*dt_pi)), logical_diff_matrices,
                   get(buffer2), mesh_of_one_element.extents());
    get(*dt_pi) *= (4 * get<0>(radius) * get(det_inverse_jacobian) *
                    get(det_inv_jacobian_tau)) *
                   (1.0 / square(R_0));
    apply_matrices(make_not_null(&get(*buffer3)), logical_diff_matrices,
                   get(psi), mesh_of_one_element.extents());
    get(*dt_pi) += (spacetime_dim - 2.0) * get(buffer1) * 4 *
                   get(det_inverse_jacobian) * get(det_inv_jacobian_tau) *
                   get(*buffer3) * (1.0 / square(R_0));
    // Compute dt_phi

    get(buffer2) = get(buffer1) * get(pi) + gamma2 * get(psi);

    apply_matrices(make_not_null(&get(*dt_phi)), logical_diff_matrices,
                   get(buffer2), mesh_of_one_element.extents());

    get(*dt_phi) *= 4 * get<0>(radius) * get(det_inverse_jacobian) *
                    get(det_inv_jacobian_tau) * (1.0 / square(R_0));
    get(*dt_phi) -= gamma2 * get(phi);
  }

  for (size_t element = mesh_of_one_element.number_of_grid_points();
       element <
       number_of_elements * mesh_of_one_element.number_of_grid_points() - 1;
       element = element + mesh_of_one_element.number_of_grid_points()) {
    get(*dt_psi)[element] =
        (get(*dt_psi)[element] + get(*dt_psi)[element - 1]) / 2;
    get(*dt_psi)[element - 1] = get(*dt_psi)[element];
    get(*dt_phi)[element] =
        (get(*dt_phi)[element] + get(*dt_phi)[element - 1]) / 2;
    get(*dt_phi)[element - 1] = get(*dt_phi)[element];
    get(*dt_pi)[element] =
        (get(*dt_pi)[element] + get(*dt_pi)[element - 1]) / 2;
    get(*dt_pi)[element - 1] = get(*dt_pi)[element];
  }
  // DG Boundary Correction, turn on to check

  // std::pair<DataVector, DataVector> pts_and_wts =
  // Spectral::compute_collocation_points_and_weights<
  // Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>(
  // mesh_of_one_element.number_of_grid_points());

  // Scalar<DataVector> w_plus{
  // (get(pi) + get(phi) +
  // gamma2 * get(psi) /
  // (get(metric_function_a) * exp(-get(metric_function_delta))) +
  // (spacetime_dim - 2) * get(psi) * 1.0 / get<0>(radius)) *
  // 0.5};
  // Scalar<DataVector> w_minus{
  // (get(pi) - get(phi) +
  // gamma2 * get(psi) /
  // (get(metric_function_a) * exp(-get(metric_function_delta))) -
  // (spacetime_dim - 2) * get(psi) * 1.0 / get<0>(radius)) *
  // 0.5};
  // for (size_t element = mesh_of_one_element.number_of_grid_points();
  // element <
  // number_of_elements * mesh_of_one_element.number_of_grid_points() - 1;
  // element = element + mesh_of_one_element.number_of_grid_points()) {
  // // correcting every end and beginning grid point by multiplying by
  // // 4r*inv_J*n_r = 4r*inv_J*1/sqrt(A)
  // get(*dt_pi)[element] +=
  // (4 * get<0>(radius)[element] * get(det_inverse_jacobian)[element] *
  // get(det_inv_jacobian_tau)[element] /
  // sqrt(get(metric_function_a)[element]) / std::get<1>(pts_and_wts)[0]) *
  // (get(metric_function_a)[element - 1] *
  // exp(-get(metric_function_delta)[element - 1])) *
  // get(w_plus)[element - 1] -
  // (get(metric_function_a)[element] *
  // exp(-get(metric_function_delta)[element])) *
  // get(w_minus)[element];

  // get(*dt_pi)[element - 1] +=
  // (-4 * get<0>(radius)[element - 1] *
  // get(det_inverse_jacobian)[element - 1] *
  // get(det_inv_jacobian_tau)[element - 1] * 1.0 /
  // sqrt(get(metric_function_a)[element - 1]) * 1.0 /
  // std::get<1>(
  // pts_and_wts)[mesh_of_one_element.number_of_grid_points() - 1]) *
  // (get(metric_function_a)[element - 1] *
  // exp(-get(metric_function_delta)[element - 1])) *
  // get(w_plus)[element - 1] -
  // (get(metric_function_a)[element] *
  // exp(-get(metric_function_delta)[element])) *
  // get(w_minus)[element];
  // get(*dt_phi)[element] +=
  // (4 * get<0>(radius)[element] * get(det_inverse_jacobian)[element] *
  // get(det_inv_jacobian_tau)[element] * 1.0 /
  // sqrt(get(metric_function_a)[element]) * 1.0 /
  // std::get<1>(pts_and_wts)[0]) *
  // (get(metric_function_a)[element - 1] *
  // exp(-get(metric_function_delta)[element - 1])) *
  // get(w_plus)[element - 1] +
  // (get(metric_function_a)[element] *
  // exp(-get(metric_function_delta)[element])) *
  // get(w_minus)[element];
  // get(*dt_phi)[element - 1] +=
  // (-4 * get<0>(radius)[element - 1] *
  // get(det_inverse_jacobian)[element - 1] * 1.0 /
  // sqrt(get(metric_function_a)[element - 1]) *
  // get(det_inv_jacobian_tau)[element - 1] * 1.0 /
  // std::get<1>(
  // pts_and_wts)[mesh_of_one_element.number_of_grid_points() - 1]) *
  // (get(metric_function_a)[element - 1] *
  // exp(-get(metric_function_delta)[element - 1])) *
  // get(w_plus)[element - 1] +
  // (get(metric_function_a)[element] *
  // exp(-get(metric_function_delta)[element])) *
  // get(w_minus)[element];
  // }
  // Apply boundary conditions at outer boundary.
  const size_t outer_boundary_index = get(psi).size() - 1;
  // get(*dt_psi)[outer_boundary_index] =
  // -get(psi)[outer_boundary_index] /
  // get<0>(radius)[outer_boundary_index] -
  // get(phi)[outer_boundary_index];

  // // Phi boundary condition
  // double logical_d_phi_at_boundary = 0.0;
  // for (size_t i = 0,
  // last_mesh_location =
  // get(phi).size() -
  // mesh_of_one_element.number_of_grid_points();
  // i < mesh_of_one_element.number_of_grid_points(); ++i) {
  // logical_d_phi_at_boundary +=
  // logical_diff_matrices[0].get()(
  // mesh_of_one_element.number_of_grid_points() - 1 , i) *
  // get(phi)[last_mesh_location + i];
  // }
  // get(*dt_phi)[outer_boundary_index] =
  // get(psi)[outer_boundary_index] /
  // square(get<0>(radius)[outer_boundary_index]) -
  // get(phi)[outer_boundary_index] / get<0>(radius)[outer_boundary_index]
  // - get(det_inverse_jacobian)[outer_boundary_index] * 4 *
  // get<0>(radius)[outer_boundary_index] * logical_d_phi_at_boundary;
  // Pi boundary condition
  get(*dt_pi)[outer_boundary_index] =
      (-get(*dt_phi)[outer_boundary_index] -
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
  double min_adapted_dt = 1e100;

  for (size_t i = 0; i < get(delta).size() - 1; i++) {
    double dt = 0.0;
    if ((get<0>(radius)[i + 1] - get<0>(radius)[i]) != 0) {
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
    const Mesh<1>& mesh_of_one_element, const size_t number_of_elements,
    const size_t refinement_level, const std::array<DataVector, 3> vars,
    const gsl::not_null<DataVector*> integrand_buffer,
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    const double gamma2, const Scalar<DataVector>& det_jacobian,
    const gsl::not_null<Matrix*> matrix_buffer, const size_t spacetime_dim,
    const double R_0, const Scalar<DataVector>& det_jacobian_tau,
    const Scalar<DataVector>& det_inv_jacobian_tau) {
  std::vector<ElementVolumeData> VolumeData;
  Scalar<DataVector> temp_phi{vars[1]};
  Scalar<DataVector> temp_pi{vars[2]};
  // compute_delta_integral_logical(delta, integrand_buffer,
  // mesh_of_one_element,
  // temp_phi, temp_pi, det_jacobian, R_0,
  // det_jacobian_tau);
  // compute_mass_integral(mass, matrix_buffer, mesh_of_one_element, temp_phi,
  // temp_pi, det_jacobian, radius, spacetime_dim, R_0,
  // det_jacobian_tau);
  // compute_metric_function_a_from_mass(metric_function_a, *mass, radius,
  // spacetime_dim);
  for (size_t i = 0; i < get<0>(radius).size();
       i = i + mesh_of_one_element.number_of_grid_points()) {
    ElementId element_id = compute_element_ids(
        number_of_elements,
        refinement_level)[i / mesh_of_one_element.number_of_grid_points()];
    std::vector<TensorComponent> Data;
    const auto add_variable = [&Data](const std::string& name,
                                      const DataVector& variable) {
      Data.push_back(TensorComponent(name, variable));
    };
    std::string in_name1{"Psi"};
    std::string in_name2{"Phi"};
    std::string in_name3{"Pi"};
    std::string in_name4{"Mass"};
    std::string in_name5{"A"};
    std::string in_name6{"Delta"};
    DataVector Psi_per_element{&const_cast<double&>(vars[0][i]),
                               mesh_of_one_element.number_of_grid_points()};
    DataVector Phi_per_element{&const_cast<double&>(vars[1][i]),
                               mesh_of_one_element.number_of_grid_points()};
    DataVector Pi_per_element{&const_cast<double&>(vars[2][i]),
                              mesh_of_one_element.number_of_grid_points()};
    // DataVector Mass_per_element{&const_cast<double&>(get(*mass)[i]),
    // mesh_of_one_element.number_of_grid_points()};
    DataVector A_per_element{&const_cast<double&>(get(*metric_function_a)[i]),
                             mesh_of_one_element.number_of_grid_points()};
    DataVector Delta_per_element{&const_cast<double&>(get(*delta)[i]),
                                 mesh_of_one_element.number_of_grid_points()};
    add_variable(in_name1, Psi_per_element);
    add_variable(in_name2, Phi_per_element);
    add_variable(in_name3, Pi_per_element);
    // add_variable(in_name4, Mass_per_element);
    add_variable(in_name5, A_per_element);
    add_variable(in_name6, Delta_per_element);

    ElementVolumeData EVdata =
        ElementVolumeData(element_id, Data, mesh_of_one_element);
    VolumeData.push_back(EVdata);
  }
  return VolumeData;
}
void find_min_A(const gsl::not_null<Scalar<DataVector>*> metric_function_a,
                bool BH_formed, const double epsilon) {
  for (size_t index = 0; index < get(*metric_function_a).size(); index++) {
    if (get(*metric_function_a)[index] < epsilon) {
      BH_formed = true;
      break;
    }
  }
}
std::array<DataVector, 3> integrate_fields_in_time(
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const gsl::not_null<Scalar<DataVector>*> dt_phi,
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    gsl::not_null<Scalar<DataVector>*> buffer3,
    const gsl::not_null<DataVector*> integrand_buffer,
    const Scalar<DataVector>& det_jacobian,
    const gsl::not_null<Matrix*> matrix_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& phi, const Scalar<DataVector>& pi,
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    const double gamma2, const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const Scalar<DataVector>& det_inverse_jacobian, const size_t spacetime_dim,
    const double R_0, const size_t refinement_level, bool BH_formed,
    const Scalar<DataVector>& det_jacobian_tau,
    const Scalar<DataVector>& det_inv_jacobian_tau,
    std::array<std::reference_wrapper<const Matrix>, 1> filter_matrices) {
  using Vars = std::array<DataVector, 3>;

  Vars vars{get(psi), get(phi), get(pi)};

  using StateDopri5 = boost::numeric::odeint::runge_kutta_dopri5<Vars>;
  const size_t number_of_elements =
      get<0>(radius).size() / mesh_of_one_element.number_of_grid_points();
  StateDopri5 st{};
  std::vector<double> times;
  const double epsilon = 0.0000001;
  double time = 0.0;
  double dt = 0.0;
  const double CFL = 0.1;
  size_t step = 0;
  while (time < 5.0 && time != -std::numeric_limits<double>::infinity()) {
    auto system = [&mesh_of_one_element, &metric_function_a, &delta, &mass,
                   &radius, &det_inverse_jacobian, &gamma2, &spacetime_dim,
                   &buffer3, &integrand_buffer, &det_jacobian, &matrix_buffer,
                   &R_0, &det_inv_jacobian_tau, &det_jacobian_tau,
                   &filter_matrices](const Vars& local_vars, Vars& local_dvars,
                                     const double current_time) {
      Scalar<DataVector> temp_psi{const_cast<DataVector&>(local_vars[0]).data(),
                                  local_vars[0].size()};
      Scalar<DataVector> temp_phi{const_cast<DataVector&>(local_vars[1]).data(),
                                  local_vars[1].size()};
      Scalar<DataVector> temp_pi{const_cast<DataVector&>(local_vars[2]).data(),
                                 local_vars[2].size()};
      apply_matrices(make_not_null(&const_cast<DataVector&>(get(temp_psi))),
                     filter_matrices, get(temp_psi),
                     mesh_of_one_element.extents());
      apply_matrices(make_not_null(&const_cast<DataVector&>(get(temp_pi))),
                     filter_matrices, get(temp_pi),
                     mesh_of_one_element.extents());
      apply_matrices(make_not_null(&const_cast<DataVector&>(get(temp_phi))),
                     filter_matrices, get(temp_phi),
                     mesh_of_one_element.extents());
      // compute_delta_integral_logical(delta, integrand_buffer,
      // mesh_of_one_element, temp_phi, temp_pi,
      // det_jacobian, R_0, det_jacobian_tau);
      // compute_mass_integral(mass, matrix_buffer, mesh_of_one_element,
      // temp_phi,
      // temp_pi, det_jacobian, radius, spacetime_dim,
      // R_0, det_jacobian_tau);
      // compute_metric_function_a_from_mass(metric_function_a, *mass, radius,
      // spacetime_dim);
      const auto size = get(temp_psi).size();
      Scalar<DataVector> temp_dtpsi{size, 0.0};
      Scalar<DataVector> temp_dtphi{size, 0.0};
      Scalar<DataVector> temp_dtpi{size, 0.0};
      compute_time_derivatives_first_order(
          make_not_null(&temp_dtpsi), make_not_null(&temp_dtphi),
          make_not_null(&temp_dtpi), buffer3, mesh_of_one_element, temp_psi,
          temp_phi, temp_pi, *metric_function_a, *delta, gamma2, radius,
          det_inverse_jacobian, spacetime_dim, R_0, det_inv_jacobian_tau);

      local_dvars[0] = get(temp_dtpsi);
      local_dvars[1] = get(temp_dtphi);
      local_dvars[2] = get(temp_dtpi);
    };

    std::string tag{"ElementData"};
    const observers::ObservationId obs_id = observers::ObservationId(time, tag);

    if (step % 500 == 0) {
      std::vector<ElementVolumeData> VolumeData = create_data_for_file(
          radius, mesh_of_one_element, number_of_elements, refinement_level,
          vars, integrand_buffer, mass, delta, metric_function_a, gamma2,
          det_jacobian, matrix_buffer, spacetime_dim, R_0, det_jacobian_tau,
          det_inv_jacobian_tau);
      std::cout << "time:\n" << time << "\n";
      std::cout << "step:\n" << step << "\n";
      // std::cout << "dt:\n" << dt << "\n";
      write_data_hd5file(VolumeData, obs_id, 0.0, R_0, refinement_level,
                         mesh_of_one_element.number_of_grid_points());
    }

    dt = compute_adaptive_step_size(*delta, *metric_function_a, radius, CFL);
    st.do_step(system, vars, time, dt);
    // times.push_back(time);

    time = time + dt;
    step = step + 1;
    // find_min_A(metric_function_a, BH_formed, epsilon);
  }
  std::cout << "time:\n" << time << "\n";
  std::cout << "Number of Steps:\n" << step << "\n";
  return vars;
}

Scalar<DataVector> differential_eq_for_A(
    const gsl::not_null<Scalar<DataVector>*> derivative_r,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& phi,
    const Scalar<DataVector>& pi, const Scalar<DataVector>& A,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const size_t spacetime_dim) {
  std::array<std::reference_wrapper<const Matrix>, 1> matrices{
      {std::cref(Spectral::differentiation_matrix(mesh_of_one_element))}};

  apply_matrices(make_not_null(&get(*derivative_r)), matrices, get(A),
                 mesh_of_one_element.extents());

  const DataVector view_A{&const_cast<double&>(get(A)[0]),  // NOLINT
                          get(A).size()};
  const DataVector view_radius{
      &const_cast<double&>(get<0>(radius)[0]),  // NOLINT
      get<0>(radius).size()};
  Scalar<DataVector> diff_eq{
      ((spacetime_dim - 3) / view_radius) * (1 - view_A) -
      2 * M_PI * view_radius * view_A * (square(get(pi)) + square(get(phi)))};
  return diff_eq;
}

void run(const size_t refinement_level, const size_t points_per_element,
         const double amp) {
  domain::creators::register_derived_with_charm();
  const size_t number_of_elements = two_to_the(refinement_level);
  const Mesh<1> mesh_of_one_element{points_per_element,
                                    Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};
  // Scalar<DataVector> delta{mesh_of_one_element.number_of_grid_points() *
  // number_of_elements};
  Scalar<DataVector> delta{
      mesh_of_one_element.number_of_grid_points() * number_of_elements, 0.0};
  Scalar<DataVector> dt_psi{mesh_of_one_element.number_of_grid_points() *
                            number_of_elements};
  Scalar<DataVector> dt_phi{mesh_of_one_element.number_of_grid_points() *
                            number_of_elements};
  Scalar<DataVector> dt_pi{mesh_of_one_element.number_of_grid_points() *
                           number_of_elements};
  Scalar<DataVector> jacobian{
      mesh_of_one_element.number_of_grid_points() * number_of_elements, 0.0};
  Scalar<DataVector> inv_jacobian{
      mesh_of_one_element.number_of_grid_points() * number_of_elements, 0.0};
  DataVector integrand_buffer{mesh_of_one_element.number_of_grid_points() *
                              number_of_elements};
  const tnsr::I<DataVector, 1, Frame::ElementLogical> logical_coord{
      logical_coordinates(mesh_of_one_element)};

  const std::array<DataVector, 1> logical_coords_array{logical_coord.get(0)};

  const std::vector<SegmentId> segids =
      compute_seg_ids(number_of_elements, refinement_level);
  tnsr::I<DataVector, 1, Frame::BlockLogical> block_logical_coords{
      mesh_of_one_element.number_of_grid_points() * number_of_elements, 0.0};
  for (size_t element_index = 0; element_index < number_of_elements;
       element_index += 1) {
    double lower = segids[element_index].endpoint(Side::Lower);
    double upper = segids[element_index].endpoint(Side::Upper);
    domain::CoordinateMaps::Affine affine_map(-1, 1, lower, upper);

    DataVector jacobian_of_element{
        std::next(
            get(jacobian).data(),
            static_cast<std::ptrdiff_t>(
                element_index * mesh_of_one_element.number_of_grid_points())),
        mesh_of_one_element.number_of_grid_points()};
    jacobian_of_element = affine_map.jacobian(logical_coords_array)[0];

    DataVector inv_jacobian_of_element{
        std::next(
            get(inv_jacobian).data(),
            static_cast<std::ptrdiff_t>(
                element_index * mesh_of_one_element.number_of_grid_points())),
        mesh_of_one_element.number_of_grid_points()};
    inv_jacobian_of_element = affine_map.inv_jacobian(logical_coords_array)[0];
    for (size_t coord_index = 0;
         coord_index < mesh_of_one_element.number_of_grid_points();
         coord_index += 1) {
      std::array<double, 1> point{};
      std::array<double, 1> point_to_check{logical_coord[0][coord_index]};
      // point = Interval(point_to_check);
      point = affine_map(point_to_check);
      block_logical_coords.get(
          0)[element_index * mesh_of_one_element.number_of_grid_points() +
             coord_index] = point[0];
    }
  }
  std::cout << "BLC:\n"
            << std::setprecision(16) << std::scientific
            << block_logical_coords.get(0) << "\n";

  domain::CoordinateMaps::Distribution distribution =
      domain::CoordinateMaps::Distribution::Logarithmic;
  double singularity = -1.0002499999999999;
  domain::CoordinateMaps::Interval Interval(-1, 1, -1, 1, distribution,
                                            singularity);
  tnsr::I<DataVector, 1, Frame::BlockLogical> int_coord_tau{
      mesh_of_one_element.number_of_grid_points() * number_of_elements, 0.0};
  for (size_t coord_index = 0;
       coord_index <
       mesh_of_one_element.number_of_grid_points() * number_of_elements;
       coord_index += 1) {
    std::array<double, 1> point{};
    std::array<double, 1> point_to_check{
        block_logical_coords.get(0)[coord_index]};
    point = Interval(point_to_check);

    int_coord_tau.get(0)[coord_index] = point[0];
  }
  const std::array<DataVector, 1> block_logical_coords_array{
      block_logical_coords.get(0)};
  Scalar<DataVector> jacobian_tau{
      Interval.jacobian(block_logical_coords_array)[0]};
  Scalar<DataVector> inv_jacobian_tau{
      Interval.inv_jacobian(block_logical_coords_array)[0]};

  const double width = 0.8;
  const double p = 2;
  const double q = 3;
  const double R_0 = 1.0;

  Scalar<DataVector> mass{
      mesh_of_one_element.number_of_grid_points() * number_of_elements, 0.0};
  Scalar<DataVector> buffer{mesh_of_one_element.number_of_grid_points() *
                            number_of_elements};

  Scalar<DataVector> metric_function_a{
      mesh_of_one_element.number_of_grid_points() * number_of_elements, 1.0};

  const tnsr::I<DataVector, 1, Frame::Inertial> radius{
      {{sqrt((int_coord_tau.get(0) + 1.0) * 0.5) * R_0}}};

  const Scalar<DataVector> phi{
      -amp * (p * pow(get<0>(radius), p + q - 1) * pow(1 / width, p) - q) *
      exp(-pow(get<0>(radius), p) / pow(width, p))};
  const Scalar<DataVector> pi{-get(phi)};
  const Scalar<DataVector> psi{amp * pow(get<0>(radius), q) *
                               exp(-pow(get<0>(radius), p) / pow(width, p))};

  // const Scalar<DataVector> phi{get<0>(radius)};
  // const Scalar<DataVector> pi{
  //     mesh_of_one_element.number_of_grid_points() * number_of_elements, 0.0};
  // const Scalar<DataVector>
  // psi{square(get<0>(radius))*exp(-pow(get<0>(radius),2))}; const
  // Scalar<DataVector> phi{(2*pow(get<0>(radius),3)-
  // 2*get<0>(radius))*exp(-square(get<0>(radius)))};
  // const Scalar<DataVector> pi{2*pow(get<0>(radius),3)};

  Matrix matrix_buffer{mesh_of_one_element.number_of_grid_points(),
                       mesh_of_one_element.number_of_grid_points()};

  const size_t spacetime_dim = 4;

  bool BH_formed = false;
  const double gamma2 = 0.0;
  [[maybe_unused]] const size_t last_point = get(mass).size() - 1;
  std::cout << "tau_jacobian:\n"
            << std::setprecision(16) << std::scientific << get(jacobian_tau)
            << "\n";
  const double alpha = 36;
  const unsigned half_power = 32;  // to remove lower mode.
  const long unsigned int FilterIndex = 0;
  Filters::Exponential<FilterIndex> exponential_filter =
      Filters::Exponential<FilterIndex>(alpha, half_power, true, std::nullopt);
  const Matrix& filter_matrix =
      exponential_filter.filter_matrix(mesh_of_one_element);
  std::array<std::reference_wrapper<const Matrix>, 1> filter_matrices{
      {std::cref(filter_matrix)}};
  // compute_delta_integral_logical(&delta, &integrand_buffer,
  // mesh_of_one_element,
  //                                phi, pi, jacobian, R_0, jacobian_tau);
  // compute_mass_integral(&mass, &matrix_buffer, mesh_of_one_element, phi, pi,
  //                       jacobian, radius, spacetime_dim, R_0, jacobian_tau);
  // compute_metric_function_a_from_mass(&metric_function_a, mass, radius,
  //                                     spacetime_dim);
  // compute_time_derivatives_first_order(
  //     &dt_psi, &dt_phi, &dt_pi, &buffer, mesh_of_one_element, psi, phi, pi,
  //     metric_function_a, delta, gamma2, radius, inv_jacobian, spacetime_dim,
  //     R_0, inv_jacobian_tau);
  std::array<DataVector, 3> evaluated_vars = integrate_fields_in_time(
      &dt_psi, &dt_phi, &dt_pi, &buffer, &integrand_buffer, jacobian,
      &matrix_buffer, mesh_of_one_element, psi, phi, pi, &mass, &delta,
      &metric_function_a, gamma2, radius, inv_jacobian, spacetime_dim, R_0,
      refinement_level, BH_formed, jacobian_tau, inv_jacobian_tau,
      filter_matrices);

  std::cout << "Psi:\n"
            << std::setprecision(16) << std::scientific << evaluated_vars[0]
            << "\n";
  std::cout << "Phi:\n"
            << std::setprecision(16) << std::scientific << evaluated_vars[1]
            << "\n";
  std::cout << "Pi:\n"
            << std::setprecision(16) << std::scientific << evaluated_vars[2]
            << "\n";
  // std::cout << "dtPsi:\n"
  //           << std::setprecision(16) << std::scientific << get(dt_psi) <<
  //           "\n";
  // std::cout << "dtPhi:\n"
  //           << std::setprecision(16) << std::scientific << get(dt_phi) <<
  //           "\n";
  // std::cout << "dtPi:\n"
  //           << std::setprecision(16) << std::scientific << get(dt_pi) <<
  //           "\n";
  std::cout << "Mass:\n"
            << std::setprecision(16) << std::scientific << get(mass)[last_point]
            << "\n";
  std::cout << "Metric A:\n"
            << std::setprecision(16) << std::scientific
            << get(metric_function_a) << "\n";
  std::cout << "delta:\n"
            << std::setprecision(16) << std::scientific << get(delta) << "\n";
  std::cout << "Radius:\n"
            << std::setprecision(16) << std::scientific << get<0>(radius)
            << "\n";
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
      "Initial Amplitude");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u or vars.count("input-file") == 0u or
      vars.count("ref") == 0u or vars.count("points") == 0u or
      vars.count("amplitude") == 0u) {
    Parallel::printf("%s\n", desc);
    return 0;
  }

  run(vars["ref"].as<size_t>(), vars["points"].as<size_t>(),
      vars["amplitude"].as<double>());
}
