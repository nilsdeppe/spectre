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
Scalar<DataVector> differential_eq_for_A(
    const Scalar<DataVector>& phi, const Scalar<DataVector>& pi,
    const Scalar<DataVector>& A,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const size_t spacetime_dim) {
  Scalar<DataVector> diff_eq{((spacetime_dim - 3) / get<0>(radius)) *
                                 (1 - get(A)) -
                             2 * M_PI * get<0>(radius) * get(A) *
                                 (square(get(pi)) + square(get(phi)))};

  // Scalar<DataVector> diff_eq{2 * M_PI * get<0>(radius) * get(A) *
  //                                (square(get(pi)) + square(get(phi)))};

  return diff_eq;
}
Scalar<DataVector> differential_eq_for_delta(
    const Scalar<DataVector>& phi, const Scalar<DataVector>& pi,
    const Scalar<DataVector>& delta,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius) {
  Scalar<DataVector> diff_eq{-4 * M_PI * get<0>(radius) *
                             (square(get(pi)) + square(get(phi)))};
  return diff_eq;
}
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
void compute_time_derivatives_first_order_2(
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const gsl::not_null<Scalar<DataVector>*> dt_phi_tilde,
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const gsl::not_null<Scalar<DataVector>*> buffer3,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& phi_tilde, const Scalar<DataVector>& pi,
    const Scalar<DataVector>& phi, const Scalar<DataVector>& metric_function_a,
    const Scalar<DataVector>& metric_function_delta, const double gamma2,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const Scalar<DataVector>& det_inverse_jacobian, const size_t spacetime_dim,
    const double R_0, const Scalar<DataVector>& det_inv_jacobian_tau,
    const std::array<std::reference_wrapper<const Matrix>, 1>&
        filter_matrices) {
  Scalar<DataVector> diff_eq_A{get(psi).size()};

  // std::cout << "phi:\n" << phi << "\n";
  // std::cout << "pi:\n" << pi << "\n";
  // std::cout << "psi:\n" << psi << "\n";
  // std::cout << "A:\n" << metric_function_a << "\n";
  // std::cout << "delta:\n" << metric_function_delta << "\n";
  // std::cout << "pi:\n" << pi << "\n";
  Scalar<DataVector> diff_eq_delta =
      differential_eq_for_delta(phi, pi, metric_function_delta, radius);
  // Scalar<DataVector> diff_eq_A{get(metric_function_a).size(), 0.0};
  // Scalar<DataVector> diff_eq_delta{get(metric_function_a).size(), 0.0};
  const double number_of_elements =
      get(pi).size() / mesh_of_one_element.number_of_grid_points();
  Scalar<DataVector> buffer1{mesh_of_one_element.number_of_grid_points() *
                             number_of_elements};
  Scalar<DataVector> buffer2{mesh_of_one_element.number_of_grid_points() *
                             number_of_elements};
  Scalar<DataVector> buffer4{mesh_of_one_element.number_of_grid_points() *
                             number_of_elements};

  std::array<std::reference_wrapper<const Matrix>, 1> logical_diff_matrices{
      {std::cref(Spectral::differentiation_matrix(mesh_of_one_element))}};
  apply_matrices(make_not_null(&get(buffer4)), logical_diff_matrices,
                 get(metric_function_a), mesh_of_one_element.extents());
  get(diff_eq_A) = 4 * get<0>(radius) * get(det_inv_jacobian_tau) *
                   get(det_inverse_jacobian) * get(buffer4);

  {
    // compute dt_psi
    get(buffer1) = get(metric_function_a) * exp(-get(metric_function_delta));
    get(*dt_psi) = get(buffer1) * get(pi);



    // compute 2nd term of dt_pi
    apply_matrices(make_not_null(&get(*dt_pi)), logical_diff_matrices,
                   get(phi_tilde), mesh_of_one_element.extents());
    get(*dt_pi) *= (16 * square(get<0>(radius)) * get(det_inv_jacobian_tau) *
                    get(det_inverse_jacobian) * get(buffer1)) *
                   (1.0 / square(R_0));
    get(*dt_pi) += get(diff_eq_A) * exp(-get(metric_function_delta)) * 4 *
                       get<0>(radius) * get(phi_tilde) -
                   get(diff_eq_delta) * get(buffer1) * 4 * get<0>(radius) *
                       get(phi_tilde) +
                   4 * get(buffer1) * get(phi_tilde);

    // compute 1st term of dt_pi
    apply_matrices(make_not_null(&get(*buffer3)), logical_diff_matrices,
                   get(psi), mesh_of_one_element.extents());
    get(*dt_pi) +=
        (spacetime_dim - 2.0) * get(buffer1) * 4 * get(*buffer3) *
        get(det_inv_jacobian_tau) * get(det_inverse_jacobian) *
        (1.0 / square(R_0));  // adding in the first term of the expansion

    // compute dt_phi
    get(buffer2) = get(buffer1) * get(pi) + gamma2 * get(psi);

    apply_matrices(make_not_null(&get(*dt_phi_tilde)), logical_diff_matrices,
                   get(buffer2), mesh_of_one_element.extents());
    get(*dt_phi_tilde) *= get(det_inv_jacobian_tau) *
                          get(det_inverse_jacobian) * (1.0 / square(R_0));
    get(*dt_phi_tilde) -= gamma2 * get(phi_tilde);
  }
  // DataVector no_filter{get(*dt_psi)};
  // apply_matrices(make_not_null(&get(*dt_psi)), filter_matrices, no_filter,
  //                mesh_of_one_element.extents());
  // no_filter = get(*dt_pi);
  // apply_matrices(make_not_null(&get(*dt_pi)), filter_matrices, no_filter,
  //                mesh_of_one_element.extents());
  // no_filter = get(*dt_phi_tilde);
  // apply_matrices(make_not_null(&get(*dt_phi_tilde)), filter_matrices,
  // no_filter,
  //                mesh_of_one_element.extents());
  for (size_t element = mesh_of_one_element.number_of_grid_points();
       element <
       number_of_elements * mesh_of_one_element.number_of_grid_points() - 1;
       element = element + mesh_of_one_element.number_of_grid_points()) {
    get(*dt_psi)[element] =
        (get(*dt_psi)[element] + get(*dt_psi)[element - 1]) / 2;
    get(*dt_psi)[element - 1] = get(*dt_psi)[element];
    get(*dt_phi_tilde)[element] =
        (get(*dt_phi_tilde)[element] + get(*dt_phi_tilde)[element - 1]) / 2;
    get(*dt_phi_tilde)[element - 1] = get(*dt_phi_tilde)[element];
    get(*dt_pi)[element] =
        (get(*dt_pi)[element] + get(*dt_pi)[element - 1]) / 2;
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
  double min_adapted_dt = exp(300);

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
  Scalar<DataVector> temp_phi{vars[1] * 4 * get<0>(radius)};
  Scalar<DataVector> temp_pi{vars[2]};
  compute_delta_integral_logical(delta, integrand_buffer, mesh_of_one_element,
                                 temp_phi, temp_pi, det_jacobian, R_0,
                                 det_jacobian_tau);
  compute_mass_integral(mass, matrix_buffer, mesh_of_one_element, temp_phi,
                        temp_pi, det_jacobian, radius, spacetime_dim, R_0,
                        det_jacobian_tau);
  compute_metric_function_a_from_mass(metric_function_a, *mass, radius,
                                      spacetime_dim);
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
    DataVector radius_in_element{&const_cast<double&>(get<0>(radius)[i]),
                                 mesh_of_one_element.number_of_grid_points()};
    add_variable("Radius", radius_in_element);
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
inline const char* const BoolToString(bool b) { return b ? "true" : "false"; }
std::array<DataVector, 3> integrate_fields_in_time(
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const gsl::not_null<Scalar<DataVector>*> dt_phi,
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    gsl::not_null<Scalar<DataVector>*> buffer3,
    const gsl::not_null<DataVector*> integrand_buffer,
    const Scalar<DataVector>& det_jacobian,
    const gsl::not_null<Matrix*> matrix_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& phi_tilde, const Scalar<DataVector>& pi,
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    const double gamma2, const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const Scalar<DataVector>& det_inverse_jacobian, const size_t spacetime_dim,
    const double R_0, const size_t refinement_level, bool BH_formed,
    const Scalar<DataVector>& det_jacobian_tau,
    const Scalar<DataVector>& det_inv_jacobian_tau,
    const std::array<std::reference_wrapper<const Matrix>, 1>&
        filter_matrices) {
  using Vars = std::array<DataVector, 3>;

  Vars vars{get(psi), get(phi_tilde), get(pi)};

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
  // size_t nan = 2;
  // size_t counter =0;
  while (time < 5.0 && time != -std::numeric_limits<double>::infinity()) {
    // std::cout << "here:\n"  << "\n";
    auto system = [&mesh_of_one_element, &metric_function_a, &delta, &mass,
                   &radius, &det_inverse_jacobian, &gamma2, &spacetime_dim,
                   &buffer3, &integrand_buffer, &det_jacobian, &matrix_buffer,
                   &R_0, &det_inv_jacobian_tau, &det_jacobian_tau,
                   &filter_matrices](const Vars& local_vars, Vars& local_dvars,
                                     const double current_time) {
      Scalar<DataVector> temp_psi{const_cast<DataVector&>(local_vars[0]).data(),
                                  local_vars[0].size()};
      Scalar<DataVector> temp_phi_tilde{
          const_cast<DataVector&>(local_vars[1]).data(), local_vars[1].size()};
      Scalar<DataVector> temp_pi{const_cast<DataVector&>(local_vars[2]).data(),
                                 local_vars[2].size()};
      // Scalar<DataVector> temp_psi{local_vars[0]};
      // Scalar<DataVector> temp_phi_tilde{local_vars[1]};
      // Scalar<DataVector> temp_pi{local_vars[2]};

      Scalar<DataVector> temp_phi{get(temp_phi_tilde) * 4 * get<0>(radius)};

      compute_delta_integral_logical(delta, integrand_buffer,
                                     mesh_of_one_element, temp_phi, temp_pi,
                                     det_jacobian, R_0, det_jacobian_tau);
      compute_mass_integral(mass, matrix_buffer, mesh_of_one_element, temp_phi,
                            temp_pi, det_jacobian, radius, spacetime_dim, R_0,
                            det_jacobian_tau);
      compute_metric_function_a_from_mass(metric_function_a, *mass, radius,
                                          spacetime_dim);

      const auto size = get(temp_psi).size();
      Scalar<DataVector> temp_dtpsi{size, 0.0};
      Scalar<DataVector> temp_dtphi_tilde{size, 0.0};
      Scalar<DataVector> temp_dtpi{size, 0.0};
      compute_time_derivatives_first_order_2(
          make_not_null(&temp_dtpsi), make_not_null(&temp_dtphi_tilde),
          make_not_null(&temp_dtpi), buffer3, mesh_of_one_element, temp_psi,
          temp_phi_tilde, temp_pi, temp_phi, *metric_function_a, *delta, gamma2,
          radius, det_inverse_jacobian, spacetime_dim, R_0,
          det_inv_jacobian_tau, filter_matrices);

      local_dvars[0] = get(temp_dtpsi);
      local_dvars[1] = get(temp_dtphi_tilde);
      local_dvars[2] = get(temp_dtpi);
    };

    std::string tag{"ElementData"};
    const observers::ObservationId obs_id = observers::ObservationId(time, tag);
    // dt = compute_adaptive_step_size(*delta, *metric_function_a, radius, CFL);
    if (step % 100 == 0) {
      std::vector<ElementVolumeData> VolumeData = create_data_for_file(
          radius, mesh_of_one_element, number_of_elements, refinement_level,
          vars, integrand_buffer, mass, delta, metric_function_a, gamma2,
          det_jacobian, matrix_buffer, spacetime_dim, R_0, det_jacobian_tau,
          det_inv_jacobian_tau);

      std::cout << "time: " << time << " step: " << step << ' ' << dt << "\n";

      write_data_hd5file(VolumeData, obs_id, 0.0, R_0, refinement_level,
                         mesh_of_one_element.number_of_grid_points());
    }

    find_min_A(metric_function_a, BH_formed, epsilon);
    if (BH_formed) {
      std::cout << "Black Hole"
                << "\n";
      return vars;
    }

    dt = compute_adaptive_step_size(*delta, *metric_function_a, radius, CFL);
    st.do_step(system, vars, time, dt);
    // std::cout << "A:\n" << get(*metric_function_a) << "\n";
    // std::cout << "where_nan:\n" << nan << "\n";

    size_t check = 0;
    if (step % 1000 == 0) {
      for (size_t element_boundary =
               mesh_of_one_element.number_of_grid_points();
           element_boundary <
           number_of_elements * mesh_of_one_element.number_of_grid_points() - 1;
           element_boundary =
               element_boundary + mesh_of_one_element.number_of_grid_points()) {
        if (vars[0][element_boundary] == vars[0][element_boundary - 1]) {
          check++;
        }
      }
      if (check == number_of_elements - 1) {
        std::cout << "check:\n"
                  << "true"
                  << "\n";
      } else {
        std::cout << "check:\n"
                  << "false"
                  << "\n";
      }
    }
    time = time + dt;
    step = step + 1;
  }
  std::cout << "time:\n" << time << "\n";
  std::cout << "Number of Steps:\n" << step << "\n";
  return vars;
}

void run(const size_t refinement_level, const size_t points_per_element,
         const double amp) {
  domain::creators::register_derived_with_charm();
  const size_t number_of_elements = two_to_the(refinement_level);
  const Mesh<1> mesh_of_one_element{points_per_element,
                                    Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};
  // const Mesh<1> mesh_of_one_element{points_per_element,
  //                                   Spectral::Basis::Legendre,
  //                                   Spectral::Quadrature::Gauss};

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

      point = affine_map(point_to_check);
      block_logical_coords.get(
          0)[element_index * mesh_of_one_element.number_of_grid_points() +
             coord_index] = point[0];
    }
  }

  //   domain::CoordinateMaps::Distribution distribution =
  //       domain::CoordinateMaps::Distribution::Logarithmic;
  domain::CoordinateMaps::Distribution distribution =
      domain::CoordinateMaps::Distribution::Linear;
  double singularity = -1.0002499999999999;
  //   domain::CoordinateMaps::Interval Interval(-1, 1, -1, 1, distribution,
  //                                             singularity);
  domain::CoordinateMaps::Interval Interval(-1, 1, -1, 1, distribution);

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

  const double width = 1.0;
  const double p = 2;
  const double q = 2;
  const double R_0 = 5.0;

  Scalar<DataVector> mass{mesh_of_one_element.number_of_grid_points() *
                          number_of_elements};
  Scalar<DataVector> buffer{mesh_of_one_element.number_of_grid_points() *
                            number_of_elements};

  Scalar<DataVector> metric_function_a{
      mesh_of_one_element.number_of_grid_points() * number_of_elements};

  const tnsr::I<DataVector, 1, Frame::Inertial> radius{
      {{sqrt((int_coord_tau.get(0) + 1.0) * 0.5) * R_0}}};

  const Scalar<DataVector> psi{amp * pow(get<0>(radius), 2 * q) *
                               exp(-pow(get<0>(radius), p) / pow(width, p))};
  // const Scalar<DataVector> psi{get(mass).size(), 0.0};
  const Scalar<DataVector> phi{
      -amp * (pow(get<0>(radius), 2 * q - 1)) *
      (p * pow(get<0>(radius), p) - 2 * q * pow(width, p)) *
      exp(-pow(get<0>(radius), p) / pow(width, p)) / pow(width, p)};
  // const Scalar<DataVector> phi{get(psi).size(), 0.0};
  const Scalar<DataVector> phi_tilde{
      -amp * 0.25 * (pow(get<0>(radius), 2 * q - 2)) *
      (p * pow(get<0>(radius), p) - 2 * q * pow(width, p)) *
      exp(-pow(get<0>(radius), p) / pow(width, p)) / pow(width, p)};
  const Scalar<DataVector> pi{-get<0>(radius) * get(phi)};

  Matrix matrix_buffer{mesh_of_one_element.number_of_grid_points(),
                       mesh_of_one_element.number_of_grid_points()};

  const size_t spacetime_dim = 4;

  bool BH_formed = false;
  const double gamma2 = 0.0;
  [[maybe_unused]] const size_t last_point = get(mass).size() - 1;

  const double alpha = 36;
  const unsigned half_power = 128;  // to remove lower mode.
  const long unsigned int FilterIndex = 0;
  Filters::Exponential<FilterIndex> exponential_filter =
      Filters::Exponential<FilterIndex>(alpha, half_power, true, std::nullopt);
  const Matrix& filter_matrix =
      exponential_filter.filter_matrix(mesh_of_one_element);
  const std::array<std::reference_wrapper<const Matrix>, 1> filter_matrices{
      {std::cref(filter_matrix)}};
  // compute_delta_integral_logical(&delta, &integrand_buffer,
  // mesh_of_one_element, phi, pi, jacobian, R_0, jacobian_tau);
  // compute_mass_integral(&mass, &matrix_buffer, mesh_of_one_element, phi, pi,
  //                       jacobian, radius, spacetime_dim, R_0, jacobian_tau);
  // compute_metric_function_a_from_mass(&metric_function_a, mass, radius,
  //                                     spacetime_dim);
  // compute_time_derivatives_first_order_2(
  //   &dt_psi,
  //   &dt_phi_tilde,
  //   &dt_pi,
  //   &buffer,
  //   mesh_of_one_element, psi,phi_tilde,  pi,phi,metric_function_a,delta,
  //   gamma2,radius, inv_jacobian, spacetime_dim, R_0, inv_jacobian_tau,
  //   filter_matrices);
  //   compute_time_derivatives_first_order(
  //       &dt_psi, &dt_phi, &dt_pi, &buffer, mesh_of_one_element, psi, phi, pi,
  //       metric_function_a, delta, gamma2, radius, inv_jacobian,
  //       spacetime_dim, R_0, inv_jacobian_tau);

  std::array<DataVector, 3> evaluated_vars = integrate_fields_in_time(
      &dt_psi, &dt_phi, &dt_pi, &buffer, &integrand_buffer, jacobian,
      &matrix_buffer, mesh_of_one_element, psi, phi_tilde, pi, &mass, &delta,
      &metric_function_a, gamma2, radius, inv_jacobian, spacetime_dim, R_0,
      refinement_level, BH_formed, jacobian_tau, inv_jacobian_tau,
      filter_matrices);

  // std::cout << "Psi:\n"
  //           << std::setprecision(16) << std::scientific << get(psi) << "\n";
  // std::cout << "Phi:\n"
  //           << std::setprecision(16) << std::scientific <<
  //           get(phi)
  //           << "\n";
  // std::cout << "Pi:\n"
  //           << std::setprecision(16) << std::scientific <<
  //           evaluated_vars[2]
  //           << "\n";
  // std::cout << "dtPsi:\n"
  //           << std::setprecision(16) << std::scientific << get(dt_psi) <<
  //           "\n";
  // std::cout << "dtPhiTilde:\n"
  //           << std::setprecision(16) << std::scientific << get(dt_phi_tilde)
  //           << "\n";
  // std::cout << "dtPi:\n"
  //           << std::setprecision(16) << std::scientific << get(dt_pi) <<
  //           "\n";
  // std::cout << "Mass:\n"
  //           << std::setprecision(16) << std::scientific <<
  //           get(mass)[last_point]
  //           << "\n";
  //   std::cout << "Metric A:\n"
  //             << std::setprecision(16) << std::scientific
  //             << get(metric_function_a) << "\n";
  //   std::cout << "delta:\n"
  //             << std::setprecision(16) << std::scientific << get(delta) <<
  //             "\n";
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
