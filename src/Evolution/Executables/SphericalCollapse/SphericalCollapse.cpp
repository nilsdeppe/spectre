// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need Boost MultiArray because it is used internally by ODEINT
#include "DataStructures/BoostMultiArray.hpp"

#include <algorithm>
#include <boost/numeric/odeint.hpp>
#include <boost/program_options.hpp>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <tuple>
#include <typeinfo>
#include <variant>
#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Blaze/IntegerPow.hpp"
#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
// #include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/Structure/ChildSize.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/LinearSolver/Lapack.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
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
  static constexpr Options::String help = {"The radius of the outer boundary."};
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

  bool operator==(const ElementId1d& other) const {
    return block_id == other.block_id &&
           segment_id ==
               other.segment_id;  // Adjust comparison as per actual structure
  }
};
std::ostream& operator<<(std::ostream& os, const ElementId1d& id) {
  return os << "(" << id.block_id << "," << id.segment_id << ")";
}

template <size_t VolumeDim>
ElementId1d id_of_parent(const ElementId1d& element_id,
                         const std::array<amr::Flag, VolumeDim>& flags) {
  using ::operator<<;
  ASSERT(alg::count(flags, amr::Flag::Join) > 0,
         "Element " << element_id << " is not joining given flags " << flags);
  ASSERT(alg::count(flags, amr::Flag::Split) == 0,
         "Splitting and joining an Element is not supported");

  return {element_id.block_id, element_id.segment_id.id_of_parent()};
}

template <size_t VolumeDim>
std::array<ElementId1d, 2> ids_of_children(
    const ElementId1d& element_id,
    const std::array<amr::Flag, VolumeDim>& flags) {
  using ::operator<<;
  ASSERT(alg::count(flags, amr::Flag::Split) > 0,
         "Element " << element_id << " has no children given flags " << flags);
  ASSERT(alg::count(flags, amr::Flag::Join) == 0,
         "Splitting and joining an Element is not supported");
  const size_t block_id = element_id.block_id;
  // const size_t grid_index = element_id.grid_index();
  if constexpr (VolumeDim == 1) {
    return {{{block_id, {{element_id.segment_id.id_of_child(Side::Lower)}}},
             {block_id, {{element_id.segment_id.id_of_child(Side::Upper)}}}}};
  }
}
std::vector<domain::CoordinateMap<Frame::ElementLogical, Frame::Grid,
                                  domain::CoordinateMaps::Affine,
                                  domain::CoordinateMaps::Interval>>
make_coordinate_map(const size_t number_of_elements,
                    std::vector<ElementId1d>& element_ids) {
  std::vector<domain::CoordinateMap<Frame::ElementLogical, Frame::Grid,
                                    domain::CoordinateMaps::Affine,
                                    domain::CoordinateMaps::Interval>>
      coordinate_maps{number_of_elements};

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
  // std::cout << "is it this vector(3)" << "\n";
  return coordinate_maps;
}
tnsr::I<DataVector, 1, Frame::Grid> initialize_grid_coords(
    std::vector<ElementId1d>& element_ids, const size_t number_of_elements,
    const Mesh<1>& mesh_of_one_element) {
  const tnsr::I<DataVector, 1, Frame::ElementLogical>
      logical_coords_one_element{logical_coordinates(mesh_of_one_element)};
  std::vector<domain::CoordinateMap<Frame::ElementLogical, Frame::Grid,
                                    domain::CoordinateMaps::Affine,
                                    domain::CoordinateMaps::Interval>>
      coordinate_maps = make_coordinate_map(number_of_elements, element_ids);
  tnsr::I<DataVector, 1, Frame::Grid> grid_coords{
      mesh_of_one_element.number_of_grid_points() * number_of_elements};
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
  }

  return grid_coords;
}
std::array<const Scalar<DataVector>, 2> create_jacobians(
    const Mesh<1>& mesh_of_one_element, std::vector<ElementId1d>& element_ids,
    const size_t number_of_elements) {
  const tnsr::I<DataVector, 1, Frame::ElementLogical>
      logical_coords_one_element{logical_coordinates(mesh_of_one_element)};

  std::vector<domain::CoordinateMap<Frame::ElementLogical, Frame::Grid,
                                    domain::CoordinateMaps::Affine,
                                    domain::CoordinateMaps::Interval>>
      coordinate_maps = make_coordinate_map(number_of_elements, element_ids);

  Jacobian<DataVector, 1, Frame::ElementLogical, Frame::Grid> jacobian{
      mesh_of_one_element.number_of_grid_points() * number_of_elements};
  InverseJacobian<DataVector, 1, Frame::ElementLogical, Frame::Grid>
      inv_jacobian{mesh_of_one_element.number_of_grid_points() *
                   number_of_elements};
  for (size_t element_index = 0; element_index < number_of_elements;
       element_index += 1) {
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
  std::array<const Scalar<DataVector>, 2> dets{det_jacobian, det_inv_jacobian};

  return dets;
}
static bool use_flat_space = false;
const DataVector read_element_data(
    const DataVector& quantity, const Mesh<1>& mesh_of_one_element,
    const size_t element_index,
    [[maybe_unused]] const size_t number_of_elements) {
  const DataVector view{mesh_of_one_element.number_of_grid_points()};
  make_const_view(make_not_null(&view), quantity,
                  element_index * mesh_of_one_element.number_of_grid_points(),
                  mesh_of_one_element.number_of_grid_points());

  return view;
}
Scalar<DataVector> differential_eq_for_A(
    const Scalar<DataVector>& phi, const Scalar<DataVector>& pi,
    const Scalar<DataVector>& A,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const double spacetime_dim, const bool intermediate) {
  Scalar<DataVector> diff_eq{get<0>(radius).size()};
  get(diff_eq)[0] = 0.0;
  for (size_t i = 1; i < get<0>(radius).size(); i++) {
    get(diff_eq)[i] =
        ((spacetime_dim - 3) / get<0>(radius))[i] * (1 - get(A)[i]) -
        2 * M_PI * get<0>(radius)[i] * get(A)[i] *
            (square(get(pi)[i]) + square(get(phi)[i]));
    if (intermediate) {
      if (use_flat_space) {
        get(diff_eq)[i] = 0.0;
      }
    }
  }
  return diff_eq;
}

Scalar<DataVector> differential_eq_for_delta(
    const Scalar<DataVector>& phi, const Scalar<DataVector>& pi,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const bool intermediate) {
  Scalar<DataVector> diff_eq{-4 * M_PI * get<0>(radius) *
                             (square(get(pi)) + square(get(phi)))};
  if (intermediate) {
    if (use_flat_space) {
      get(diff_eq) = 0.0;
    }
  }

  return diff_eq;
}

void compute_delta_integral_logical(
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<DataVector*> integrand_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& phi,
    const Scalar<DataVector>& pi, const Scalar<DataVector>& det_jacobian,
    const tnsr::I<DataVector, 1, Frame::Inertial>& /*radius*/,
    const double outer_boundary_radius, const bool intermediate) {
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
  if (intermediate) {
    if (use_flat_space) {
      get(*delta) = 0.0;
    }
  }
}

void basic_lu(const gsl::not_null<Matrix*> alu,
              const gsl::not_null<DataVector*> b) {
  const size_t n = alu->columns();
  for (size_t j = 0; j < alu->columns(); ++j) {
    for (size_t i = 0; i <= j; ++i) {
      for (size_t k = 0; k < i; ++k) {
        (*alu)(i, j) -= (k == i ? 1.0 : (*alu)(i, k)) * (*alu)(j, k);
      }
    }
    for (size_t i = j + 1; i < alu->rows(); ++i) {
      for (size_t k = 0; k < j; ++k) {
        (*alu)(i, j) -= (k == i ? 1.0 : (*alu)(i, k)) * (*alu)(j, k);
      }
      (*alu)(i, j) /= (*alu)(j, j);
    }
  }
  // Now alu is the alphas and betas of the LU decomp
  for (size_t i = 0; i < n; i++) {
    double sum = (*b)[i];
    for (size_t j = 0; j < i; j++) {
      sum -= (*alu)(i, j) * (*b)[j];
    }
    (*b)[i] = sum;
  }
  for (size_t i = n-1; i < n; i--) {
    double sum = (*b)[i];
    for (size_t j = i + 1; j < n; j++) {
      sum -= (*alu)(i, j) * (*b)[j];
    }
    (*b)[i] = sum / (*alu)(i, i);
  }
}

void compute_mass_integral(
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Matrix*> matrix_buffer,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& phi,
    const Scalar<DataVector>& pi, const Scalar<DataVector>& det_jacobian,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const double spacetime_dim, const double outer_boundary_radius,
    const bool intermediate) {
  const size_t pts_per_element = mesh_of_one_element.number_of_grid_points();
  const size_t number_of_grids = get(pi).size() / pts_per_element;
  const Matrix& integration_matrix =
      Spectral::integration_matrix(mesh_of_one_element);
  static std::vector<int> ipiv_cache(
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre>);
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
                   integer_pow(get<0>(radius)[index],
                               static_cast<int>(spacetime_dim) - 3);
        matrix_buffer->operator()(i, k) =
            (i == k ? 1.0 : 0.0) +
            integration_matrix(i, k) * sigma * get(det_jacobian)[index];
      }
    }
    // Solve the linear system A m = b for m (the mass)
    // NOTE: Can't use Cholesky because the matrix is not symmetric .
    basic_lu(matrix_buffer, make_not_null(&view));
    // lapack::general_matrix_linear_solve(
    //     make_not_null(&view), make_not_null(&ipiv_cache), matrix_buffer);
  }
  if (intermediate) {
    if (use_flat_space) {
      get(*mass) = 0.0;
    }
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
  view_a =
      1.0 - 2.0 * view_mass /
                integer_pow(view_radius, static_cast<int>(spacetime_dim) - 3);
  // for (size_t i = 1; i < get(mass).size(); ++i) {
  //   get(*metric_function_a)[i] =
  //       1.0 - 2.0 * get(mass)[i] / pow(get<0>(radius)[i], spacetime_dim
  //       - 3.0);
  // }
}
double truncation_error_estimate(const DataVector& variable_to_check,
                                 const Mesh<1>& mesh_of_one_element,
                                 const size_t element_index,
                                 const size_t number_of_elements,
                                 const db::Access& box) {
  double abs_error = PowerMonitors::absolute_truncation_error(
      read_element_data(variable_to_check, mesh_of_one_element, element_index,
                        number_of_elements),
      mesh_of_one_element)[0];
  double rel_error = PowerMonitors::relative_truncation_error(
      read_element_data(variable_to_check, mesh_of_one_element, element_index,
                        number_of_elements),
      mesh_of_one_element)[0];
  double rel_tolerance = 1e-5;
  double abs_tolerance = 1e-6 * get<Tags::Amplitude>(box);
  return std::abs(abs_error) /
         (abs_tolerance + std::abs(rel_error) * rel_tolerance);
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
  // std::cout << "is it this vector(4)" << "\n";
  return element_ids;
}
// std::vector<std::variant<std::vector<ElementId1d>, std::array<DataVector,
// 3>>>
std::tuple<std::vector<ElementId1d>, std::array<DataVector, 3>, size_t>
determine_bad_truncation_error(const DataVector& variable_to_check,
                               const size_t number_of_elements,
                               std::vector<ElementId1d> element_ids,
                               const Mesh<1>& mesh_of_one_element,
                               const db::Access& box,
                               std::array<DataVector, 3> vars) {
  // need to define this "condition1"
  std::vector<ElementId1d> new_elements{};
  // std::cout << "check1" << number_of_elements << "\n";
  // std::cout << "check2" << element_ids.size() << "\n";
  std::vector<int> changed{};
  size_t changed_val = 0;
  bool max_reached = false;
  // 0: nothing changed, 1: refined, 2: coarsened
  for (size_t element_index = 0; element_index < number_of_elements;
       element_index += 1) {
    // bool changed_any_element = false;
    std::array<ElementId1d, 2> child_ids = ids_of_children(
        element_ids[element_index], std::array{amr::Flag::Split});

    const double condition1 = 5;
    const bool refine =
        truncation_error_estimate(variable_to_check, mesh_of_one_element,
                                  element_index, number_of_elements,
                                  box) > condition1;
    // const bool refine =true;
    // const bool coarse = false;

    const size_t max_refinement_level = 28;
    if ((element_ids[element_index].segment_id.refinement_level() <
         max_refinement_level) &&
        std::all_of(
            child_ids.begin(), child_ids.end(), [](const ElementId1d& id) {
              return id.segment_id.refinement_level() < max_refinement_level;
            })) {
      if (refine) {
        changed.push_back(1);
        for (size_t ind = 0; ind < child_ids.size(); ind += 1) {
          new_elements.push_back(child_ids[ind]);
          changed_val += 1;
        }

      }
      // else if (coarse){
      //   // refine less (combine to get parent)
      // changed.push_back(2);
      //   // element_ids.erase(element_ids.begin() + element_index);
      //   // element_ids.erase(element_ids.begin() + index_of_sibling);
      //   ElementId1d parent_id_new{0, parent_id.segment_id(0)};
      //   new_elements.push_back(parent_id_new);

      // }
      else if (!refine) {
        changed.push_back(0);
        new_elements.push_back(element_ids[element_index]);
        // sum_nochange += 1;
      }
    } else {
      max_reached = true;
    }
  }
  if (max_reached == true) {
    new_elements = element_ids;
    changed = std::vector<int>(new_elements.size(), 0);
    changed_val = 0;
  }

  // check if changed is all 0, no refinement change
  if (changed_val != 0) {
    std::array<DataVector, 3> new_vars_copy = vars;

    new_vars_copy[0].destructive_resize(
        mesh_of_one_element.number_of_grid_points() * new_elements.size());

    new_vars_copy[1].destructive_resize(
        mesh_of_one_element.number_of_grid_points() * new_elements.size());
    new_vars_copy[2].destructive_resize(
        mesh_of_one_element.number_of_grid_points() * new_elements.size());
    for (size_t var_index = 0; var_index < 3; ++var_index) {
      for (size_t index_old = 0, index_new = 0;
           index_old < element_ids.size();) {
        if (changed[index_old] == 0) {
          DataVector view_old{
              &vars[var_index]
                   [index_old * mesh_of_one_element.number_of_grid_points()],
              mesh_of_one_element.number_of_grid_points()};

          DataVector view_new{
              &new_vars_copy[var_index]
                            [index_new *
                             mesh_of_one_element.number_of_grid_points()],
              mesh_of_one_element.number_of_grid_points()};

          view_new = view_old;

          ++index_old;
          ++index_new;

        } else if (changed[index_old] == 1) {
          std::array<std::reference_wrapper<const Matrix>, 2> matrices{
              {std::cref(Spectral::projection_matrix_parent_to_child(
                   mesh_of_one_element, mesh_of_one_element,
                   Spectral::ChildSize::LowerHalf)),
               std::cref(Spectral::projection_matrix_parent_to_child(
                   mesh_of_one_element, mesh_of_one_element,
                   Spectral::ChildSize::UpperHalf))}};

          for (size_t i = 0; i < 2; ++i) {  // loop over lower & upper
            DataVector view_old{
                &vars[var_index]
                     [index_old * mesh_of_one_element.number_of_grid_points()],
                mesh_of_one_element.number_of_grid_points()};
            // std::cout << index_new *
            // mesh_of_one_element.number_of_grid_points()
            //           << "\n";
            DataVector view_new{
                &new_vars_copy[var_index]
                              [index_new *
                               mesh_of_one_element.number_of_grid_points()],
                mesh_of_one_element.number_of_grid_points()};
            // std::cout << var_index << "\n";

            // std::cout << new_vars_copy[var_index]<< "\n";

            apply_matrices(make_not_null(&view_new),
                           std::array<Matrix, 1>{matrices[i].get()}, view_old,
                           mesh_of_one_element.extents());
            // if (var_index ==0){std::cout << new_vars_copy[var_index]<< "\n";
            // std::cout << view_new<< "\n";}

            ++index_new;
          }

          ++index_old;
        }
        // else {
        //   // assert(coarsening);
        //   index_old += 2;
        //   ++index_new;
        // }
      }
    }
    return std::make_tuple(new_elements, new_vars_copy, changed_val);
  } else {
    return std::make_tuple(element_ids, vars, changed_val);
  }
}

void compute_time_derivatives_first_order_2(
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const gsl::not_null<Scalar<DataVector>*> dt_phi_tilde,
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const Mesh<1>& mesh_of_one_element, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& phi_tilde, const Scalar<DataVector>& pi,
    [[maybe_unused]] const Scalar<DataVector>& phi,
    const Scalar<DataVector>& metric_function_a,
    const Scalar<DataVector>& metric_function_delta, const double gamma2,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const Scalar<DataVector>& det_inverse_jacobian, const double spacetime_dim,
    const double outer_boundary_radius,
    const std::array<std::reference_wrapper<const Matrix>, 1>& filter_matrices,
    [[maybe_unused]] const bool intermediate) {
  // Scalar<DataVector> diff_eq_A = differential_eq_for_A(
  //     phi, pi, metric_function_a, radius, spacetime_dim, intermediate);
  // Scalar<DataVector> diff_eq_delta =
  //     differential_eq_for_delta(phi, pi, radius, intermediate);
  const size_t num_pts = get(pi).size();
  const size_t number_of_elements =
      num_pts / mesh_of_one_element.number_of_grid_points();
  // Scalar<DataVector> buffer1{mesh_of_one_element.number_of_grid_points() *
  //                            number_of_elements};

  // We can extend this once the dt_VARS are contiguous so that we can operate
  // on all evolved variables at once.
  DataVector full_buffer{num_pts * 4};
  Scalar<DataVector> buffer2{full_buffer.data(), num_pts};
  Scalar<DataVector> buf_dt_psi{
      std::next(full_buffer.data(), static_cast<std::ptrdiff_t>(num_pts)),
      num_pts};
  Scalar<DataVector> buf_dt_phi_tilde{
      std::next(full_buffer.data(), static_cast<std::ptrdiff_t>(2 * num_pts)),
      num_pts};
  Scalar<DataVector> buf_dt_pi{
      std::next(full_buffer.data(), static_cast<std::ptrdiff_t>(3 * num_pts)),
      num_pts};
  // Scalar<DataVector> buffer4{mesh_of_one_element.number_of_grid_points() *
  //                            number_of_elements};

  std::array<std::reference_wrapper<const Matrix>, 1> logical_diff_matrices{
      {std::cref(Spectral::differentiation_matrix(mesh_of_one_element))}};

  // compute dt_psi
  // get(buffer1) = get(metric_function_a) * exp(-get(metric_function_delta));
  // get(*dt_psi) = get(buffer1) * get(pi);

  // get(*dt_psi) = get(metric_function_a) * exp(-get(metric_function_delta));
  get(buf_dt_psi) = get(metric_function_a) * exp(-get(metric_function_delta));

  // compute 2nd term of dt_pi
  get(buffer2) = get(buf_dt_psi) * get(phi_tilde);
  apply_matrices(make_not_null(&get(buf_dt_pi)), logical_diff_matrices,
                 get(buffer2), mesh_of_one_element.extents());
  // 16 because other factor of 4 from tilde_phi
  get(buf_dt_pi) *= square(4.0 / outer_boundary_radius) *
                    square(get<0>(radius)) * get(det_inverse_jacobian);

  // A form that expands the partial derivative out
  // apply_matrices(make_not_null(&get(*dt_pi)), logical_diff_matrices,
  //                get(phi_tilde), mesh_of_one_element.extents());
  // get(*dt_pi) *= (4.0 * get<0>(radius) / square(outer_boundary_radius));
  // get(*dt_pi) *= get(det_inverse_jacobian) * get(*dt_psi);
  // get(*dt_pi) += (get(diff_eq_A) * exp(-get(metric_function_delta)) -
  //                 get(diff_eq_delta) * get(*dt_psi)) *
  //                *get(phi_tilde);
  // get(*dt_pi) *= 4.0 * get<0>(radius);

  // compute 1st term of dt_pi
  get(buf_dt_pi) += 4.0 * (spacetime_dim - 1.0) * get(buf_dt_psi) *
                    get(phi_tilde);  // adding in the first term
                                     // of the expansion

  // compute dt_phi_tilde
  get(buf_dt_psi) *= get(pi);
  get(buffer2) = get(buf_dt_psi) + gamma2 * get(psi);

  apply_matrices(make_not_null(&get(buf_dt_phi_tilde)), logical_diff_matrices,
                 get(buffer2), mesh_of_one_element.extents());
  get(buf_dt_phi_tilde) *=
      get(det_inverse_jacobian);  //  * (1.0 / square(outer_boundary_radius));
  get(buf_dt_phi_tilde) *= (1.0 / square(outer_boundary_radius));

  get(buf_dt_phi_tilde) -= gamma2 * get(phi_tilde);

  {
    // DataVector& pre_filter_data = get(buffer2);
    // pre_filter_data = get(*dt_psi);
    // DataVector pre_filter_data{get(*dt_psi)};
    apply_matrices(make_not_null(&get(*dt_psi)), filter_matrices,
                   get(buf_dt_psi), mesh_of_one_element.extents());
    // pre_filter_data = get(*dt_pi);
    apply_matrices(make_not_null(&get(*dt_pi)), filter_matrices, get(buf_dt_pi),
                   mesh_of_one_element.extents());
    // pre_filter_data = get(*dt_phi_tilde);
    apply_matrices(make_not_null(&get(*dt_phi_tilde)), filter_matrices,
                   get(buf_dt_phi_tilde), mesh_of_one_element.extents());
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
    const Mesh<1>& mesh, const Scalar<DataVector>& delta,
    const Scalar<DataVector>& metric_function_a,
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const double CFL_safety_factor) {
  double min_adapted_dt = 1.0e300;

  const size_t num_pts = mesh.number_of_grid_points();
  const size_t num_elements = get(delta).size() / num_pts;
  for (size_t element_index = 0; element_index < num_elements;
       ++element_index) {
    const size_t i = (element_index + 1) * num_pts - 2;
    if ((get<0>(radius)[i + 1] - get<0>(radius)[i]) >
        1.0e-14 * get<0>(radius)[i]) {
      min_adapted_dt = std::min(
          min_adapted_dt, (get<0>(radius)[i + 1] - get<0>(radius)[i]) *
                              exp(get(delta)[i]) / get(metric_function_a)[i]);
    }
  }

  // for (size_t i = 1; i < get(delta).size() - 1; i++) {
  //   if ((get<0>(radius)[i + 1] - get<0>(radius)[i]) >
  //       1.0e-14 * get<0>(radius)[i]) {
  //     min_adapted_dt = std::min(
  //         min_adapted_dt, (get<0>(radius)[i + 1] - get<0>(radius)[i]) *
  //                             exp(get(delta)[i]) /
  //                             get(metric_function_a)[i]);
  //   }
  // }

  min_adapted_dt = CFL_safety_factor * min_adapted_dt;
  if (min_adapted_dt > 1) {
    std::cout << get(metric_function_a) << "\n";
  }
  return min_adapted_dt;
}

// void write_data_hd5file(const std::vector<ElementVolumeData>& volume_data,
//                         const observers::ObservationId& observation_id,
//                         const double lower_r, const double upper_r,
//                         const double time) {
//   const std::string h5_file_name{"VolumeDataForFields"};
//   const std::string input_source{""};
//   const std::string subfile_path{"/ElementData"};
//   const uint32_t version_number = 0;
//   h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name + ".h5"s, true,
//                                                 input_source};
//   auto& volume_file =
//       h5_file.try_insert<h5::VolumeData>(subfile_path, version_number);

//   // Just write an invalid domain for now.
//   domain::creators::Interval interval{std::array{lower_r},
//   std::array{upper_r},
//                                       std::array{0_st}, std::array{10_st}};
//   Domain<1> domain = interval.create_domain();
//   const auto serialized_domain = serialize(domain);
//   volume_file.write_volume_data(observation_id.hash(), time, volume_data,
//                                 serialized_domain);
// }

void create_data_for_file(
    const tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const Mesh<1>& mesh_of_one_element,
    [[maybe_unused]] const std::vector<ElementId1d>& element_ids,
    const std::array<DataVector, 3>& vars,
    const gsl::not_null<DataVector*> integrand_buffer,
    const gsl::not_null<Scalar<DataVector>*> mass,
    const gsl::not_null<Scalar<DataVector>*> delta,
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    const double /*gamma2*/, const Scalar<DataVector>& det_jacobian,
    const gsl::not_null<Matrix*> matrix_buffer, const double spacetime_dim,
    const double outer_boundary_radius, const size_t step_number,
    const double time, const std::string& volume_data_directory,
    const Scalar<DataVector>& det_inverse_jacobian, const db::Access& box,
    const std::array<std::reference_wrapper<const Matrix>, 1>& filter_matrices,
    const size_t number_of_elements) {
  const Scalar<DataVector> temp_phi{vars[1] * 4 * get<0>(radius)};
  const Scalar<DataVector> temp_pi{vars[2]};
  const Scalar<DataVector> temp_psi{vars[0]};
  const Scalar<DataVector> temp_phi_tilde{vars[1]};
  const bool intermediate = false;
  // const Scalar<DataVector>& det_inverse_jacobian
  compute_delta_integral_logical(delta, integrand_buffer, mesh_of_one_element,
                                 temp_phi, temp_pi, det_jacobian, radius,
                                 outer_boundary_radius, intermediate);
  compute_mass_integral(mass, matrix_buffer, mesh_of_one_element, temp_phi,
                        temp_pi, det_jacobian, radius, spacetime_dim,
                        outer_boundary_radius, intermediate);
  compute_metric_function_a_from_mass(metric_function_a, *mass, radius,
                                      spacetime_dim);
  const auto size = get(temp_psi).size();
  Scalar<DataVector> temp_dtpsi{size, 0.0};
  Scalar<DataVector> temp_dtphi_tilde{size, 0.0};
  Scalar<DataVector> temp_dtpi{size, 0.0};
  compute_time_derivatives_first_order_2(
      make_not_null(&temp_dtpsi), make_not_null(&temp_dtphi_tilde),
      make_not_null(&temp_dtpi), mesh_of_one_element, temp_psi, temp_phi_tilde,
      temp_pi, temp_phi, *metric_function_a, *delta, get<Tags::Gamma2>(box),
      radius, det_inverse_jacobian, get<Tags::SpacetimeDimensions>(box),
      get<Tags::OuterBoundaryRadius>(box), filter_matrices, intermediate);
  std::stringstream data_to_write{};
  // std::stringstream truncation_error_data{};
  data_to_write
      << std::setprecision(18) << std::scientific << "# Time: " << time
      << "\n# 0 radius\n# 1 psi\n# 2 phi\n# 3 phi_tilde\n# 4 pi\n# 5 delta\n"
      << "# 6 mass\n# 7 A\n# 8 dt_psi\n# 9 dt_phi_tilde\n# 10 dt_pi\n";
  for (size_t i = 0; i < get<0>(radius).size(); ++i) {
    data_to_write << std::setprecision(18) << get<0>(radius)[i] << ' '
                  << vars[0][i] << ' ' << get(temp_phi)[i] << ' ' << vars[1][i]
                  << ' ' << get(temp_pi)[i] << ' ' << get(*delta)[i] << ' '
                  << get(*mass)[i] << ' ' << get(*metric_function_a)[i] << ' '
                  << get(temp_dtpsi)[i] << ' ' << get(temp_dtphi_tilde)[i]
                  << ' ' << get(temp_dtpi)[i] << "\n";
  }
  std::ofstream out_file{volume_data_directory + "/Step" +
                         std::to_string(step_number) + ".txt"};
  out_file << data_to_write.str();
  out_file.close();

  // std::stringstream mass_data_to_write{};
  // mass_data_to_write << std::setprecision(18) << std::scientific << time;
  // mass_data_to_write << std::setprecision(18) << get(*mass)[-1] << "\n";
  // std::ofstream mass_file;
  // mass_file.open(volume_data_directory + "/Mass.txt", std::ios::app);
  // mass_file << mass_data_to_write.str();
  // mass_file.close();

  std::stringstream truncation_error_data{};
  truncation_error_data
      << std::setprecision(18) << "# Time: " << time
      << "\n# 1 psi\n# 2 phi\n# 3 phi_tilde\n# 4 pi\n# 5 delta\n"
      << "# 6 mass\n# 7 A\n";
  // std::cout << number_of_elements << "\n";
  for (size_t element_index = 0; element_index < number_of_elements;
       element_index += 1) {
    truncation_error_data
        << std::setprecision(18)
        << PowerMonitors::relative_truncation_error(
               read_element_data(vars[0], mesh_of_one_element, element_index,
                                 number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::relative_truncation_error(
               read_element_data(get(temp_phi), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::relative_truncation_error(
               read_element_data(vars[1], mesh_of_one_element, element_index,
                                 number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::relative_truncation_error(
               read_element_data(get(temp_pi), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::relative_truncation_error(
               read_element_data(get(*delta), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::relative_truncation_error(
               read_element_data(get(*mass), mesh_of_one_element, element_index,
                                 number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::relative_truncation_error(
               read_element_data(get(*metric_function_a), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << "\n";
  }

  std::ofstream truncation_error_file;
  truncation_error_file.open(volume_data_directory + "/RelTruncationError.txt",
                             std::ios::app);
  truncation_error_file << truncation_error_data.str();
  truncation_error_file.close();

  std::stringstream abs_truncation_error_data{};
  abs_truncation_error_data
      << std::setprecision(18) << "# Time: " << time
      << "\n# 1 psi\n# 2 phi\n# 3 phi_tilde\n# 4 pi\n# 5 delta\n"
      << "# 6 mass\n# 7 A\n";
  for (size_t element_index = 0; element_index < number_of_elements;
       element_index += 1) {
    abs_truncation_error_data
        << std::setprecision(18)
        << PowerMonitors::absolute_truncation_error(
               read_element_data(vars[0], mesh_of_one_element, element_index,
                                 number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::absolute_truncation_error(
               read_element_data(get(temp_phi), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::absolute_truncation_error(
               read_element_data(vars[1], mesh_of_one_element, element_index,
                                 number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::absolute_truncation_error(
               read_element_data(get(temp_pi), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::absolute_truncation_error(
               read_element_data(get(*delta), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::absolute_truncation_error(
               read_element_data(get(*mass), mesh_of_one_element, element_index,
                                 number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::absolute_truncation_error(
               read_element_data(get(*metric_function_a), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << "\n";
  }

  std::ofstream abs_truncation_error_file;
  abs_truncation_error_file.open(
      volume_data_directory + "/AbsTruncationError.txt", std::ios::app);
  abs_truncation_error_file << abs_truncation_error_data.str();
  abs_truncation_error_file.close();

  std::stringstream power_monitors_data{};
  power_monitors_data
      << std::setprecision(18) << "# Time: " << time
      << "\n# 1 psi\n# 2 phi\n# 3 phi_tilde\n# 4 pi\n# 5 delta\n"
      << "# 6 mass\n# 7 A\n";
  for (size_t element_index = 0; element_index < number_of_elements;
       element_index += 1) {
    power_monitors_data
        << std::setprecision(18)
        << PowerMonitors::power_monitors(
               read_element_data(vars[0], mesh_of_one_element, element_index,
                                 number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::power_monitors(
               read_element_data(get(temp_phi), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::power_monitors(
               read_element_data(vars[1], mesh_of_one_element, element_index,
                                 number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::power_monitors(
               read_element_data(get(temp_pi), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::power_monitors(
               read_element_data(get(*delta), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::power_monitors(
               read_element_data(get(*mass), mesh_of_one_element, element_index,
                                 number_of_elements),
               mesh_of_one_element)[0]
        << ' '
        << PowerMonitors::power_monitors(
               read_element_data(get(*metric_function_a), mesh_of_one_element,
                                 element_index, number_of_elements),
               mesh_of_one_element)[0]
        << "\n";
  }

  std::ofstream power_monitors_file;
  power_monitors_file.open(volume_data_directory + "/PowerMonitors.txt",
                           std::ios::app);
  power_monitors_file << power_monitors_data.str();
  power_monitors_file.close();
  return;
}

std::optional<double> find_min_A(
    const gsl::not_null<Scalar<DataVector>*> metric_function_a,
    tnsr::I<DataVector, 1, Frame::Inertial>& radius,
    const double horizon_tolerance) {
  for (size_t index = 0; index < get(*metric_function_a).size(); index++) {
    // std::cout <<"metric value" << "\n";
    // std::cout << abs(get(*metric_function_a)[index]) << "\n";
    if (abs(get(*metric_function_a)[index]) < horizon_tolerance) {
      std::cout << "A: " << abs(get(*metric_function_a)[index]) << "\n";
      return get<0>(radius)[index];
    }
  }
  return std::nullopt;
}

std::array<DataVector, 3> integrate_fields_in_time(
    const gsl::not_null<DataVector*> integrand_buffer,
    const Scalar<DataVector>& det_jacobian,
    const gsl::not_null<Matrix*> matrix_buffer,
    const Mesh<1>& mesh_of_one_element, std::vector<ElementId1d>& element_ids,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& phi_tilde,
    const Scalar<DataVector>& pi, const gsl::not_null<Scalar<DataVector>*> mass,
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

  // using StateDopri5 = boost::numeric::odeint::runge_kutta_dopri5<Vars>;
  // StateDopri5 st{};
  std::vector<double> times;

  double time = 0.0;
  double dt = 0.0;
  const bool filter_evolved_vars = false;
  size_t step = 0;
  std::optional<double> black_hole_radius{};
  const bool intermediate = true;
  size_t number_of_elements = element_ids.size();

  // make all the variables non-const

  Scalar<DataVector> mutable_det_inverse_jacobian = det_inverse_jacobian;
  Scalar<DataVector> mutable_det_jacobian = det_jacobian;
  tnsr::I<DataVector, 1, Frame::Inertial> mutable_radius = radius;
  DataVector no_filter{};
  Scalar<DataVector> temp_phi{};

  using std::abs;
  using StateDopri5 = boost::numeric::odeint::runge_kutta_dopri5<Vars>;
  StateDopri5 st{};
  while (abs(time) <= (get<Tags::FinalTime>(box))) {
    // std::cout << "size_check" << "\n";
    // std::cout<< get(*metric_function_a).size() << "\n";
    // std::cout<< get(*delta).size() << "\n";
    // std::cout<< get(*mass).size() << "\n";
    // std::cout<< get(mutable_det_inverse_jacobian).size() << "\n";
    // std::cout<< get(mutable_det_jacobian).size() << "\n";
    // std::cout<< (*integrand_buffer).size() << "\n";
    // std::cout<< get<0>(mutable_radius).size() << "\n";
    // std::cout << element_ids.size() << "\n";
    auto system = [&mesh_of_one_element, &metric_function_a, &delta, &mass,
                   &mutable_radius, &mutable_det_inverse_jacobian,
                   &integrand_buffer, &mutable_det_jacobian, &matrix_buffer,
                   &box, &filter_matrices, &element_ids, filter_evolved_vars,
                   &no_filter,
                   &temp_phi](const Vars& local_vars, Vars& local_dvars,
                              [[maybe_unused]] const double current_time) {
      (void)filter_evolved_vars;  // silence compiler warning
      // std::array<const Scalar<DataVector>,7>
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
        if (no_filter.size() != get(temp_psi).size()) {
          no_filter.destructive_resize(get(temp_psi).size());
        }
        no_filter = get(temp_psi);
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

      if (const size_t expected_size = get(temp_phi_tilde).size();
          get(temp_phi).size() != expected_size) {
        get(temp_phi).destructive_resize(expected_size);
      }
      get(temp_phi) = 4 * get(temp_phi_tilde) * get<0>(mutable_radius);

      compute_delta_integral_logical(
          delta, integrand_buffer, mesh_of_one_element, temp_phi, temp_pi,
          mutable_det_jacobian, mutable_radius,
          get<Tags::OuterBoundaryRadius>(box), intermediate);
      compute_mass_integral(mass, matrix_buffer, mesh_of_one_element, temp_phi,
                            temp_pi, mutable_det_jacobian, mutable_radius,
                            get<Tags::SpacetimeDimensions>(box),
                            get<Tags::OuterBoundaryRadius>(box), intermediate);
      compute_metric_function_a_from_mass(
          metric_function_a, *mass, mutable_radius,
          get<Tags::SpacetimeDimensions>(box));

      const auto size = get(temp_psi).size();
      if (local_dvars[0].size() != size) {
        for (DataVector&  t : local_dvars) {
          t.destructive_resize(size);
        }
      }
      Scalar<DataVector> temp_dtpsi{local_dvars[0].data(), size};
      Scalar<DataVector> temp_dtphi_tilde{local_dvars[1].data(), size};
      Scalar<DataVector> temp_dtpi{local_dvars[2].data(), size};
      compute_time_derivatives_first_order_2(
          make_not_null(&temp_dtpsi), make_not_null(&temp_dtphi_tilde),
          make_not_null(&temp_dtpi), mesh_of_one_element, temp_psi,
          temp_phi_tilde, temp_pi, temp_phi, *metric_function_a, *delta,
          get<Tags::Gamma2>(box), mutable_radius, mutable_det_inverse_jacobian,
          get<Tags::SpacetimeDimensions>(box),
          get<Tags::OuterBoundaryRadius>(box), filter_matrices, intermediate);
    };

    if (not use_flat_space) {
      // if (number_of_elements == 49){std::cout<<get(*mass) << "\n";}

      black_hole_radius = find_min_A(metric_function_a, mutable_radius,
                                     get<Tags::HorizonFinderTolerance>(box));
    }
    if (step % observation_frequency == 0 or black_hole_radius.has_value()) {
      std::cout << "The step is: " << step << "\n";
      std::cout << "no of elements is: " << number_of_elements << "\n";
      create_data_for_file(mutable_radius, mesh_of_one_element, element_ids,
                           vars, integrand_buffer, mass, delta,
                           metric_function_a, get<Tags::Gamma2>(box),
                           mutable_det_jacobian, matrix_buffer,
                           get<Tags::SpacetimeDimensions>(box),
                           get<Tags::OuterBoundaryRadius>(box), step, time,
                           volume_data_directory, mutable_det_inverse_jacobian,
                           box, filter_matrices, number_of_elements);
      if (black_hole_radius.has_value()) {
        std::cout << "Found black hole!!\nRadius: " << black_hole_radius.value()
                  << "\nTime: " << time << "\nStep: " << step << "\n";
        return vars;
      }
    }
    if (time_print_frequency.has_value() and
        (step % time_print_frequency.value() == 0)) {
      std::cout << "time: " << time << " step: " << step << " dt: " << dt
                << "\n";
    }

    dt = compute_adaptive_step_size(mesh_of_one_element, *delta,
                                    *metric_function_a, mutable_radius,
                                    get<Tags::CflFactor>(box));
    if (dt == 0.0) {
      std::cout << "dt " << dt << "\n";
    }

    if (time + dt > get<Tags::FinalTime>(box)) {
      dt = get<Tags::FinalTime>(box) - time;
    }
    if (dt == 0.0) {
      std::cout << "time " << time << "\n";
      return vars;
    }
    st.do_step(system, vars, time, dt);

    time = time + dt;
    ++step;
    // using MyVariant =
    //     std::variant<std::vector<ElementId1d>, std::array<DataVector, 3>>;
    // std::array<DataVector, 3> new_vars;
    std::tuple<std::vector<ElementId1d>, std::array<DataVector, 3>, size_t>
        returned_vec = determine_bad_truncation_error(
            vars[2], number_of_elements, element_ids, mesh_of_one_element, box,
            vars);

    element_ids = std::get<0>(returned_vec);

    vars = std::get<1>(returned_vec);
    size_t changed_val = std::get<2>(returned_vec);
    number_of_elements = element_ids.size();
    if (changed_val != 0) {
      tnsr::I<DataVector, 1, Frame::Grid> new_grid_coords =
          initialize_grid_coords(element_ids, number_of_elements,
                                 mesh_of_one_element);
      std::array<const Scalar<DataVector>, 2> determinants = create_jacobians(
          mesh_of_one_element, element_ids, number_of_elements);
      mutable_det_jacobian = determinants[0];
      mutable_det_inverse_jacobian = determinants[1];
      get<0>(mutable_radius) = sqrt((get<0>(new_grid_coords) + 1.0) * 0.5) *
                               get<Tags::OuterBoundaryRadius>(box);
      // if (number_of_elements ==49){std::cout<<vars[0]<< "\n";}
      get(*metric_function_a)
          .destructive_resize(mesh_of_one_element.number_of_grid_points() *
                              number_of_elements);
      get(*delta).destructive_resize(
          mesh_of_one_element.number_of_grid_points() * number_of_elements);
      get(*mass).destructive_resize(
          mesh_of_one_element.number_of_grid_points() * number_of_elements);
      (*integrand_buffer)
          .destructive_resize(mesh_of_one_element.number_of_grid_points() *
                              number_of_elements);

      Scalar<DataVector> phi_new{vars[1] * 4 * get<0>(mutable_radius)};
      Scalar<DataVector> pi_new{vars[2]};

      compute_delta_integral_logical(
          delta, integrand_buffer, mesh_of_one_element, phi_new, pi_new,
          mutable_det_jacobian, mutable_radius,
          get<Tags::OuterBoundaryRadius>(box), intermediate);
      compute_mass_integral(mass, matrix_buffer, mesh_of_one_element, phi_new,
                            pi_new, mutable_det_jacobian, mutable_radius,
                            get<Tags::SpacetimeDimensions>(box),
                            get<Tags::OuterBoundaryRadius>(box), intermediate);
      compute_metric_function_a_from_mass(
          metric_function_a, *mass, mutable_radius,
          get<Tags::SpacetimeDimensions>(box));
      st = StateDopri5{};
    }

    // std::cout << number_of_elements << "\n";

    // std::cout << "vector sizes match" << "\n";
    //  std::cout <<dt << "\n";
  }
  std::cout << "time " << time << "\n";
  return vars;
}

void run(const db::Access& box) {
  domain::creators::register_derived_with_charm();
  std::vector<ElementId1d> element_ids =
      compute_element_ids(get<Tags::InnerRefinementLevel>(box),
                          get<Tags::OuterRefinementLevel>(box));
  const size_t number_of_elements = element_ids.size();

  const Mesh<1> mesh_of_one_element{get<Tags::PointsPerElement>(box),
                                    Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};
  // std::cout << number_of_elements << '\n';

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

  tnsr::I<DataVector, 1, Frame::Grid> grid_coords = initialize_grid_coords(
      element_ids, number_of_elements, mesh_of_one_element);

  std::array<const Scalar<DataVector>, 2> determinants =
      create_jacobians(mesh_of_one_element, element_ids, number_of_elements);
  const Scalar<DataVector> det_jacobian = determinants[0];
  const Scalar<DataVector> det_inv_jacobian = determinants[1];

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
  const bool intermediate = true;
  DataVector integrand_buffer{mesh_of_one_element.number_of_grid_points() *
                              number_of_elements};
  compute_delta_integral_logical(
      &delta, &integrand_buffer, mesh_of_one_element, phi, pi, det_jacobian,
      radius, get<Tags::OuterBoundaryRadius>(box), intermediate);
  compute_mass_integral(&mass, &matrix_buffer, mesh_of_one_element, phi, pi,
                        det_jacobian, radius,
                        get<Tags::SpacetimeDimensions>(box),
                        get<Tags::OuterBoundaryRadius>(box), intermediate);
  compute_metric_function_a_from_mass(&metric_function_a, mass, radius,
                                      get<Tags::SpacetimeDimensions>(box));
  // get(delta) = 0.0;
  // get(mass) = 1.0;

  compute_time_derivatives_first_order_2(
      &dt_psi, &dt_phi_tilde, &dt_pi, mesh_of_one_element, psi, phi_tilde, pi,
      phi, metric_function_a, delta, get<Tags::Gamma2>(box), radius,
      det_inv_jacobian, get<Tags::SpacetimeDimensions>(box),
      get<Tags::OuterBoundaryRadius>(box), filter_matrices, intermediate);
  // compute_delta_integral_logical(&delta, &integrand_buffer,
  // mesh_of_one_element,
  //                                phi, pi, det_jacobian, radius,
  //                                get<Tags::OuterBoundaryRadius>(box));
  // compute_mass_integral(
  //     &mass, &matrix_buffer, mesh_of_one_element, phi, pi, det_jacobian,
  //     radius, get<Tags::SpacetimeDimensions>(box),
  //     get<Tags::OuterBoundaryRadius>(box));
  // std::cout << "mass" << get(mass) << "\n";
  // compute_metric_function_a_from_mass(&metric_function_a, mass, radius,
  //                                     get<Tags::SpacetimeDimensions>(box));
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
