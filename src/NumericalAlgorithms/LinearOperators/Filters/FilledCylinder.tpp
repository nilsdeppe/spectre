// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/LinearOperators/Filters/FilledCylinder.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <pup_stl.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/FilteringB2.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace Filters {
namespace FilledCylinder_detail {
inline std::optional<unsigned> to_unsigned(
    const std::optional<size_t> half_power) {
  if (not half_power.has_value()) {
    return std::nullopt;
  }
  return static_cast<unsigned>(*half_power);
}

inline bool is_legendre_or_chebyshev(const Spectral::Basis basis,
                                     const Spectral::Quadrature quadrature) {
  return (basis == Spectral::Basis::Legendre or
          basis == Spectral::Basis::Chebyshev) and
         (quadrature == Spectral::Quadrature::Gauss or
          quadrature == Spectral::Quadrature::GaussLobatto);
}
}  // namespace FilledCylinder_detail

template <typename TagList>
FilledCylinder<TagList>::FilledCylinder(
    const size_t num_modes_to_kill,
    const std::optional<size_t> radial_angular_half_power,
    const std::optional<size_t> z_half_power, const bool enable,
    const std::optional<std::vector<std::string>>& blocks_to_filter,
    const bool volume_filter_on_substep, const bool boundary_filter_on_substep,
    const std::optional<size_t> volume_filter_every_n_steps,
    const std::optional<size_t> boundary_filter_every_n_steps,
    const Options::Context& context)
    : num_modes_to_kill_(num_modes_to_kill),
      radial_angular_half_power_(radial_angular_half_power),
      z_half_power_(z_half_power),
      enable_(enable),
      volume_filter_on_substep_(volume_filter_on_substep),
      boundary_filter_on_substep_(boundary_filter_on_substep),
      volume_filter_every_n_steps_(volume_filter_every_n_steps),
      boundary_filter_every_n_steps_(boundary_filter_every_n_steps) {
  if (blocks_to_filter.has_value()) {
    blocks_and_groups_to_filter_ = std::vector<std::string>{};
    std::unordered_set<std::string> seen{};
    for (const std::string& block_name : blocks_to_filter.value()) {
      if (not seen.emplace(block_name).second) {
        PARSE_ERROR(context,
                    "Duplicate block name '"
                        << block_name
                        << "' found when creating a FilledCylinder filter.");
      }
      blocks_and_groups_to_filter_->push_back(block_name);
    }
  }
}

template <typename TagList>
void FilledCylinder<TagList>::pup(PUP::er& p) {
  Filter<3, TagList>::pup(p);
  p | num_modes_to_kill_;
  p | radial_angular_half_power_;
  p | z_half_power_;
  p | enable_;
  p | blocks_and_groups_to_filter_;
  p | blocks_to_filter_;
  p | volume_filter_on_substep_;
  p | boundary_filter_on_substep_;
  p | volume_filter_every_n_steps_;
  p | boundary_filter_every_n_steps_;
}

template <typename TagList>
std::unique_ptr<Filter<3, TagList>> FilledCylinder<TagList>::get_clone() const {
  return std::make_unique<FilledCylinder>(*this);
}

template <typename TagList>
bool FilledCylinder<TagList>::apply_volume_filter_on_substep() const {
  return enable_ and volume_filter_on_substep_;
}

template <typename TagList>
bool FilledCylinder<TagList>::apply_volume_filter_on_this_step(
    const size_t step_number) const {
  if (not enable_) {
    return false;
  }
  if (volume_filter_every_n_steps_.has_value()) {
    return step_number % volume_filter_every_n_steps_.value() == 0;
  } else {
    return false;
  }
}

template <typename TagList>
bool FilledCylinder<TagList>::apply_boundary_filter_on_substep() const {
  return enable_ and boundary_filter_on_substep_;
}

template <typename TagList>
bool FilledCylinder<TagList>::apply_boundary_filter_on_this_step(
    const size_t step_number) const {
  if (not enable_) {
    return false;
  }
  if (boundary_filter_every_n_steps_.has_value()) {
    return step_number % boundary_filter_every_n_steps_.value() == 0;
  } else {
    return false;
  }
}

template <typename TagList>
const std::optional<std::vector<size_t>>&
FilledCylinder<TagList>::blocks_to_filter() const {
  return blocks_to_filter_;
}

template <typename TagList>
void FilledCylinder<TagList>::set_blocks_to_filter(
    const std::vector<std::string>& all_block_names,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        block_groups) {
  if (not blocks_and_groups_to_filter_.has_value()) {
    blocks_to_filter_ = std::nullopt;
    return;
  }
  if (all_block_names.empty()) {
    ERROR(
        "The domain chosen doesn't use block names, but the filter has "
        "specified block names to use.");
  }
  blocks_to_filter_ = domain::block_ids_from_names(
      blocks_and_groups_to_filter_.value(), all_block_names, block_groups);
}

template <typename TagList>
bool FilledCylinder<TagList>::supports_mesh(const Mesh<3>& mesh) const {
  // Radial direction (dim 0): ZernikeB2 with GaussRadauUpper quadrature
  if (mesh.basis(0) != Spectral::Basis::ZernikeB2 or
      mesh.quadrature(0) != Spectral::Quadrature::GaussRadauUpper) {
    return false;
  }
  // Angular direction (dim 1): ZernikeB2 with Equiangular quadrature
  if (mesh.basis(1) != Spectral::Basis::ZernikeB2 or
      mesh.quadrature(1) != Spectral::Quadrature::Equiangular) {
    return false;
  }
  // Axial z direction (dim 2): Legendre or Chebyshev
  return FilledCylinder_detail::is_legendre_or_chebyshev(mesh.basis(2),
                                                         mesh.quadrature(2));
}

template <typename TagList>
const Matrix& FilledCylinder<TagList>::z_filter_matrix(
    const std::optional<size_t> half_power, const Mesh<1>& mesh_1d) const {
  if (not half_power.has_value()) {
    return empty_matrix_;
  }
  const size_t extent = mesh_1d.extents(0);
  const auto it = cached_z_filters_.find(extent);
  if (it != cached_z_filters_.end()) {
    return it->second;
  }
  return cached_z_filters_
      .emplace(extent, Spectral::filtering::exponential_filter(
                           mesh_1d, 36.0, static_cast<unsigned>(*half_power)))
      .first->second;
}

template <typename TagList>
const Matrix& FilledCylinder<TagList>::mantle_angular_filter_matrix(
    const size_t num_angular_points) const {
  if (num_modes_to_kill_ == 0 and not radial_angular_half_power_.has_value()) {
    return empty_matrix_;
  }
  const auto it = cached_mantle_angular_filters_.find(num_angular_points);
  if (it != cached_mantle_angular_filters_.end()) {
    return it->second;
  }
  // The angular direction of a ZernikeB2 mesh is collocated on equiangular
  // points, so a Fourier exponential roll-off and top-mode cutoff can be
  // applied to it directly via a matching Fourier/Equiangular 1-D mesh.
  const Mesh<1> fourier_mesh{num_angular_points, Spectral::Basis::Fourier,
                             Spectral::Quadrature::Equiangular};
  Matrix combined{};
  if (radial_angular_half_power_.has_value()) {
    combined = Spectral::filtering::exponential_filter(
        fourier_mesh, 36.0, static_cast<unsigned>(*radial_angular_half_power_));
  }
  if (num_modes_to_kill_ > 0) {
    const Matrix& cutoff = Spectral::filtering::zero_highest_modes(
        fourier_mesh, num_modes_to_kill_);
    combined = radial_angular_half_power_.has_value()
                   ? Matrix{cutoff * combined}
                   : cutoff;
  }
  return cached_mantle_angular_filters_
      .emplace(num_angular_points, std::move(combined))
      .first->second;
}

template <typename TagList>
void FilledCylinder<TagList>::apply_in_volume(
    const gsl::not_null<Variables<TagList>*> vars, const Mesh<3>& mesh,
    const std::optional<
        InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
    /*inv_jac_grid_to_inertial*/,
    const std::optional<Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
    /*jac_grid_to_inertial*/) const {
  Spectral::filtering::zernike_b2_cylinder_filter(
      vars, mesh, 36.0,
      FilledCylinder_detail::to_unsigned(radial_angular_half_power_),
      FilledCylinder_detail::to_unsigned(z_half_power_), num_modes_to_kill_);
}

template <typename TagList>
void FilledCylinder<TagList>::apply_on_boundary(
    const gsl::not_null<Variables<TagList>*> vars, const Mesh<2>& mesh,
    const std::optional<
        InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
    /*inv_jac_grid_to_inertial*/,
    const std::optional<Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
    /*jac_grid_to_inertial*/) const {
  const bool zernike_0 = mesh.basis(0) == Spectral::Basis::ZernikeB2;
  const bool zernike_1 = mesh.basis(1) == Spectral::Basis::ZernikeB2;
  // A filled-cylinder face drops one volume dimension while preserving the
  // order of the remaining two, so the basis (and quadrature) pattern of the
  // 2-D face mesh uniquely identifies which physical directions it spans:
  //   (ZernikeB2, ZernikeB2)              -> (radial, angular)  [axial face]
  //   (ZernikeB2/Equiangular, Legendre)   -> (angular, z)       [mantle face]
  //   (ZernikeB2/GaussRadauUpper, Legendre) -> (radial, z)      [angular face]
  if (zernike_0 and zernike_1) {
    // Axial face: a full disk.
    Spectral::filtering::zernike_b2_disk_filter(
        vars, mesh, 36.0,
        FilledCylinder_detail::to_unsigned(radial_angular_half_power_),
        num_modes_to_kill_);
    return;
  }
  std::array<std::reference_wrapper<const Matrix>, 2> filter{
      std::cref(empty_matrix_), std::cref(empty_matrix_)};
  if (zernike_0 and not zernike_1) {
    // dim 0 is a lone ZernikeB2 direction, dim 1 is z (Legendre/Chebyshev).
    if (mesh.quadrature(0) == Spectral::Quadrature::Equiangular) {
      // Mantle face: dim 0 is the angular direction, filtered as Fourier.
      filter[0] = std::cref(mantle_angular_filter_matrix(mesh.extents(0)));
    }
    // Otherwise dim 0 is the radial direction (GaussRadauUpper), which is left
    // untouched: a lone 1-D radial Zernike slice has no filter engine.
    filter[1] =
        std::cref(z_filter_matrix(z_half_power_, mesh.slice_through(1)));
  } else {
    ERROR(
        "FilledCylinder filter called on a face mesh with an unexpected basis "
        "combination. Got basis ("
        << mesh.basis(0) << ", " << mesh.basis(1)
        << "); a filled-cylinder face must have either two ZernikeB2 "
           "directions or exactly one.");
  }
  *vars = apply_matrices(filter, *vars, mesh.extents());
}

template <typename TagList>
bool operator==(const FilledCylinder<TagList>& lhs,
                const FilledCylinder<TagList>& rhs) {
  return lhs.num_modes_to_kill_ == rhs.num_modes_to_kill_ and
         lhs.radial_angular_half_power_ == rhs.radial_angular_half_power_ and
         lhs.z_half_power_ == rhs.z_half_power_ and
         lhs.enable_ == rhs.enable_ and
         lhs.blocks_and_groups_to_filter_ ==
             rhs.blocks_and_groups_to_filter_ and
         lhs.blocks_to_filter_ == rhs.blocks_to_filter_ and
         lhs.volume_filter_on_substep_ == rhs.volume_filter_on_substep_ and
         lhs.boundary_filter_on_substep_ == rhs.boundary_filter_on_substep_ and
         lhs.volume_filter_every_n_steps_ ==
             rhs.volume_filter_every_n_steps_ and
         lhs.boundary_filter_every_n_steps_ ==
             rhs.boundary_filter_every_n_steps_;
}

template <typename TagList>
bool operator!=(const FilledCylinder<TagList>& lhs,
                const FilledCylinder<TagList>& rhs) {
  return not(lhs == rhs);
}

template <typename TagList>
bool FilledCylinder<TagList>::is_equal(const Filter<3, TagList>& other) const {
  const auto* const other_cylinder =
      dynamic_cast<const FilledCylinder<TagList>*>(&other);
  if (other_cylinder == nullptr) {
    return false;
  }
  return *this == *other_cylinder;
}

template <typename TagList>
PUP::able::PUP_ID FilledCylinder<TagList>::my_PUP_ID = 0;  // NOLINT
}  // namespace Filters
