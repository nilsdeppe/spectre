// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"

#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalEndcap.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalFlatEndcap.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "Options/ParseError.hpp"

namespace {
std::array<double, 3> rotate_to_z_axis(const std::array<double, 3> input) {
  return discrete_rotation(
      OrientationMap<3>{std::array<Direction<3>, 3>{Direction<3>::lower_zeta(),
                                                    Direction<3>::upper_eta(),
                                                    Direction<3>::upper_xi()}},
      input);
}
std::array<double, 3> rotate_from_z_to_x_axis(
    const std::array<double, 3> input) {
  return discrete_rotation(
      OrientationMap<3>{std::array<Direction<3>, 3>{Direction<3>::upper_zeta(),
                                                    Direction<3>::upper_eta(),
                                                    Direction<3>::lower_xi()}},
      input);
}
std::array<double, 3> flip_about_xy_plane(const std::array<double, 3> input) {
  return std::array<double, 3>{input[0], input[1], -input[2]};
}
}  // namespace

namespace domain::creators {
CylindricalBinaryCompactObject::CylindricalBinaryCompactObject(
    std::array<double, 3> center_A, std::array<double, 3> center_B,
    double radius_A, double radius_B, bool include_inner_sphere_A,
    bool include_inner_sphere_B, bool include_outer_sphere, double outer_radius,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_grid_points,
    std::optional<bco::TimeDependentMapOptions<true>> time_dependent_options,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : center_A_(rotate_to_z_axis(center_A)),
      center_B_(rotate_to_z_axis(center_B)),
      radius_A_(radius_A),
      radius_B_(radius_B),
      include_inner_sphere_A_(include_inner_sphere_A),
      include_inner_sphere_B_(include_inner_sphere_B),
      include_outer_sphere_(include_outer_sphere),
      outer_radius_(outer_radius),
      inner_boundary_condition_(std::move(inner_boundary_condition)),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      time_dependent_options_(std::move(time_dependent_options)) {
  if (center_A_[2] <= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of the input CenterA is expected to be positive");
  }
  if (center_B_[2] >= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of the input CenterB is expected to be negative");
  }
  if (radius_A_ <= 0.0 or radius_B_ <= 0.0) {
    PARSE_ERROR(context, "RadiusA and RadiusB are expected to be positive");
  }
  if (radius_A_ < radius_B_) {
    PARSE_ERROR(context, "RadiusA should not be smaller than RadiusB");
  }
  if (std::abs(center_A_[2]) > std::abs(center_B_[2])) {
    PARSE_ERROR(context,
                "We expect |x_A| <= |x_B|, for x the x-coordinate of either "
                "CenterA or CenterB.  We should roughly have "
                "RadiusA x_A + RadiusB x_B = 0 (i.e. for BBHs the "
                "center of mass should be about at the origin).");
  }
  // The value 3.0 * (center_A_[2] - center_B_[2]) is what is
  // chosen in SpEC as the inner radius of the innermost outer sphere.
  if (outer_radius_ < 3.0 * (center_A_[2] - center_B_[2])) {
    PARSE_ERROR(context,
                "OuterRadius is too small. Please increase it "
                "beyond "
                    << 3.0 * (center_A_[2] - center_B_[2]));
  }

  if ((outer_boundary_condition_ == nullptr) xor
      (inner_boundary_condition_ == nullptr)) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(inner_boundary_condition_) or
      is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with a binary domain");
  }

  // The choices made below for the quantities xi, z_cutting_plane_,
  // and xi_min_sphere_e are the ones made in SpEC, and in the
  // Appendix of https://arxiv.org/abs/1206.3015.  Other choices could
  // be made that would still result in a reasonable Domain. In
  // particular, during a SpEC BBH evolution the excision boundaries
  // can sometimes get too close to z_cutting_plane_, and the
  // simulation must be halted and regridded with a different choice
  // of z_cutting_plane_, so it may be possible to choose a different
  // initial value of z_cutting_plane_ that reduces the number of such
  // regrids or eliminates them.

  // xi is the quantity in Eq. (A10) of
  // https://arxiv.org/abs/1206.3015 that represents how close the
  // cutting plane is to either center.  Unfortunately, there is a
  // discrepancy between what xi means in the paper and what it is in
  // the code.  I (Mark) think that this is a typo in the paper,
  // because otherwise the domain doesn't make sense.  To fix this,
  // either Eq. (A9) in the paper should have xi -> 1-xi, or Eq. (A10)
  // should have x_A and x_B swapped.
  // Here we will use the same definition of xi in Eq. (A10), but we
  // will swap xi -> 1-xi in Eq. (A9).
  // Therefore, xi = 0 means that the cutting plane passes through the center of
  // object B, and xi = 1 means that the cutting plane passes through
  // the center of object A.  Note that for |x_A| <= |x_B| (as assumed
  // above), xi is always <= 1/2.
  constexpr double xi_min = 0.25;
  // Same as Eq. (A10)
  const double xi =
      std::max(xi_min, std::abs(center_A_[2]) /
                           (std::abs(center_A_[2]) + std::abs(center_B_[2])));

  // Compute cutting plane
  // This is Eq. (A9) with xi -> 1-xi.
  z_cutting_plane_ = cut_spheres_offset_factor_ *
                     ((1.0 - xi) * center_B_[2] + xi * center_A_[2]);

  // outer_radius_A is the outer radius of the inner sphere A, if it exists.
  // If the inner sphere A does not exist, then outer_radius_A is the same
  // as radius_A_.
  // If the inner sphere does exist, the algorithm for computing
  // outer_radius_A is the same as in SpEC when there is one inner shell.
  outer_radius_A_ =
      include_inner_sphere_A_
          ? radius_A_ +
                0.5 * (std::abs(z_cutting_plane_ - center_A_[2]) - radius_A_)
          : radius_A_;

  // outer_radius_B is the outer radius of the inner sphere B, if it exists.
  // If the inner sphere B does not exist, then outer_radius_B is the same
  // as radius_B_.
  // If the inner sphere does exist, the algorithm for computing
  // outer_radius_B is the same as in SpEC when there is one inner shell.
  outer_radius_B_ =
      include_inner_sphere_B_
          ? radius_B_ +
                0.5 * (std::abs(z_cutting_plane_ - center_B_[2]) - radius_B_)
          : radius_B_;

  // Each former filled cylinder (5 wedge blocks) and each former side cylinder
  // (4 wedge blocks) is now a single block. The main region has 10 blocks; each
  // inner sphere adds 3 (E filled, M filled, E side) and the outer sphere adds
  // 4 (CA filled, CB filled, CA side, CB side).
  // Each inner sphere and the wavezone are single S2 spherical-harmonic shells.
  number_of_blocks_ = 10;
  if (include_inner_sphere_A) {
    number_of_blocks_ += 1;
  }
  if (include_inner_sphere_B) {
    number_of_blocks_ += 1;
  }
  if (include_outer_sphere) {
    number_of_blocks_ += 1;
  }

  // Add SphereE blocks if necessary.  Note that
  // https://arxiv.org/abs/1206.3015 has a mistake just above
  // Eq. (A.11) and the same mistake above Eq. (A.20), where it lists
  // the wrong mass ratio (for BBHs). The correct statement is that if
  // xi <= 1/3, this means that the mass ratio (for BBH) is large (>=2)
  // and we should add SphereE blocks.
  constexpr double xi_min_sphere_e = 1.0 / 3.0;
  if (xi <= xi_min_sphere_e) {
    // The following ERROR will be removed in an upcoming PR that
    // will support higher mass ratios.
    ERROR(
        "We currently only support domains where objects A and B are "
        "approximately the same size, and approximately the same distance from "
        "the origin.  More technically, we support xi > "
        << xi_min_sphere_e << ", but the value of xi is " << xi
        << ". Support for more general domains will be added in the near "
           "future");
  }

  // Create grid anchors in x direction from unrotated input centers
  grid_anchors_ = bco::create_grid_anchors(center_A, center_B);

  // Create block names and groups (one block per former cylinder group). The
  // order here must match the order in which the maps are constructed in
  // create_domain().
  auto add_block_name = [this](const std::string& name,
                               const std::string& group_name) {
    block_names_.push_back(name);
    block_groups_[group_name].insert(name);
  };

  // Main region (10 blocks).
  add_block_name("CAFilledCylinder", "Outer");
  add_block_name("CACylinder", "Outer");
  add_block_name("EAFilledCylinder", "InnerA");
  add_block_name("EACylinder", "InnerA");
  add_block_name("EBFilledCylinder", "InnerB");
  add_block_name("EBCylinder", "InnerB");
  add_block_name("MAFilledCylinder", "InnerA");
  add_block_name("MBFilledCylinder", "InnerB");
  add_block_name("CBFilledCylinder", "Outer");
  add_block_name("CBCylinder", "Outer");

  if (include_inner_sphere_A) {
    add_block_name("SphereA", "InnerSphereA");
  }
  if (include_inner_sphere_B) {
    add_block_name("SphereB", "InnerSphereB");
  }
  first_outer_shell_block = block_names_.size();
  if (include_outer_sphere) {
    add_block_name("SphereC", "OuterSphere");
  }

  // Expand initial refinement over all blocks
  const ExpandOverBlocks<std::array<size_t, 3>> expand_over_blocks{
      block_names_, block_groups_};
  try {
    initial_refinement_ = std::visit(expand_over_blocks, initial_refinement);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialRefinement': " << error.what());
  }
  try {
    initial_grid_points_ = std::visit(expand_over_blocks, initial_grid_points);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialGridPoints': " << error.what());
  }

  // The ZernikeB2 filled-cylinder blocks and the Fourier side blocks cannot be
  // h-refined in the radial (xi) or angular (eta) directions, and the angular
  // direction requires an odd number of grid points for numerical stability.
  // Only the perpendicular/axial (zeta) direction may be refined. For the
  // filled cylinders the ZernikeB2 radial extent is determined by the angular
  // extent (see domain::creators::AngularCylinder): with the maximum angular
  // mode M = N_phi / 2, the radial extent is M/2 + 1 + (M % 2).
  for (size_t block = 0; block < number_of_blocks_; ++block) {
    if (block_names_[block].rfind("Sphere", 0) == 0) {
      // S2 spherical-harmonic shell, extents [radial, colatitude, longitude].
      // The angular directions cannot be h-refined, and Spherepack requires
      // longitude = 2 * colatitude - 1. The radial direction may be refined.
      gsl::at(initial_refinement_[block], 1) = 0;
      gsl::at(initial_refinement_[block], 2) = 0;
      gsl::at(initial_grid_points_[block], 2) =
          2 * gsl::at(initial_grid_points_[block], 1) - 1;
      continue;
    }
    gsl::at(initial_refinement_[block], 0) = 0;
    gsl::at(initial_refinement_[block], 1) = 0;
    if (gsl::at(initial_grid_points_[block], 1) % 2 == 0) {
      gsl::at(initial_grid_points_[block], 1) += 1;
    }
    if (block_names_[block].find("FilledCylinder") != std::string::npos) {
      const size_t theta_modes = gsl::at(initial_grid_points_[block], 1) / 2;
      gsl::at(initial_grid_points_[block], 0) =
          theta_modes / 2 + 1 + theta_modes % 2;
    }
  }

  // Build time-dependent maps
  // The size map, which is applied from the grid to distorted frame, currently
  // needs to start and stop at certain radii around each excision. If the inner
  // spheres aren't included, the outer radii would have to be in the middle of
  // a block. With the inner spheres, the outer radii can be at block
  // boundaries. The outer sphere must be specified because the time-dependent
  // maps use piecewise functions for `Expansion` and `Translation`. This means
  // an inner common radius must be specified for the piecewise bounds.
  if (time_dependent_options_.has_value() and
      not(include_inner_sphere_A and include_inner_sphere_B and
          include_outer_sphere)) {
    PARSE_ERROR(context,
                "To use the CylindricalBBH domain with time-dependent maps, "
                "you must include the inner spheres for both objects and "
                "the outer sphere. "
                "Currently, one or both objects is missing the inner spheres or"
                " the outer sphere is missing.");
  }

  if (time_dependent_options_.has_value()) {
    const double inner_common_radius = 3.0 * (center_A_[2] - center_B_[2]);
    const auto center_A_aligned = rotate_from_z_to_x_axis(center_A_);
    const auto center_B_aligned = rotate_from_z_to_x_axis(center_B_);
    time_dependent_options_->build_maps(
        std::array{center_A_aligned, center_B_aligned}, std::nullopt,
        std::nullopt,
        std::array{z_cutting_plane_,
                   0.5 * (center_A_aligned[1] + center_B_aligned[1]),
                   0.5 * (center_A_aligned[2] + center_B_aligned[2])},
        std::array{radius_A_, outer_radius_A_},
        std::array{radius_B_, outer_radius_B_}, false, false,
        inner_common_radius, outer_radius_);
  }
}

Domain<3> CylindricalBinaryCompactObject::create_domain() const {
  // Each former "filled cylinder" (a central square block plus four
  // surrounding wedge blocks) is now a single ZernikeB2 filled-cylinder block
  // with topology {B2Radial, B2Angular, I1} == [r, phi, perp], and each former
  // "cylinder"/side (four wedge blocks) is now a single Fourier annular block
  // with topology {I1, S1, I1} == [r, phi, z]. Both single-block
  // representations share the same azimuthal convention (phi via
  // PolarToCartesian), and block neighbors are specified explicitly below
  // because the automatic corner-based neighbor detection cannot handle the
  // degenerate r=0 axis of the disk or the periodic angular seam of the
  // annulus.
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      coordinate_maps{};
  // Group tag for each block, used to wire up the explicit neighbor graph.
  std::vector<std::string> block_tags{};

  const OrientationMap<3> rotate_to_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
      Direction<3>::lower_xi()}};

  const OrientationMap<3> rotate_to_minus_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
      Direction<3>::upper_xi()}};

  // A 180-degree rotation about the cylinder axis (negates the cross-section).
  // The minus-x-axis blocks have the opposite azimuthal handedness from the
  // plus-x-axis blocks (their shared seams relate by phi -> pi - phi, which a
  // block OrientationMap cannot represent on a polar disk). Pre-rotating the
  // minus-x-axis blocks' cross sections by pi turns every cross-handedness seam
  // into phi -> -phi, which is an expressible angular (eta) flip and is
  // node-conforming for the odd equiangular grids. The endcap/side maps are
  // axisymmetric, so this pre-rotation does not change the physical domain.
  const OrientationMap<3> half_turn_about_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_xi(), Direction<3>::lower_eta(),
      Direction<3>::upper_zeta()}};

  const std::array<double, 3> center_cutting_plane = {0.0, 0.0,
                                                      z_cutting_plane_};

  // The labels EA, EB, EE, etc are from Figure 20 of
  // https://arxiv.org/abs/1206.3015
  //
  // center_EA and radius_EA are the center and outer-radius of the
  // cylindered-sphere EA in Figure 20.
  //
  // center_EB and radius_EB are the center and outer-radius of the
  // cylindered-sphere EB in Figure 20.
  //
  // radius_MB is eq. A16 or A23 in the paper (depending on whether
  // the EE spheres exist), and is the radius of the circle where the EB
  // sphere intersects the cutting plane.
  const std::array<double, 3> center_EA = {
      0.0, 0.0, cut_spheres_offset_factor_ * center_A_[2]};
  const std::array<double, 3> center_EB = {
      0.0, 0.0, center_B_[2] * cut_spheres_offset_factor_};
  const double radius_MB =
      std::abs(cut_spheres_offset_factor_ * center_B_[2] - z_cutting_plane_);
  const double radius_EA =
      sqrt(square(center_EA[2] - z_cutting_plane_) + square(radius_MB));
  const double radius_EB =
      sqrt(2.0) * std::abs(center_EB[2] - z_cutting_plane_);

  // Lambda that builds a single ZernikeB2 filled-cylinder block from a
  // UniformCylindricalEndcap or UniformCylindricalFlatEndcap map and a
  // DiscreteRotation map. The logical coordinates are [xi, eta, zeta] =
  // [r, phi, perp]: xi in [-1, 1] maps to r in [0, 1], eta is the azimuthal
  // angle (passed through Identity to PolarToCartesian), and zeta in [-1, 1]
  // is the perpendicular direction. PolarToCartesian produces the unit right
  // cylinder (x^2 + y^2 <= 1, -1 <= z <= 1) that the endcap maps expect.
  const auto add_filled_cylinder_to_list_of_maps =
      [&coordinate_maps, &block_tags, &rotate_to_minus_x_axis,
       &half_turn_about_axis](
          const auto& endcap_map,
          const CoordinateMaps::DiscreteRotation<3>& rotation_map,
          const std::string& tag) {
        using Affine = CoordinateMaps::Affine;
        using Identity1D = CoordinateMaps::Identity<1>;
        using Interval = CoordinateMaps::Interval;
        const bool minus_side =
            rotation_map ==
            CoordinateMaps::DiscreteRotation<3>{rotate_to_minus_x_axis};
        coordinate_maps.push_back(
            domain::make_coordinate_map_base<Frame::BlockLogical,
                                             Frame::Inertial>(
                CoordinateMaps::ProductOf3Maps<Affine, Identity1D, Interval>{
                    Affine{-1.0, 1.0, 0.0, 1.0}, Identity1D{},
                    Interval{-1.0, 1.0, -1.0, 1.0,
                             CoordinateMaps::Distribution::Linear}},
                CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                               Identity1D>{
                    CoordinateMaps::PolarToCartesian{}, Identity1D{}},
                CoordinateMaps::DiscreteRotation<3>{
                    minus_side ? half_turn_about_axis
                               : OrientationMap<3>::create_aligned()},
                endcap_map, rotation_map));
        block_tags.push_back(tag);
      };

  // Lambda that builds a single Fourier annular block (a hollow cylindrical
  // shell) from a UniformCylindricalSide map and a DiscreteRotation map. The
  // logical coordinates are [xi, eta, zeta] = [r, phi, z]: xi in [-1, 1] maps
  // to r in [1, 2] (the shell radii expected by UniformCylindricalSide), eta
  // is the azimuthal angle, and zeta in [-1, 1] is the axial direction.
  const auto add_side_to_list_of_maps =
      [&coordinate_maps, &block_tags, &rotate_to_minus_x_axis,
       &half_turn_about_axis](
          const CoordinateMaps::UniformCylindricalSide& side_map,
          const CoordinateMaps::DiscreteRotation<3>& rotation_map,
          const std::string& tag) {
        using Affine = CoordinateMaps::Affine;
        using Identity1D = CoordinateMaps::Identity<1>;
        using Interval = CoordinateMaps::Interval;
        const bool minus_side =
            rotation_map ==
            CoordinateMaps::DiscreteRotation<3>{rotate_to_minus_x_axis};
        coordinate_maps.push_back(
            domain::make_coordinate_map_base<Frame::BlockLogical,
                                             Frame::Inertial>(
                CoordinateMaps::ProductOf3Maps<Affine, Identity1D, Interval>{
                    Affine{-1.0, 1.0, 1.0, 2.0}, Identity1D{},
                    Interval{-1.0, 1.0, -1.0, 1.0,
                             CoordinateMaps::Distribution::Linear}},
                CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                               Identity1D>{
                    CoordinateMaps::PolarToCartesian{}, Identity1D{}},
                CoordinateMaps::DiscreteRotation<3>{
                    minus_side ? half_turn_about_axis
                               : OrientationMap<3>::create_aligned()},
                side_map, rotation_map));
        block_tags.push_back(tag);
      };

  // Lambda that builds a single S2 spherical-harmonic shell block (topology
  // {I1, S2Colatitude, S2Longitude}) from inner_radius to outer_radius,
  // recentered on 'center'. The logical coordinates are [xi, eta, zeta] =
  // [radial, colatitude, longitude]. SphericalToCartesianPfaffian produces an
  // origin-centered shell, and the trailing ProductOf3Maps translates it to
  // 'center' (identity translation for origin-centered shells). These shells
  // connect to the surrounding ZernikeB2/Fourier blocks through non-conforming
  // interfaces.
  const auto add_sphere_shell_to_list_of_maps = [&coordinate_maps, &block_tags](
                                                    const double inner_radius,
                                                    const double outer_radius,
                                                    const std::array<double, 3>&
                                                        center,
                                                    const std::string& tag) {
    using Affine = CoordinateMaps::Affine;
    coordinate_maps.push_back(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf2Maps<Affine, CoordinateMaps::Identity<2>>{
                Affine{-1.0, 1.0, inner_radius, outer_radius},
                CoordinateMaps::Identity<2>{}},
            CoordinateMaps::SphericalToCartesianPfaffian{},
            CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>{
                Affine{-1.0, 1.0, -1.0 + center[0], 1.0 + center[0]},
                Affine{-1.0, 1.0, -1.0 + center[1], 1.0 + center[1]},
                Affine{-1.0, 1.0, -1.0 + center[2], 1.0 + center[2]}}));
    block_tags.push_back(tag);
  };

  // Inner radius of the outer C shell, if it exists.
  // If it doesn't exist, then it is the same as the outer_radius_.
  const double inner_radius_C = include_outer_sphere_
                                    ? 3.0 * (center_A_[2] - center_B_[2])
                                    : outer_radius_;

  // z_cut_CA_lower is the lower z_plane position for the CA endcap,
  // defined by https://arxiv.org/abs/1206.3015 in the bulleted list
  // after Eq. (A.19) EXCEPT that here we use a factor of 1.6 instead of 1.5
  // to put the plane farther from center_A.
  const double z_cut_CA_lower =
      z_cutting_plane_ + 1.6 * (center_EA[2] - z_cutting_plane_);
  // z_cut_CA_upper is the upper z_plane position for the CA endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.
  const double z_cut_CA_upper =
      std::max(0.5 * (z_cut_CA_lower + inner_radius_C), 0.7 * inner_radius_C);
  // z_cut_EA_upper is the upper z_plane position for the EA endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.
  const double z_cut_EA_upper = center_A_[2] + 0.7 * outer_radius_A_;
  // z_cut_EA_lower is the lower z_plane position for the EA endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.
  const double z_cut_EA_lower = center_A_[2] - 0.7 * outer_radius_A_;

  // CA Filled Cylinder
  add_filled_cylinder_to_list_of_maps(
      CoordinateMaps::UniformCylindricalEndcap(center_EA, make_array<3>(0.0),
                                               radius_EA, inner_radius_C,
                                               z_cut_CA_lower, z_cut_CA_upper),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis), "CA_filled");

  // CA Cylinder
  add_side_to_list_of_maps(
      CoordinateMaps::UniformCylindricalSide(
          // codecov complains about the next line being untested.
          // No idea why, since this entire function is called.
          // LCOV_EXCL_START
          center_EA, make_array<3>(0.0), radius_EA, inner_radius_C,
          // LCOV_EXCL_STOP
          z_cut_CA_lower, z_cutting_plane_, z_cut_CA_upper, z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis), "CA_side");

  // EA Filled Cylinder
  add_filled_cylinder_to_list_of_maps(
      CoordinateMaps::UniformCylindricalEndcap(center_A_, center_EA,
                                               outer_radius_A_, radius_EA,
                                               z_cut_EA_upper, z_cut_CA_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis), "EA_filled");

  // EA Cylinder
  add_side_to_list_of_maps(
      // For some reason codecov complains about the next line.
      CoordinateMaps::UniformCylindricalSide(  // LCOV_EXCL_LINE
          center_A_, center_EA, outer_radius_A_, radius_EA, z_cut_EA_upper,
          z_cut_EA_lower, z_cut_CA_lower, z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis), "EA_side");

  // z_cut_CB_lower is the lower z_plane position for the CB endcap,
  // defined by https://arxiv.org/abs/1206.3015 in the bulleted list
  // after Eq. (A.19) EXCEPT that here we use a factor of 1.6 instead of 1.5
  // to put the plane farther from center_B.
  // Note here that 'lower' means 'farther from z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_CB_lower =
      z_cutting_plane_ + 1.6 * (center_EB[2] - z_cutting_plane_);
  // z_cut_CB_upper is the upper z_plane position for the CB endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme. Note here that 'upper' means 'closer to z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_CB_upper =
      std::min(0.5 * (z_cut_CB_lower - inner_radius_C), -0.7 * inner_radius_C);
  // z_cut_EB_upper is the upper z_plane position for the EB endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.  Note here that 'upper' means 'closer to z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_EB_upper = center_B_[2] - 0.7 * outer_radius_B_;
  // z_cut_EB_lower is the lower z_plane position for the EB endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme. Note here that 'lower' means 'farther from z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_EB_lower = center_B_[2] + 0.7 * outer_radius_B_;

  // EB Filled Cylinder
  add_filled_cylinder_to_list_of_maps(
      CoordinateMaps::UniformCylindricalEndcap(
          flip_about_xy_plane(center_B_), flip_about_xy_plane(center_EB),
          outer_radius_B_, radius_EB, -z_cut_EB_upper, -z_cut_CB_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis), "EB_filled");

  // EB Cylinder
  add_side_to_list_of_maps(
      CoordinateMaps::UniformCylindricalSide(
          flip_about_xy_plane(center_B_), flip_about_xy_plane(center_EB),
          outer_radius_B_, radius_EB, -z_cut_EB_upper, -z_cut_EB_lower,
          -z_cut_CB_lower, -z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis), "EB_side");

  // MA Filled Cylinder
  add_filled_cylinder_to_list_of_maps(
      CoordinateMaps::UniformCylindricalFlatEndcap(
          flip_about_xy_plane(center_A_),
          flip_about_xy_plane(center_cutting_plane), outer_radius_A_, radius_MB,
          -z_cut_EA_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis), "MA_filled");
  // MB Filled Cylinder
  add_filled_cylinder_to_list_of_maps(
      // For some reason codecov complains about the next line.
      CoordinateMaps::UniformCylindricalFlatEndcap(  // LCOV_EXCL_LINE
          center_B_, center_cutting_plane, outer_radius_B_, radius_MB,
          z_cut_EB_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis), "MB_filled");

  // CB Filled Cylinder
  add_filled_cylinder_to_list_of_maps(
      CoordinateMaps::UniformCylindricalEndcap(
          flip_about_xy_plane(center_EB), make_array<3>(0.0), radius_EB,
          inner_radius_C, -z_cut_CB_lower, -z_cut_CB_upper),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis), "CB_filled");

  // CB Cylinder
  add_side_to_list_of_maps(
      CoordinateMaps::UniformCylindricalSide(
          flip_about_xy_plane(center_EB), make_array<3>(0.0), radius_EB,
          inner_radius_C, -z_cut_CB_lower, -z_cutting_plane_, -z_cut_CB_upper,
          -z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis), "CB_side");

  if (include_inner_sphere_A_) {
    // S2 spherical-harmonic shell wrapping excision A, from radius_A out to
    // outer_radius_A, centered on object A (in the inertial frame, which is
    // related to the internal z-axis frame by rotate_from_z_to_x_axis).
    add_sphere_shell_to_list_of_maps(radius_A_, outer_radius_A_,
                                     rotate_from_z_to_x_axis(center_A_),
                                     "SphereA");
  }
  if (include_inner_sphere_B_) {
    add_sphere_shell_to_list_of_maps(radius_B_, outer_radius_B_,
                                     rotate_from_z_to_x_axis(center_B_),
                                     "SphereB");
  }
  if (include_outer_sphere_) {
    // The wavezone is a single S2 spherical-harmonic shell centered at the
    // origin, spanning the inner common radius out to the outer radius.
    add_sphere_shell_to_list_of_maps(inner_radius_C, outer_radius_,
                                     make_array<3>(0.0), "SphereC");
  }

  // Map each block group tag to its block index.
  std::unordered_map<std::string, size_t> tag_to_index{};
  for (size_t i = 0; i < block_tags.size(); ++i) {
    tag_to_index.emplace(block_tags[i], i);
  }

  // The distinct relative orientations occurring in the cylindered-sphere
  // construction (orientation of the neighboring block relative to this one).
  // Notation [d_xi, d_eta, d_zeta] gives the direction each logical axis of the
  // neighbor maps to.
  using Dir = Direction<3>;
  const OrientationMap<3> orient_id = OrientationMap<3>::create_aligned();
  // Filled-cylinder mantle (up_xi) -> abutting side, for the curved endcaps.
  const OrientationMap<3> orient_fs{
      std::array<Dir, 3>{Dir::lower_zeta(), Dir::upper_eta(), Dir::upper_xi()}};
  // Side z-end (up_zeta) -> abutting filled cylinder (inverse of orient_fs).
  const OrientationMap<3> orient_sf{
      std::array<Dir, 3>{Dir::upper_zeta(), Dir::upper_eta(), Dir::lower_xi()}};
  // Flat-endcap (M) filled cylinder <-> side.
  const OrientationMap<3> orient_mside{
      std::array<Dir, 3>{Dir::upper_zeta(), Dir::lower_eta(), Dir::upper_xi()}};
  // Cutting-plane / outer-shell seam (angular reflection + perp flip).
  const OrientationMap<3> orient_zseam{
      std::array<Dir, 3>{Dir::upper_xi(), Dir::lower_eta(), Dir::lower_zeta()}};

  // Directed neighbor graph keyed by block tag:
  // {self tag, self direction, neighbor tag, orientation of neighbor}.
  // Edges whose neighbor group is absent (sphere not included) are skipped;
  // those faces become external boundaries or excision-sphere abutting faces.
  struct EdgeSpec {
    std::string self_tag;
    Direction<3> self_direction;
    std::string neighbor_tag;
    const OrientationMap<3>* orientation;
  };
  const std::vector<EdgeSpec> edges{
      {"CA_filled", Dir::lower_zeta(), "EA_filled", &orient_id},
      {"CA_filled", Dir::upper_xi(), "CA_side", &orient_fs},
      {"CA_side", Dir::lower_xi(), "EA_side", &orient_id},
      {"CA_side", Dir::lower_zeta(), "CB_side", &orient_zseam},
      {"CA_side", Dir::upper_zeta(), "CA_filled", &orient_sf},
      {"EA_filled", Dir::upper_xi(), "EA_side", &orient_fs},
      {"EA_filled", Dir::upper_zeta(), "CA_filled", &orient_id},
      {"EA_side", Dir::lower_zeta(), "MA_filled", &orient_mside},
      {"EA_side", Dir::upper_xi(), "CA_side", &orient_id},
      {"EA_side", Dir::upper_zeta(), "EA_filled", &orient_sf},
      {"EB_filled", Dir::upper_xi(), "EB_side", &orient_fs},
      {"EB_filled", Dir::upper_zeta(), "CB_filled", &orient_id},
      {"EB_side", Dir::lower_zeta(), "MB_filled", &orient_mside},
      {"EB_side", Dir::upper_xi(), "CB_side", &orient_id},
      {"EB_side", Dir::upper_zeta(), "EB_filled", &orient_sf},
      {"MA_filled", Dir::upper_xi(), "EA_side", &orient_mside},
      {"MA_filled", Dir::upper_zeta(), "MB_filled", &orient_zseam},
      {"MB_filled", Dir::upper_xi(), "EB_side", &orient_mside},
      {"MB_filled", Dir::upper_zeta(), "MA_filled", &orient_zseam},
      {"CB_filled", Dir::lower_zeta(), "EB_filled", &orient_id},
      {"CB_filled", Dir::upper_xi(), "CB_side", &orient_fs},
      {"CB_side", Dir::lower_xi(), "EB_side", &orient_id},
      {"CB_side", Dir::lower_zeta(), "CA_side", &orient_zseam},
      {"CB_side", Dir::upper_zeta(), "CB_filled", &orient_sf},
  };

  std::vector<DirectionMap<3, BlockNeighbors<3>>> neighbors_of_all_blocks(
      coordinate_maps.size());
  for (const auto& edge : edges) {
    const auto self_it = tag_to_index.find(edge.self_tag);
    const auto neighbor_it = tag_to_index.find(edge.neighbor_tag);
    if (self_it == tag_to_index.end() or neighbor_it == tag_to_index.end()) {
      continue;
    }
    neighbors_of_all_blocks[self_it->second].emplace(
        edge.self_direction,
        BlockNeighbors<3>(neighbor_it->second, *edge.orientation));
  }

  // Non-conforming interface between the wavezone S2 shell (SphereC) and the
  // outer cylindered C blocks. SphereC's inner radial face abuts all four
  // C-region outer faces (the filled cylinders via their upper_zeta face, the
  // sides via their upper_xi face). The angular layouts differ (spherical
  // vs. polar), so this is a non-conforming (mortar) interface; only the radial
  // direction is mapped, the angular directions use Direction::self().
  if (include_outer_sphere_) {
    const OrientationMap<3> sphere_to_filled{
        std::array<Dir, 3>{Dir::upper_zeta(), Dir::self(), Dir::self()}};
    const OrientationMap<3> sphere_to_side{
        std::array<Dir, 3>{Dir::upper_xi(), Dir::self(), Dir::self()}};
    const size_t c = tag_to_index.at("SphereC");
    const size_t ca_f = tag_to_index.at("CA_filled");
    const size_t ca_s = tag_to_index.at("CA_side");
    const size_t cb_f = tag_to_index.at("CB_filled");
    const size_t cb_s = tag_to_index.at("CB_side");
    neighbors_of_all_blocks[c].emplace(
        Dir::lower_xi(), BlockNeighbors<3>{{ca_f, ca_s, cb_f, cb_s},
                                           {{ca_f, sphere_to_filled},
                                            {ca_s, sphere_to_side},
                                            {cb_f, sphere_to_filled},
                                            {cb_s, sphere_to_side}},
                                           false});
    neighbors_of_all_blocks[ca_f].emplace(
        Dir::upper_zeta(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_filled.inverse_map()}}, false});
    neighbors_of_all_blocks[cb_f].emplace(
        Dir::upper_zeta(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_filled.inverse_map()}}, false});
    neighbors_of_all_blocks[ca_s].emplace(
        Dir::upper_xi(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_side.inverse_map()}}, false});
    neighbors_of_all_blocks[cb_s].emplace(
        Dir::upper_xi(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_side.inverse_map()}}, false});
  }

  // Non-conforming interfaces between each inner S2 shell (SphereA/SphereB) and
  // the surrounding cylindered E/M blocks. The shell's outer radial face abuts
  // the inner faces of the E filled cylinder (lower_zeta), the M filled
  // cylinder (lower_zeta), and the E side (lower_xi).
  const auto wire_inner_sphere = [&neighbors_of_all_blocks, &tag_to_index](
                                     const std::string& sphere_tag,
                                     const std::string& e_filled,
                                     const std::string& m_filled,
                                     const std::string& e_side) {
    const OrientationMap<3> to_filled{std::array<Direction<3>, 3>{
        Direction<3>::lower_zeta(), Direction<3>::self(),
        Direction<3>::self()}};
    const OrientationMap<3> to_side{std::array<Direction<3>, 3>{
        Direction<3>::lower_xi(), Direction<3>::self(), Direction<3>::self()}};
    const size_t s = tag_to_index.at(sphere_tag);
    const size_t ef = tag_to_index.at(e_filled);
    const size_t mf = tag_to_index.at(m_filled);
    const size_t es = tag_to_index.at(e_side);
    neighbors_of_all_blocks[s].emplace(
        Direction<3>::upper_xi(),
        BlockNeighbors<3>{{ef, mf, es},
                          {{ef, to_filled}, {mf, to_filled}, {es, to_side}},
                          false});
    neighbors_of_all_blocks[ef].emplace(
        Direction<3>::lower_zeta(),
        BlockNeighbors<3>{{s}, {{s, to_filled.inverse_map()}}, false});
    neighbors_of_all_blocks[mf].emplace(
        Direction<3>::lower_zeta(),
        BlockNeighbors<3>{{s}, {{s, to_filled.inverse_map()}}, false});
    neighbors_of_all_blocks[es].emplace(
        Direction<3>::lower_xi(),
        BlockNeighbors<3>{{s}, {{s, to_side.inverse_map()}}, false});
  };
  if (include_inner_sphere_A_) {
    wire_inner_sphere("SphereA", "EA_filled", "MA_filled", "EA_side");
  }
  if (include_inner_sphere_B_) {
    wire_inner_sphere("SphereB", "EB_filled", "MB_filled", "EB_side");
  }

  // Excision spheres. The faces abutting each excision are the inner
  // (lower_zeta for filled, lower_xi for side) faces of the innermost layer:
  // the inner-sphere blocks if present, otherwise the E and M blocks.
  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};
  std::unordered_map<size_t, Direction<3>> abutting_directions_A;
  if (include_inner_sphere_A_) {
    // The inner S2 shell's inner radial face is the excision boundary.
    abutting_directions_A.emplace(tag_to_index.at("SphereA"),
                                  Direction<3>::lower_xi());
  } else {
    abutting_directions_A.emplace(tag_to_index.at("EA_filled"),
                                  Direction<3>::lower_zeta());
    abutting_directions_A.emplace(tag_to_index.at("MA_filled"),
                                  Direction<3>::lower_zeta());
    abutting_directions_A.emplace(tag_to_index.at("EA_side"),
                                  Direction<3>::lower_xi());
  }
  excision_spheres.emplace(
      "ExcisionSphereA",
      ExcisionSphere<3>{
          radius_A_,
          tnsr::I<double, 3, Frame::Grid>(rotate_from_z_to_x_axis(center_A_)),
          abutting_directions_A});

  std::unordered_map<size_t, Direction<3>> abutting_directions_B;
  if (include_inner_sphere_B_) {
    abutting_directions_B.emplace(tag_to_index.at("SphereB"),
                                  Direction<3>::lower_xi());
  } else {
    abutting_directions_B.emplace(tag_to_index.at("EB_filled"),
                                  Direction<3>::lower_zeta());
    abutting_directions_B.emplace(tag_to_index.at("MB_filled"),
                                  Direction<3>::lower_zeta());
    abutting_directions_B.emplace(tag_to_index.at("EB_side"),
                                  Direction<3>::lower_xi());
  }
  excision_spheres.emplace(
      "ExcisionSphereB",
      ExcisionSphere<3>{
          radius_B_,
          tnsr::I<double, 3, Frame::Grid>(rotate_from_z_to_x_axis(center_B_)),
          abutting_directions_B});

  // Assemble the blocks. Filled cylinders use the ZernikeB2 full_cylinder
  // topology, sides use the Fourier cylindrical_shell topology, and the
  // spherical-harmonic shells (tags starting with "Sphere") use the S2
  // spherical_shell topology.
  std::vector<Block<3>> blocks{};
  blocks.reserve(coordinate_maps.size());
  for (size_t i = 0; i < coordinate_maps.size(); ++i) {
    const std::array<domain::Topology, 3>& topology =
        block_tags[i].rfind("Sphere", 0) == 0
            ? domain::topologies::spherical_shell
            : (block_tags[i].find("_filled") != std::string::npos
                   ? domain::topologies::full_cylinder
                   : domain::topologies::cylindrical_shell);
    blocks.emplace_back(std::move(coordinate_maps[i]), i,
                        std::move(neighbors_of_all_blocks[i]), block_names_[i],
                        topology);
  }

  Domain<3> domain{std::move(blocks), std::move(excision_spheres),
                   block_groups_};

  if (time_dependent_options_.has_value()) {
    ASSERT(include_inner_sphere_A_ and include_inner_sphere_B_,
           "When using time dependent maps for the CylindricalBBH domain, you "
           "must include both inner spheres.");
    // Default initialize everything to nullptr so that we only need to set the
    // appropriate block maps for the specific frames
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        grid_to_inertial_block_maps{number_of_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
        grid_to_distorted_block_maps{number_of_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
        distorted_to_inertial_block_maps{number_of_blocks_};

    // The 0th block always exists and will only need an rigid expansion +
    // rotation + translation map from the grid to inertial frame. No maps to
    // the distorted frame
    grid_to_inertial_block_maps[0] =
        time_dependent_options_
            ->grid_to_inertial_map<domain::ObjectLabel::None>(false, true);

    // The first block in the outer shell needs the transition expansion +
    // rotation + translation map from the grid to inertial frame. No maps to
    // the distorted frame
    grid_to_inertial_block_maps[first_outer_shell_block] =
        time_dependent_options_
            ->grid_to_inertial_map<domain::ObjectLabel::None>(false, false);

    // Inside the excision sphere we add the grid to inertial map from the
    // outer shell. This allows the center of the excisions/horizons to be
    // mapped properly to the inertial frame.
    domain.inject_time_dependent_map_for_excision_sphere(
        "ExcisionSphereA",
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::A>(
            true, true, true));
    domain.inject_time_dependent_map_for_excision_sphere(
        "ExcisionSphereB",
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::B>(
            true, true, true));

    // Because we require that both objects have inner shells, the distorted
    // (shape/size) maps live in the single inner S2 shell of each object
    // (SphereA, SphereB). The wavezone shell (starting at
    // first_outer_shell_block) uses the transition map; all remaining blocks
    // use the rigid block-0 map. The `true` being passed specifies that the
    // size map *should* be included in the distorted frame.
    const size_t a_block = tag_to_index.at("SphereA");
    const size_t b_block = tag_to_index.at("SphereB");
    constexpr size_t inner_sphere_block_count = 1;

    grid_to_inertial_block_maps[a_block] =
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::A>(
            true, true);
    grid_to_distorted_block_maps[a_block] =
        time_dependent_options_->grid_to_distorted_map<domain::ObjectLabel::A>(
            true);
    distorted_to_inertial_block_maps[a_block] =
        time_dependent_options_
            ->distorted_to_inertial_map<domain::ObjectLabel::A>(true, true);

    grid_to_inertial_block_maps[b_block] =
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::B>(
            true, true);
    grid_to_distorted_block_maps[b_block] =
        time_dependent_options_->grid_to_distorted_map<domain::ObjectLabel::B>(
            true);
    distorted_to_inertial_block_maps[b_block] =
        time_dependent_options_
            ->distorted_to_inertial_map<domain::ObjectLabel::B>(true, true);

    for (size_t block = 1; block < number_of_blocks_; ++block) {
      if (block == a_block or block == b_block or
          block == first_outer_shell_block) {
        continue;  // Already initialized
      } else if (block > a_block and
                 block < a_block + inner_sphere_block_count) {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[a_block]->get_clone();
        if (grid_to_distorted_block_maps[a_block] != nullptr) {
          grid_to_distorted_block_maps[block] =
              grid_to_distorted_block_maps[a_block]->get_clone();
          distorted_to_inertial_block_maps[block] =
              distorted_to_inertial_block_maps[a_block]->get_clone();
        }
      } else if (block > b_block and
                 block < b_block + inner_sphere_block_count) {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[b_block]->get_clone();
        if (grid_to_distorted_block_maps[b_block] != nullptr) {
          grid_to_distorted_block_maps[block] =
              grid_to_distorted_block_maps[b_block]->get_clone();
          distorted_to_inertial_block_maps[block] =
              distorted_to_inertial_block_maps[b_block]->get_clone();
        }
      } else if (block > first_outer_shell_block) {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[first_outer_shell_block]->get_clone();
      } else {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[0]->get_clone();
      }
    }

    for (size_t block = 0; block < number_of_blocks_; ++block) {
      domain.inject_time_dependent_map_for_block(
          block, std::move(grid_to_inertial_block_maps[block]),
          std::move(grid_to_distorted_block_maps[block]),
          std::move(distorted_to_inertial_block_maps[block]));
    }
  }

  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
CylindricalBinaryCompactObject::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{number_of_blocks_};

  std::unordered_map<std::string, size_t> index_of{};
  for (size_t i = 0; i < block_names_.size(); ++i) {
    index_of.emplace(block_names_[i], i);
  }
  const auto set_outer = [this, &boundary_conditions, &index_of](
                             const std::string& name, const Direction<3>& dir) {
    boundary_conditions[index_of.at(name)][dir] =
        outer_boundary_condition_->get_clone();
  };
  const auto set_inner = [this, &boundary_conditions, &index_of](
                             const std::string& name, const Direction<3>& dir) {
    boundary_conditions[index_of.at(name)][dir] =
        inner_boundary_condition_->get_clone();
  };

  // Outer boundary: the outer radial face of the wavezone shell, or (without
  // the wavezone shell) the outer faces of the outermost cylindered C blocks.
  if (include_outer_sphere_) {
    set_outer("SphereC", Direction<3>::upper_xi());
  } else {
    set_outer("CAFilledCylinder", Direction<3>::upper_zeta());
    set_outer("CBFilledCylinder", Direction<3>::upper_zeta());
    set_outer("CACylinder", Direction<3>::upper_xi());
    set_outer("CBCylinder", Direction<3>::upper_xi());
  }

  // Inner (excision) boundary for object A: the lower-zeta face of the
  // innermost filled cylinders and the lower-xi face of the innermost side.
  if (include_inner_sphere_A_) {
    set_inner("InnerSphereEAFilledCylinder", Direction<3>::lower_zeta());
    set_inner("InnerSphereMAFilledCylinder", Direction<3>::lower_zeta());
    set_inner("InnerSphereEACylinder", Direction<3>::lower_xi());
  } else {
    set_inner("EAFilledCylinder", Direction<3>::lower_zeta());
    set_inner("MAFilledCylinder", Direction<3>::lower_zeta());
    set_inner("EACylinder", Direction<3>::lower_xi());
  }
  if (include_inner_sphere_B_) {
    set_inner("InnerSphereEBFilledCylinder", Direction<3>::lower_zeta());
    set_inner("InnerSphereMBFilledCylinder", Direction<3>::lower_zeta());
    set_inner("InnerSphereEBCylinder", Direction<3>::lower_xi());
  } else {
    set_inner("EBFilledCylinder", Direction<3>::lower_zeta());
    set_inner("MBFilledCylinder", Direction<3>::lower_zeta());
    set_inner("EBCylinder", Direction<3>::lower_xi());
  }
  return boundary_conditions;
}

std::vector<std::array<size_t, 3>>
CylindricalBinaryCompactObject::initial_extents() const {
  return initial_grid_points_;
}

std::vector<std::array<size_t, 3>>
CylindricalBinaryCompactObject::initial_refinement_levels() const {
  return initial_refinement_;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
CylindricalBinaryCompactObject::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  return time_dependent_options_.has_value()
             ? time_dependent_options_->create_functions_of_time(
                   initial_expiration_times)
             : std::unordered_map<
                   std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{};
}
}  // namespace domain::creators
