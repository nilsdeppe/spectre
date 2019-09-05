// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/KeplerianDisk.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

template <size_t Dim>
struct KeplerianDiskProxy : NewtonianEuler::Solutions::KeplerianDisk<Dim> {
  using NewtonianEuler::Solutions::KeplerianDisk<Dim>::KeplerianDisk;

  template <typename DataType>
  using variables_tags =
      tmpl::list<NewtonianEuler::Tags::MassDensity<DataType>,
                 NewtonianEuler::Tags::Velocity<DataType, Dim, Frame::Inertial>,
                 NewtonianEuler::Tags::SpecificInternalEnergy<DataType>,
                 NewtonianEuler::Tags::Pressure<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>>
  primitive_variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x) const
      noexcept {
    const double dummy_time = 0.1;
    return this->variables(x, dummy_time, variables_tags<DataType>{});
  }
};

template <size_t Dim, typename DataType>
void test_solution(const DataType& used_for_size,
                   const std::array<double, Dim>& disk_center,
                   const std::string& disk_center_option) noexcept {
  const double adiabatic_index = 1.6;
  const double ambient_mass_density = 1.1e-5;
  const double ambient_pressure = 4.2e-5;
  const double disk_mass_density = 0.1;
  const double disk_inner_radius = 0.53;
  const double disk_outer_radius = 1.3;
  const double smoothing_parameter = 0.02;
  const double transition_width = 0.04;
  KeplerianDiskProxy<Dim> disk(adiabatic_index, ambient_mass_density,
                               ambient_pressure, disk_center, disk_mass_density,
                               disk_inner_radius, disk_outer_radius,
                               smoothing_parameter, transition_width);
  pypp::check_with_random_values<
      1, typename KeplerianDiskProxy<Dim>::template variables_tags<DataType>>(
      &KeplerianDiskProxy<Dim>::template primitive_variables<DataType>, disk,
      "KeplerianDisk",
      {"mass_density", "velocity", "specific_internal_energy", "pressure"},
      {{{-1.5, 1.5}}},
      std::make_tuple(adiabatic_index, ambient_mass_density, ambient_pressure,
                      disk_center, disk_mass_density, disk_inner_radius,
                      disk_outer_radius, smoothing_parameter, transition_width),
      used_for_size);

  const auto disk_from_options =
      test_creation<NewtonianEuler::Solutions::KeplerianDisk<Dim>>(
          "  AdiabaticIndex: 1.6\n"
          "  AmbientMassDensity: 1.1e-5\n"
          "  AmbientPressure: 4.2e-5\n"
          "  DiskCenter: " +
          disk_center_option +
          "\n"
          "  DiskMassDensity: 0.1\n"
          "  DiskInnerRadius: 0.53\n"
          "  DiskOuterRadius: 1.3\n"
          "  SmoothingParameter: 0.02\n"
          "  TransitionWidth: 0.04");
  CHECK(disk_from_options == disk);

  KeplerianDiskProxy<Dim> disk_to_move(
      adiabatic_index, ambient_mass_density, ambient_pressure, disk_center,
      disk_mass_density, disk_inner_radius, disk_outer_radius,
      smoothing_parameter, transition_width);
  test_move_semantics(std::move(disk_to_move), disk);  //  NOLINT

  test_serialization(disk);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/NewtonianEuler"};

  const std::array<double, 2> disk_center_2d = {{-0.54, 1.2}};
  test_solution<2>(std::numeric_limits<double>::signaling_NaN(), disk_center_2d,
                   "[-0.54, 1.2]");
  test_solution<2>(DataVector(5), disk_center_2d, "[-0.54, 1.2]");

  const std::array<double, 3> disk_center_3d = {{1.67, -0.5, -1.2}};
  test_solution<3>(std::numeric_limits<double>::signaling_NaN(), disk_center_3d,
                   "[1.67, -0.5, -1.2]");
  test_solution<3>(DataVector(5), disk_center_3d, "[1.67, -0.5, -1.2]");
}

// [[OutputRegex, The ambient mass density must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.AmbDens2D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<2> test_disk(
      1.41, -1.4, 2.1, {{3.1, 2.8}}, 1.3, 0.4, 1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The ambient mass density must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.AmbDens3D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<3> test_disk(
      1.41, -1.4, 2.1, {{3.1, 2.8, -0.3}}, 1.3, 0.4, 1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The ambient pressure must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.AmbPres2D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<2> test_disk(
      1.41, 1.4, -2.1, {{3.1, 2.8}}, 1.3, 0.4, 1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The ambient pressure must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.AmbPres3D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<3> test_disk(
      1.41, 1.4, -2.1, {{3.1, 2.8, -0.3}}, 1.3, 0.4, 1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The disk mass density must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.DiskDens2D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<2> test_disk(
      1.41, 1.4, 2.1, {{3.1, 2.8}}, -1.3, 0.4, 1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The disk mass density must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.DiskDens3D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<3> test_disk(
      1.41, 1.4, 2.1, {{3.1, 2.8, -0.3}}, -1.3, 0.4, 1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The disk inner radius must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.DiskInnRad2D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<2> test_disk(
      1.41, 1.4, 2.1, {{3.1, 2.8}}, 1.3, -0.4, 1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The disk inner radius must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.DiskInnRad3D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<3> test_disk(
      1.41, 1.4, 2.1, {{3.1, 2.8, -0.3}}, 1.3, -0.4, 1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The disk outer radius must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.DiskOutRad2D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<2> test_disk(
      1.41, 1.4, 2.1, {{3.1, 2.8}}, 1.3, 0.4, -1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The disk outer radius must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.DiskOutRad3D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<3> test_disk(
      1.41, 1.4, 2.1, {{3.1, 2.8, -0.3}}, 1.3, 0.4, -1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The disk outer radius must be greater than the inner radius.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.DiskRadii2D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<2> test_disk(
      1.41, 1.4, 2.1, {{3.1, 2.8}}, 1.3, 2.4, 1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The disk outer radius must be greater than the inner radius.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.DiskRadii3D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<3> test_disk(
      1.41, 1.4, 2.1, {{3.1, 2.8, -0.3}}, 1.3, 2.4, 1.5, 0.02, 0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The transition width must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.TransWidth2D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<2> test_disk(
      1.41, 1.4, 2.1, {{3.1, 2.8}}, 1.3, 0.4, 1.5, 0.02, -0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The transition width must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.KeplDisk.TransWidth3D",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::KeplerianDisk<3> test_disk(
      1.41, 1.4, 2.1, {{3.1, 2.8, -0.3}}, 1.3, 0.4, 1.5, 0.02, -0.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
