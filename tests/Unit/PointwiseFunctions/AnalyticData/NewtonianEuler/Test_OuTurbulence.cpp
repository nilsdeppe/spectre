// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianEuler/OuTurbulence.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

template <size_t Dim, typename DataType>
void test_initial_data(const DataType& used_for_size) noexcept {
  const double polytropic_exponent = 1.333333333333333;
  const double initial_density = 2.012;
  const double decay_time = 0.5;
  const double energy_input_per_mode = 0.002;
  const double min_stirring_wavenumber = 6.283;
  const double max_stirring_wavenumber = 18.95;
  const double solenoidal_weight = 1.0;
  const double anisotropy_factor = 1.0;

  NewtonianEuler::AnalyticData::OuTurbulence<Dim> initial_data(
      polytropic_exponent, initial_density, decay_time, energy_input_per_mode,
      min_stirring_wavenumber, max_stirring_wavenumber, solenoidal_weight,
      anisotropy_factor);

  const auto data_from_options = TestHelpers::test_creation<
      NewtonianEuler::AnalyticData::OuTurbulence<Dim>>(
      "  PolytropicExponent: 1.333333333333333\n"
      "  InitialDensity: 2.012\n"
      "  DecayTime: 0.5\n"
      "  EnergyPerMode: 0.002\n"
      "  MinWavenumber: 6.283\n"
      "  MaxWavenumber: 18.95\n"
      "  SolenoidalWeight: 1.0\n"
      "  AnisotropyFactor: 1.0");

  CHECK(data_from_options == initial_data);

  NewtonianEuler::AnalyticData::OuTurbulence<Dim> data_to_move(
      polytropic_exponent, initial_density, decay_time, energy_input_per_mode,
      min_stirring_wavenumber, max_stirring_wavenumber, solenoidal_weight,
      anisotropy_factor);
  test_move_semantics(std::move(data_to_move), initial_data);  //  NOLINT

  test_serialization(initial_data);

  const auto x = make_with_value<tnsr::I<DataType, Dim>>(used_for_size, 19.87);
  const auto primitives = initial_data.variables(
      x, tmpl::list<NewtonianEuler::Tags::MassDensity<DataType>,
                    NewtonianEuler::Tags::Velocity<DataType, Dim>,
                    NewtonianEuler::Tags::SpecificInternalEnergy<DataType>,
                    NewtonianEuler::Tags::Pressure<DataType>>{});

  const auto expected_mass_density =
      make_with_value<Scalar<DataType>>(used_for_size, initial_density);
  const auto expected_velocity =
      make_with_value<tnsr::I<DataType, Dim>>(used_for_size, 0.0);
  const auto expected_specific_internal_energy =
      make_with_value<Scalar<DataType>>(
          used_for_size, pow(initial_density, polytropic_exponent - 1.0) /
                             (polytropic_exponent - 1.0));
  const auto expected_pressure = make_with_value<Scalar<DataType>>(
      used_for_size, pow(initial_density, polytropic_exponent));

  CHECK(get<NewtonianEuler::Tags::MassDensity<DataType>>(primitives) ==
        expected_mass_density);
  CHECK(get<NewtonianEuler::Tags::Velocity<DataType, Dim>>(primitives) ==
        expected_velocity);
  CHECK(get<NewtonianEuler::Tags::SpecificInternalEnergy<DataType>>(
            primitives) == expected_specific_internal_energy);
  CHECK(get<NewtonianEuler::Tags::Pressure<DataType>>(primitives) ==
        expected_pressure);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.NewtEuler.OuTurbulence",
                  "[Unit][PointwiseFunctions]") {
  test_initial_data<3>(std::numeric_limits<double>::signaling_NaN());
  test_initial_data<3>(DataVector(5));
}

// [[OutputRegex, The initial density must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.NewtEuler.OuTurbulenceDens",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::AnalyticData::OuTurbulence<3> test_data(
      1.34, -0.1, 0.4, 0.231, 6.23, 18.0, 0.4, 1.5);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

template <size_t Dim>
struct OuTurb {
  using type = NewtonianEuler::AnalyticData::OuTurbulence<Dim>;
  static constexpr OptionString help = {"Homogeneous fluid at rest."};
};

// [[OutputRegex, In string:.*At line 3 column 19:.Value -0.2 is below the lower
// bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.NewtEuler.OuTurbulenceDensOpt",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<OuTurb<3>>> test_options("");
  test_options.parse(
      "OuTurb:\n"
      "  PolytropicExponent: 1.4\n"
      "  InitialDensity: -0.2\n"
      "  DecayTime: 0.5\n"
      "  EnergyPerMode: 0.002\n"
      "  MinWavenumber: 6.283\n"
      "  MaxWavenumber: 18.95\n"
      "  SolenoidalWeight: 0.3\n"
      "  AnisotropyFactor: 1.5");
  test_options.get<OuTurb<3>>();
}
