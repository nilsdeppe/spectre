// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianEuler/RayleighTaylorInst.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

template <size_t Dim>
struct RayleighTaylorInstProxy
    : NewtonianEuler::AnalyticData::RayleighTaylorInst<Dim> {
  using NewtonianEuler::AnalyticData::RayleighTaylorInst<
      Dim>::RayleighTaylorInst;

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
void test_analytic_data(const DataType& used_for_size) noexcept {
  const double adiabatic_index = 1.41;
  const double lower_mass_density = 2.5;
  const double upper_mass_density = 1.2;
  const double background_pressure = 4.3;
  const double perturbation_amplitude = 0.01;
  const double damping_factor = 0.1;
  const double interface_height = 0.3;
  const double grav_acceleration = 0.34;

  RayleighTaylorInstProxy<Dim> rt_inst(adiabatic_index, lower_mass_density,
                                       upper_mass_density, background_pressure,
                                       perturbation_amplitude, damping_factor,
                                       interface_height, grav_acceleration);
  pypp::check_with_random_values<
      1,
      typename RayleighTaylorInstProxy<Dim>::template variables_tags<DataType>>(
      &RayleighTaylorInstProxy<Dim>::template primitive_variables<DataType>,
      rt_inst, "RayleighTaylorInst",
      {"mass_density", "velocity", "specific_internal_energy", "pressure"},
      {{{0.0, 1.0}}},
      std::make_tuple(adiabatic_index, lower_mass_density, upper_mass_density,
                      background_pressure, perturbation_amplitude,
                      damping_factor, interface_height, grav_acceleration),
      used_for_size);

  const auto rt_inst_from_options =
      test_creation<NewtonianEuler::AnalyticData::RayleighTaylorInst<Dim>>(
          "  AdiabaticIndex: 1.41\n"
          "  LowerMassDensity: 2.5\n"
          "  UpperMassDensity: 1.2\n"
          "  BackgroundPressure: 4.3\n"
          "  PerturbAmplitude: 0.01\n"
          "  DampingFactor: 0.1\n"
          "  InterfaceHeight: 0.3\n"
          "  GravAcceleration: 0.34");
  CHECK(rt_inst_from_options == rt_inst);

  RayleighTaylorInstProxy<Dim> rt_inst_to_move(
      adiabatic_index, lower_mass_density, upper_mass_density,
      background_pressure, perturbation_amplitude, damping_factor,
      interface_height, grav_acceleration);
  test_move_semantics(std::move(rt_inst_to_move), rt_inst);  //  NOLINT

  test_serialization(rt_inst);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.NewtEuler.RtInst",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/NewtonianEuler"};

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_analytic_data, (2, 3));
}

template <size_t Dim>
struct Instability {
  using type = NewtonianEuler::AnalyticData::RayleighTaylorInst<Dim>;
  static constexpr OptionString help = {
      "Initial data to simulate the RT instability."};
};

// [[OutputRegex, In string:.*At line 3 column 21:.Value -2.1 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.NewtEuler.RtInst.RhoLower2d",
    "[Unit][PointwiseFunctions]") {
  ERROR_TEST();
  Options<tmpl::list<Instability<2>>> test_options("");
  test_options.parse(
      "Instability:\n"
      "  AdiabaticIndex: 1.43\n"
      "  LowerMassDensity: -2.1\n"
      "  UpperMassDensity: 2.0\n"
      "  BackgroundPressure: 1.1\n"
      "  PerturbAmplitude: 0.1\n"
      "  DampingFactor: 0.01\n"
      "  InterfaceHeight: 0.4\n"
      "  GravAcceleration: 0.91");
  test_options.get<Instability<2>>();
}

// [[OutputRegex, In string:.*At line 3 column 21:.Value -2.1 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.NewtEuler.RtInst.RhoLower3d",
    "[Unit][PointwiseFunctions]") {
  ERROR_TEST();
  Options<tmpl::list<Instability<3>>> test_options("");
  test_options.parse(
      "Instability:\n"
      "  AdiabaticIndex: 1.43\n"
      "  LowerMassDensity: -2.1\n"
      "  UpperMassDensity: 2.0\n"
      "  BackgroundPressure: 1.1\n"
      "  PerturbAmplitude: 0.1\n"
      "  DampingFactor: 0.01\n"
      "  InterfaceHeight: 0.4\n"
      "  GravAcceleration: 0.91");
  test_options.get<Instability<3>>();
}

// [[OutputRegex, In string:.*At line 4 column 21:.Value -2 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.NewtEuler.RtInst.RhoUpper2d",
    "[Unit][PointwiseFunctions]") {
  ERROR_TEST();
  Options<tmpl::list<Instability<2>>> test_options("");
  test_options.parse(
      "Instability:\n"
      "  AdiabaticIndex: 1.43\n"
      "  LowerMassDensity: 2.1\n"
      "  UpperMassDensity: -2.0\n"
      "  BackgroundPressure: 1.1\n"
      "  PerturbAmplitude: 0.1\n"
      "  DampingFactor: 0.01\n"
      "  InterfaceHeight: 0.4\n"
      "  GravAcceleration: 0.91");
  test_options.get<Instability<2>>();
}

// [[OutputRegex, In string:.*At line 4 column 21:.Value -2 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.NewtEuler.RtInst.RhoUpper3d",
    "[Unit][PointwiseFunctions]") {
  ERROR_TEST();
  Options<tmpl::list<Instability<3>>> test_options("");
  test_options.parse(
      "Instability:\n"
      "  AdiabaticIndex: 1.43\n"
      "  LowerMassDensity: 2.1\n"
      "  UpperMassDensity: -2.0\n"
      "  BackgroundPressure: 1.1\n"
      "  PerturbAmplitude: 0.1\n"
      "  DampingFactor: 0.01\n"
      "  InterfaceHeight: 0.4\n"
      "  GravAcceleration: 0.91");
  test_options.get<Instability<3>>();
}

// [[OutputRegex, In string:.*At line 5 column 23:.Value -1.1 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.NewtEuler.RtInst.Pressure2d",
    "[Unit][PointwiseFunctions]") {
  ERROR_TEST();
  Options<tmpl::list<Instability<2>>> test_options("");
  test_options.parse(
      "Instability:\n"
      "  AdiabaticIndex: 1.43\n"
      "  LowerMassDensity: 2.1\n"
      "  UpperMassDensity: 2.0\n"
      "  BackgroundPressure: -1.1\n"
      "  PerturbAmplitude: 0.1\n"
      "  DampingFactor: 0.01\n"
      "  InterfaceHeight: 0.4\n"
      "  GravAcceleration: 0.91");
  test_options.get<Instability<2>>();
}

// [[OutputRegex, In string:.*At line 5 column 23:.Value -1.1 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.NewtEuler.RtInst.Pressure3d",
    "[Unit][PointwiseFunctions]") {
  ERROR_TEST();
  Options<tmpl::list<Instability<3>>> test_options("");
  test_options.parse(
      "Instability:\n"
      "  AdiabaticIndex: 1.43\n"
      "  LowerMassDensity: 2.1\n"
      "  UpperMassDensity: 2.0\n"
      "  BackgroundPressure: -1.1\n"
      "  PerturbAmplitude: 0.1\n"
      "  DampingFactor: 0.01\n"
      "  InterfaceHeight: 0.4\n"
      "  GravAcceleration: 0.91");
  test_options.get<Instability<3>>();
}

// [[OutputRegex, In string:.*At line 9 column 21:.Value -0.91 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.NewtEuler.RtInst.Grav2d",
    "[Unit][PointwiseFunctions]") {
  ERROR_TEST();
  Options<tmpl::list<Instability<2>>> test_options("");
  test_options.parse(
      "Instability:\n"
      "  AdiabaticIndex: 1.43\n"
      "  LowerMassDensity: 2.1\n"
      "  UpperMassDensity: 2.0\n"
      "  BackgroundPressure: 1.1\n"
      "  PerturbAmplitude: 0.1\n"
      "  DampingFactor: 0.01\n"
      "  InterfaceHeight: 0.4\n"
      "  GravAcceleration: -0.91");
  test_options.get<Instability<2>>();
}

// [[OutputRegex, In string:.*At line 9 column 21:.Value -0.91 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.NewtEuler.RtInst.Grav3d",
    "[Unit][PointwiseFunctions]") {
  ERROR_TEST();
  Options<tmpl::list<Instability<3>>> test_options("");
  test_options.parse(
      "Instability:\n"
      "  AdiabaticIndex: 1.43\n"
      "  LowerMassDensity: 2.1\n"
      "  UpperMassDensity: 2.0\n"
      "  BackgroundPressure: 1.1\n"
      "  PerturbAmplitude: 0.1\n"
      "  DampingFactor: 0.01\n"
      "  InterfaceHeight: 0.4\n"
      "  GravAcceleration: 0.91");
  test_options.get<Instability<3>>();
}
