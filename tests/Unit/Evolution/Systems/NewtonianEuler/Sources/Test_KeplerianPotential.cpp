// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/KeplerianPotential.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

template <size_t Dim>
void test_sources(const std::array<double, Dim>& potential_center) noexcept {
  const double smoothing_parameter = 0.05;
  const double transition_width = 0.2;
  NewtonianEuler::Sources::KeplerianPotential<Dim> source(
      potential_center, smoothing_parameter, transition_width);
  pypp::check_with_random_values<1>(
      &NewtonianEuler::Sources::KeplerianPotential<Dim>::apply, source,
      "Evolution.Systems.NewtonianEuler.Sources.KeplerianPotential",
      {"source_momentum_density", "source_energy_density"}, {{{-0.5, 0.5}}},
      std::make_tuple(potential_center, smoothing_parameter, transition_width),
      DataVector(5));

  NewtonianEuler::Sources::KeplerianPotential<Dim> source_to_move(
      potential_center, smoothing_parameter, transition_width);
  test_move_semantics(std::move(source_to_move), source);  // NOLINT

  test_serialization(source);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Sources.KeplPotential",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test_sources<2>({{0.1, -0.4}});
  test_sources<3>({{-2.4, 3.5, 0.9}});
}

// [[OutputRegex, The smoothing parameter must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Sources.KeplPotential.SmoothParam2D",
    "[Unit][Evolution]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Sources::KeplerianPotential<2> test_disk({{0.123, -4.54}},
                                                           -1.3, 0.12);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The smoothing parameter must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Sources.KeplPotential.SmoothParam3D",
    "[Unit][Evolution]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Sources::KeplerianPotential<3> test_disk(
      {{0.123, -4.54, 4.12}}, -1.3, 0.12);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
