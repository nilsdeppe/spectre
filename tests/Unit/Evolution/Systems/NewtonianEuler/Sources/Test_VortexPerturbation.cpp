// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/VortexPerturbation.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <string>

// IWYU pragma: no_include "DataStructures/Tensor/Tensor.hpp"
// IWYU pragma: no_include "Utilities/Gsl.hpp"

namespace {
// Need this proxy in order for pypp to evaluate function whose arguments
// include a `Time` variable.
template <size_t Dim>
struct VortexPerturbationProxy
    : NewtonianEuler::Sources::VortexPerturbation<Dim> {
  using NewtonianEuler::Sources::VortexPerturbation<Dim>::VortexPerturbation;
  void apply_helper(
      const gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
      const gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const tnsr::I<DataVector, Dim>& x, const double t) const noexcept {
    this->apply(source_mass_density_cons, source_momentum_density,
                source_energy_density, x, t);
  }
};

template <size_t Dim>
void test_sources(
    const std::array<double, Dim>& vortex_center,
    const std::array<double, Dim>& vortex_mean_velocity) noexcept {
  const double adiabatic_index = 1.2;
  const double perturbation_amplitude = 0.1987;
  const double vortex_strength = 2.0;
  VortexPerturbationProxy<Dim> source(adiabatic_index, perturbation_amplitude,
                                      vortex_center, vortex_mean_velocity,
                                      vortex_strength);
  if (Dim == 3) {
    pypp::check_with_random_values<1>(
        &VortexPerturbationProxy<Dim>::apply_helper, source,
        "Evolution.Systems.NewtonianEuler.Sources.VortexPerturbation",
        {"source_mass_density_cons", "source_momentum_density",
         "source_energy_density"},
        {{{-1.0, 1.0}}},
        std::make_tuple(adiabatic_index, perturbation_amplitude, vortex_center,
                        vortex_mean_velocity, vortex_strength),
        DataVector(5));
  }

  VortexPerturbationProxy<Dim> source_to_move(
      adiabatic_index, perturbation_amplitude, vortex_center,
      vortex_mean_velocity, vortex_strength);
  test_move_semantics(std::move(source_to_move), source);  // NOLINT

  test_serialization(source);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Sources.VortexPerturb",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test_sources<2>({{-1.1, 0.4}}, {{0.89, -0.19}});
  test_sources<3>({{0.7, -5.0, 5.5}}, {{1.8, 0.6, -0.57}});
}
