// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/OuAnisotropicForcing.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace NewtonianEuler {
namespace AnalyticData {

/*!
 * \brief Homogeneous fluid at rest used for OU turbulence
 */
template <size_t Dim>
class OuTurbulence : public MarkAsAnalyticData {
 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<false>;
  using source_term_type = Sources::OuAnisotropicForcing<Dim>;

  /// The polytropic exponent of the fluid.
  struct PolytropicExponent {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic exponent of the fluid."};
  };

  /// The initial mass density in the box
  struct InitialDensity {
    using type = double;
    static constexpr OptionString help = {
        "The initial mass density in the box."};
    static constexpr type lower_bound() noexcept { return 0.0; }
  };

  /// The decay (or correlation) time of the OU forcing
  struct DecayTime {
    using type = double;
    static constexpr OptionString help = {
        "The decay (or correlation) time of the OU forcing."};
    static constexpr type lower_bound() noexcept { return 0.0; }
  };

  /// The energy input per mode of the OU forcing
  struct EnergyPerMode {
    using type = double;
    static constexpr OptionString help = {
        "The energy input per mode of the OU forcing."};
    static constexpr type lower_bound() noexcept { return 0.0; }
  };

  /// The minimum wavenumber for the OU forcing
  struct MinWavenumber {
    using type = double;
    static constexpr OptionString help = {
        "The minumum wavenumber for the OU forcing."};
    static constexpr type lower_bound() noexcept { return 0.0; }
  };

  /// The maximum wavenumber for the OU forcing
  struct MaxWavenumber {
    using type = double;
    static constexpr OptionString help = {
        "The maximum wavenumber for the OU forcing."};
    static constexpr type lower_bound() noexcept { return 0.0; }
  };

  /// The solenoidal weight of the OU forcing
  struct SolenoidalWeight {
    using type = double;
    static constexpr OptionString help = {
        "The solenoidal weight of the OU forcing."};
    static constexpr type lower_bound() noexcept { return 0.0; }
    static constexpr type upper_bound() noexcept { return 1.0; }
  };

  /// The anisotropy factor of the OU forcing
  struct AnisotropyFactor {
    using type = double;
    static constexpr OptionString help = {
        "The anisotropy factor of the OU forcing"};
    static constexpr type lower_bound() noexcept { return 1.0; }
  };

  using options = tmpl::list<PolytropicExponent, InitialDensity, DecayTime,
                             EnergyPerMode, MinWavenumber, MaxWavenumber,
                             SolenoidalWeight, AnisotropyFactor>;

  static constexpr OptionString help = {"Homogeneous fluid at rest."};

  OuTurbulence() = default;
  OuTurbulence(const OuTurbulence& /*rhs*/) = delete;
  OuTurbulence& operator=(const OuTurbulence& /*rhs*/) = delete;
  OuTurbulence(OuTurbulence&& /*rhs*/) noexcept = default;
  OuTurbulence& operator=(OuTurbulence&& /*rhs*/) noexcept = default;
  ~OuTurbulence() = default;

  OuTurbulence(double polytropic_exponent, double initial_density,
               double decay_time, double energy_input_per_mode,
               double min_stirring_wavenumber, double max_stirring_wavenumber,
               double solenoidal_weight, double anisotropy_factor) noexcept;

  /// Retrieve a collection of hydrodynamic variables at position x and time t
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  const EquationsOfState::PolytropicFluid<false>& equation_of_state() const
      noexcept {
    return equation_of_state_;
  }

  const Sources::OuAnisotropicForcing<Dim>& source_term() const noexcept {
    return source_term_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

 private:
  // @{
  /// Retrieve hydro variable at `(x, t)`
  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                 tmpl::list<Tags::MassDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<Tags::MassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x,
      tmpl::list<Tags::Velocity<DataType, Dim, Frame::Inertial>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<Tags::Velocity<DataType, Dim, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x,
      tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                 tmpl::list<Tags::Pressure<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Pressure<DataType>>;
  // @}

  template <size_t SpatialDim>
  friend bool
  operator==(  // NOLINT (clang-tidy: readability-redundant-declaration)
      const OuTurbulence<SpatialDim>& lhs,
      const OuTurbulence<SpatialDim>& rhs) noexcept;

  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  double initial_density_ = std::numeric_limits<double>::signaling_NaN();

  EquationsOfState::PolytropicFluid<false> equation_of_state_{};
  Sources::OuAnisotropicForcing<Dim> source_term_{};
};

template <size_t Dim>
bool operator!=(const OuTurbulence<Dim>& lhs,
                const OuTurbulence<Dim>& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace NewtonianEuler
