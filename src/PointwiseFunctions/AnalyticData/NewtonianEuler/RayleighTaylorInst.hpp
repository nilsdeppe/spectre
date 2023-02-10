// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace NewtonianEuler {
namespace AnalyticData {

/*!
 * \brief Initial data to simulate the Rayleigh-Taylor instability.
 *
 * \note This class can be used to initialize 2D and 3D data. In 3D,
 * the vertical direction is taken to be the \f$z-\f$axis, and the
 * interface is parallel to the \f$x-y\f$ plane.
 */
template <size_t Dim>
class RayleighTaylorInst {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<false>;

  /// The adiabatic index of the fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr OptionString help = {"The adiabatic index of the fluid."};
  };

  /// The mass density below the interface
  struct LowerMassDensity {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr OptionString help = {
        "The mass density below the interface."};
  };

  /// The mass density above the interface
  struct UpperMassDensity {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr OptionString help = {
        "The mass density above the interface."};
  };

  /// The background pressure of the fluid
  struct BackgroundPressure {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr OptionString help = {"The background pressure."};
  };

  /// The amplitude of the perturbation in the vertical velocity
  struct PerturbAmplitude {
    using type = double;
    static constexpr OptionString help = {"The amplitude of the perturbation."};
  };

  /// The damping factor of the perturbation
  struct DampingFactor {
    using type = double;
    static constexpr OptionString help = {
        "The dampingf factor of the perturbation."};
  };

  /// The vertical coordinate of the interface
  struct InterfaceHeight {
    using type = double;
    static constexpr OptionString help = {
        "The vertical coordinate of the interface."};
  };

  /// The magnitude of the gravitational acceleration
  struct GravAcceleration {
    using type = double;
    static constexpr OptionString help = {
        "The magnitude of the gravitational acceleration."};
  };

  using options = tmpl::list<AdiabaticIndex, LowerMassDensity, UpperMassDensity,
                             BackgroundPressure, PerturbAmplitude,
                             DampingFactor, InterfaceHeight, GravAcceleration>;

  static constexpr OptionString help = {
      "Initial data to simulate the RT instability."};

  RayleighTaylorInst() = default;
  RayleighTaylorInst(const RayleighTaylorInst& /*rhs*/) = delete;
  RayleighTaylorInst& operator=(const RayleighTaylorInst& /*rhs*/) = delete;
  RayleighTaylorInst(RayleighTaylorInst&& /*rhs*/) noexcept = default;
  RayleighTaylorInst& operator=(RayleighTaylorInst&& /*rhs*/) noexcept =
      default;
  ~RayleighTaylorInst() = default;

  RayleighTaylorInst(double adiabatic_index, double lower_mass_density,
                     double upper_mass_density, double background_pressure,
                     double perturbation_amplitude, double damping_factor,
                     double interface_height, double grav_acceleration);

  /// Retrieve a collection of hydrodynamic variables at position x
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x, const double t,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  const EquationsOfState::IdealFluid<false>& equation_of_state() const
      noexcept {
    return equation_of_state_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

 private:
  // @{
  /// Retrieve hydro variable at `x`
  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                 const double t,
                 tmpl::list<Tags::MassDensity<DataType>> /*meta*/
                 ) const noexcept
      -> tuples::TaggedTuple<Tags::MassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x, const double t,
      tmpl::list<Tags::Velocity<DataType, Dim, Frame::Inertial>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<Tags::Velocity<DataType, Dim, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                 const double t,
                 tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/
                 ) const noexcept
      -> tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                 const double t, tmpl::list<Tags::Pressure<DataType>> /*meta*/
                 ) const noexcept
      -> tuples::TaggedTuple<Tags::Pressure<DataType>>;
  // @}

  template <size_t SpatialDim>
  friend bool
  operator==(  // NOLINT (clang-tidy: readability-redundant-declaration)
      const RayleighTaylorInst<SpatialDim>& lhs,
      const RayleighTaylorInst<SpatialDim>& rhs) noexcept;

  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double lower_mass_density_ = std::numeric_limits<double>::signaling_NaN();
  double upper_mass_density_ = std::numeric_limits<double>::signaling_NaN();
  double background_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double perturbation_amplitude_ = std::numeric_limits<double>::signaling_NaN();
  double damping_factor_ = std::numeric_limits<double>::signaling_NaN();
  double interface_height_ = std::numeric_limits<double>::signaling_NaN();
  // should be retrieved from EvolutionSystem option in input file
  double grav_acceleration_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::IdealFluid<false> equation_of_state_{};
};

template <size_t Dim>
bool operator!=(const RayleighTaylorInst<Dim>& lhs,
                const RayleighTaylorInst<Dim>& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace NewtonianEuler
