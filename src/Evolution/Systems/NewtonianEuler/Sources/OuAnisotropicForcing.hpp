// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain

namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace NewtonianEuler {
namespace Sources {

/*!
 * \brief Source terms for driving turbulence as a time sequence
 * following an Ornstein-Uhlenbeck (OU) process
 *
 * TO-DO: REWORK THIS DOCUMENTATION!!
 *
 * The OU process is a stochastic differential equation describing the
 * evolution of the forcing term in terms of Fourier coefficients.
 * The modes and coefficients are computed as the evolution proceeds.
 *
 * The subroutine AnisotropicForcingOU::ouNoiseUpdate
 * updates a vector of real values according to an algorithm
 * that generates an Ornstein-Uhlenbeck sequence.
 *
 * The sequence x_n is a Markov process that takes the previous value,
 * weights by an exponential damping factor with a given correlation
 * time "ts" (t_correlation) and drives by adding a Gaussian random variable
 * with variance "variance", weighted by a second damping factor, also
 * with correlation time "ts". For a timestep of dt, this sequence
 * can be written as :
 *
 *      x_n+1 = f x_n + sigma * sqrt (1 - f**2) z_n
 *
 * where f = exp (-dt / ts), z_n is a Gaussian random variable drawn
 * from a Gaussian distribution with unit variance, and sigma is the
 * desired variance of the OU sequence. (See Bartosch, 2001).
 *
 * The resulting sequence should satisfy the properties of zero mean,
 * and stationary (independent of portion of sequence) RMS equal to
 * "variance". Its power spectrum in the time domain can vary from
 *  white noise to "brown" noise (P (f) = const. to 1 / f^2).
 *
 * Input parameters:
 * - OuDeltaT (double): the Ornstein-Uhlenbeck model's DeltaT
 * - SpectrumForm (size_t): 0: Band; 1: Parabola in k-space
 * - DecayTime (double): autocorrelation time for forcing
 * - Energy (double): energy input/mode
 * - StirMin (double): minimum stirring wavenumber
 * - StirMax (double): maximum stirring wavenumber
 * - SolenoidalWeight (double): param in [0, 1]. 0 = compressive, 1 = solenoidal
 * - Anisotropy (double): anisotropy factor in modes.
 *                          increase from 1 makes x-modes anisotropic
 * - SeedForRng (int): seed for random number generator
 *
 *****************************
 *
 * TO-DO:
 * - Box size and initial/final times need to come from input file.
 *   Related: assert kmax to be in k-space box in initialize_stirring_modes().
 *   Maybe dims of k-space box can be chosen so that condition is met
 * - Wavevectors and amplitudes are quadruplicated info. They're doubles though
 * - Generalize projections for any solenoidal weight
 * - k dot x is computed at each timestep, could be cached
 * - Spectrum form could be an enum
 * - Test for anisotropic source needs rework. Related: get rid of
 *   `FourierData(size_t)` overload and `fourier_data()`
 * - Maybe OU sequence and methods and members related can be in its own struct
 */
template <size_t Dim>
struct OuAnisotropicForcing {
  OuAnisotropicForcing() noexcept = default;
  OuAnisotropicForcing(const OuAnisotropicForcing& /*rhs*/) = default;
  OuAnisotropicForcing& operator=(const OuAnisotropicForcing& /*rhs*/) =
      default;
  OuAnisotropicForcing(OuAnisotropicForcing&& /*rhs*/) noexcept = default;
  OuAnisotropicForcing& operator=(OuAnisotropicForcing&& /*rhs*/) noexcept =
      default;
  ~OuAnisotropicForcing() = default;

  OuAnisotropicForcing(double ou_delta_t, size_t spectrum_form,
                       double decay_time, double energy_input_per_mode,
                       double min_stirring_wavenumber,
                       double max_stirring_wavenumber, double solenoidal_weight,
                       double anisotropy_factor, int seed_for_rng) noexcept;

  void pup(PUP::er& /*p*/) noexcept;  // NOLINT( google-runtime-references)

  /// Holds data for computing forcing modes
  struct FourierData {
    FourierData() noexcept = default;

    // overload for testing purposes. Not used in production.
    explicit FourierData(const size_t number_of_modes) noexcept
        : mode_amplitudes(number_of_modes), mode_wavevectors(number_of_modes) {}

    void pup(PUP::er& p) noexcept {  // NOLINT(google-runtime-references)
      p | mode_amplitudes;
      p | mode_wavevectors;
    }

    friend bool operator==(const FourierData& lhs,
                           const FourierData& rhs) noexcept {
      return lhs.mode_amplitudes == rhs.mode_amplitudes and
             lhs.mode_wavevectors == rhs.mode_wavevectors;
    }

    std::vector<double> mode_amplitudes{};
    std::vector<std::array<double, Dim>> mode_wavevectors{};
  };

  using sourced_variables =
      tmpl::list<Tags::MomentumDensity<Dim>, Tags::EnergyDensity>;

  using argument_tags =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>, ::Tags::Time>;

  void apply(gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
             gsl::not_null<Scalar<DataVector>*> source_energy_density,
             const Scalar<DataVector>& mass_density_cons,
             const tnsr::I<DataVector, Dim>& momentum_density,
             const tnsr::I<DataVector, Dim>& x, double t) const noexcept;

  // Retrieve for testing purposes only.
  const FourierData& fourier_data() const noexcept { return fourier_data_; }

 private:
  template <size_t SpatialDim>
  friend bool operator==(  // NOLINT(readability-redundant-declaration)
      const OuAnisotropicForcing<SpatialDim>& lhs,
      const OuAnisotropicForcing<SpatialDim>& rhs) noexcept;

  // FLASH's ran1s()
  // Taken from Numerical Recipes.
  // non-const because it mutates seed_for_rng_
  double ran1s(gsl::not_null<int*> idum) noexcept;

  // FLASH's st_grn()
  // Draws a number randomly from a Gaussian distribution with the standard
  // uniform distribution function using the Box-Muller transformation in polar
  // coordinates. The resulting Gaussian has unit variance.
  // non-const mutates seed_for_rng_ through a call to ran1s
  void grn(gsl::not_null<double*> grnval) noexcept;

  // FLASH's stir_init()
  // Initializes the Fourier modes to be used in the expansion.
  // Called from constructor. non-const as it initializes fourier_data_,
  // as well as number_of_modes_
  void initialize_stirring_modes() noexcept;

  // FLASH's st_ounoiseinit()
  // Initializes pseudo-random sequence for the OU process.
  // Called from generate_ou_sequence()
  void initialize_ou_phases(
      gsl::not_null<std::vector<double>*> phases) noexcept;

  // FLASH's st_ounoiseupdate()
  // Updates OU phases (noise) applying the formula for a Markov process
  // Called from generate_ou_sequence()
  void update_ou_phases(gsl::not_null<std::vector<double>*> phases) noexcept;

  // Calculates and stores the OU phases for all the evolution
  // Called from constructor. non-const as it initializes ou_sequence_
  void generate_ou_sequence() noexcept;

  // FLASH's st_calcPhases(), adapted for introducing anisotropies.
  // Updates the stirring phases from the OU phases at a given time
  // for a given mode. It multiplies x-modes by anisotropy factor,
  // then it applies the projection operator.
  std::array<double, 2 * Dim> stirring_phases_for_mode(size_t time_id,
                                                       size_t mode_id) const
      noexcept;

  // FLASH's st_calcAccel()
  // Computes the acceleration as an explicit (i.e no FFT) Fourier series
  void compute_acceleration(
      gsl::not_null<tnsr::I<DataVector, Dim>*> acceleration,
      const tnsr::I<DataVector, Dim>& x, double t,
      const FourierData& fourier_data) const noexcept;

  double ou_delta_t_ = std::numeric_limits<double>::signaling_NaN();
  size_t spectrum_form_{};  // 0 (band) or 1 (parabola). Write as an enum?
  double decay_time_ = std::numeric_limits<double>::signaling_NaN();
  double energy_input_per_mode_ = std::numeric_limits<double>::signaling_NaN();
  double min_stirring_wavenumber_ =
      std::numeric_limits<double>::signaling_NaN();
  double max_stirring_wavenumber_ =
      std::numeric_limits<double>::signaling_NaN();
  double solenoidal_weight_ = std::numeric_limits<double>::signaling_NaN();
  double anisotropy_factor_ = std::numeric_limits<double>::signaling_NaN();
  int seed_for_rng_{};

  // Used to normalize the force when solenoidal_weight_ varied
  double solenoidal_weight_norm_ = std::numeric_limits<double>::signaling_NaN();

  // Total number of modes in Fourier representation of the forcing term
  size_t number_of_modes_{};

  // OU variance corresponding to decay time and energy input rate
  double ou_variance_ = std::numeric_limits<double>::signaling_NaN();
  // Damping factor for the OU sequence
  double ou_damping_factor_ = std::numeric_limits<double>::signaling_NaN();
  // OU driving factor, sqrt(1 - (OU damping_factor)**2), times OU variance
  double ou_variance_times_driving_factor_ =
      std::numeric_limits<double>::signaling_NaN();

  // Time sequence of OU phases for each complex mode in the forcing term.
  // Size of the sequence includes the initial phases (initialize_ou_phases())
  // plus the updated phases after 'n' OU steps, where n = T / ou_delta_,
  // T being the total simulation time.
  std::vector<std::vector<double>> ou_sequence_{};

  // Set of modes for representing the forcing term
  FourierData fourier_data_{};
};

template <size_t Dim>
bool operator!=(const OuAnisotropicForcing<Dim>& lhs,
                const OuAnisotropicForcing<Dim>& rhs) noexcept;
}  // namespace Sources
}  // namespace NewtonianEuler
