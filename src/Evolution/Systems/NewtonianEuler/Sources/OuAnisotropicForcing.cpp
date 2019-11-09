// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/OuAnisotropicForcing.hpp"

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace NewtonianEuler {
namespace Sources {

template <size_t Dim>
double OuAnisotropicForcing<Dim>::ran1s(
    const gsl::not_null<int*> idum) noexcept {
  const int IA = 16807, IM = 2147483647, IQ = 127773, IR = 2836;
  const double AM = 1.0 / IM, EPS = 1.2e-7, RNMX = 1.0 - EPS;

  int k, iy;
  if (*idum < 0) {
    *idum = std::max(-*idum, 1);
  }

  k = *idum / IQ;
  *idum = IA * (*idum - k * IQ) - IR * k;
  if (*idum < 0) {
    *idum += IM;
  }

  iy = *idum;
  return std::min(AM * iy, RNMX);
}

template <size_t Dim>
void OuAnisotropicForcing<Dim>::grn(
    const gsl::not_null<double*> grnval) noexcept {
  double r1 = ran1s(make_not_null(&seed_for_rng_));
  double r2 = ran1s(make_not_null(&seed_for_rng_));
  *grnval = sqrt(2.0 * log(1.0 / r1)) * cos(2.0 * M_PI * r2);
}

template <>
void OuAnisotropicForcing<3>::initialize_stirring_modes() noexcept {
  // minimum happens at both ends of the spectrum
  const double minimum_amplitude_of_parabolic_spectrum = 0.0;
  // average, as parabola is symmetric about k = central_wavenumber line
  const double parabolic_central_wavenumber =
      0.5 * (min_stirring_wavenumber_ + max_stirring_wavenumber_);
  const double parabolic_amplitude_prefactor =
      4.0 * (minimum_amplitude_of_parabolic_spectrum - 1.0) /
      square(max_stirring_wavenumber_ - min_stirring_wavenumber_);

  // NOTE: if max stirring wavenumber is really large, increase maxima
  // How large? kmax^2 > (2pi)^2 [(nx_max/Lx)^2 + (ny_max/Ly)^2 + (ny_max/Ly)^2]
  const int nx_min = 0, ny_min = 0, nz_min = 0;
  const int nx_max = 20, ny_max = 20, nz_max = 20;

  // sides of the physical domain, assumed to be a unit box centered at origin
  const double two_pi = 2.0 * M_PI;
  const double two_pi_over_side_x = two_pi / 1.0;
  const double two_pi_over_side_y = two_pi / 1.0;
  const double two_pi_over_side_z = two_pi / 1.0;

  number_of_modes_ = 0;

  for (int nx = nx_min; nx <= nx_max; ++nx) {
    const double kx = nx * two_pi_over_side_x;
    for (int ny = ny_min; ny <= ny_max; ++ny) {
      const double ky = ny * two_pi_over_side_y;
      for (int nz = nz_min; nz <= nz_max; ++nz) {
        const double kz = nz * two_pi_over_side_z;

        const double wavenumber = sqrt(square(kx) + square(ky) + square(kz));
        if (wavenumber >= min_stirring_wavenumber_ and
            wavenumber <= max_stirring_wavenumber_) {
          double amplitude = 1.0;     // initialize to band amplitude
          if (spectrum_form_ == 1) {  // but if parabolic, correct value
            amplitude += parabolic_amplitude_prefactor *
                         square(wavenumber - parabolic_central_wavenumber);
          }

          fourier_data_.mode_amplitudes.push_back(amplitude);
          fourier_data_.mode_wavevectors.push_back({{kx, ky, kz}});

          fourier_data_.mode_amplitudes.push_back(amplitude);
          fourier_data_.mode_wavevectors.push_back({{kx, -ky, kz}});

          fourier_data_.mode_amplitudes.push_back(amplitude);
          fourier_data_.mode_wavevectors.push_back({{kx, ky, -kz}});

          fourier_data_.mode_amplitudes.push_back(amplitude);
          fourier_data_.mode_wavevectors.push_back({{kx, -ky, -kz}});

          number_of_modes_ = number_of_modes_ + 4;
        }
      }
    }
  }
}

template <>
void OuAnisotropicForcing<3>::initialize_ou_phases(
    const gsl::not_null<std::vector<double>*> phases) noexcept {
  double grnval;
  for (auto& elem : *phases) {
    grn(make_not_null(&grnval));
    elem = grnval * ou_variance_;
  }
}

template <>
void OuAnisotropicForcing<3>::update_ou_phases(
    const gsl::not_null<std::vector<double>*> phases) noexcept {
  double grnval;
  for (auto& elem : *phases) {
    grn(make_not_null(&grnval));
    elem =
        elem * ou_damping_factor_ + ou_variance_times_driving_factor_ * grnval;
  }
}

template <>
void OuAnisotropicForcing<3>::generate_ou_sequence() noexcept {
  const double initial_time = 0.0;
  const double final_time = 30.0;

  const auto sequence_size =
      static_cast<size_t>(1 + (final_time - initial_time) / ou_delta_t_);

  ou_sequence_.resize(sequence_size);

  // size is (2 * spatial dim * number of modes)
  std::vector<double> previous_phases(6 * number_of_modes_);
  this->initialize_ou_phases(make_not_null(&previous_phases));

  for (size_t time_id = 0; time_id < sequence_size; ++time_id) {
    this->update_ou_phases(make_not_null(&previous_phases));
    ou_sequence_[time_id] = previous_phases;
  }
}

template <>
std::array<double, 6> OuAnisotropicForcing<3>::stirring_phases_for_mode(
    const size_t time_id, const size_t mode_id) const noexcept {
  constexpr size_t dim = 3;
  std::array<double, 2 * dim> stirring_phases;
  for (size_t i = 0; i < dim; ++i) {
    const size_t index = 2 * (dim * mode_id + i);
    stirring_phases[2 * i] = ou_sequence_[time_id][index];
    stirring_phases[2 * i + 1] = ou_sequence_[time_id][index + 1];
  }

  // x-axis will be the "radial" axis containing anisotropies
  stirring_phases[0] *= anisotropy_factor_;
  stirring_phases[1] *= anisotropy_factor_;

  double k_dot_stirring_phases_for_re_eikx_over_k_squared = 0.0;
  double k_dot_stirring_phases_for_im_eikx_over_k_squared = 0.0;
  double one_over_k_squared = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    one_over_k_squared += square(fourier_data_.mode_wavevectors[mode_id][i]);
    k_dot_stirring_phases_for_re_eikx_over_k_squared +=
        fourier_data_.mode_wavevectors[mode_id][i] * stirring_phases[2 * i];
    k_dot_stirring_phases_for_im_eikx_over_k_squared +=
        fourier_data_.mode_wavevectors[mode_id][i] * stirring_phases[2 * i + 1];
  }
  one_over_k_squared = 1.0 / one_over_k_squared;
  k_dot_stirring_phases_for_re_eikx_over_k_squared *= one_over_k_squared;
  k_dot_stirring_phases_for_im_eikx_over_k_squared *= one_over_k_squared;

  // the following expressions are only valid for solenoidal modes.
  for (size_t i = 0; i < dim; ++i) {
    stirring_phases[2 * i] =
        stirring_phases[2 * i] -
        fourier_data_.mode_wavevectors[mode_id][i] *
            k_dot_stirring_phases_for_re_eikx_over_k_squared;
    stirring_phases[2 * i + 1] =
        stirring_phases[2 * i + 1] -
        fourier_data_.mode_wavevectors[mode_id][i] *
            k_dot_stirring_phases_for_im_eikx_over_k_squared;
  }
  return stirring_phases;
}

template <>
void OuAnisotropicForcing<3>::compute_acceleration(
    const gsl::not_null<tnsr::I<DataVector, 3>*> acceleration,
    const tnsr::I<DataVector, 3>& x, const double t,
    const FourierData& fourier_data) const noexcept {
  constexpr size_t dim = 3;
  const double initial_time = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    acceleration->get(i) = 0.0;
  }
  const auto time_id = static_cast<size_t>((t - initial_time) / ou_delta_t_);
  for (size_t mode = 0; mode < number_of_modes_; ++mode) {
    // store k_dot_x here to save allocation later
    DataVector sin_k_dot_x = fourier_data.mode_wavevectors[mode][0] * get<0>(x);
    for (size_t i = 1; i < dim; ++i) {
      sin_k_dot_x += fourier_data.mode_wavevectors[mode][i] * x.get(i);
    }
    const DataVector cos_k_dot_x = cos(sin_k_dot_x);
    sin_k_dot_x = sin(sin_k_dot_x);
    const auto stirring_phases_mode = stirring_phases_for_mode(time_id, mode);
    for (size_t i = 0; i < dim; ++i) {
      acceleration->get(i) += fourier_data.mode_amplitudes[mode] *
                              (stirring_phases_mode[2 * i] * cos_k_dot_x -
                               stirring_phases_mode[2 * i + 1] * sin_k_dot_x);
    }
  }
  for (size_t i = 0; i < dim; ++i) {
    acceleration->get(i) *= solenoidal_weight_norm_;
  }
}

template <>
OuAnisotropicForcing<3>::OuAnisotropicForcing(
    const double ou_delta_t, const size_t spectrum_form,
    const double decay_time, const double energy_input_per_mode,
    const double min_stirring_wavenumber, const double max_stirring_wavenumber,
    const double solenoidal_weight, const double anisotropy_factor,
    const int seed_for_rng) noexcept
    : ou_delta_t_(ou_delta_t),
      spectrum_form_(spectrum_form),
      decay_time_(decay_time),
      energy_input_per_mode_(energy_input_per_mode),
      min_stirring_wavenumber_(min_stirring_wavenumber),
      max_stirring_wavenumber_(max_stirring_wavenumber),
      solenoidal_weight_(solenoidal_weight),
      anisotropy_factor_(anisotropy_factor),
      seed_for_rng_(seed_for_rng) {
  ASSERT(spectrum_form_ == 0 or spectrum_form_ == 1,
         "Spectrum form must be 0 (band) or 1 (parabolic). Value given: "
             << spectrum_form_);

  const size_t dim = 3;
  // Next expression is valid for any dim. Denominator equals norm
  // of the full projection operator (Eqn. (8) of Federrath et al. 2010)
  solenoidal_weight_norm_ =
      2.0 * sqrt(3.0 / dim) * sqrt(3.0) * 1.0 /
      sqrt(1.0 - 2.0 * solenoidal_weight_ + dim * square(solenoidal_weight_));

  // number_of_modes_ is initialized here
  initialize_stirring_modes();

  ou_variance_ = sqrt(energy_input_per_mode_ / decay_time_);
  ou_damping_factor_ = exp(-ou_delta_t_ / decay_time_);
  ou_variance_times_driving_factor_ =
      ou_variance_ * sqrt(1.0 - square(ou_damping_factor_));

  // Initializes ou_sequence_.
  // Uses number_of_modes_, so needs to go after stirring modes initialization
  generate_ou_sequence();
}

template <size_t Dim>
void OuAnisotropicForcing<Dim>::pup(PUP::er& p) noexcept {
  p | ou_delta_t_;
  p | spectrum_form_;
  p | decay_time_;
  p | energy_input_per_mode_;
  p | min_stirring_wavenumber_;
  p | max_stirring_wavenumber_;
  p | solenoidal_weight_;
  p | anisotropy_factor_;
  p | seed_for_rng_;
  p | solenoidal_weight_norm_;
  p | number_of_modes_;
  p | ou_variance_;
  p | ou_damping_factor_;
  p | ou_variance_times_driving_factor_;
  p | ou_sequence_;
  p | fourier_data_;
}

template <>
void OuAnisotropicForcing<3>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> source_energy_density,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::I<DataVector, 3>& x, const double t) const noexcept {
  constexpr size_t dim = 3;
  // save allocation by storing acceleration in momentum density source
  compute_acceleration(source_momentum_density, x, t, fourier_data_);
  get(*source_energy_density) = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    get(*source_energy_density) +=
        source_momentum_density->get(i) * momentum_density.get(i);
    source_momentum_density->get(i) *= get(mass_density_cons);
  }
}

template <size_t Dim>
bool operator==(const OuAnisotropicForcing<Dim>& lhs,
                const OuAnisotropicForcing<Dim>& rhs) noexcept {
  return lhs.ou_delta_t_ == rhs.ou_delta_t_ and
         lhs.spectrum_form_ == rhs.spectrum_form_ and
         lhs.decay_time_ == rhs.decay_time_ and
         lhs.energy_input_per_mode_ == rhs.energy_input_per_mode_ and
         lhs.min_stirring_wavenumber_ == rhs.min_stirring_wavenumber_ and
         lhs.max_stirring_wavenumber_ == rhs.max_stirring_wavenumber_ and
         lhs.solenoidal_weight_ == rhs.solenoidal_weight_ and
         lhs.anisotropy_factor_ == rhs.anisotropy_factor_ and
         lhs.seed_for_rng_ == rhs.seed_for_rng_ and
         lhs.solenoidal_weight_norm_ == rhs.solenoidal_weight_norm_ and
         lhs.number_of_modes_ == rhs.number_of_modes_ and
         lhs.ou_variance_ == rhs.ou_variance_ and
         lhs.ou_damping_factor_ == rhs.ou_damping_factor_ and
         lhs.ou_variance_times_driving_factor_ ==
             rhs.ou_variance_times_driving_factor_ and
         lhs.ou_sequence_ == rhs.ou_sequence_ and
         lhs.fourier_data_ == rhs.fourier_data_;
}

template <size_t Dim>
bool operator!=(const OuAnisotropicForcing<Dim>& lhs,
                const OuAnisotropicForcing<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template struct OuAnisotropicForcing<DIM(data)>;                           \
  template bool operator==(const OuAnisotropicForcing<DIM(data)>&,           \
                           const OuAnisotropicForcing<DIM(data)>&) noexcept; \
  template bool operator!=(const OuAnisotropicForcing<DIM(data)>&,           \
                           const OuAnisotropicForcing<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (3))

#undef INSTANTIATE
#undef DIM
}  // namespace Sources
}  // namespace NewtonianEuler
/// \endcond
