// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Domain.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/OuAnisotropicForcing.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

template <size_t Dim>
void test_sources() noexcept;

template <>
void test_sources<3>() noexcept {
  const double ou_delta_t = 0.05;
  const size_t spectrum_form = 1;
  const double decay_time = 0.5;
  const double energy_input_per_mode = 0.002;
  const double min_stirring_wavenumber = 6.283;
  const double max_stirring_wavenumber = 18.95;
  const double solenoidal_weight = 1.0;
  const double anisotropy_factor = 1.0;
  const int seed_for_rng = 140281;

  const size_t dim = 3;

  NewtonianEuler::Sources::OuAnisotropicForcing<dim> source(
      ou_delta_t, spectrum_form, decay_time, energy_input_per_mode,
      min_stirring_wavenumber, max_stirring_wavenumber, solenoidal_weight,
      anisotropy_factor, seed_for_rng);

  NewtonianEuler::Sources::OuAnisotropicForcing<3> source_to_move(
      ou_delta_t, spectrum_form, decay_time, energy_input_per_mode,
      min_stirring_wavenumber, max_stirring_wavenumber, solenoidal_weight,
      anisotropy_factor, seed_for_rng);

  test_move_semantics(std::move(source_to_move), source);  // NOLINT
  test_serialization(source);

  // Check acceleration on unit cube centered in origin with isotropic mesh
  domain::creators::Brick brick({{-0.5, -0.5, -0.5}}, {{0.5, 0.5, 0.5}},
                                {{false, false, false}}, {{0, 0, 0}},
                                {{4, 4, 4}});

  // In order to reproduce FLASH's checks, we test acceleration at
  // Chebyshev points. (Quadrature method is irrelevant; any can be used.)
  Mesh<dim> mesh{brick.initial_extents()[0], Spectral::Basis::Chebyshev,
                 Spectral::Quadrature::GaussLobatto};

  const auto domain = brick.create_domain();
  const auto x_logical = logical_coordinates(mesh);
  const auto x = domain.blocks()[0].stationary_map()(x_logical);

  const DataVector used_for_size(4 * 4 * 4);
  tnsr::I<DataVector, dim> source_momentum_density(used_for_size);
  Scalar<DataVector> source_energy_density(used_for_size);

  // Check for unit mass density, so that the source for the momentum density
  // is simply the acceleration.
  const auto mass_density =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  const auto momentum_density =
      make_with_value<tnsr::I<DataVector, dim>>(used_for_size, 1.0);

  // Check initial acceleration field. OU phases should be updated once.
  double time = 0.0;
  source.apply(make_not_null(&source_momentum_density),
               make_not_null(&source_energy_density), mass_density,
               momentum_density, x, time);

  // Numbers directly taken from FLASH output
  tnsr::I<DataVector, dim> expected_acceleration(used_for_size);
  get<0>(expected_acceleration) = DataVector{
      0.359558463,  -0.354885340, -0.257174045, 0.359558642,  0.244305700,
      -0.528956354, -0.893715918, 0.244305730,  0.921816826,  1.180804372,
      0.017493542,  0.921817124,  0.359558821,  -0.354885131, -0.257173926,
      0.359558821,  0.702402115,  -0.170512423, 0.071722075,  0.702402055,
      0.198517114,  0.171157092,  -0.186277568, 0.198517039,  -0.988990426,
      -1.387112856, -0.409688264, -0.988990605, 0.702402472,  -0.170512050,
      0.071722299,  0.702402413,  -0.465478778, -0.444989741, 0.462594211,
      -0.465478957, -0.260842085, 0.381776005,  -0.469114870, -0.260841966,
      -0.018607603, -0.633787274, -0.431185365, -0.018607697, -0.465478539,
      -0.444989532, 0.462594539,  -0.465478450, 0.359558791,  -0.354885697,
      -0.257173985, 0.359558791,  0.244305894,  -0.528956175, -0.893715918,
      0.244305760,  0.921817422,  1.180804372,  0.017493706,  0.921817303,
      0.359558791,  -0.354885459, -0.257173896, 0.359559029};
  get<1>(expected_acceleration) = DataVector{
      0.192679733,  0.077039503,  -0.553771734, 0.192679405,  -0.805521429,
      1.347630858,  -0.145878747, -0.805521429, 0.767339468,  -0.145981863,
      -1.629491091, 0.767339468,  0.192679435,  0.077039666,  -0.553771436,
      0.192679301,  -0.184885159, -0.592708409, -0.168027565, -0.184884995,
      -0.170858353, 1.218118787,  -0.515849948, -0.170858234, 0.512308657,
      -0.486582845, 1.110558867,  0.512308776,  -0.184885129, -0.592708230,
      -0.168027639, -0.184885293, -0.144760296, -0.421522737, 0.645737886,
      -0.144760266, 0.216090575,  -1.424385786, 0.040519584,  0.216090485,
      -0.459312499, -1.083575487, 0.338395327,  -0.459312379, -0.144760191,
      -0.421522945, 0.645738006,  -0.144760281, 0.192679867,  0.077039905,
      -0.553771853, 0.192679659,  -0.805521250, 1.347631216,  -0.145878911,
      -0.805521369, 0.767339468,  -0.145981759, -1.629491091, 0.767339647,
      0.192679614,  0.077039830,  -0.553771675, 0.192679465};
  get<2>(expected_acceleration) = DataVector{
      -0.827792943, 0.457538515,  0.378999263,  -0.827793002, -0.142222717,
      0.445413351,  1.664377332,  -0.142223135, -0.215157911, 0.207674593,
      0.574851394,  -0.215158060, -0.827792943, 0.457538873,  0.378999174,
      -0.827792943, 1.406713367,  0.067447267,  -0.049583845, 1.406713963,
      0.284058124,  -1.000855565, -0.984913588, 0.284058154,  -0.256729960,
      -0.818216801, -1.038423896, -0.256729960, 1.406713128,  0.067447037,
      -0.049583837, 1.406713963,  -0.541639745, 0.087746866,  0.034075834,
      -0.541639984, 0.908880472,  1.159749746,  1.073274016,  0.908879817,
      -0.858328879, 0.585866511,  0.358794093,  -0.858329058, -0.541639745,
      0.087747060,  0.034075581,  -0.541640043, -0.827792823, 0.457538515,
      0.378999174,  -0.827792883, -0.142222762, 0.445413262,  1.664377093,
      -0.142223209, -0.215157926, 0.207674742,  0.574851155,  -0.215158015,
      -0.827792823, 0.457538873,  0.378999114,  -0.827792943};

  // FLASH doesn't use double precision, so we must use a larger tolerance
  Approx larger_approx = Approx::custom().epsilon(1.e-4);
  // For 1.e-5, 8/208 of the following checks fail.
  CHECK_ITERABLE_CUSTOM_APPROX(source_momentum_density, expected_acceleration,
                               larger_approx);

  // Evolving the source term to a time < ou_delta_t should not modify
  // the acceleration field
  time = 0.8 * ou_delta_t;
  source.apply(make_not_null(&source_momentum_density),
               make_not_null(&source_energy_density), mass_density,
               momentum_density, x, time);

  CHECK_ITERABLE_CUSTOM_APPROX(source_momentum_density, expected_acceleration,
                               larger_approx);

  // Evolving the source term to a time > 7 * ou_delta_ should make
  // the OU phases be updated 7 extra times, and the acceleration field
  // should be updated accordingly
  time = 7.1 * ou_delta_t;
  source.apply(make_not_null(&source_momentum_density),
               make_not_null(&source_energy_density), mass_density,
               momentum_density, x, time);

  // Expected updated acceleration taken from FLASH output
  get<0>(expected_acceleration) = DataVector{
      0.723793566,  0.372507215,  0.330281019,  0.723793805,  -0.596015871,
      -0.630119026, -1.534743905, -0.596015692, -0.015715929, -0.340293288,
      0.032995611,  -0.015716089, 0.723793447,  0.372507244,  0.330280691,
      0.723793685,  -0.065003127, -0.065508090, 0.235098243,  -0.065003261,
      0.337305814,  0.664957583,  0.749031186,  0.337305963,  0.157680824,
      0.138813749,  0.179712728,  0.157680735,  -0.065003283, -0.065508269,
      0.235098064,  -0.065003276, 0.806331098,  0.634324491,  -0.161210835,
      0.806331217,  -0.116751157, -1.416200638, 1.052628398,  -0.116751283,
      0.009807470,  -1.076503873, -0.069040470, 0.009807314,  0.806330979,
      0.634324849,  -0.161210746, 0.806331098,  0.723793387,  0.372506708,
      0.330281049,  0.723793566,  -0.596015871, -0.630118728, -1.534743667,
      -0.596015632, -0.015715666, -0.340293050, 0.032995537,  -0.015715836,
      0.723793447,  0.372506738,  0.330280721,  0.723793447};
  get<1>(expected_acceleration) = DataVector{
      1.962321401,  0.055314653,  -1.098832250, 1.962321281,  1.088863611,
      0.109190583,  0.565828562,  1.088863254,  1.639242649,  -0.683996856,
      -0.908741295, 1.639242649,  1.962321401,  0.055314582,  -1.098831892,
      1.962321401,  -0.671193182, -1.326146007, -0.309497535, -0.671193421,
      -1.791109085, 0.137599155,  -1.248979330, -1.791109681, -0.253643811,
      -0.489285469, 0.257918030,  -0.253644139, -0.671193480, -1.326146007,
      -0.309497774, -0.671193600, -0.137558982, -1.661292195, -0.634115696,
      -0.137559146, -0.740324736, -0.758530736, -1.025721312, -0.740324438,
      -0.491052836, -1.877454400, 0.320271283,  -0.491052866, -0.137559131,
      -1.661292076, -0.634115815, -0.137559175, 1.962321281,  0.055314489,
      -1.098832488, 1.962320924,  1.088863611,  0.109190948,  0.565828800,
      1.088863373,  1.639242291,  -0.683997035, -0.908741415, 1.639242530,
      1.962321401,  0.055314582,  -1.098832130, 1.962321401};
  get<2>(expected_acceleration) = DataVector{
      -0.848850548, 0.466904759,  0.839702904,  -0.848850369, -0.572037995,
      1.561245084,  1.088881493,  -0.572038054, -0.069203928, 0.177758738,
      0.075625449,  -0.069204085, -0.848850667, 0.466904938,  0.839702845,
      -0.848850608, 0.721365273,  0.322648525,  0.116335809,  0.721365690,
      -0.858567953, 0.110452667,  -0.507351875, -0.858568072, 0.173544899,
      -0.178324863, -0.245339781, 0.173544973,  0.721365094,  0.322648615,
      0.116335914,  0.721365631,  -0.360681891, -0.665434837, -0.164414912,
      -0.360681921, -0.279678464, 1.217414021,  0.459692210,  -0.279678702,
      -0.427179575, 0.777880192,  -0.092401579, -0.427179426, -0.360681981,
      -0.665434837, -0.164415002, -0.360681891, -0.848850667, 0.466904819,
      0.839702904,  -0.848850548, -0.572038054, 1.561245084,  1.088881493,
      -0.572038174, -0.069203869, 0.177758604,  0.075625628,  -0.069204047,
      -0.848850787, 0.466904819,  0.839702785,  -0.848850727};

  // For larger_approx = Approx::custom().epsilon(1.e-5),
  // 13/208 of the following checks fail.
  CHECK_ITERABLE_CUSTOM_APPROX(source_momentum_density, expected_acceleration,
                               larger_approx);

  // Check at another later time. Acceleration should be updated.
  time = 100.0 * ou_delta_t;
  source.apply(make_not_null(&source_momentum_density),
               make_not_null(&source_energy_density), mass_density,
               momentum_density, x, time);

  // Expected updated acceleration taken from FLASH output
  get<0>(expected_acceleration) = DataVector{
      -0.556914151, -0.637148321, 0.053436439,  -0.556914151, -1.300711274,
      -0.937246144, 0.760500491,  -1.300711989, 1.244518280,  1.056457996,
      1.207041025,  1.244518161,  -0.556914449, -0.637148976, 0.053436264,
      -0.556914330, 0.813670874,  1.720007658,  1.052539945,  0.813670695,
      0.286147565,  -0.018342767, 0.691423297,  0.286147445,  -1.148079395,
      -1.193653464, -1.241400719, -1.148079872, 0.813670635,  1.720007181,
      1.052539945,  0.813670933,  0.243927836,  0.105125152,  -0.700930417,
      0.243927881,  0.023006566,  1.093086123,  0.540652871,  0.023006717,
      0.290858865,  0.112713486,  -0.438424498, 0.290858924,  0.243927851,
      0.105124943,  -0.700929999, 0.243927881,  -0.556913912, -0.637147903,
      0.053436443,  -0.556913912, -1.300711274, -0.937245548, 0.760500610,
      -1.300711513, 1.244517922,  1.056458235,  1.207040906,  1.244518399,
      -0.556914210, -0.637148619, 0.053436331,  -0.556914449};
  get<1>(expected_acceleration) = DataVector{
      -2.688777924, -1.148483753, -0.510826468, -2.688778639, 0.191100329,
      0.044548564,  -1.549178839, 0.191100299,  -2.174107552, -0.377515256,
      -1.178269625, -2.174107075, -2.688777924, -1.148483992, -0.510826528,
      -2.688778400, 0.496389538,  1.177015781,  1.717574239,  0.496389389,
      0.811947703,  0.057582729,  1.377627492,  0.811947703,  1.001319766,
      2.310067892,  1.530762196,  1.001319885,  0.496389300,  1.177015305,
      1.717574000,  0.496389091,  1.198991179,  0.619376838,  0.290079445,
      1.198991060,  1.402981758,  0.626666903,  -0.120638661, 1.402981400,
      1.214308739,  1.405632138,  0.837993503,  1.214308381,  1.198991537,
      0.619377196,  0.290079415,  1.198991418,  -2.688778162, -1.148483753,
      -0.510826647, -2.688779116, 0.191099912,  0.044548597,  -1.549178958,
      0.191100031,  -2.174107790, -0.377514869, -1.178269982, -2.174107075,
      -2.688778400, -1.148483515, -0.510826886, -2.688778400};
  get<2>(expected_acceleration) = DataVector{
      -0.245502219, -1.247855663, -0.251526356, -0.245502129, 1.723744869,
      0.346356243,  -0.223153710, 1.723744750,  0.393881202,  -0.023674408,
      -0.375756800, 0.393881232,  -0.245502263, -1.247855544, -0.251526177,
      -0.245502114, -1.230679154, -0.292164028, -0.410028219, -1.230678678,
      1.077987909,  -0.353959471, -0.823592424, 1.077988029,  0.166739568,
      2.291502237,  -0.060134519, 0.166739792,  -1.230679154, -0.292164445,
      -0.410028070, -1.230679154, 0.639072835,  -0.504944444, -1.747784019,
      0.639073014,  0.173869312,  0.957803845,  0.566286147,  0.173869759,
      0.209283128,  -1.050804377, 0.273862958,  0.209282935,  0.639072776,
      -0.504944324, -1.747784019, 0.639072895,  -0.245502427, -1.247855663,
      -0.251526266, -0.245502234, 1.723744869,  0.346356302,  -0.223153830,
      1.723745227,  0.393881053,  -0.023674086, -0.375756860, 0.393881261,
      -0.245502397, -1.247855544, -0.251526088, -0.245502219};

  // For larger_approx = Approx::custom().epsilon(1.e-5),
  // 7/208 of the following checks fail.
  CHECK_ITERABLE_CUSTOM_APPROX(source_momentum_density, expected_acceleration,
                               larger_approx);

  // One last time. Source terms should not change.
  time = 100.8 * ou_delta_t;
  source.apply(make_not_null(&source_momentum_density),
               make_not_null(&source_energy_density), mass_density,
               momentum_density, x, time);

  CHECK_ITERABLE_CUSTOM_APPROX(source_momentum_density, expected_acceleration,
                               larger_approx);
}

template <size_t Dim>
void test_anisotropic_modes() noexcept;

template <>
void test_anisotropic_modes<3>() noexcept {
  const double ou_delta_t = 0.1;
  const size_t spectrum_form = 1;
  const double decay_time = 0.5;
  const double energy_input_per_mode = 0.000074;
  const double min_stirring_wavenumber = 6.2832;
  const double max_stirring_wavenumber = 12.56637;
  const double solenoidal_weight = 1.0;
  const double anisotropy_factor = 2.0;
  const int seed_for_rng = 140281;

  const double two_pi = 2.0 * M_PI;
  const double k_c = 0.5 * (min_stirring_wavenumber + max_stirring_wavenumber);
  const auto expected_amplitude_from_wavevector =
      [&min_stirring_wavenumber, &max_stirring_wavenumber,
       &k_c ](const std::array<double, 3>& vector_k) noexcept {
    const double k =
        sqrt(square(vector_k[0]) + square(vector_k[1]) + square(vector_k[2]));
    return 1.0 - 4.0 * square((k - k_c) / (max_stirring_wavenumber -
                                           min_stirring_wavenumber));
  };

  // Set expected Fourier data for later comparison
  // Full mode set is expected to contain 4*4 = 16 modes.
  const size_t expected_number_of_modes = 16;
  NewtonianEuler::Sources::OuAnisotropicForcing<3>::FourierData
      expected_fourier_data(expected_number_of_modes);
  expected_fourier_data.mode_wavevectors[0] = {{0., two_pi, two_pi}};
  expected_fourier_data.mode_wavevectors[1] = {{0., -two_pi, two_pi}};
  expected_fourier_data.mode_wavevectors[2] = {{0., two_pi, -two_pi}};
  expected_fourier_data.mode_wavevectors[3] = {{0., -two_pi, -two_pi}};
  expected_fourier_data.mode_wavevectors[4] = {{two_pi, 0., two_pi}};
  expected_fourier_data.mode_wavevectors[5] = {{two_pi, 0., two_pi}};
  expected_fourier_data.mode_wavevectors[6] = {{two_pi, 0., -two_pi}};
  expected_fourier_data.mode_wavevectors[7] = {{two_pi, 0., -two_pi}};
  expected_fourier_data.mode_wavevectors[8] = {{two_pi, two_pi, 0.}};
  expected_fourier_data.mode_wavevectors[9] = {{two_pi, -two_pi, 0.}};
  expected_fourier_data.mode_wavevectors[10] = {{two_pi, two_pi, 0.}};
  expected_fourier_data.mode_wavevectors[11] = {{two_pi, -two_pi, 0.}};
  expected_fourier_data.mode_wavevectors[12] = {{two_pi, two_pi, two_pi}};
  expected_fourier_data.mode_wavevectors[13] = {{two_pi, -two_pi, two_pi}};
  expected_fourier_data.mode_wavevectors[14] = {{two_pi, two_pi, -two_pi}};
  expected_fourier_data.mode_wavevectors[15] = {{two_pi, -two_pi, -two_pi}};
  for (size_t mode = 0; mode < expected_number_of_modes; ++mode) {
    expected_fourier_data.mode_amplitudes[mode] =
        expected_amplitude_from_wavevector(
            expected_fourier_data.mode_wavevectors[mode]);
  }

  NewtonianEuler::Sources::OuAnisotropicForcing<3> source(
      ou_delta_t, spectrum_form, decay_time, energy_input_per_mode,
      min_stirring_wavenumber, max_stirring_wavenumber, solenoidal_weight,
      anisotropy_factor, seed_for_rng);

  // computation of sources modify members so test move semantics
  // and serialization before anything else
  NewtonianEuler::Sources::OuAnisotropicForcing<3> source_to_move(
      ou_delta_t, spectrum_form, decay_time, energy_input_per_mode,
      min_stirring_wavenumber, max_stirring_wavenumber, solenoidal_weight,
      anisotropy_factor, seed_for_rng);

  test_move_semantics(std::move(source_to_move), source);  // NOLINT
  test_serialization(source);

  const auto& fourier_data = source.fourier_data();

  const DataVector used_for_size(4);
  tnsr::I<DataVector, 3> source_momentum_density(used_for_size);
  Scalar<DataVector> source_energy_density(used_for_size);
  const auto mass_density =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  const auto momentum_density =
      make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 1.0);

  auto grid_points =
      make_with_value<tnsr::I<DataVector, 3>>(used_for_size, -1.0);
  get<0>(grid_points)[1] = -0.447214;
  get<0>(grid_points)[2] = 0.447214;
  get<0>(grid_points)[3] = 1.0;

  tnsr::I<DataVector, 3> expected_acceleration(used_for_size);
  // at x = -1, -0.447214, 0.447214, and 1, respectively
  get<0>(expected_acceleration) =
      DataVector{0.239302, 0.00465424, 0.0532466, 0.239302};
  get<1>(expected_acceleration) =
      DataVector{-0.0272266, 0.136412, 0.0519921, -0.0272267};
  get<2>(expected_acceleration) =
      DataVector{-0.00877922, -0.048271, -0.133017, -0.00877926};

  // OuAnisotropicForcing assumes that evolution starts at t = 0.0
  double time = 0.0;
  source.apply(make_not_null(&source_momentum_density),
               make_not_null(&source_energy_density), mass_density,
               momentum_density, grid_points, time);

  // inititialized in constructor, never updated throughout simulation
  CHECK(fourier_data.mode_wavevectors ==
        expected_fourier_data.mode_wavevectors);
  CHECK(fourier_data.mode_amplitudes == expected_fourier_data.mode_amplitudes);
  // Acceleration is stored in source_momentum_density, so, since mass
  // density = 1.0, both should be equal
  CHECK(source_momentum_density == expected_acceleration);

  time = 0.05;
  // since after the first update next_time_to_update_modes_ = ou_delta_t_
  // and thus time = 0.05 < next_time_to_update_modes_, the sources should be
  // computed with the same Fourier data, i.e., nothing should change
  source.apply(make_not_null(&source_momentum_density),
               make_not_null(&source_energy_density), mass_density,
               momentum_density, grid_points, time);

  CHECK(fourier_data.mode_wavevectors ==
        expected_fourier_data.mode_wavevectors);
  CHECK(fourier_data.mode_amplitudes == expected_fourier_data.mode_amplitudes);
  CHECK(source_momentum_density == expected_acceleration);

  time = 0.1;
  // now time = next_time_to_update_modes_ so Fourier data (specifically
  // stirring_phases_for_re_eikx and stirring_phases_for_im_eikx) should be
  // updated, and next_time_to_update_modes = next_time_to_update_modes_ +
  // ou_delta_t_;
  source.apply(make_not_null(&source_momentum_density),
               make_not_null(&source_energy_density), mass_density,
               momentum_density, grid_points, time);

  // check first mode only
  NewtonianEuler::Sources::OuAnisotropicForcing<3>::FourierData
      new_expected_data(1);
  new_expected_data.mode_amplitudes[0] = 0.9705618361651783;
  new_expected_data.mode_wavevectors[0] = {{0, two_pi, two_pi}};

  CHECK(fourier_data.mode_wavevectors[0] ==
        new_expected_data.mode_wavevectors[0]);
  CHECK(fourier_data.mode_amplitudes[0] ==
        new_expected_data.mode_amplitudes[0]);

  time = 0.15;
  // now time < next_time_to_update_modes_ so, again, nothing should change
  source.apply(make_not_null(&source_momentum_density),
               make_not_null(&source_energy_density), mass_density,
               momentum_density, grid_points, time);

  // check first mode only
  CHECK(fourier_data.mode_wavevectors[0] ==
        new_expected_data.mode_wavevectors[0]);
  CHECK(fourier_data.mode_amplitudes[0] ==
        new_expected_data.mode_amplitudes[0]);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Sources.OuAnisotropicForcing",
    "[Unit][Evolution]") {
  test_sources<3>();
  // test_anisotropic_modes<3>();
}
