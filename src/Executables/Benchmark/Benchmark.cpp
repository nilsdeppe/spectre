// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop
#include <string>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Structure/Element.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

// This file is an example of how to do microbenchmark with Google Benchmark
// https://github.com/google/benchmark
// For two examples in different anonymous namespaces

namespace {
// Benchmark of push_back() in std::vector, following Chandler Carruth's talk
// at CppCon in 2015,
// https://www.youtube.com/watch?v=nXaxk27zwlk

// void bench_create(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    benchmark::DoNotOptimize(&v);
//    static_cast<void>(v);
//  }
// }
// BENCHMARK(bench_create);

// void bench_reserve(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//  }
// }
// BENCHMARK(bench_reserve);

// void bench_push_back(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//    v.push_back(42);
//    benchmark::ClobberMemory();
//  }
// }
// BENCHMARK(bench_push_back);
}  // namespace

namespace {
// In this anonymous namespace is an example of microbenchmarking the
// all_gradient routine for the GH system

// template <size_t Dim>
// struct Kappa : db::SimpleTag {
//   using type = tnsr::abb<DataVector, Dim, Frame::Grid>;
// };
// template <size_t Dim>
// struct Psi : db::SimpleTag {
//   using type = tnsr::aa<DataVector, Dim, Frame::Grid>;
// };

// // clang-tidy: don't pass be non-const reference
// void bench_all_gradient(benchmark::State& state) {  // NOLINT
//   constexpr const size_t pts_1d = 4;
//   constexpr const size_t Dim = 3;
//   const Mesh<Dim> mesh{pts_1d, Spectral::Basis::Legendre,
//                        Spectral::Quadrature::GaussLobatto};
//   domain::CoordinateMaps::Affine map1d(-1.0, 1.0, -1.0, 1.0);
//   using Map3d =
//       domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
//                                              domain::CoordinateMaps::Affine,
//                                              domain::CoordinateMaps::Affine>;
//   domain::CoordinateMap<Frame::ElementLogical, Frame::Grid, Map3d> map(
//       Map3d{map1d, map1d, map1d});

//   using VarTags = tmpl::list<Kappa<Dim>, Psi<Dim>>;
//   const InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Grid>
//       inv_jac = map.inv_jacobian(logical_coordinates(mesh));
//   const auto grid_coords = map(logical_coordinates(mesh));
//   Variables<VarTags> vars(mesh.number_of_grid_points(), 0.0);

//   while (state.KeepRunning()) {
//     benchmark::DoNotOptimize(partial_derivatives<VarTags>(vars, mesh,
//     inv_jac));
//   }
// }
// BENCHMARK(bench_all_gradient);  // NOLINT

// void benchmark_fix_cons(benchmark::State& state) {
//   const size_t num_points = 1331;
//   Scalar<DataVector> tilde_d{DataVector(num_points)};
//   Scalar<DataVector> tilde_tau{DataVector(num_points)};
//   auto tilde_s =
//       make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
//   auto tilde_b =
//       make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
//   for (size_t i = 0; i < num_points; ++i) {
//     if (i % 2 == 0) {
//       get(tilde_d)[i] = 2.e-12;
//       get(tilde_tau)[i] = 4.5;
//       tilde_s.get(0)[i] = 3.0;
//       tilde_b.get(0)[i] = 2.0;
//     } else {
//       get(tilde_d)[i] = 1.0;
//       get(tilde_tau)[i] = 1.5;
//       tilde_s.get(0)[i] = 0.0;
//       tilde_b.get(0)[i] = 2.0;
//     }
//   }

//   auto spatial_metric =
// make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
//   auto inv_spatial_metric =
// make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
//   auto sqrt_det_spatial_metric =
//       make_with_value<Scalar<DataVector>>(tilde_d, 1.0);
//   for (size_t d = 0; d < 3; ++d) {
//     spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
//     inv_spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
//   }

//   const grmhd::ValenciaDivClean::FixConservatives variable_fixer{
//       1.e-12, 1.0e-11, 0.0, 0.0};

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(
// variable_fixer(&tilde_d, &tilde_tau, &tilde_s, tilde_b, spatial_metric,
//                        inv_spatial_metric, sqrt_det_spatial_metric));
//   }
// }
// BENCHMARK(benchmark_fix_cons);

// void benchmark_cubic_roots_simd(benchmark::State& state) {
//   enable_floating_point_exceptions();
//   // TODO: make unified with double type. This needs a type trait to either
//   //       grab value_type or return T, I think.
//   //
//   // TODO: check if return by reference speeds things up at all.
//   //
//   // TODO: Remove simd wrapper lib?
//   //
//   // x3 - 6x2 + 11x - 6
//   using SimdType = typename xsimd::make_sized_batch<double, 8>::type;
//   const auto f = [](const auto& x) {
//     return xsimd::fms(x, xsimd::fma(x, (x - 6.0), SimdType(11.0)),
//                       SimdType(6.0));
//   };
//   const SimdType lower_bound(1.5);
//   SimdType upper_bound{};
//   // const SimdType upper_bound(2.5);
//   if (state.range(0) == 0) {
//     upper_bound = SimdType(2.5 + 1.0e-100) + 1.0e-8;
//   } else if (state.range(0) == 1) {
//     upper_bound = SimdType(2.1 + 1.0e-10, 2.1 + 1.0e-10, 2.1 + 1.0e-10,
//                            2.1 + 1.0e-10, 2.5, 2.5, 2.5, 2.5) +
//                   1.0e-8;
//   } else if (state.range(0) == 2) {
//     upper_bound = SimdType(2.1 + 1.0e-10, 2.2 + 1.0e-10, 2.3 + 1.0e-10,
//                            2.4 + 1.0e-10, 2.5, 2.6, 2.7, 2.8) +
//                   1.0e-8;
//   } else if (state.range(0) == 3) {
//     const double base = 2.8 + 1.0e-8;
//     upper_bound = SimdType(base + 1.0e-10, base + 2.0e-10, base + 3.0e-10,
//                            base + 4.0e-10, base + 5.0e-10, base + 6.0e-10,
//                            base + 7.0e-10, base + 8.0e-10);
//   }
//   const auto f_at_lower_bound = f(lower_bound);
//   const auto f_at_upper_bound = f(upper_bound);
//   for (auto _ : state) {
//     for (size_t i = 0; i < (xsimd::batch<double>::size / SimdType::size);
//     ++i) {
//       benchmark::DoNotOptimize(
//           RootFinder::toms748(f, lower_bound, upper_bound, f_at_lower_bound,
//                               f_at_upper_bound, 1.0e-14, 1.0e-12));
//     }
//   }
// }
// BENCHMARK(benchmark_cubic_roots_simd)  //
//     ->DenseRange(0, 3, 1);             //

// // ->Iterations(2515925);

// void benchmark_cubic_roots_scalar(benchmark::State& state) {
//   enable_floating_point_exceptions();
//   // x3 - 6x2 + 11x - 6
//   const auto f = [](const auto& x) {
//     using std::fma;
//     using xsimd::fma;
//     using Type = std::decay_t<decltype(x)>;
//     return fma(x, fma(x, (x - 6.0), Type(11.0)), Type(-6.0));
//   };
//   // const double lower_bound(1.5);
//   // const double upper_bound(2.1 + 1.0e-10);
//   // const auto f_at_lower_bound = f(lower_bound);
//   // const auto f_at_upper_bound = f(upper_bound);
//   using SimdType = typename xsimd::make_sized_batch<double, 8>::type;
//   const SimdType lower_bound(1.5);
//   SimdType upper_bound{};
//   // const SimdType upper_bound(2.5);
//   if (state.range(0) == 0) {
//     upper_bound = SimdType(2.5 + 1.0e-100) + 1.0e-8;
//   } else if (state.range(0) == 1) {
//     upper_bound = SimdType(2.1 + 1.0e-10, 2.1 + 1.0e-10, 2.1 + 1.0e-10,
//                            2.1 + 1.0e-10, 2.5, 2.5, 2.5, 2.5) +
//                   1.0e-8;
//   } else if (state.range(0) == 2) {
//     upper_bound = SimdType(2.1 + 1.0e-10, 2.2 + 1.0e-10, 2.3 + 1.0e-10,
//                            2.4 + 1.0e-10, 2.5, 2.6, 2.7, 2.8) +
//                   1.0e-8;
//   } else if (state.range(0) == 3) {
//     const double base = 2.8 + 1.0e-8;
//     upper_bound = SimdType(base + 1.0e-10, base + 2.0e-10, base + 3.0e-10,
//                            base + 4.0e-10, base + 5.0e-10, base + 6.0e-10,
//                            base + 7.0e-10, base + 8.0e-10);
//   }
//   const auto f_at_lower_bound = f(lower_bound);
//   const auto f_at_upper_bound = f(upper_bound);
//   for (auto _ : state) {
//     for (size_t i = 0; i < SimdType::size; ++i) {
//       benchmark::DoNotOptimize(RootFinder::toms748(
//           f, lower_bound.get(i), upper_bound.get(i), f_at_lower_bound.get(i),
//           f_at_upper_bound.get(i), 1.0e-14, 1.0e-12));
//     }
//   }
// }
// BENCHMARK(benchmark_cubic_roots_scalar)  //
//     ->DenseRange(0, 3, 1);               //



void benchmark_fix_cons(benchmark::State& state) {
  const size_t num_points = 513;
  Scalar<DataVector> tilde_d{DataVector(num_points)};
  Scalar<DataVector> tilde_ye{DataVector(num_points)};
  Scalar<DataVector> tilde_tau{DataVector(num_points)};
  auto tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto tilde_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  for (size_t i = 0; i < num_points; ++i) {
    if (i % 2 == 0) {
      // tilde_d is too small, should be raised to limit
      get(tilde_d)[i] = 2.e-12;
      get(tilde_ye)[i] = 2.e-13;
      get(tilde_tau)[i] = 4.5;
      tilde_s.get(0)[i] = 3.0;
      tilde_b.get(0)[i] = 2.0;
    } else {
      // tilde_tau is too small, raise to level of needed, which also
      // causes tilde_s to be zeroed
      get(tilde_d)[i] = 1.0;
      get(tilde_ye)[i] = 1.0;
      get(tilde_tau)[i] = 1.5;
      tilde_s.get(0)[i] = 0.0;
      tilde_b.get(0)[i] = 2.0;
    }
  }

  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto sqrt_det_spatial_metric =
      make_with_value<Scalar<DataVector>>(tilde_d, 1.0);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
    inv_spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
  }

  const grmhd::ValenciaDivClean::FixConservatives variable_fixer{
      1.e-12, 1.0e-11, 1.0e-10, 1.0e-9, 0.0, 0.0};

  for (auto _ : state) {
    benchmark::DoNotOptimize(variable_fixer(
        &tilde_d, &tilde_ye, &tilde_tau, &tilde_s, tilde_b, spatial_metric,
        inv_spatial_metric, sqrt_det_spatial_metric));
  }
}
// BENCHMARK(benchmark_fix_cons)->Iterations(40380);

template <size_t ThermodynamicDim>
void benchmark_c2p_impl(
    benchmark::State& state,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  std::mt19937 generator_value{};
  gsl::not_null<std::mt19937*> generator(&generator_value);

  const size_t number_of_points = 512;
  const DataVector used_for_size{number_of_points};
  // generate random primitives with interesting astrophysical values
  const auto expected_rest_mass_density =
      TestHelpers::hydro::random_density(generator, used_for_size);
  const auto expected_electron_fraction =
      TestHelpers::hydro::random_electron_fraction(generator, used_for_size);
  const auto expected_lorentz_factor =
      TestHelpers::hydro::random_lorentz_factor(generator, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3>(generator, used_for_size);
  const auto expected_spatial_velocity = TestHelpers::hydro::random_velocity(
      generator, expected_lorentz_factor, spatial_metric);
  Scalar<DataVector> expected_specific_internal_energy{};
  Scalar<DataVector> expected_pressure{};
  if constexpr (ThermodynamicDim == 1) {
    expected_specific_internal_energy =
        equation_of_state.specific_internal_energy_from_density(
            expected_rest_mass_density);
    expected_pressure =
        equation_of_state.pressure_from_density(expected_rest_mass_density);
  } else if constexpr (ThermodynamicDim == 2) {
    // note this call assumes an ideal fluid
    expected_specific_internal_energy =
        TestHelpers::hydro::random_specific_internal_energy(generator,
                                                            used_for_size);
    expected_pressure = equation_of_state.pressure_from_density_and_energy(
        expected_rest_mass_density, expected_specific_internal_energy);
  }

  const auto expected_specific_enthalpy = hydro::relativistic_specific_enthalpy(
      expected_rest_mass_density, expected_specific_internal_energy,
      expected_pressure);
  const auto expected_magnetic_field =
      TestHelpers::hydro::random_magnetic_field(generator, expected_pressure,
                                                spatial_metric);
  const auto expected_divergence_cleaning_field =
      TestHelpers::hydro::random_divergence_cleaning_field(generator,
                                                           used_for_size);

  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const Scalar<DataVector> sqrt_det_spatial_metric =
      Scalar<DataVector>{sqrt(get(det_and_inv.first))};

  Scalar<DataVector> tilde_d(number_of_points);
  Scalar<DataVector> tilde_ye(number_of_points);
  Scalar<DataVector> tilde_tau(number_of_points);
  tnsr::i<DataVector, 3> tilde_s(number_of_points);
  tnsr::I<DataVector, 3> tilde_b(number_of_points);
  Scalar<DataVector> tilde_phi(number_of_points);

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&tilde_d), make_not_null(&tilde_ye),
      make_not_null(&tilde_tau), make_not_null(&tilde_s),
      make_not_null(&tilde_b), make_not_null(&tilde_phi),
      expected_rest_mass_density, expected_electron_fraction,
      expected_specific_internal_energy, expected_specific_enthalpy,
      expected_pressure, expected_spatial_velocity, expected_lorentz_factor,
      expected_magnetic_field, sqrt_det_spatial_metric, spatial_metric,
      expected_divergence_cleaning_field);

  Scalar<DataVector> rest_mass_density(number_of_points);
  Scalar<DataVector> electron_fraction(number_of_points);
  Scalar<DataVector> specific_internal_energy(number_of_points);
  tnsr::I<DataVector, 3> spatial_velocity(number_of_points);
  tnsr::I<DataVector, 3> magnetic_field(number_of_points);
  Scalar<DataVector> divergence_cleaning_field(number_of_points);
  Scalar<DataVector> lorentz_factor(number_of_points);
  Scalar<DataVector> pressure(number_of_points, 0.0);
  Scalar<DataVector> specific_enthalpy(number_of_points);
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        grmhd::ValenciaDivClean::PrimitiveFromConservative<tmpl::list<
            grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>::
            apply(make_not_null(&rest_mass_density),
                  make_not_null(&electron_fraction),
                  make_not_null(&specific_internal_energy),
                  make_not_null(&spatial_velocity),
                  make_not_null(&magnetic_field),
                  make_not_null(&divergence_cleaning_field),
                  make_not_null(&lorentz_factor), make_not_null(&pressure),
                  make_not_null(&specific_enthalpy), tilde_d, tilde_ye,
                  tilde_tau, tilde_s, tilde_b, tilde_phi, spatial_metric,
                  inv_spatial_metric, sqrt_det_spatial_metric,
                  equation_of_state));
  }
}

void benchmark_c2p_polytrope(
    benchmark::State& state) {
  const EquationsOfState::PolytropicFluid<true> polytropic_fluid(100.0, 2.0);
  benchmark_c2p_impl(state, polytropic_fluid);
}
BENCHMARK(benchmark_c2p_polytrope); //->Iterations(40380);

void benchmark_c2p_ideal_fluid(
    benchmark::State& state) {
  const EquationsOfState::IdealFluid<true> ideal_fluid(4.0 / 3.0);
  benchmark_c2p_impl(state, ideal_fluid);
}
BENCHMARK(benchmark_c2p_ideal_fluid); //->Iterations(40380);

// Reference times:
// benchmark_c2p_polytrope       775022 ns       774529 ns          740
// benchmark_c2p_ideal_fluid     612335 ns       611891 ns          963


}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
