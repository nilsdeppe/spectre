// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Harmonic.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop
#include <charm++.h>
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

#include "DataStructures/TaggedContainers.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

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

template <class Tm1, class T0, class T1, class T2, typename... TempTags,
          typename... ArgTags>
void forward_to(Tm1&& dt_vars, const Mesh<3>& mesh, T0&& temp_vars,
                T1&& evolved_vars, T2&& d_vars,
                const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
                const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                      Frame::Inertial>& inverse_jacobian,
                tmpl::list<TempTags...>, tmpl::list<ArgTags...>) {
  using Ats = tmpl::list<ArgTags...>;
  const Scalar<DataVector> gamma{evolved_vars.number_of_grid_points(), 0.0};
  const double time = 1.0;
  GeneralizedHarmonic::gauges::Harmonic gauge{};
  using vars_list = typename std::decay_t<T2>::tags_list;
  GeneralizedHarmonic::TimeDerivative<3>::apply(
      get<tmpl::at_c<Ats, 0>>(dt_vars), get<tmpl::at_c<Ats, 1>>(dt_vars),
      get<tmpl::at_c<Ats, 2>>(dt_vars), get<TempTags>(temp_vars)...,
      get<tmpl::at_c<vars_list, 0>>(d_vars),
      get<tmpl::at_c<vars_list, 1>>(d_vars),
      get<tmpl::at_c<vars_list, 2>>(d_vars),

      get<tmpl::at_c<Ats, 0>>(evolved_vars),
      get<tmpl::at_c<Ats, 1>>(evolved_vars),
      get<tmpl::at_c<Ats, 2>>(evolved_vars), gamma, gamma, gamma, gauge, mesh,
      time, inertial_coords, inverse_jacobian,

      {});
}

// clang-tidy: don't pass be non-const reference
void bench_all_gradient(benchmark::State& state) {  // NOLINT
  constexpr const size_t pts_1d = 5;
  constexpr const size_t Dim = 3;
  const Mesh<Dim> mesh{pts_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  domain::CoordinateMaps::Affine map1d(-1.0, 1.0, -1.0, 1.0);
  using Map3d =
      domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>;
  domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial, Map3d> map(
      Map3d{map1d, map1d, map1d});

  using VarTags = tmpl::list<gr::Tags::SpacetimeMetric<3>,
                             GeneralizedHarmonic::Tags::Pi<Dim>,
                             GeneralizedHarmonic::Tags::Phi<Dim>>;
  const InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      inv_jac = map.inv_jacobian(logical_coordinates(mesh));
  const auto grid_coords = map(logical_coordinates(mesh));
  Variables<VarTags> vars(mesh.number_of_grid_points(), 0.0);
  for (size_t i = 0; i < 4; ++i) {
    get<gr::Tags::SpacetimeMetric<3>>(vars).get(i, i) = 1.0;
  }
  Variables<VarTags> dt_vars(mesh.number_of_grid_points(), 0.0);
  benchmark::DoNotOptimize(dt_vars);

  while (state.KeepRunning()) {
    Variables<typename GeneralizedHarmonic::TimeDerivative<3>::temporary_tags>
      temp_tags{mesh.number_of_grid_points(), 0.0};
    const auto d_vars = partial_derivatives<VarTags>(vars, mesh, inv_jac);
    benchmark::DoNotOptimize(d_vars);
    forward_to(
        make_not_null(&dt_vars), mesh, make_not_null(&temp_tags), vars, d_vars,
        grid_coords, inv_jac,
        typename GeneralizedHarmonic::TimeDerivative<3>::temporary_tags{},
        tmpl::pop_back<
            typename GeneralizedHarmonic::TimeDerivative<3>::argument_tags>{});
  }
}
BENCHMARK(bench_all_gradient);  // NOLINT
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
