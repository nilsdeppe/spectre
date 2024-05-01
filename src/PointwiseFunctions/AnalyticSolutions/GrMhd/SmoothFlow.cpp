// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Expressions/Divide.hpp"

namespace grmhd::Solutions {

SmoothFlow::SmoothFlow(const std::array<double, 3>& mean_velocity,
                       const std::array<double, 3>& wavevector,
                       const double pressure, const double adiabatic_index,
                       const double perturbation_size)
    : RelativisticEuler::Solutions::SmoothFlow<3>(mean_velocity, wavevector,
                                                  pressure, adiabatic_index,
                                                  perturbation_size) {}

std::unique_ptr<evolution::initial_data::InitialData> SmoothFlow::get_clone()
    const {
  return std::make_unique<SmoothFlow>(*this);
}

SmoothFlow::SmoothFlow(CkMigrateMessage* msg)
    : RelativisticEuler::Solutions::SmoothFlow<3>(msg) {}

void SmoothFlow::pup(PUP::er& p) {
  RelativisticEuler::Solutions::SmoothFlow<3>::pup(p);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
SmoothFlow::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  //  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)};
  /*
  auto result = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  result.get(2) = 1.0;
  return  {std::move(result)}; // infinite sheet of charge */
  /*
  auto result = x;
  const Scalar<DataType> radius = magnitude(x);
  for (size_t i = 0; i < 3; ++i) {
    result.get(i) /= square(get(radius));
  }
  return {std::move(result)}; // ~point charge? */

  /*
  auto result = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  result.get(2) = 1.0 * x.get(2);
  return {std::move(result)}; // slab spanning the size of the domain
  */

  auto result = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  result.get(2) = 1.0*x.get(2) * exp(-0.5 * square(x.get(2)));
  return {std::move(result)};
// smooth function that is ~0 at 0 and pi and has some maximum between
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
SmoothFlow::variables(
    const tnsr::I<DataType, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

PUP::able::PUP_ID SmoothFlow::my_PUP_ID = 0;

bool operator==(const SmoothFlow& lhs, const SmoothFlow& rhs) {
  using smooth_flow = RelativisticEuler::Solutions::SmoothFlow<3>;
  return *static_cast<const smooth_flow*>(&lhs) ==
         *static_cast<const smooth_flow*>(&rhs);
}

bool operator!=(const SmoothFlow& lhs, const SmoothFlow& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                                        \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)> >                   \
      SmoothFlow::variables(const tnsr::I<DTYPE(data), 3>& x, double t,     \
                            tmpl::list<TAG(data) < DTYPE(data)> > /*meta*/) \
          const;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (double, DataVector),
                        (hydro::Tags::DivergenceCleaningField))

#define INSTANTIATE_VECTORS(_, data)                                           \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3> >                   \
      SmoothFlow::variables(const tnsr::I<DTYPE(data), 3>& x, double t,        \
                            tmpl::list<TAG(data) < DTYPE(data), 3> > /*meta*/) \
          const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::MagneticField))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS
}  // namespace grmhd::Solutions
