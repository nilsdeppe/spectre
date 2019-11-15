// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/NewtonianEuler/OuTurbulence.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/OuAnisotropicForcing.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace NewtonianEuler {
namespace AnalyticData {

template <size_t Dim>
OuTurbulence<Dim>::OuTurbulence(
    const double polytropic_exponent, const double initial_density,
    const double decay_time, const double energy_input_per_mode,
    const double min_stirring_wavenumber, const double max_stirring_wavenumber,
    const double solenoidal_weight, const double anisotropy_factor) noexcept
    : polytropic_exponent_(polytropic_exponent),
      initial_density_(initial_density),
      // Polytropic constant is set equal to 1.0
      equation_of_state_(1.0, polytropic_exponent),
      source_term_(0.1, 1, decay_time, energy_input_per_mode,
                   min_stirring_wavenumber, max_stirring_wavenumber,
                   solenoidal_weight, anisotropy_factor, 140281) {
  ASSERT(initial_density_ > 0.0,
         "The initial density must be positive. The value given "
         "was "
             << initial_density_ << ".");
}

template <size_t Dim>
void OuTurbulence<Dim>::pup(PUP::er& p) noexcept {
  p | polytropic_exponent_;
  p | initial_density_;
  p | equation_of_state_;
  p | source_term_;
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::MassDensity<DataType>> OuTurbulence<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::MassDensity<DataType>> /*meta*/) const noexcept {
  return make_with_value<Scalar<DataType>>(x, initial_density_);
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Velocity<DataType, Dim, Frame::Inertial>>
OuTurbulence<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::Velocity<DataType, Dim, Frame::Inertial>> /*meta*/) const
    noexcept {
  return make_with_value<tnsr::I<DataType, Dim>>(x, 0.0);
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>
OuTurbulence<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
    noexcept {
  return equation_of_state_.specific_internal_energy_from_density(
      get<Tags::MassDensity<DataType>>(
          variables(x, tmpl::list<Tags::MassDensity<DataType>>{})));
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Pressure<DataType>> OuTurbulence<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::Pressure<DataType>> /*meta*/) const noexcept {
  return equation_of_state_.pressure_from_density(
      get<Tags::MassDensity<DataType>>(
          variables(x, tmpl::list<Tags::MassDensity<DataType>>{})));
}

template <size_t Dim>
bool operator==(const OuTurbulence<Dim>& lhs,
                const OuTurbulence<Dim>& rhs) noexcept {
  // No comparison for equation_of_state_. Comparing polytropic_exponent_
  // should suffice.
  return lhs.polytropic_exponent_ == rhs.polytropic_exponent_ and
         lhs.initial_density_ == rhs.initial_density_ and
         lhs.source_term_ == rhs.source_term_;
}

template <size_t Dim>
bool operator!=(const OuTurbulence<Dim>& lhs,
                const OuTurbulence<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_CLASS(_, data)                                   \
  template class OuTurbulence<DIM(data)>;                            \
  template bool operator==(const OuTurbulence<DIM(data)>&,           \
                           const OuTurbulence<DIM(data)>&) noexcept; \
  template bool operator!=(const OuTurbulence<DIM(data)>&,           \
                           const OuTurbulence<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_CLASS, (3))

#define INSTANTIATE_SCALARS(_, data)                                 \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>>             \
      OuTurbulence<DIM(data)>::variables(                            \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x, \
          tmpl::list<TAG(data) < DTYPE(data)>>) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (3), (double, DataVector),
                        (Tags::MassDensity, Tags::SpecificInternalEnergy,
                         Tags::Pressure))

#define INSTANTIATE_VELOCITY(_, data)                                       \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), DIM(data),          \
                               Frame::Inertial>>                            \
      OuTurbulence<DIM(data)>::variables(                                   \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x,        \
          tmpl::list<TAG(data) < DTYPE(data), DIM(data), Frame::Inertial>>) \
          const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VELOCITY, (3), (double, DataVector),
                        (Tags::Velocity))

#undef DIM
#undef DTYPE
#undef TAG
#undef INSTANTIATE_CLASS
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VELOCITY

}  // namespace AnalyticData
}  // namespace NewtonianEuler
/// \endcond
