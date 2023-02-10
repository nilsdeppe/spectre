// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/NewtonianEuler/RayleighTaylorInst.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace NewtonianEuler {
namespace AnalyticData {

template <size_t Dim>
RayleighTaylorInst<Dim>::RayleighTaylorInst(
    const double adiabatic_index, const double lower_mass_density,
    const double upper_mass_density, const double background_pressure,
    const double perturbation_amplitude, const double damping_factor,
    const double interface_height, const double grav_acceleration)
    : adiabatic_index_(adiabatic_index),
      lower_mass_density_(lower_mass_density),
      upper_mass_density_(upper_mass_density),
      background_pressure_(background_pressure),
      perturbation_amplitude_(perturbation_amplitude),
      damping_factor_(damping_factor),
      interface_height_(interface_height),
      grav_acceleration_(grav_acceleration),
      equation_of_state_(adiabatic_index) {
  ASSERT(lower_mass_density_ > 0. and upper_mass_density_ > 0.,
         "The mass density must be positive everywhere. Lower "
         "density: "
             << lower_mass_density_
             << ", Upper density: " << upper_mass_density_ << ".");
  ASSERT(background_pressure_ > 0.,
         "The background pressure must be positive. The value given was "
             << background_pressure_ << ".");
  ASSERT(grav_acceleration_ > 0.,
         "The gravitational acceleration must be positive. The value given was "
             << grav_acceleration_ << ".");
}

template <size_t Dim>
void RayleighTaylorInst<Dim>::pup(PUP::er& p) noexcept {
  p | adiabatic_index_;
  p | lower_mass_density_;
  p | upper_mass_density_;
  p | background_pressure_;
  p | perturbation_amplitude_;
  p | damping_factor_;
  p | interface_height_;
  p | grav_acceleration_;
  p | equation_of_state_;
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::MassDensity<DataType>>
RayleighTaylorInst<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x, const double t,
    tmpl::list<Tags::MassDensity<DataType>> /*meta*/) const noexcept {
  auto result = make_with_value<Scalar<DataType>>(x, 0.0);
  const size_t n_pts = get_size(get<0>(x));
  for (size_t s = 0; s < n_pts; ++s) {
    get_element(get(result), s) =
        get_element(get<Dim - 1>(x), s) < interface_height_
            ? lower_mass_density_
            : upper_mass_density_;
  }
  return result;
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Velocity<DataType, Dim, Frame::Inertial>>
RayleighTaylorInst<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x, const double t,
    tmpl::list<Tags::Velocity<DataType, Dim, Frame::Inertial>> /*meta*/) const
    noexcept {
  auto result =
      make_with_value<tnsr::I<DataType, Dim, Frame::Inertial>>(x, 0.0);
  get<Dim - 1>(result) =
      perturbation_amplitude_ * sin(2.0 * M_PI * get<0>(x) / 0.5) *
      exp(-square((get<Dim - 1>(x) - interface_height_) / damping_factor_));
  return result;
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Pressure<DataType>>
RayleighTaylorInst<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x, const double t,
    tmpl::list<Tags::Pressure<DataType>> /*meta*/) const noexcept {
  auto result = make_with_value<Scalar<DataType>>(x, background_pressure_);
  const size_t n_pts = get_size(get<0>(x));
  const double lower_specific_weight = lower_mass_density_ * grav_acceleration_;
  const double upper_specific_weight = upper_mass_density_ * grav_acceleration_;
  for (size_t s = 0; s < n_pts; ++s) {
    const double& vertical_coord = get_element(get<Dim - 1>(x), s);
    get_element(get(result), s) -=
        (vertical_coord < interface_height_
             ? lower_specific_weight * vertical_coord
             : lower_specific_weight * interface_height_ +
                   upper_specific_weight *
                       (vertical_coord - interface_height_));
    // Depending on the coords and the input paratemers, the pressure
    // might become negative at some heights. (Physically educated values
    // shouldn't give rise to problems, but blind numerical experiments might.)
    ASSERT(get_element(get(result), s) > 0.0,
           "Pressure is negative at vertical coordinate: "
               << vertical_coord << ". Value of pressure: "
               << get_element(get(result), s) << ".");
  }
  return result;
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>
RayleighTaylorInst<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x, const double t,
    tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
    noexcept {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<Tags::MassDensity<DataType>>(
          variables(x, t, tmpl::list<Tags::MassDensity<DataType>>{})),
      get<Tags::Pressure<DataType>>(
          variables(x, t, tmpl::list<Tags::Pressure<DataType>>{})));
}

template <size_t Dim>
bool operator==(const RayleighTaylorInst<Dim>& lhs,
                const RayleighTaylorInst<Dim>& rhs) noexcept {
  // No comparison for equation_of_state_. Comparing adiabatic_index_ should
  // suffice.
  return lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.lower_mass_density_ == rhs.lower_mass_density_ and
         lhs.upper_mass_density_ == rhs.upper_mass_density_ and
         lhs.background_pressure_ == rhs.background_pressure_ and
         lhs.perturbation_amplitude_ == rhs.perturbation_amplitude_ and
         lhs.damping_factor_ == rhs.damping_factor_ and
         lhs.interface_height_ == rhs.interface_height_ and
         lhs.grav_acceleration_ == rhs.grav_acceleration_;
}

template <size_t Dim>
bool operator!=(const RayleighTaylorInst<Dim>& lhs,
                const RayleighTaylorInst<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_CLASS(_, data)                                         \
  template class RayleighTaylorInst<DIM(data)>;                            \
  template bool operator==(const RayleighTaylorInst<DIM(data)>&,           \
                           const RayleighTaylorInst<DIM(data)>&) noexcept; \
  template bool operator!=(const RayleighTaylorInst<DIM(data)>&,           \
                           const RayleighTaylorInst<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_CLASS, (2, 3))

#define INSTANTIATE_SCALARS(_, data)                                 \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>>             \
      RayleighTaylorInst<DIM(data)>::variables(                      \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x, \
          const double t, tmpl::list<TAG(data) < DTYPE(data)>>)      \
          const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (2, 3), (double, DataVector),
                        (Tags::MassDensity, Tags::SpecificInternalEnergy,
                         Tags::Pressure))

#define INSTANTIATE_VELOCITY(_, data)                                       \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), DIM(data),          \
                               Frame::Inertial>>                            \
      RayleighTaylorInst<DIM(data)>::variables(                             \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x,        \
          const double t,                                                   \
          tmpl::list<TAG(data) < DTYPE(data), DIM(data), Frame::Inertial>>) \
          const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VELOCITY, (2, 3), (double, DataVector),
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
