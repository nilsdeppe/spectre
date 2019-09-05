// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/KeplerianDisk.hpp"

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
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace NewtonianEuler {
namespace Solutions {

template <size_t Dim>
KeplerianDisk<Dim>::KeplerianDisk(
    const double adiabatic_index, const double ambient_mass_density,
    const double ambient_pressure, const std::array<double, Dim>& disk_center,
    const double disk_mass_density, const double disk_inner_radius,
    const double disk_outer_radius, const double smoothing_parameter,
    const double transition_width)
    : adiabatic_index_(adiabatic_index),
      ambient_mass_density_(ambient_mass_density),
      ambient_pressure_(ambient_pressure),
      disk_center_(disk_center),
      disk_mass_density_(disk_mass_density),
      disk_inner_radius_(disk_inner_radius),
      disk_outer_radius_(disk_outer_radius),
      smoothing_parameter_(smoothing_parameter),
      transition_width_(transition_width),
      equation_of_state_(adiabatic_index),
      source_term_(disk_center, smoothing_parameter, transition_width) {
  ASSERT(ambient_mass_density_ > 0.0,
         "The ambient mass density must be positive. The value given was "
             << ambient_mass_density_ << ".");
  ASSERT(ambient_pressure_ > 0.0,
         "The ambient pressure must be positive. The value given was "
             << ambient_pressure_ << ".");
  ASSERT(disk_mass_density_ > 0.0,
         "The disk mass density must be positive. The value given was "
             << disk_mass_density_ << ".");
  ASSERT(disk_inner_radius_ > 0.0,
         "The disk inner radius must be positive. The value given was "
             << disk_inner_radius_ << ".");
  ASSERT(disk_outer_radius_ > 0.0,
         "The disk outer radius must be positive. The value given was "
             << disk_outer_radius_ << ".");
  ASSERT(disk_outer_radius_ > disk_inner_radius_,
         "The disk outer radius must be greater than the inner radius. Given "
         "outer radius: "
             << disk_outer_radius_
             << ". Given inner radius: " << disk_inner_radius_ << ".");
  ASSERT(transition_width_ > 0.0,
         "The transition width must be positive. The value given was "
             << transition_width_ << ".");
}

template <size_t Dim>
void KeplerianDisk<Dim>::pup(PUP::er& p) noexcept {
  p | adiabatic_index_;
  p | ambient_mass_density_;
  p | ambient_pressure_;
  p | disk_center_;
  p | disk_mass_density_;
  p | disk_inner_radius_;
  p | disk_outer_radius_;
  p | smoothing_parameter_;
  p | transition_width_;
  p | equation_of_state_;
  p | source_term_;
}

template <size_t Dim>
template <typename DataType>
KeplerianDisk<Dim>::IntermediateVariables<DataType>::IntermediateVariables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x,
    const std::array<double, Dim>& disk_center) noexcept {
  x_prime = get<0>(x) - disk_center[0];
  y_prime = get<1>(x) - disk_center[1];
  r_prime = square(x_prime) + square(y_prime);
  if (Dim == 3) {
    r_prime += square(get<Dim - 1>(x) - disk_center[Dim - 1]);
  }
  r_prime = sqrt(r_prime);
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::MassDensity<DataType>> KeplerianDisk<Dim>::variables(
    tmpl::list<Tags::MassDensity<DataType>> /*meta*/,
    const IntermediateVariables<DataType>& vars) const noexcept {
  auto result =
      make_with_value<Scalar<DataType>>(vars.r_prime, ambient_mass_density_);
  const double r_inn_minus = disk_inner_radius_ - 0.5 * transition_width_;
  const double r_inn_plus = r_inn_minus + transition_width_;
  const double r_out_minus = disk_outer_radius_ - 0.5 * transition_width_;
  const double r_out_plus = r_out_minus + transition_width_;
  const double rho_d_minus_rho_a_over_dr =
      (disk_mass_density_ - ambient_mass_density_) / transition_width_;
  for (size_t s = 0; s < get_size(vars.r_prime); ++s) {
    const double r_prime_s = get_element(vars.r_prime, s);
    if (r_prime_s > r_inn_minus and r_prime_s <= r_inn_plus) {
      get_element(get(result), s) +=
          rho_d_minus_rho_a_over_dr * (r_prime_s - r_inn_minus);
    } else if (r_prime_s > r_inn_plus and r_prime_s <= r_out_minus) {
      get_element(get(result), s) = disk_mass_density_;
    } else if (r_prime_s > r_out_minus and r_prime_s <= r_out_plus) {
      get_element(get(result), s) =
          disk_mass_density_ -
          rho_d_minus_rho_a_over_dr * (r_prime_s - r_out_minus);
    }
  }
  return std::move(result);
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Velocity<DataType, Dim, Frame::Inertial>>
KeplerianDisk<Dim>::variables(
    tmpl::list<Tags::Velocity<DataType, Dim, Frame::Inertial>> /*meta*/,
    const IntermediateVariables<DataType>& vars) const noexcept {
  auto result = make_with_value<tnsr::I<DataType, Dim, Frame::Inertial>>(
      vars.r_prime, 0.0);
  const double r_min = disk_inner_radius_ - 2.0 * transition_width_;
  const double r_max = disk_outer_radius_ + 2.0 * transition_width_;
  for (size_t s = 0; s < get_size(vars.r_prime); ++s) {
    const double r_prime_s = get_element(vars.r_prime, s);
    if (r_prime_s >= r_min and r_prime_s < r_max) {
      const double prefactor = 1.0 / pow(r_prime_s, 1.5);
      get_element(get<0>(result), s) =
          -get_element(vars.y_prime, s) * prefactor;
      get_element(get<1>(result), s) = get_element(vars.x_prime, s) * prefactor;
    }
  }
  return std::move(result);
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Pressure<DataType>> KeplerianDisk<Dim>::variables(
    tmpl::list<Tags::Pressure<DataType>> /*meta*/,
    const IntermediateVariables<DataType>& vars) const noexcept {
  return make_with_value<Scalar<DataType>>(vars.x_prime, ambient_pressure_);
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>
KeplerianDisk<Dim>::variables(
    tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/,
    const IntermediateVariables<DataType>& vars) const noexcept {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<Tags::MassDensity<DataType>>(
          variables(tmpl::list<Tags::MassDensity<DataType>>{}, vars)),
      get<Tags::Pressure<DataType>>(
          variables(tmpl::list<Tags::Pressure<DataType>>{}, vars)));
}

template <size_t Dim>
bool operator==(const KeplerianDisk<Dim>& lhs,
                const KeplerianDisk<Dim>& rhs) noexcept {
  // No comparison for equation_of_state_ or source_term_. Comparing
  // member variables should suffice.
  return lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.ambient_mass_density_ == rhs.ambient_mass_density_ and
         lhs.ambient_pressure_ == rhs.ambient_pressure_ and
         lhs.disk_center_ == rhs.disk_center_ and
         lhs.disk_mass_density_ == rhs.disk_mass_density_ and
         lhs.disk_inner_radius_ == rhs.disk_inner_radius_ and
         lhs.disk_outer_radius_ == rhs.disk_outer_radius_ and
         lhs.smoothing_parameter_ == rhs.smoothing_parameter_ and
         lhs.transition_width_ == rhs.transition_width_;
}

template <size_t Dim>
bool operator!=(const KeplerianDisk<Dim>& lhs,
                const KeplerianDisk<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_CLASS(_, data)                                             \
  template class KeplerianDisk<DIM(data)>;                                     \
  template struct KeplerianDisk<DIM(data)>::IntermediateVariables<double>;     \
  template struct KeplerianDisk<DIM(data)>::IntermediateVariables<DataVector>; \
  template bool operator==(const KeplerianDisk<DIM(data)>&,                    \
                           const KeplerianDisk<DIM(data)>&) noexcept;          \
  template bool operator!=(const KeplerianDisk<DIM(data)>&,                    \
                           const KeplerianDisk<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_CLASS, (2, 3))

#define INSTANTIATE_SCALARS(_, data)                     \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>> \
      KeplerianDisk<DIM(data)>::variables(               \
          tmpl::list<TAG(data) < DTYPE(data)>>,          \
          const IntermediateVariables<DTYPE(data)>&) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (2, 3), (double, DataVector),
                        (Tags::MassDensity, Tags::Pressure,
                         Tags::SpecificInternalEnergy))

#define INSTANTIATE_VELOCITY(_, data)                                       \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), DIM(data),          \
                               Frame::Inertial>>                            \
      KeplerianDisk<DIM(data)>::variables(                                  \
          tmpl::list<TAG(data) < DTYPE(data), DIM(data), Frame::Inertial>>, \
          const IntermediateVariables<DTYPE(data)>&) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VELOCITY, (2, 3), (double, DataVector),
                        (Tags::Velocity))

#undef DIM
#undef DTYPE
#undef TAG
#undef INSTANTIATE_CLASS
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VELOCITY

}  // namespace Solutions
}  // namespace NewtonianEuler
/// \endcond
