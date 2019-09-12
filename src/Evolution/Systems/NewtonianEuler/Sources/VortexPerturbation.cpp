// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/VortexPerturbation.hpp"

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace NewtonianEuler {
namespace Sources {
template <size_t Dim>
VortexPerturbation<Dim>::VortexPerturbation(
    const double adiabatic_index, const double perturbation_amplitude,
    const std::array<double, Dim>& vortex_center,
    const std::array<double, Dim>& vortex_mean_velocity,
    const double vortex_strength) noexcept
    : adiabatic_index_(adiabatic_index),
      perturbation_amplitude_(perturbation_amplitude),
      vortex_center_(vortex_center),
      vortex_mean_velocity_(vortex_mean_velocity),
      vortex_strength_(vortex_strength) {}

template <size_t Dim>
void VortexPerturbation<Dim>::pup(PUP::er& p) noexcept {
  p | adiabatic_index_;
  p | perturbation_amplitude_;
  p | vortex_center_;
  p | vortex_mean_velocity_;
  p | vortex_strength_;
}

template <>
void VortexPerturbation<2>::apply() const noexcept {}

template <>
void VortexPerturbation<3>::apply(
    const gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> source_energy_density,
    const tnsr::I<DataVector, 3>& x, const double time) const noexcept {
  Solutions::IsentropicVortex<3> vortex(
      adiabatic_index_, vortex_center_, vortex_mean_velocity_,
      perturbation_amplitude_, vortex_strength_);
  const size_t size = get<0>(x).size();
  const auto vortex_primitives = vortex.variables(
      x, time,
      tmpl::list<Tags::MassDensity<DataVector>, Tags::Velocity<DataVector, 3>,
                 Tags::SpecificInternalEnergy<DataVector>,
                 Tags::Pressure<DataVector>>{});

  Variables<
      tmpl::list<::Tags::TempScalar<0>, ::Tags::TempI<1, 3, Frame::Inertial>,
                 ::Tags::TempScalar<2>>>
      temp_buffer(size);
  auto& vortex_mass_density_cons = get<::Tags::TempScalar<0>>(temp_buffer);
  auto& vortex_momentum_density =
      get<::Tags::TempI<1, 3, Frame::Inertial>>(temp_buffer);
  auto& vortex_energy_density = get<::Tags::TempScalar<2>>(temp_buffer);

  NewtonianEuler::ConservativeFromPrimitive<3>::apply(
      make_not_null(&vortex_mass_density_cons),
      make_not_null(&vortex_momentum_density),
      make_not_null(&vortex_energy_density),
      get<Tags::MassDensity<DataVector>>(vortex_primitives),
      get<Tags::Velocity<DataVector, 3, Frame::Inertial>>(vortex_primitives),
      get<Tags::SpecificInternalEnergy<DataVector>>(vortex_primitives));

  DataVector dz_vortex_velocity_z =
      perturbation_amplitude_ * vortex.dz_function_of_z(get<2>(x));

  get(*source_mass_density_cons) =
      get(vortex_mass_density_cons) * dz_vortex_velocity_z;

  for (size_t i = 0; i < 3; ++i) {
    source_momentum_density->get(i) =
        vortex_momentum_density.get(i) * dz_vortex_velocity_z;
  }
  source_momentum_density->get(2) *= 2.0;

  get(*source_energy_density) =
      (get(vortex_energy_density) +
       get(get<Tags::Pressure<DataVector>>(vortex_primitives)) +
       vortex_momentum_density.get(2) *
           get<2>(get<Tags::Velocity<DataVector, 3, Frame::Inertial>>(
               vortex_primitives))) *
      dz_vortex_velocity_z;
}

template <size_t Dim>
bool operator==(const VortexPerturbation<Dim>& lhs,
                const VortexPerturbation<Dim>& rhs) noexcept {
  return lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.perturbation_amplitude_ == rhs.perturbation_amplitude_ and
         lhs.vortex_center_ == rhs.vortex_center_ and
         lhs.vortex_mean_velocity_ == rhs.vortex_mean_velocity_ and
         lhs.vortex_strength_ == rhs.vortex_strength_;
}

template <size_t Dim>
bool operator!=(const VortexPerturbation<Dim>& lhs,
                const VortexPerturbation<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template struct VortexPerturbation<DIM(data)>;                           \
  template bool operator==(const VortexPerturbation<DIM(data)>&,           \
                           const VortexPerturbation<DIM(data)>&) noexcept; \
  template bool operator!=(const VortexPerturbation<DIM(data)>&,           \
                           const VortexPerturbation<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

}  // Namespace Sources
}  // namespace NewtonianEuler

/// \endcond
