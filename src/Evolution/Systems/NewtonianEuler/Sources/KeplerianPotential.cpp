// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/KeplerianPotential.hpp"

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace NewtonianEuler {
namespace Sources {
template <size_t Dim>
KeplerianPotential<Dim>::KeplerianPotential(
    const std::array<double, Dim>& potential_center,
    const double smoothing_parameter, const double transition_width) noexcept
    : potential_center_(potential_center),
      smoothing_parameter_(smoothing_parameter),
      transition_width_(transition_width) {
  ASSERT(smoothing_parameter_ > 0.0,
         "The smoothing parameter must be positive. The value given was: "
             << smoothing_parameter_);
}

template <size_t Dim>
void KeplerianPotential<Dim>::pup(PUP::er& p) noexcept {
  p | potential_center_;
  p | smoothing_parameter_;
  p | transition_width_;
}

template <size_t Dim>
void KeplerianPotential<Dim>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> source_energy_density,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const tnsr::I<DataVector, Dim>& x) const noexcept {
  // We precompute the acceleration, but we store it in
  // source_momentum_density in order to save memory.
  for (size_t i = 0; i < Dim; ++i) {
    source_momentum_density->get(i) = x.get(i) - gsl::at(potential_center_, i);
  }

  const auto r_prime = magnitude(*source_momentum_density);
  const double one_minus_dr_halves = 0.5 - 0.5 * transition_width_;
  for (size_t s = 0; s < get<0>(x).size(); ++s) {
    const double r_prime_s = get(r_prime)[s];

    // Arbitrary upper bound below which we take the limit x'/r' --> 1
    if (UNLIKELY(r_prime_s < 1.e-15)) {
      for (size_t i = 0; i < Dim; ++i) {
        source_momentum_density->get(i)[s] =
            -1.0 / (square(r_prime_s) + square(smoothing_parameter_));
      }
    } else {
      for (size_t i = 0; i < Dim; ++i) {
        source_momentum_density->get(i)[s] *=
            (r_prime_s > one_minus_dr_halves
                 ? -1.0 / cube(r_prime_s)
                 : -1.0 / r_prime_s /
                       (square(r_prime_s) + square(smoothing_parameter_)));
      }
    }
  }

  get(*source_energy_density) =
      get(dot_product(momentum_density, *source_momentum_density));
  for (size_t i = 0; i < Dim; ++i) {
    source_momentum_density->get(i) *= get(mass_density_cons);
  }
}

template <size_t Dim>
bool operator==(const KeplerianPotential<Dim>& lhs,
                const KeplerianPotential<Dim>& rhs) noexcept {
  return lhs.potential_center_ == rhs.potential_center_ and
         lhs.smoothing_parameter_ == rhs.smoothing_parameter_ and
         lhs.transition_width_ == rhs.transition_width_;
}

template <size_t Dim>
bool operator!=(const KeplerianPotential<Dim>& lhs,
                const KeplerianPotential<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template struct KeplerianPotential<DIM(data)>;                           \
  template bool operator==(const KeplerianPotential<DIM(data)>&,           \
                           const KeplerianPotential<DIM(data)>&) noexcept; \
  template bool operator!=(const KeplerianPotential<DIM(data)>&,           \
                           const KeplerianPotential<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef DIM
#undef INSTANTIATE

}  // Namespace Sources
}  // namespace NewtonianEuler
/// \endcond
