// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/KerrHorizonConforming.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace domain::CoordinateMaps {

KerrHorizonConforming::KerrHorizonConforming(
    const double mass, const std::array<double, 3> dimensionless_spin)
    : spin_parameter_(mass * dimensionless_spin),
      spin_mag_sq_(square(mass) * dot(dimensionless_spin, dimensionless_spin)) {
  ASSERT(magnitude(dimensionless_spin) < 1.,
         "Dimensionless spin magnitude must be < 1. Given dimensionless spin: "
             << dimensionless_spin << " with magnitude "
             << magnitude(dimensionless_spin));
  ASSERT(mass > 0., "Mass must be positive. Given mass: " << mass);
}

template <typename T>
void KerrHorizonConforming::operator()(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*> result,
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  if constexpr (std::is_same_v<tt::remove_cvref_wrap_t<T>, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    (*result)[0].destructive_resize(size);
    (*result)[1].destructive_resize(size);
    (*result)[2].destructive_resize(size);
  }
  ReturnType& stretch_fac = (*result)[2];
  stretch_factor_square(make_not_null(&stretch_fac), source_coords);
  stretch_fac = sqrt(stretch_fac);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(*result, i) = gsl::at(source_coords, i) * stretch_fac;
  }
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> KerrHorizonConforming::operator()(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  std::array<ReturnType, 3> result{};
  (*this)(make_not_null(&result), source_coords);
  return result;
}

std::optional<std::array<double, 3>> KerrHorizonConforming::inverse(
    const std::array<double, 3>& target_coords) const {
  const auto coords_mag_sq = dot(target_coords, target_coords);
  const auto coords_sq_min_spin_sq = coords_mag_sq - spin_mag_sq_;
  const auto coords_dot_spin = dot(target_coords, spin_parameter_);
  const auto fac = (coords_sq_min_spin_sq +
                    sqrt(coords_sq_min_spin_sq * coords_sq_min_spin_sq +
                         4. * square(coords_dot_spin))) /
                   (2. * coords_mag_sq);
  // this is the way it was written in spec, but I dont think `fac` can
  // ever be smaller than 0
  return fac >= 0.
             ? std::optional<std::array<double, 3>>(target_coords * sqrt(fac))
             : std::nullopt;
}

template <typename T>
void KerrHorizonConforming::jacobian(
    const gsl::not_null<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>*>
        result,
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  if constexpr (std::is_same_v<ReturnType, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    for (auto& component : *result) {
      component.destructive_resize(size);
    }
  }

  // use allocations from `*result` for auxiliaries
  ReturnType& fac = get<0, 0>(*result);
  ReturnType& source_coords_sq = get<0, 1>(*result);
  ReturnType& coords_dot_spin = get<0, 2>(*result);
  ReturnType& subexpr_1 = get<1, 0>(*result);
  ReturnType& subexpr_2 = get<1, 1>(*result);

  std::array<ReturnType, 3> dfac_dx{};
  if constexpr (std::is_same_v<ReturnType, DataVector>) {
    dfac_dx[0].set_data_ref(&get<2, 0>(*result));
    dfac_dx[1].set_data_ref(&get<2, 1>(*result));
    dfac_dx[2].set_data_ref(&get<2, 2>(*result));
  }

  stretch_factor_square(make_not_null(&fac), source_coords);
  source_coords_sq = dot(source_coords, source_coords);
  coords_dot_spin = dot(source_coords, spin_parameter_);
  subexpr_1 = 4. * source_coords_sq * (1. - fac);
  subexpr_2 = 2. * fac * coords_dot_spin;

  for (size_t i = 0; i < 3; ++i) {
    gsl::at(dfac_dx, i) = subexpr_1 * gsl::at(source_coords, i) +
                          2. * spin_mag_sq_ * gsl::at(source_coords, i) -
                          subexpr_2 * gsl::at(spin_parameter_, i);
  }
  dfac_dx = dfac_dx / (square(source_coords_sq) + square(coords_dot_spin));

  const ReturnType sqrt_fac = sqrt(fac);

  // not mathematically a part of `dfac_dx` but can be absorbed to avoid
  // allocation for temporary
  dfac_dx = dfac_dx / (2. * sqrt_fac);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result->get(i, j) = gsl::at(dfac_dx, j) * gsl::at(source_coords, i);
    }
    result->get(i, i) += sqrt_fac;
  }
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
KerrHorizonConforming::jacobian(const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  tnsr::Ij<ReturnType, 3, Frame::NoFrame> result(
      get_size(dereference_wrapper(source_coords[0])));
  jacobian(make_not_null(&result), source_coords);
  return result;
}

template <typename T>
void KerrHorizonConforming::stretch_factor_square(
    const gsl::not_null<tt::remove_cvref_wrap_t<T>*> result,
    const std::array<T, 3>& source_coords) const {
  auto& source_coords_sq = *result;
  source_coords_sq = dot(source_coords, source_coords);
  *result = source_coords_sq * (source_coords_sq + spin_mag_sq_) /
            (source_coords_sq * source_coords_sq +
             square(dot(source_coords, spin_parameter_)));
}

void KerrHorizonConforming::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | spin_parameter_;
    p | spin_mag_sq_;
  }
}

bool operator==(const KerrHorizonConforming& lhs,
                const KerrHorizonConforming& rhs) {
  return lhs.spin_parameter_ == rhs.spin_parameter_;
}

bool operator!=(const KerrHorizonConforming& lhs,
                const KerrHorizonConforming& rhs) {
  return not(lhs == rhs);
}
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  KerrHorizonConforming::operator()(                                         \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  KerrHorizonConforming::jacobian(                                           \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef INSTANTIATE

#define INSTANTIATE_NOT_NULL(_, data)                                     \
  template void KerrHorizonConforming::operator()(                        \
      gsl::not_null<std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*> \
          result,                                                         \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_NOT_NULL,
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef INSTANTIATE_NOT_NULL

#define INSTANTIATE_JACOBIAN_NOT_NULL(_, data)                                \
  template void KerrHorizonConforming::jacobian(                              \
      gsl::not_null<                                                          \
          tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>*> \
          result,                                                             \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_JACOBIAN_NOT_NULL,
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE_JACOBIAN_NOT_NULL
}  // namespace domain::CoordinateMaps
