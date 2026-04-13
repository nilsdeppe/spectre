// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {
SphericalToCartesianPfaffian::SphericalToCartesianPfaffian() = default;
SphericalToCartesianPfaffian::SphericalToCartesianPfaffian(
    SphericalToCartesianPfaffian&&) = default;
SphericalToCartesianPfaffian::SphericalToCartesianPfaffian(
    const SphericalToCartesianPfaffian&) = default;
SphericalToCartesianPfaffian& SphericalToCartesianPfaffian::operator=(
    const SphericalToCartesianPfaffian&) = default;
SphericalToCartesianPfaffian& SphericalToCartesianPfaffian::operator=(
    SphericalToCartesianPfaffian&&) = default;

template <typename T>
void SphericalToCartesianPfaffian::operator()(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*> result,
    const std::array<T, 3>& source_coords) const {
  if constexpr (std::is_same_v<tt::remove_cvref_wrap_t<T>, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    (*result)[0].destructive_resize(size);
    (*result)[1].destructive_resize(size);
    (*result)[2].destructive_resize(size);
  }
  const auto& [r, theta, phi] = source_coords;
  (*result)[0] = r * sin(theta) * cos(phi);
  (*result)[1] = r * sin(theta) * sin(phi);
  (*result)[2] = r * cos(theta);
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3>
SphericalToCartesianPfaffian::operator()(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  std::array<ReturnType, 3> result{};
  (*this)(make_not_null(&result), source_coords);
  return result;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
std::optional<std::array<double, 3>> SphericalToCartesianPfaffian::inverse(
    const std::array<double, 3>& target_coords) const {
  const auto& [x, y, z] = target_coords;
  if (UNLIKELY(y == 0.0 and x == 0.0)) {
    if (UNLIKELY(z == 0.0)) {
      return std::array{0.0, 0.5 * M_PI, 0.0};
    } else {
      return std::array{std::abs(z), z > 0.0 ? 0.0 : M_PI, 0.0};
    }
  } else {
    const double r = std::hypot(x, y, z);
    const double phi = atan2(y, x);
    return std::array{r, acos(z / r), phi < 0.0 ? phi + 2.0 * M_PI : phi};
  }
}

template <typename T>
void SphericalToCartesianPfaffian::jacobian(
    const gsl::not_null<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>*>
        result,
    const std::array<T, 3>& source_coords) const {
  const auto& [r, theta, phi] = source_coords;
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  if constexpr (std::is_same_v<ReturnType, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    for (auto& component : *result) {
      component.destructive_resize(size);
    }
  }
  // Zero all components first (get<2, 2> is intentionally left zero)
  for (auto& component : *result) {
    component = 0.0;
  }
  // Pfaffian basis means phi components are 1 / sin_theta times those of a
  // coord basis
  const auto& cos_theta = get<2, 0>(*result) = cos(theta);
  const auto& sin_theta = get<2, 1>(*result) = sin(theta);
  const auto& cos_phi = get<1, 2>(*result) = cos(phi);
  const auto& sin_phi = get<0, 2>(*result) = sin(phi);
  get<0, 0>(*result) = sin_theta * cos_phi;
  get<1, 0>(*result) = sin_theta * sin_phi;
  get<0, 1>(*result) = r * cos_theta * cos_phi;
  get<1, 1>(*result) = r * cos_theta * sin_phi;
  get<2, 1>(*result) *= -r;
  get<0, 2>(*result) *= -r;
  get<1, 2>(*result) *= r;
  // get<2, 2>(*result) is zero
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
SphericalToCartesianPfaffian::jacobian(
    const std::array<T, 3>& source_coords) const {
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> result{};
  jacobian(make_not_null(&result), source_coords);
  return result;
}

void SphericalToCartesianPfaffian::pup(PUP::er& /*p*/) {}

bool operator==(const SphericalToCartesianPfaffian& /*lhs*/,
                const SphericalToCartesianPfaffian& /*rhs*/) {
  return true;
}

bool operator!=(const SphericalToCartesianPfaffian& lhs,
                const SphericalToCartesianPfaffian& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_DTYPE(_, data)                                           \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  SphericalToCartesianPfaffian::operator()(                                  \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  SphericalToCartesianPfaffian::jacobian(                                    \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE,
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef INSTANTIATE_DTYPE

#define INSTANTIATE_NOT_NULL(_, data)                                     \
  template void SphericalToCartesianPfaffian::operator()(                 \
      gsl::not_null<std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*> \
          result,                                                         \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_NOT_NULL,
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef INSTANTIATE_NOT_NULL

#define INSTANTIATE_JACOBIAN_NOT_NULL(_, data)                                \
  template void SphericalToCartesianPfaffian::jacobian(                       \
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
