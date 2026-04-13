// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/PolarToCartesian.hpp"

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
PolarToCartesian::PolarToCartesian() = default;
PolarToCartesian::PolarToCartesian(PolarToCartesian&&) = default;
PolarToCartesian::PolarToCartesian(const PolarToCartesian&) = default;
PolarToCartesian& PolarToCartesian::operator=(const PolarToCartesian&) =
    default;
PolarToCartesian& PolarToCartesian::operator=(PolarToCartesian&&) = default;

template <typename T>
void PolarToCartesian::operator()(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 2>*> result,
    const std::array<T, 2>& source_coords) const {
  if constexpr (std::is_same_v<tt::remove_cvref_wrap_t<T>, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    (*result)[0].destructive_resize(size);
    (*result)[1].destructive_resize(size);
  }
  const auto& [r, phi] = source_coords;
  (*result)[0] = r * cos(phi);
  (*result)[1] = r * sin(phi);
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 2> PolarToCartesian::operator()(
    const std::array<T, 2>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  std::array<ReturnType, 2> result{};
  (*this)(make_not_null(&result), source_coords);
  return result;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
std::optional<std::array<double, 2>> PolarToCartesian::inverse(
    const std::array<double, 2>& target_coords) const {
  const auto& [x, y] = target_coords;
  if (UNLIKELY(y == 0.0 and x == 0.0)) {
    return std::array{0.0, 0.0};
  } else {
    const double r = std::hypot(x, y);
    const double phi = atan2(y, x);
    return std::array{r, phi < 0.0 ? phi + 2.0 * M_PI : phi};
  }
}

template <typename T>
void PolarToCartesian::jacobian(
    const gsl::not_null<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame>*>
        result,
    const std::array<T, 2>& source_coords) const {
  if constexpr (std::is_same_v<tt::remove_cvref_wrap_t<T>, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    for (auto& component : *result) {
      component.destructive_resize(size);
    }
  }
  const auto& [r, phi] = source_coords;
  const auto& cos_phi = get<0, 0>(*result) = cos(phi);
  const auto& sin_phi = get<1, 0>(*result) = sin(phi);
  get<0, 1>(*result) = -r * sin_phi;
  get<1, 1>(*result) = r * cos_phi;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame>
PolarToCartesian::jacobian(const std::array<T, 2>& source_coords) const {
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> result{};
  jacobian(make_not_null(&result), source_coords);
  return result;
}

void PolarToCartesian::pup(PUP::er& /*p*/) {}

bool operator==(const PolarToCartesian& /*lhs*/,
                const PolarToCartesian& /*rhs*/) {
  return true;
}

bool operator!=(const PolarToCartesian& lhs, const PolarToCartesian& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_DTYPE(_, data)                                            \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 2>                \
  PolarToCartesian::operator()(                                               \
      const std::array<DTYPE(data), 2>& source_coords) const;                 \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 2, Frame::NoFrame>  \
  PolarToCartesian::jacobian(const std::array<DTYPE(data), 2>& source_coords) \
      const;

GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE,
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef INSTANTIATE_DTYPE

#define INSTANTIATE_NOT_NULL(_, data)                                     \
  template void PolarToCartesian::operator()(                             \
      gsl::not_null<std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 2>*> \
          result,                                                         \
      const std::array<DTYPE(data), 2>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_NOT_NULL,
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef INSTANTIATE_NOT_NULL

#define INSTANTIATE_JACOBIAN_NOT_NULL(_, data)                                \
  template void PolarToCartesian::jacobian(                                   \
      gsl::not_null<                                                          \
          tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 2, Frame::NoFrame>*> \
          result,                                                             \
      const std::array<DTYPE(data), 2>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_JACOBIAN_NOT_NULL,
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE_JACOBIAN_NOT_NULL
}  // namespace domain::CoordinateMaps
