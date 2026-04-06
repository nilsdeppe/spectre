// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"

namespace fd {
std::ostream& operator<<(std::ostream& os, DerivativeOrder der_order) {
  switch (der_order) {
    case DerivativeOrder::OneHigherThanReconsMnd:
      return os << "OneHigherThanReconsMnd";
    case DerivativeOrder::OneHigherThanReconsButFiveToFourMnd:
      return os << "OneHigherThanReconsButFiveToFourMnd";
    case DerivativeOrder::Two:
      return os << "2";
    case DerivativeOrder::FourMnd:
      return os << "4 MidpointAndNode";
    case DerivativeOrder::SixMnd:
      return os << "6 MidpointAndNode";
    case DerivativeOrder::EightMnd:
      return os << "8 MidpointAndNode";
    case DerivativeOrder::TenMnd:
      return os << "10 MidpointAndNode";
    case DerivativeOrder::FourMd:
      return os << "4 Midpoint";
    case DerivativeOrder::SixMd:
      return os << "6 Midpoint";
    case DerivativeOrder::EightMd:
      return os << "8 Midpoint";
    case DerivativeOrder::TenMd:
      return os << "10 Midpoint";
    default:
      ERROR("Unknown value for DerivativeOrder");
  };
}
}  // namespace fd

template <>
fd::DerivativeOrder
Options::create_from_yaml<fd::DerivativeOrder>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if (type_read == get_output(fd::DerivativeOrder::OneHigherThanReconsMnd)) {
    return fd::DerivativeOrder::OneHigherThanReconsMnd;
  } else if (type_read ==
             get_output(
                 fd::DerivativeOrder::OneHigherThanReconsButFiveToFourMnd)) {
    return fd::DerivativeOrder::OneHigherThanReconsButFiveToFourMnd;
  } else if (type_read == get_output(fd::DerivativeOrder::Two)) {
    return fd::DerivativeOrder::Two;
  } else if (type_read == get_output(fd::DerivativeOrder::FourMnd)) {
    return fd::DerivativeOrder::FourMnd;
  } else if (type_read == get_output(fd::DerivativeOrder::SixMnd)) {
    return fd::DerivativeOrder::SixMnd;
  } else if (type_read == get_output(fd::DerivativeOrder::EightMnd)) {
    return fd::DerivativeOrder::EightMnd;
  } else if (type_read == get_output(fd::DerivativeOrder::TenMnd)) {
    return fd::DerivativeOrder::TenMnd;
  } else if (type_read == get_output(fd::DerivativeOrder::FourMd)) {
    return fd::DerivativeOrder::FourMd;
  } else if (type_read == get_output(fd::DerivativeOrder::SixMd)) {
    return fd::DerivativeOrder::SixMd;
  } else if (type_read == get_output(fd::DerivativeOrder::EightMd)) {
    return fd::DerivativeOrder::EightMd;
  } else if (type_read == get_output(fd::DerivativeOrder::TenMd)) {
    return fd::DerivativeOrder::TenMd;
  }
  PARSE_ERROR(
      options.context(),
      "Failed to convert \""
          << type_read << "\" to DerivativeOrder. Must be one of '"
          << get_output(fd::DerivativeOrder::OneHigherThanReconsMnd) << "', '"
          << get_output(
                 fd::DerivativeOrder::OneHigherThanReconsButFiveToFourMnd)
          << "', '2', '4 MidpointAndNode', '6 MidpointAndNode', "
             "'8 MidpointAndNode', '10 MidpointAndNode', "
             "'4 Midpoint', '6 Midpoint', '8 Midpoint', or '10 Midpoint'.");
}
