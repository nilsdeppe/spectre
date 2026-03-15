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
    case DerivativeOrder::OneHigherThanRecons:
      return os << "OneHigherThanRecons";
    case DerivativeOrder::OneHigherThanReconsButFiveToFour:
      return os << "OneHigherThanReconsButFiveToFour";
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
  if (type_read == get_output(fd::DerivativeOrder::OneHigherThanRecons)) {
    return fd::DerivativeOrder::OneHigherThanRecons;
  } else if (type_read ==
             get_output(
                 fd::DerivativeOrder::OneHigherThanReconsButFiveToFour)) {
    return fd::DerivativeOrder::OneHigherThanReconsButFiveToFour;
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
  }
  PARSE_ERROR(
      options.context(),
      "Failed to convert \""
          << type_read << "\" to DerivativeOrder. Must be one of '"
          << get_output(fd::DerivativeOrder::OneHigherThanRecons) << "', '"
          << get_output(fd::DerivativeOrder::OneHigherThanReconsButFiveToFour)
          << "', '2', '4 MidpointAndNode', '6 MidpointAndNode', "
             "'8 MidpointAndNode', or '10 MidpointAndNode'.");
}
