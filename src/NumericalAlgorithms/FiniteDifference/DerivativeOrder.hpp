// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace fd {
/// Controls which FD derivative order is used.
enum class DerivativeOrder : int {
  /// \brief Use one order high derivative.
  ///
  /// For example, if fifth order reconstruction is used, then a sixth-order
  /// derivative is used.
  OneHigherThanRecons = -1,
  /// \brief Same as `OneHigherThanRecons` except uses a fourth-order derivative
  /// if fifth-order reconstruction was used.
  OneHigherThanReconsButFiveToFour = -2,
  /// \brief Use 2nd order derivatives
  Two = 2,
  /// \brief Use 4th order midpoint-and-node-to-node-difference derivatives
  FourMnd = 4,
  /// \brief Use 6th order midpoint-and-node-to-node-difference derivatives
  SixMnd = 6,
  /// \brief Use 8th order midpoint-and-node-to-node-difference derivatives
  EightMnd = 8,
  /// \brief Use 10th order midpoint-and-node-to-node-difference derivatives
  TenMnd = 10,
  /// \brief Use 4th order midpoint-only derivatives
  FourMd = 14,
  /// \brief Use 6th order midpoint-only derivatives
  SixMd = 16,
  /// \brief Use 8th order midpoint-only derivatives
  EightMd = 18,
  /// \brief Use 10th order midpoint-only derivatives
  TenMd = 20
};

std::ostream& operator<<(std::ostream& os, DerivativeOrder der_order);

/// Returns the FD order corresponding to a `DerivativeOrder` value.
///
/// For adaptive orders (`OneHigherThanRecons` and
/// `OneHigherThanReconsButFiveToFour`) the raw negative int value is returned.
constexpr int fd_order(const DerivativeOrder der_order) {
  switch (der_order) {
    case DerivativeOrder::OneHigherThanRecons:
      return static_cast<int>(DerivativeOrder::OneHigherThanRecons);
    case DerivativeOrder::OneHigherThanReconsButFiveToFour:
      return static_cast<int>(
          DerivativeOrder::OneHigherThanReconsButFiveToFour);
    case DerivativeOrder::Two:
      return 2;
    case DerivativeOrder::FourMnd:
    case DerivativeOrder::FourMd:
      return 4;
    case DerivativeOrder::SixMnd:
    case DerivativeOrder::SixMd:
      return 6;
    case DerivativeOrder::EightMnd:
    case DerivativeOrder::EightMd:
      return 8;
    case DerivativeOrder::TenMnd:
    case DerivativeOrder::TenMd:
      return 10;
    default:
      return static_cast<int>(der_order);
  }
}
}  // namespace fd

template <>
struct Options::create_from_yaml<fd::DerivativeOrder> {
  template <typename Metavariables>
  static fd::DerivativeOrder create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
fd::DerivativeOrder
Options::create_from_yaml<fd::DerivativeOrder>::create<void>(
    const Options::Option& options);
