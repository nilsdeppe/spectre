// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <fd::DerivativeOrder DerivOrder>
void test_construct_from_options(const std::string& expected_output) {
  CHECK(get_output(DerivOrder) == expected_output);
  const auto created =
      TestHelpers::test_creation<fd::DerivativeOrder>(get_output(DerivOrder));
  CHECK(created == DerivOrder);
}
}  // namespace


SPECTRE_TEST_CASE("Unit.FiniteDifference.DerivativeOrder",
                  "[Unit][NumericalAlgorithms]") {
  test_construct_from_options<fd::DerivativeOrder::OneHigherThanReconsMnd>(
      "OneHigherThanReconsMnd");
  test_construct_from_options<
      fd::DerivativeOrder::OneHigherThanReconsButFiveToFourMnd>(
      "OneHigherThanReconsButFiveToFourMnd");
  test_construct_from_options<fd::DerivativeOrder::Two>("2");
  test_construct_from_options<fd::DerivativeOrder::FourMnd>(
      "4 MidpointAndNode");
  test_construct_from_options<fd::DerivativeOrder::SixMnd>("6 MidpointAndNode");
  test_construct_from_options<fd::DerivativeOrder::EightMnd>(
      "8 MidpointAndNode");
  test_construct_from_options<fd::DerivativeOrder::TenMnd>(
      "10 MidpointAndNode");
  test_construct_from_options<fd::DerivativeOrder::FourMd>("4 Midpoint");
  test_construct_from_options<fd::DerivativeOrder::SixMd>("6 Midpoint");
  test_construct_from_options<fd::DerivativeOrder::EightMd>("8 Midpoint");
  test_construct_from_options<fd::DerivativeOrder::TenMd>("10 Midpoint");

  // Test fd_order() helper
  CHECK(fd::fd_order(fd::DerivativeOrder::Two) == 2);
  CHECK(fd::fd_order(fd::DerivativeOrder::FourMnd) == 4);
  CHECK(fd::fd_order(fd::DerivativeOrder::FourMd) == 4);
  CHECK(fd::fd_order(fd::DerivativeOrder::SixMnd) == 6);
  CHECK(fd::fd_order(fd::DerivativeOrder::SixMd) == 6);
  CHECK(fd::fd_order(fd::DerivativeOrder::EightMnd) == 8);
  CHECK(fd::fd_order(fd::DerivativeOrder::EightMd) == 8);
  CHECK(fd::fd_order(fd::DerivativeOrder::TenMnd) == 10);
  CHECK(fd::fd_order(fd::DerivativeOrder::TenMd) == 10);
  CHECK(fd::fd_order(fd::DerivativeOrder::OneHigherThanReconsMnd) == -1);
  CHECK(fd::fd_order(
            fd::DerivativeOrder::OneHigherThanReconsButFiveToFourMnd) == -2);
}
