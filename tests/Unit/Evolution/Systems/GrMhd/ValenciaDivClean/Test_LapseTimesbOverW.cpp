// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/LapseTimesbOverW.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.LapseTimesbOverW",
                  "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  pypp::check_with_random_values<1>(
      &grmhd::ValenciaDivClean::
          magnetic_field_dot_spatial_velocity_and_lapse_times_b_over_w,
      "LapseTimesbOverW",
      {"magnetic_field_dot_spatial_velocity", "lapse_b_over_w"}, {{{0.1, 1.0}}},
      DataVector{5});
}
