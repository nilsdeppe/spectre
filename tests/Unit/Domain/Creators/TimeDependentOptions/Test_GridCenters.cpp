// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/Creators/TimeDependentOptions/GridCenters.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/Creators/TimeDependent/TestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace domain::creators::time_dependent_options {

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.GridCenters",
                  "[Domain][Unit]") {
  domain::FunctionsOfTime::register_derived_with_charm();

  {
    const auto grid_centers_options =
        TestHelpers::test_option_tag<GridCentersOptions>(
            "SpecEvolutionParametersPerlFile: " + unit_test_src_path() +
            "../InputFiles/GrMhd/GhValenciaDivClean/EvolutionParameters.perl\n"
            "ScaleInspiralRateBy: Auto\n");
    REQUIRE(grid_centers_options.has_value());
    CHECK(std::get<GridCentersOptions>(*grid_centers_options).initial_values ==
          std::array{DataVector{16.1996, 3.59764e-5, 0.0, -16.2, 0.0, 0.0},
                     DataVector{-0.00008095, 0.0, 0.0, 0.00008095, 0.0, 0.0},
                     DataVector{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});
    CHECK(not std::get<GridCentersOptions>(*grid_centers_options)
                  .scale_inspiral_rate_by.has_value());
  }

  {
    const auto grid_centers_options =
        TestHelpers::test_option_tag<GridCentersOptions>(
            "SpecEvolutionParametersPerlFile: " + unit_test_src_path() +
            "../InputFiles/GrMhd/GhValenciaDivClean/EvolutionParameters.perl\n"
            "ScaleInspiralRateBy: 0.9\n");
    REQUIRE(grid_centers_options.has_value());
    CHECK(std::get<GridCentersOptions>(*grid_centers_options).initial_values ==
          std::array{DataVector{16.1996, 3.59764e-5, 0.0, -16.2, 0.0, 0.0},
                     DataVector{-0.00008095 * 0.9, 0.0, 0.0, 0.00008095 * 0.9,
                                0.0, 0.0},
                     DataVector{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});
    CHECK(std::get<GridCentersOptions>(*grid_centers_options)
              .scale_inspiral_rate_by.value() == 0.9);
  }
  {
    INFO("FromVolumeFile");
    constexpr size_t Dim = 3;
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};
    functions_of_time["GridCenters"] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
            0.0,
            std::array{DataVector{2 * Dim, 1.0}, DataVector{2 * Dim, 2.0},
                       DataVector{2 * Dim, 3.0}},
            100.0);
    const std::string filename{"NewFinalActualFinal2Final.h5"};
    const std::string subfile_name{"VolumeData"};
    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }

    TestHelpers::domain::creators::write_volume_data(filename, subfile_name,
                                                     functions_of_time);

    const auto grid_centers_map_options = TestHelpers::test_option_tag<
        domain::creators::time_dependent_options::GridCentersOptions>(
        "H5Filename: " + filename + "\nSubfileName: " + subfile_name);

    REQUIRE(grid_centers_map_options.has_value());
    CHECK(std::holds_alternative<FromVolumeFile>(
        grid_centers_map_options.value()));
    const std::array initial_values{DataVector{Dim, 1.0}, DataVector{Dim, 2.0},
                                    DataVector{Dim, 3.0}};

    const auto grid_centers_ptr =
        get_grid_centers(grid_centers_map_options.value(), 0.1, 65.8);

    const auto* grid_centers =
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
            grid_centers_ptr.get());

    CHECK(grid_centers != nullptr);

    CHECK(grid_centers->time_bounds() == std::array{0.1, 65.8});
    CHECK_ITERABLE_APPROX(
        grid_centers->func_and_2_derivs(0.3),
        functions_of_time.at("GridCenters")->func_and_2_derivs(0.3));

    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }
  }
}
}  // namespace domain::creators::time_dependent_options
