# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(AnalyticSolutions)

set(LIBRARY "CceHelpers")

set(LIBRARY_SOURCES
  CceComputationTestHelpers.cpp
  KleinGordonBoundaryTestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  BoundaryTestHelpers.hpp
  CceComputationTestHelpers.hpp
  KleinGordonBoundaryTestHelpers.hpp
  WriteToWorldtubeH5.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  Cce
  DataStructuresHelpers
  Framework
  GeneralRelativity
  GeneralRelativitySolutions
  H5

  PUBLIC
  DataStructures
  Spectral
  SpinWeightedSphericalHarmonics
  Utilities
  )
