# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "DomainHelpers")

set(LIBRARY_SOURCES
  DomainTestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit/
  HEADERS
  CoordinateMaps/TestMapHelpers.hpp
  Creators/TestHelpers.hpp
  Creators/TimeDependence/TestHelpers.hpp
  DomainTestHelpers.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  CoordinateMaps
  DataStructuresHelpers
  DomainTimeDependence

  PRIVATE
  DomainBoundaryConditions
  ErrorHandling
  Spectral

  PUBLIC
  DataStructures
  Domain
  DomainBoundaryConditionsHelpers
  DomainCreators
  DomainStructure
  Framework
  FunctionsOfTime
  Utilities
  )

add_subdirectory(Amr)
add_subdirectory(BoundaryConditions)
add_subdirectory(Structure)
