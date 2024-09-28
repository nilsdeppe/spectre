# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "ControlSystemHelpers")

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  Examples.hpp
  SystemHelpers.hpp
  TestStructs.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Actions
  Charmxx::pup
  ControlSystem
  ControlSystemMeasurements
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  ErrorHandling
  EventsAndTriggers
  Framework
  FunctionsOfTime
  Observer
  Options
  Parallel
  Serialization
  SphericalHarmonics
  Time
  Utilities
  )
