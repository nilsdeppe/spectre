# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "DenseTriggerHelpers")

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  TestTrigger.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit/
  HEADERS
  TestTrigger.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Charmxx::pup
  DataStructures
  EventsAndDenseTriggers
  Framework
  Options
  Serialization
  Utilities
  )
