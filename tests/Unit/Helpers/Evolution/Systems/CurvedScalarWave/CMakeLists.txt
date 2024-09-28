# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "CurvedScalarWaveHelpers")

set(LIBRARY_SOURCES
  TestHelpers.cpp
  Worldtube/TestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit/
  HEADERS
  TestHelpers.hpp
  Worldtube/TestHelpers.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DomainHelpers

  PUBLIC
  DataStructures
  DomainBoundaryConditionsHelpers
  DomainCreators
  Framework
  Utilities
  )
