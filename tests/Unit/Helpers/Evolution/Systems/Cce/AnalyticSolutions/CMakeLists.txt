# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "CceAnalyticSolutionsHelpers")

set(LIBRARY_SOURCES
  AnalyticDataHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  AnalyticDataHelpers.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Framework
  GeneralRelativity
  SpinWeightedSphericalHarmonics

  PUBLIC
  Cce
  DataStructures
  Utilities
  )
