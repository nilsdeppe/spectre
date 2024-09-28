# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "DataStructuresHelpers")

set(LIBRARY_SOURCES
  MathWrapperDetail.cpp
  RandomUnitNormal.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  CustomStaticSizeVector.hpp
  MakeRandomVectorInMagnitudeRange.hpp
  MakeWithRandomValues.hpp
  MathWrapper.hpp
  MathWrapperDetail.hpp
  RandomUnitNormal.hpp
  TestTags.hpp
  VectorImplTestHelper.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Framework

  PUBLIC
  DataStructures
  Utilities
  )

add_subdirectory(DataBox)
