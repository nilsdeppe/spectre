# Distributed under the MIT License.
# See LICENSE.txt for details.

if ("${SPECTRE_PARALLEL_LIB}" STREQUAL "charm")
  add_charm_module(Printf)
endif()

set(LIBRARY Printf)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Printf.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Printf.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ErrorHandling
  INTERFACE
  Utilities
  PRIVATE
  SystemUtilities
  )

if ("${SPECTRE_PARALLEL_LIB}" STREQUAL "charm")
  add_dependencies(
    ${LIBRARY}
    module_Printf
  )
endif()
