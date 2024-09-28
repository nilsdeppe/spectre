# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY AmrCriteria)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Constraints.cpp
  DriveToTarget.cpp
  IncreaseResolution.cpp
  Loehner.cpp
  Persson.cpp
  Random.cpp
  TruncationError.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Constraints.hpp
  Criteria.hpp
  Criterion.hpp
  DriveToTarget.hpp
  Factory.hpp
  IncreaseResolution.hpp
  Loehner.hpp
  Persson.hpp
  Random.hpp
  TruncationError.hpp
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Domain
  Events
  Options
  Parallel
  ParallelAmr
  Serialization

  PRIVATE
  ErrorHandling
  LinearOperators

  PUBLIC
  Amr
  Charmxx::pup
  DataStructures
  DomainStructure
  Spectral
  Utilities
  )

add_subdirectory(Tags)
