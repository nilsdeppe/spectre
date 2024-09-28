# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EllipticDg)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Initialization.cpp
  Penalty.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DgElementArray.hpp
  DgOperator.hpp
  Initialization.hpp
  Penalty.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  AmrProjectors
  Charmxx::charmxx
  DomainCreators
  Elliptic
  EllipticSystems
  FunctionsOfTime
  Options
  Parallel
  ParallelAmr
  Printf
  SystemUtilities

  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainStructure
  ErrorHandling
  LinearOperators
  Spectral
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(SubdomainOperator)
