# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EllipticDgSubdomainOperator)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  InitializeSubdomain.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InitializeSubdomain.hpp
  SubdomainOperator.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  Charmxx::pup
  Domain
  DomainCreators
  Elliptic
  EllipticDg
  EllipticSystems
  ErrorHandling
  Parallel
  Serialization

  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  DomainStructure
  ParallelSchwarz
  Spectral
  Utilities
)
