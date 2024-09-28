# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EllipticSubdomainPreconditioners)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  RegisterDerived.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  MinusLaplacian.hpp
  RegisterDerived.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  Convergence
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  Elliptic
  EllipticDgSubdomainOperator
  ErrorHandling
  Options
  PoissonBoundaryConditions

  PUBLIC
  LinearSolver
  ParallelSchwarz
  Poisson
  Serialization
  Utilities
  )
