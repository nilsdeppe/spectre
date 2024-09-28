# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Burgers)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  Divergence.cpp
  Fluxes.cpp
  TimeDerivativeTerms.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  Fluxes.hpp
  System.hpp
  Tags.hpp
  TimeDerivativeTerms.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  AnalyticData
  AnalyticSolutions
  CoordinateMaps
  DomainCreators
  EvolutionBoundaryConditions
  FunctionsOfTime
  InitialDataUtilities
  Options
  Parallel

  PRIVATE
  BurgersAnalyticData
  BurgersSolutions
  FiniteDifference
  LinearOperators

  PUBLIC
  Charmxx::pup
  DataStructures
  DgSubcell
  DiscontinuousGalerkin
  Domain
  DomainBoundaryConditions
  DomainStructure
  ErrorHandling
  Evolution
  Serialization
  Spectral
  Utilities
)

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(FiniteDifference)
add_subdirectory(Subcell)
