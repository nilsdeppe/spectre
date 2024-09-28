# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizonFinder)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FastFlow.cpp
  InterpolationTarget.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComputeExcisionBoundaryVolumeQuantities.hpp
  ComputeExcisionBoundaryVolumeQuantities.tpp
  ComputeHorizonVolumeQuantities.hpp
  ComputeHorizonVolumeQuantities.tpp
  FastFlow.hpp
  HorizonAliases.hpp
  InterpolationTarget.hpp
  ObserveCenters.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Domain
  DomainCreators
  FunctionsOfTime
  GeneralRelativity
  GeneralizedHarmonic
  Initialization
  LinearOperators
  Observer
  Parallel
  ParallelInterpolation
  Printf
  Spectral

  PRIVATE
  Charmxx::pup

  PUBLIC
  DataStructures
  ErrorHandling
  GrSurfaces
  Logging
  Options
  SphericalHarmonics
  Utilities
)

add_subdirectory(Callbacks)
add_subdirectory(Python)
