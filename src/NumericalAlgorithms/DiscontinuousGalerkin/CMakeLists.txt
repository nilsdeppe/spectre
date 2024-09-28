# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DiscontinuousGalerkin)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ApplyMassMatrix.cpp
  Formulation.cpp
  InterpolateFromBoundary.cpp
  LiftFromBoundary.cpp
  MetricIdentityJacobian.cpp
  MortarHelpers.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApplyMassMatrix.hpp
  Formulation.hpp
  HasReceivedFromAllMortars.hpp
  InterpolateFromBoundary.hpp
  LiftFlux.hpp
  LiftFromBoundary.hpp
  MetricIdentityJacobian.hpp
  MortarHelpers.hpp
  NormalDotFlux.hpp
  ProjectToBoundary.hpp
  SimpleBoundaryData.hpp
  SimpleMortarData.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Charmxx::pup
  Serialization
  SpatialDiscretization

  PUBLIC
  DataStructures
  DomainStructure
  ErrorHandling
  Options
  Spectral
  Utilities
  )

add_subdirectory(Tags)
