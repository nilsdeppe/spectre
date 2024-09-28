# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ElasticitySolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BentBeam.cpp
  HalfSpaceMirror.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BentBeam.hpp
  Factory.hpp
  HalfSpaceMirror.hpp
  Zero.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  InitialDataUtilities
  LinearOperators
  Options
  Serialization

  PRIVATE
  ElasticityPointwiseFunctions
  GSL::gsl
  Integration

  PUBLIC
  Charmxx::pup
  ConstitutiveRelations
  DataStructures
  Elasticity
  ErrorHandling
  Utilities
  )
