# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ConstitutiveRelations)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CubicCrystal.cpp
  IsotropicHomogeneous.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ConstitutiveRelation.hpp
  CubicCrystal.hpp
  Factory.hpp
  IsotropicHomogeneous.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Options
  Serialization

  PRIVATE
  ErrorHandling

  PUBLIC
  Charmxx::pup
  DataStructures
  Utilities
  )
