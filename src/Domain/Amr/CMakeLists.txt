# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Amr)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Flag.cpp
  Helpers.cpp
  Info.cpp
  NeighborsOfChild.cpp
  NeighborsOfParent.cpp
  NewNeighborIds.cpp
  UpdateAmrDecision.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Amr.hpp
  Flag.hpp
  Helpers.hpp
  Info.hpp
  NeighborsOfChild.hpp
  NeighborsOfParent.hpp
  NewNeighborIds.hpp
  UpdateAmrDecision.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Charmxx::pup
  DomainStructure
  ErrorHandling
  Options
  Utilities
  PUBLIC
  Boost::boost
  DataStructures
  Spectral
  )

add_subdirectory(Tags)
