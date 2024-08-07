# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY KokkosTest)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Test.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  SpectreKokkos
  )
