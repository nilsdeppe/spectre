## Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE SphericalCollapse)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  SphericalCollapse.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Boost::boost
  Boost::program_options
  CoordinateMaps
  DataStructures
  Domain
  DomainStructure
  IO
  Interpolation
  Informer
  LinearSolver
  Spectral
  Utilities
  )

# add_dependencies(test-executables ${EXECUTABLE})
