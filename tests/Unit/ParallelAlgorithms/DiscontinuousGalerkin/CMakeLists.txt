# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelDiscontinuousGalerkin")

set(LIBRARY_SOURCES
  Test_CollectDataForFluxes.cpp
  Test_FluxCommunication.cpp
  Test_InitializeInterfaces.cpp
  Test_InitializeMortars.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "CoordinateMaps;DataStructures;Domain;DiscontinuousGalerkin;EllipticDg;ErrorHandling;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
