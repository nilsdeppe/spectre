# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DgSubcell)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ActiveGrid.hpp
  CartesianFluxDivergence.hpp
  ComputeBoundaryTerms.hpp
  CorrectPackagedData.hpp
  DgSubcell.hpp
  Matrices.hpp
  Mesh.hpp
  NeighborData.hpp
  NeighborReconstructedFaceSolution.hpp
  PerssonTci.hpp
  PrepareNeighborData.hpp
  Projection.hpp
  RdmpTci.hpp
  Reconstruction.hpp
  SliceData.hpp
  SubcellOptions.hpp
  TciStatus.hpp
  TwoMeshRdmpTci.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ActiveGrid.cpp
  Matrices.cpp
  Mesh.cpp
  NeighborData.cpp
  PerssonTci.cpp
  Projection.cpp
  Reconstruction.cpp
  SliceData.cpp
  SubcellOptions.cpp
  TciStatus.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  IO
  Spectral

  INTERFACE
  Events
  Parallel

  PRIVATE
  Blas
  ErrorHandling
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(Events)
add_subdirectory(Tags)
