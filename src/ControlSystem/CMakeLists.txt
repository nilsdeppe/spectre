# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ControlSystem)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Averager.cpp
  CalculateMeasurementTimescales.cpp
  Controller.cpp
  ExpirationTimes.cpp
  FutureMeasurements.cpp
  TimescaleTuner.cpp
  UpdateFunctionOfTime.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Averager.hpp
  CalculateMeasurementTimescales.hpp
  CombinedName.hpp
  Component.hpp
  Controller.hpp
  ExpirationTimes.hpp
  FutureMeasurements.hpp
  IsSize.hpp
  Metafunctions.hpp
  NamespaceDocs.hpp
  RunCallbacks.hpp
  TimescaleTuner.hpp
  Trigger.hpp
  UpdateControlSystem.hpp
  UpdateFunctionOfTime.hpp
  WriteData.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Actions
  ApparentHorizonFinder
  Boost::boost
  ControlSystemMeasurements
  CoordinateMaps
  Domain
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  GeneralRelativity
  Logging
  Observer
  Options
  Parallel
  ParallelInterpolation
  Printf
  Time

  PRIVATE
  FiniteDifference

  PUBLIC
  Charmxx::pup
  DataStructures
  DomainStructure
  ErrorHandling
  FunctionsOfTime
  GrSurfaces
  Interpolation
  Serialization
  SphericalHarmonics
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(ControlErrors)
add_subdirectory(Measurements)
add_subdirectory(Protocols)
add_subdirectory(Systems)
add_subdirectory(Tags)
