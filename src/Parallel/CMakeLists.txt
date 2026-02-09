# Distributed under the MIT License.
# See LICENSE.txt for details.

if ("${SPECTRE_PARALLEL_LIB}" STREQUAL "charm")
  add_charm_module(GlobalCache)
  add_charm_module(Main)
else()
  add_library(module_GlobalCache INTERFACE)
  add_library(module_Main INTERFACE)
endif()

set(LIBRARY Parallel)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ArrayComponentId.cpp
  CharmRegistration.cpp
  ExitCode.cpp
  InitializationFunctions.cpp
  Main.cpp
  NodeLock.cpp
  Phase.cpp
  Reduction.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AlgorithmExecution.hpp
  AlgorithmMetafunctions.hpp
  ArrayComponentId.hpp
  ArrayIndex.hpp
  Callback.hpp
  CharmMain.tpp
  CharmRegistration.hpp
  CreateFromOptions.hpp
  DistributedObject.hpp
  DomainDiagnosticInfo.hpp
  ElementRegistration.hpp
  ExitCode.hpp
  FifoCache.hpp
  GetSection.hpp
  GlobalCache.hpp
  GlobalCacheDeclare.hpp
  InboxInserters.hpp
  Info.hpp
  InitializationFunctions.hpp
  InitializationTag.hpp
  Invoke.hpp
  Local.hpp
  Main.hpp
  MaxInlineMethodsReached.hpp
  MultiReaderSpinlock.hpp
  NodeLock.hpp
  OutputInbox.hpp
  ParallelComponentHelpers.hpp
  Phase.hpp
  PhaseControlReductionHelpers.hpp
  PhaseDependentActionList.hpp
  Reduction.hpp
  ReductionDeclare.hpp
  ResourceInfo.hpp
  Section.hpp
  Spinlock.hpp
  StaticSpscQueue.hpp
  TypeTraits.hpp
  )

# Depends only on relatively low-level libraries so that compiling small
# executables (and low-level unit tests) doesn't require building the entire
# code base. E.g. doesn't depend on Domain.
target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  Boost::program_options
  DataStructures
  Informer
  PUBLIC
  Charmxx::charmxx
  ErrorHandling
  Options
  Printf
  Serialization
  SystemUtilities
  Utilities
  )

if ("${SPECTRE_PARALLEL_LIB}" STREQUAL "charm")
  add_dependencies(
    ${LIBRARY}
    module_GlobalCache
    module_Main          # GlobalCache module depends on Main module
  )
endif()

if ("${SPECTRE_PARALLEL_LIB}" STREQUAL "charm")
  add_subdirectory(Algorithms)
endif()
add_subdirectory(ArrayCollection)
add_subdirectory(MemoryMonitor)
add_subdirectory(PhaseControl)
add_subdirectory(Printf)
add_subdirectory(Protocols)
add_subdirectory(Tags)
