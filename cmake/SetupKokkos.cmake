# Distributed under the MIT License.
# See LICENSE.txt for details.

option(SPECTRE_KOKKOS "Use Kokkos" OFF)

if(SPECTRE_KOKKOS)
  set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
     "Kokkos aggressive vectorization")

   if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR SPECTRE_DEBUG)
     message(STATUS "Enabling Kokkos debug mode")
     set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
     set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
       "Bounds checking on Kokkos views")
     set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
       "Sanity checks on Kokkos DualView")
   endif()

  if(Kokkos_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL
      "Enable lambda expressions in CUDA")
  endif()

  find_package(Kokkos)

  if (NOT Kokkos_FOUND)
    if (NOT SPECTRE_FETCH_MISSING_DEPS)
      message(FATAL_ERROR "Could not find Kokkos. If you want to fetch "
        "missing dependencies automatically, set SPECTRE_FETCH_MISSING_DEPS=ON.")
    endif()
    message(STATUS "Fetching Kokkos")
    include(FetchContent)
    FetchContent_Declare(Kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG 4.4.00
      GIT_SHALLOW TRUE
      ${SPECTRE_FETCHCONTENT_BASE_ARGS}
    )
    FetchContent_MakeAvailable(Kokkos)
  endif()

  if (TARGET Kokkos::kokkos
      AND Kokkos_ENABLE_CUDA
      AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_property(TARGET Kokkos::kokkos
      APPEND PROPERTY
      INTERFACE_COMPILE_OPTIONS
      $<$<COMPILE_LANGUAGE:CXX>:
      -Xcudafe;"--diag_suppress=114,177,186,191,550,554,1301,1305,2189,3060">
    )
  endif()

  if (TARGET Kokkos::kokkos
      AND Kokkos_ENABLE_CUDA
      AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if (${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}
        VERSION_GREATER 12.0)
      message(STATUS "CUDA 12.1 is only partially supported by Clang")
      set_property(TARGET Kokkos::kokkos
        APPEND PROPERTY
        INTERFACE_COMPILE_OPTIONS
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-cuda-version>
      )
    endif()
  endif()
endif()
