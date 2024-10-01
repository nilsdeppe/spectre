# Distributed under the MIT License.
# See LICENSE.txt for details.

option(USE_SLEEF "Use Sleef to add more vectorized instructions." OFF)

if(USE_SLEEF)
  # Try to find Sleef to increase vectorization
  find_package(Sleef)
endif()

if(SLEEF_FOUND)
  message(STATUS "Sleef libs: ${SLEEF_LIBRARIES}")
  message(STATUS "Sleef incl: ${SLEEF_INCLUDE_DIR}")
  message(STATUS "Sleef vers: ${SLEEF_VERSION}")

  file(APPEND
    "${CMAKE_BINARY_DIR}/BuildInfo.txt"
    "Sleef version: ${SLEEF_VERSION}\n"
  )
endif()

# Every time we've upgraded blaze compatibility in the past, we've had to change
# vector code, so we should expect to need changes again on each subsequent
# release, so we should specify an exact version requirement. However, Blaze
# hasn't been consistent in naming releases (version 3.8.2 has 3.9.0 written
# in Version.h).
find_package(Blaze 3.8)

if (NOT Blaze_FOUND)
  if (NOT SPECTRE_FETCH_MISSING_DEPS)
    message(FATAL_ERROR "Could not find Blaze. If you want to fetch "
      "missing dependencies automatically, set SPECTRE_FETCH_MISSING_DEPS=ON.")
  endif()
  message(STATUS "Fetching Blaze")
  include(FetchContent)
  FetchContent_Declare(Blaze
    URL https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.8.2.tar.gz
    ${SPECTRE_FETCHCONTENT_BASE_ARGS}
  )
  # Configure Blaze CMake variables. Most configuration is done below.
  set(BLAZE_SHARED_MEMORY_PARALLELIZATION 0 CACHE INTERNAL "Blaze SMP mode")
  FetchContent_MakeAvailable(Blaze)
  set(BLAZE_INCLUDE_DIR ${blaze_SOURCE_DIR})
  set(BLAZE_VERSION "3.8.2")
endif()

message(STATUS "Blaze incl: ${BLAZE_INCLUDE_DIR}")
message(STATUS "Blaze vers: ${BLAZE_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "Blaze version: ${BLAZE_VERSION}\n"
  )

find_package(BLAS REQUIRED)
find_package(GSL REQUIRED)
find_package(LAPACK REQUIRED)

add_library(Blaze INTERFACE IMPORTED)
set_property(TARGET Blaze PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${BLAZE_INCLUDE_DIR})
target_link_libraries(
  Blaze
  INTERFACE
  BLAS::BLAS
  GSL::gsl # for BLAS header
  LAPACK::LAPACK
  )
set(_BLAZE_USE_SLEEF 0)

if(SLEEF_FOUND)
  target_link_libraries(
    Blaze
    INTERFACE
    Sleef
    )
  set_property(
    GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
    Sleef
    )
  set(_BLAZE_USE_SLEEF 1)
endif()

# If BLAZE_USE_STRONG_INLINE=ON, Blaze will use this keyword to increase the
# likelihood of inlining. If BLAZE_USE_STRONG_INLINE=OFF, uses inline keyword
# as a fallback.
option(BLAZE_USE_STRONG_INLINE "Increase likelihood of Blaze inlining." ON)

set(_BLAZE_USE_STRONG_INLINE 0)

if(BLAZE_USE_STRONG_INLINE)
  set(_BLAZE_USE_STRONG_INLINE 1)
endif()

# If BLAZE_USE_ALWAYS_INLINE=ON, Blaze will use this keyword to force inlining.
# If BLAZE_USE_ALWAYS_INLINE=OFF or if the platform being used cannot 100%
# guarantee inlining, uses BLAZE_STRONG_INLINE as a fallback.
option(BLAZE_USE_ALWAYS_INLINE "Force Blaze inlining." ON)

set(_BLAZE_USE_ALWAYS_INLINE 0)

if(BLAZE_USE_ALWAYS_INLINE)
  set(_BLAZE_USE_ALWAYS_INLINE 1)
endif()

# Configure Blaze. Some of the Blaze configuration options could be optimized
# for the machine we are running on. See documentation:
# https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation#!step-2-configuration
target_compile_definitions(Blaze
  INTERFACE
  # - Enable external BLAS kernels
  BLAZE_BLAS_MODE=1
  # - Use BLAS header from GSL. We could also find and include a <cblas.h> (or
  #   similarly named) header that may be distributed with the BLAS
  #   implementation, but it's not guaranteed to be available and may conflict
  #   with the GSL header. Since we use GSL anyway, it's easier to use their
  #   BLAS header.
  BLAZE_BLAS_INCLUDE_FILE=<gsl/gsl_cblas.h>
  # - Set default matrix storage order to column-major, since many of our
  #   functions are implemented for column-major layout. This default reduces
  #   conversions.
  BLAZE_DEFAULT_STORAGE_ORDER=blaze::columnMajor
  # - Disable SMP parallelization. This disables SMP parallelization for all
  #   possible backends (OpenMP, C++11 threads, Boost, HPX):
  #   https://bitbucket.org/blaze-lib/blaze/wiki/Serial%20Execution#!option-3-deactivation-of-parallel-execution
  BLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0
  # - Disable MPI parallelization
  BLAZE_MPI_PARALLEL_MODE=0
  # - Using the default cache size, which may have been configured automatically
  #   by the Blaze CMake configuration for the machine we are running on. We
  #   could override it here explicitly to tune performance.
  # BLAZE_CACHE_SIZE
  # - Disable padding for dynamic matrices.
  #   Blaze warns that this may decrease performance:
  #   https://bitbucket.org/blaze-lib/blaze/src/c4d9e85414370e880e5e79c86e3c8d4d38dcde7a/blaze/config/Optimizations.h#lines-52
  #   We haven't tested this much, so we may want to try enabling padding again.
  #   To support padding, explicit calls to LAPACK functions need to pass
  #   `.spacing()` instead of `.rows()/.columns()` to the `LDA`, `LDB`, etc.
  #   parameters (see `[matrix_spacing]` in `Test_Spectral.cpp`).
  BLAZE_USE_PADDING=0
  # - Always enable non-temporal stores for cache optimization of large data
  #   structures: https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20Files#!streaming-non-temporal-stores
  BLAZE_USE_STREAMING=1
  # - Skip initializing default-constructed structures for fundamental types
  BLAZE_USE_DEFAULT_INITIALIZATON=0
  # Use Sleef for vectorization of more math functions
  BLAZE_USE_SLEEF=${_BLAZE_USE_SLEEF}
  # Set inlining settings
  BLAZE_USE_STRONG_INLINE=${_BLAZE_USE_STRONG_INLINE}
  BLAZE_USE_ALWAYS_INLINE=${_BLAZE_USE_ALWAYS_INLINE}
  )

# We need to make sure `BlazeExceptions.hpp` is included. It is included in the
# PCH (see tools/SpectrePch.hpp). If there's no PCH, we need to include it here.
if (NOT USE_PCH)
  target_compile_options(Blaze
    INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include Utilities/BlazeExceptions.hpp>")
endif()

add_interface_lib_headers(
  TARGET Blaze
  HEADERS
  blaze/math/AlignmentFlag.h
  blaze/math/Column.h
  blaze/math/CompressedMatrix.h
  blaze/math/CompressedVector.h
  blaze/math/CustomVector.h
  blaze/math/DenseMatrix.h
  blaze/math/DenseVector.h
  blaze/math/DynamicMatrix.h
  blaze/math/DynamicVector.h
  blaze/math/GroupTag.h
  blaze/math/Matrix.h
  blaze/math/PaddingFlag.h
  blaze/math/StaticMatrix.h
  blaze/math/StaticVector.h
  blaze/math/Submatrix.h
  blaze/math/Subvector.h
  blaze/math/TransposeFlag.h
  blaze/math/Vector.h
  blaze/math/constraints/SIMDPack.h
  blaze/math/lapack/trsv.h
  blaze/math/traits/MultTrait.h
  blaze/math/typetraits/IsColumnMajorMatrix.h
  blaze/math/typetraits/IsDenseMatrix.h
  blaze/math/typetraits/IsDenseMatrix.h
  blaze/math/typetraits/IsSparseMatrix.h
  blaze/math/typetraits/IsVector.h
  blaze/math/simd/BasicTypes.h
  blaze/system/Inline.h
  blaze/system/Optimizations.h
  blaze/system/Vectorization.h
  blaze/system/Version.h
  blaze/util/typetraits/RemoveConst.h
  )

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Blaze
  )
