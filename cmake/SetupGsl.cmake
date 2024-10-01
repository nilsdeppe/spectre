# Distributed under the MIT License.
# See LICENSE.txt for details.

option(GSL_STATIC
  "Link static versions of GNU Scientific Library" OFF)

if(${GSL_STATIC})
  set(_BACKUP_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif(${GSL_STATIC})
find_package(GSL REQUIRED)
if(${GSL_STATIC})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_BACKUP_CMAKE_FIND_LIBRARY_SUFFIXES})
endif(${GSL_STATIC})

message(STATUS "GSL libs: ${GSL_LIBRARIES}")
message(STATUS "GSL incl: ${GSL_INCLUDE_DIR}")
message(STATUS "GSL vers: ${GSL_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "GSL version: ${GSL_VERSION}\n"
  )

set_property(
  TARGET GSL::gsl
  APPEND
  PROPERTY
  PUBLIC_HEADER
  gsl/gsl_cblas.h
  gsl/gsl_errno.h
  gsl/gsl_integration.h
  gsl/gsl_matrix_double.h
  gsl/gsl_multifit.h
  gsl/gsl_multiroots.h
  gsl/gsl_poly.h
  gsl/gsl_sf_bessel.h
  gsl/gsl_spline.h
  gsl/gsl_vector_double.h
)

# Link external BLAS library. We don't need the GSL::gslcblas target.
find_package(BLAS REQUIRED)
target_link_libraries(
  GSL::gsl
  INTERFACE
  BLAS::BLAS
  )

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  GSL::gsl GSL::gslcblas
  )
