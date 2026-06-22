################################################################################
#
# \file      cmake/CodeCoverageDetection.cmake
# \author    J. Bakosi
# \copyright 2012-2015, Jozsef Bakosi, 2016, Los Alamos National Security, LLC.
# \brief     Detect prerequesites for code coverage analysis
# \date      Fri 03 Mar 2017 11:50:24 AM MST
#
# Modifications for SpECTRE:
# 1) Auto find either llvm-cov or llvm-cov-${LLVM_VERSION} instead of using
#    a shell script that hard codes the LLVM version
# 2) Formatting changes
################################################################################

# Attempt to find tools required for code coverage analysis

if(NOT LLVM_COV_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(LLVM_COV_ROOT "")
  set(LLVM_COV_ROOT $ENV{LLVM_COV_ROOT})
endif()

option(COVERAGE "Enable code coverage analysis." OFF)

# Whether to also generate the HTML coverage report. The lcov `.info` file
# (consumed by codecov) is always produced; the HTML report is extra work that
# CI doesn't need, so it can be disabled with `-D COVERAGE_HTML_REPORT=OFF`.
option(COVERAGE_HTML_REPORT
  "Generate the HTML coverage report in addition to the lcov .info file." ON)

# Number of workers passed to lcov 2.x `--parallel` when processing the
# per-translation-unit gcov data during capture. The capture is the dominant
# cost of the coverage target, so raising this (e.g. to the number of available
# cores) substantially speeds it up. Defaults to 1 (serial).
set(SPECTRE_LCOV_CORES "1" CACHE STRING
  "Number of parallel workers lcov uses when capturing coverage data.")

# Code coverage analysis only supported if all prerequisites found and the user
# has requested it via the cmake variable COVERAGE=on..
if(COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
    string(
      REGEX MATCH "^[0-9]+.[0-9]+" LLVM_VERSION
      "${CMAKE_CXX_COMPILER_VERSION}"
      )
    find_program(
      LLVM_COV_BIN
      NAMES "llvm-cov-${LLVM_VERSION}" "llvm-cov"
      HINTS ${LLVM_COV_ROOT}
      )
    configure_file(
      "${CMAKE_SOURCE_DIR}/tools/llvm-gcov.sh"
      "${CMAKE_BINARY_DIR}/llvm-gcov.sh"
      )
    set(GCOV "${CMAKE_BINARY_DIR}/llvm-gcov.sh")
  elseif( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
    # Use the gcov matching the compiler version. The unversioned `gcov` on the
    # system may belong to a different GCC than the one used to build, which
    # produces .gcno/.gcda files gcov cannot read (version mismatch warnings
    # like "version 'B05*', prefer version 'B14*'" and empty coverage data).
    string(
      REGEX MATCH "^[0-9]+" GCC_MAJOR_VERSION
      "${CMAKE_CXX_COMPILER_VERSION}"
      )
    find_program(
      GCOV
      NAMES "gcov-${GCC_MAJOR_VERSION}" "gcov"
      REQUIRED
      )
  endif()

  find_program(LCOV lcov REQUIRED)
  find_program(GENHTML genhtml REQUIRED)
  find_program(SED sed REQUIRED)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")

  # Enable code coverage analysis.
  SET(CODE_COVERAGE ON)

  # Make flag enabling code coverage analysis available in parent cmake scope
  mark_as_advanced(CODE_COVERAGE)

  # Only include code coverage cmake functions if all prerequisites are met
  include(CodeCoverage)
elseif(COVERAGE)
  message(FATAL_ERROR "Failed to enable code coverage analysis. Not all "
    "prerequisites found: gcov:${GCOV}, lcov:${LCOV}, genhtml:${GENHTML},"
    " sed:${SED}")
endif()
