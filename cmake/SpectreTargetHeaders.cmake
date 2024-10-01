# Distributed under the MIT License.
# See LICENSE.txt for details.

# Add header files for CMake to track for a given library
# or executable.
#
# The header and source files together are used to track dependencies between
# targets automatically in SpECTRE. Specifically, CI checks that the
# dependencies of a target supply all the header files used in a library.
# While this checking still requires includes to be correct (be they checked
# manually or automatically), it significantly lowers the burden of correct
# dependency management on developers.
#
# Additionally, some IDEs perform better if the header files for libraries
# are known.
#
# Usage:
#  spectre_target_headers(
#    TARGET_NAME
#    INCLUDE_DIRECTORY /path/to/include/relative/to
#    HEADERS
#    A.hpp
#    B.hpp
#    B.tpp
#    C.hpp)
#
# Arguments:
# - TARGET_NAME: the name of the library or executable
# - INCLUDE_DIRECTORY: the directory relative to which the header
#                      files must be included in C++ files. E.g.
#                      `${CMAKE_SOURCE_DIR}/src` if the include paths are
#                      relative to `${CMAKE_SOURCE_DIR}/src`.
# - HEADERS: the HEADERS argument is a list of all the headers files
#            to append, including tpp files, relative to the current
#            source directory.
function(spectre_target_headers TARGET_NAME)
  cmake_parse_arguments(
    ARG "" "INCLUDE_DIRECTORY" "HEADERS"
    ${ARGN})

  get_target_property(
    TARGET_TYPE
    ${TARGET_NAME}
    TYPE
    )

  if(NOT ARG_INCLUDE_DIRECTORY)
    message(FATAL_ERROR
      "Must specify the include directory relative to which the "
      "header files for a library will be included when calling "
      "spectre_target_headers. The named argument is INCLUDE_DIRECTORY. "
      "Call was made for target ${TARGET_NAME}.")
  endif(NOT ARG_INCLUDE_DIRECTORY)
  # Note: Search the list because an empty HEADERS is okay.
  list(FIND ARGN "HEADERS" _FOUND_HEADERS)
  if(${_FOUND_HEADERS} EQUAL -1)
    message(FATAL_ERROR
      "No HEADERS section specified in call to spectre_target_headers "
      "for target ${TARGET_NAME}. You must specificy HEADERS before "
      "listing the header files.")
  endif()

  if(NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
    get_property(
      INCLUDE_DIRS
      TARGET ${TARGET_NAME}
      PROPERTY INCLUDE_DIRECTORIES
      )
    if(NOT ${ARG_INCLUDE_DIRECTORY} IN_LIST INCLUDE_DIRS)
      set_property(
        TARGET ${TARGET_NAME}
        APPEND
        PROPERTY INCLUDE_DIRECTORIES ${ARG_INCLUDE_DIRECTORY}
        )
    endif(NOT ${ARG_INCLUDE_DIRECTORY} IN_LIST INCLUDE_DIRS)
  endif(NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)

  get_property(
    INTERFACE_INCLUDE_DIRS
    TARGET ${TARGET_NAME}
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    )
  if(NOT ${ARG_INCLUDE_DIRECTORY} IN_LIST INTERFACE_INCLUDE_DIRS)
    set_property(
      TARGET ${TARGET_NAME}
      APPEND
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ARG_INCLUDE_DIRECTORY}
      )
  endif(NOT ${ARG_INCLUDE_DIRECTORY} IN_LIST INTERFACE_INCLUDE_DIRS)

  unset(_HEADER_FILES)
  foreach(HEADER ${ARG_HEADERS})
    if(NOT HEADER STREQUAL "PRIVATE" AND
        NOT HEADER STREQUAL "PUBLIC" AND
        NOT HEADER STREQUAL "INTERFACE" AND
        NOT IS_ABSOLUTE "${HEADER}")
      set(_ABSOLUTE_PATH "${CMAKE_CURRENT_LIST_DIR}/${HEADER}")
      file(RELATIVE_PATH HEADER ${ARG_INCLUDE_DIRECTORY} ${_ABSOLUTE_PATH})
    endif()
    list(APPEND _HEADER_FILES ${HEADER})
  endforeach()

  if(${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
    add_interface_lib_headers(
      TARGET ${TARGET_NAME}
      HEADERS
      ${_HEADER_FILES}
      )
  else(${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
    set_property(
      TARGET ${TARGET_NAME}
      APPEND
      PROPERTY
      PUBLIC_HEADER
      ${_HEADER_FILES}
      )
  endif(${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
endfunction(spectre_target_headers TARGET_NAME)
