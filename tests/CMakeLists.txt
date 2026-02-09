# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(support)
add_subdirectory(tools)
if ("${SPECTRE_PARALLEL_LIB}" STREQUAL "charm")
  add_subdirectory(InputFiles)
endif()
add_subdirectory(Unit)
