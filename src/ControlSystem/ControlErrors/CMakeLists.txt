# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Size)

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  GridCenters.cpp
  Size.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Expansion.hpp
  GridCenters.hpp
  Rotation.hpp
  Shape.hpp
  Size.hpp
  Translation.hpp
  )
