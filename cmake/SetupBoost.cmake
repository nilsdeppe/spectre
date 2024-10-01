# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Boost 1.60.0 REQUIRED COMPONENTS program_options)

# CMake versions don't set this consistently
set(Boost_VERSION "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")

message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
message(STATUS "Boost include: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost version: ${Boost_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "Boost version: ${Boost_VERSION}\n"
  )

# Boost organizes targets as:
# - Boost::boost is the header-only parts of Boost
# - Boost::COMPONENT are the components that need linking, e.g. program_options

add_interface_lib_headers(
  TARGET Boost::boost
  HEADERS
  boost/algorithm/string.hpp
  boost/algorithm/string/join.hpp
  boost/config.hpp
  boost/core/demangle.hpp
  boost/functional/hash.hpp
  boost/integer/common_factor_rt.hpp
  boost/iterator/transform_iterator.hpp
  boost/iterator/zip_iterator.hpp
  boost/make_shared.hpp
  boost/math/interpolators/barycentric_rational.hpp
  boost/math/quaternion.hpp
  boost/math/special_functions/binomial.hpp
  boost/math/special_functions/sign.hpp
  boost/math/tools/roots.hpp
  boost/multi_array.hpp
  boost/multi_array/base.hpp
  boost/multi_array/extent_gen.hpp
  boost/numeric/odeint.hpp
  boost/numeric/odeint/integrate/integrate_adaptive.hpp>
  boost/numeric/odeint/stepper/controlled_runge_kutta.hpp
  boost/numeric/odeint/stepper/dense_output_runge_kutta.hpp
  boost/numeric/odeint/stepper/generation/make_dense_output.hpp
  boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp
  boost/parameter/name.hpp
  boost/preprocessor.hpp
  boost/preprocessor/arithmetic/dec.hpp
  boost/preprocessor/arithmetic/inc.hpp
  boost/preprocessor/arithmetic/sub.hpp
  boost/preprocessor/control/expr_iif.hpp
  boost/preprocessor/control/iif.hpp
  boost/preprocessor/control/while.hpp
  boost/preprocessor/list/adt.hpp
  boost/preprocessor/list/fold_left.hpp
  boost/preprocessor/list/fold_right.hpp
  boost/preprocessor/list/for_each.hpp
  boost/preprocessor/list/for_each_product.hpp
  boost/preprocessor/list/size.hpp
  boost/preprocessor/list/to_tuple.hpp
  boost/preprocessor/list/transform.hpp
  boost/preprocessor/logical/bitand.hpp
  boost/preprocessor/logical/bool.hpp
  boost/preprocessor/logical/compl.hpp
  boost/preprocessor/punctuation/comma_if.hpp
  boost/preprocessor/repetition/for.hpp
  boost/preprocessor/repetition/repeat.hpp
  boost/preprocessor/tuple/elem.hpp
  boost/preprocessor/tuple/enum.hpp
  boost/preprocessor/tuple/reverse.hpp
  boost/preprocessor/tuple/size.hpp
  boost/preprocessor/tuple/to_list.hpp
  boost/preprocessor/variadic/elem.hpp
  boost/preprocessor/variadic/to_list.hpp
  boost/range/combine.hpp
  boost/rational.hpp
  boost/shared_ptr.hpp
  boost/stacktrace.hpp
  boost/stacktrace/stacktrace.hpp
  boost/stacktrace/stacktrace_fwd.hpp
  boost/tuple/tuple.hpp
  boost/tuple/tuple_comparison.hpp
  boost/variant.hpp
  boost/variant/get.hpp
  boost/variant/variant.hpp
  )

set_property(
  TARGET Boost::program_options
  APPEND PROPERTY PUBLIC_HEADER
  boost/program_options.hpp
  )

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Boost::boost Boost::program_options
  )

# We disable thread safety of Boost::shared_ptr since it makes them faster
# to use and we do not share them between threads. If a thread-safe
# shared_ptr is desired it must be implemented to work with Charm++'s threads
# anyway.
set_property(TARGET Boost::boost
  APPEND PROPERTY
  INTERFACE_COMPILE_DEFINITIONS
  $<$<COMPILE_LANGUAGE:CXX>:BOOST_SP_DISABLE_THREADS>)

# With newer versions of boost, sometimes there are internal boost warnings
# about deprecated headers. This disables those as we have no control over what
# boost does internally.
set_property(TARGET Boost::boost
  APPEND PROPERTY
  INTERFACE_COMPILE_DEFINITIONS
  $<$<COMPILE_LANGUAGE:CXX>:BOOST_ALLOW_DEPRECATED_HEADERS>)

# Override the boost index type to match the STL for Boost.MultiArray
# (std::ptrdiff_t to std::size_t)
# Note: This header guard changed in Boost 1.73.0
if(Boost_VERSION VERSION_GREATER_EQUAL 1.73.0)
  set(BOOST_MULTI_ARRAY_TYPES_HEADER_GUARD BOOST_MULTI_ARRAY_TYPES_HPP)
else()
  set(BOOST_MULTI_ARRAY_TYPES_HEADER_GUARD BOOST_MULTI_ARRAY_TYPES_RG071801_HPP)
endif()
set_property(TARGET Boost::boost
  APPEND PROPERTY
  INTERFACE_COMPILE_DEFINITIONS
  $<$<COMPILE_LANGUAGE:CXX>:${BOOST_MULTI_ARRAY_TYPES_HEADER_GUARD}>)

# Work around boost not building with clang 15 (fixed in boost 1.83.0)
# (https://github.com/boostorg/functional/pull/21)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND BOOST_VERSION VERSION_LESS 1.83.0)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "15.0.0" OR
      CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "15.0.0")
        target_compile_definitions(Boost::boost INTERFACE _HAS_AUTO_PTR_ETC=0)
  endif()
endif()

