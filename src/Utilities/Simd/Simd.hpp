// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#ifdef SPECTRE_USE_XSIMD
#include <xsimd/xsimd.hpp>

/// Namespace containing SIMD functions based on XSIMD.
namespace simd = xsimd;

namespace MakeWithValueImpls {
template <typename U, typename T, typename Arch>
struct MakeWithValueImpl<xsimd::batch<U, Arch>, T> {
  static SPECTRE_ALWAYS_INLINE xsimd::batch<U, Arch> apply(const T& /* input */,
                                                           const U value) {
    return xsimd::batch<U, Arch>(value);
  }
};
}  // namespace MakeWithValueImpls
#endif
