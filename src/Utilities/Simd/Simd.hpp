// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#ifdef SPECTRE_USE_XSIMD
#include <xsimd/xsimd.hpp>

/// Namespace containing SIMD functions based on XSIMD.
namespace simd = xsimd;

namespace MakeWithValueImpls {
template <typename U, typename T>
struct MakeWithValueImpl<simd::batch<U>, T> {
  static SPECTRE_ALWAYS_INLINE simd::batch<U> apply(const T& /* input */,
                                                    const U value) {
    return simd::batch<U>(value);
  }
};
}  // namespace MakeWithValueImpls
#endif
