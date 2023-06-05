// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace sys {
/*!
 * \brief Info about the CPU cache.
 */
struct CacheInfo {
  /// The level of the cache, 1, 2, or 3 representing the L1, L2, or L3 cache.
  uint64_t level;
  /// The size of the cache in bytes.
  uint64_t size;
  /// The linesize of the cache in bytes.
  ///
  /// This can be particularly important for parallel programming to avoid false
  /// sharing.
  uint64_t linesize;
};

namespace detail {
std::array<CacheInfo, 3> cache_info();
}  // namespace detail

enum class CacheLevel : int {
  /// L1 cache level. This is nearest to the CPU and generally quite small,
  /// e.g. 32kB.
  L1 = 0,
  /// L2 cache level. Second closest to the CPU and split between instruction
  /// and data as well as two hyperthreads. Typical size can be as large as
  /// 512kB total, but check for your CPU.
  L2 = 1,
  /// L3 cache level. Furthest from the CPU and split between instruction
  /// and data as well as several cores (e.g. 4 on AMD Threadripper 3970X).
  /// Typical size is can be as large as 16MB total, but check for your CPU.
  L3 = 2
};

/*!
 * \brief Get cache size and linesize in bytes for either the level 1, 2, or 3
 * cache.
 */
inline CacheInfo cache_info(const CacheLevel level) {
  static const auto info = detail::cache_info();
  return info[static_cast<size_t>(level)];
}

/*!
 * \brief The cache line size.
 *
 * This is 64 bytes on the vast majority of systems. IF we need to, we can
 * have CMake figure this out on different hardware, but since the hardware
 * code is compiled on doesn't need to exactly match the runtime hardware,
 * this could be an issue.
 */
static constexpr uint64_t compile_time_cache_line_size = 64;

/*!
 * \brief The L1 cache line size.
 *
 * This is typically the relevant line size since it's how much data is loaded
 * into the CPU at once and so causes problems with false
 * sharing. Additionally, the L2 and L3 cache line sizes are typically the same.
 */
static const uint64_t cache_line_size = []() {
  return cache_info(CacheLevel::L1).linesize;
}();
}  // namespace sys
