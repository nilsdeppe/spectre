// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <atomic>

#include "Utilities/ErrorHandling/Assert.hpp"

namespace Parallel {
/*!
 * \brief a Multi-produce no-consumer array.
 *
 * This allay allows threadsafe wait-free insertion, but does not provide a
 * threadsafe way to read the data. This means that the user must ensure that
 * no thread will modify the data in the array, including inserts, when a
 * consumer starts using it.
 */
template <typename T, size_t Capacity>
class MpncArray {
 public:
  size_t add_back() { return size_.fetch_add(1, std::memory_order_acq_rel); }

  T& operator[](const size_t index) {
    ASSERT(index < size_.load(std::memory_order_acquire), "");
    return data_[index];
  }

  const T& operator[](const size_t index) const {
    ASSERT(index < size_.load(std::memory_order_acquire), "");
    return data_[index];
  }

  T& access_without_size_check(const size_t index) {
    ASSERT(index < Capacity, "");
    return data_[index];
  }

  const T& access_without_size_check(const size_t index) const {
    ASSERT(index < Capacity, "");
    return data_[index];
  }

  size_t size() const { return size_.load(std::memory_order_relaxed); }

 private:
  std::atomic<size_t> size_{0};
  std::array<T, Capacity> data_{};
};
}  // namespace Parallel
